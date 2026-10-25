"""Tests for rl/train.py. Run with: uv run pytest rl/test_train.py -q

Nothing here trains: `algos.build` and `algos.load` are replaced by a fake agent, so the
tests pin the warm-start wiring rather than any learning behaviour.

The one SB3 behaviour worth stating, because it is invisible and wrong by default:
`DQN.exploration_rate` is recomputed from `_current_progress_remaining`, which every
`learn()` call resets. A warm-started chain would therefore restart exploration at
`exploration_initial_eps` (1.0) on every fold and re-randomise itself.
"""
import pickle
from datetime import datetime

import pandas as pd
import pytest

from degradation.schade import SchadeDegradation
from rl import algos, train
from rl.features import PriceScaler

CAP_2024, CAP_2025 = 17500.0, 20300.0


@pytest.fixture
def prices():
    index = pd.date_range(datetime(2024, 1, 10, 4, 5), periods=24, freq="5min")
    return pd.Series([20.0, 15.0, -50.0, 10.0, 300.0, 25.0] * 4, index=index, name="RRP")


@pytest.fixture
def battery():
    return SchadeDegradation(0.060, 0.100, 150000.0)


class FakeAgent:
    """Stands in for an SB3 agent. On-policy by default: no exploration schedule."""

    def __init__(self):
        self.learnedFor = None
        self.savedTo = None
        self.seededWith = None

    def learn(self, total_timesteps, progress_bar=False):
        self.learnedFor = total_timesteps

    def save(self, path):
        self.savedTo = str(path)

    def set_random_seed(self, seed):
        self.seededWith = seed


class FakeOffPolicyAgent(FakeAgent):
    """An agent with DQN's exploration schedule, which resets on every learn() call."""

    def __init__(self):
        super().__init__()
        self.exploration_final_eps = 0.05
        self.exploration_schedule = lambda remaining: 1.0 * remaining


@pytest.fixture
def spy(monkeypatch):
    """Capture how trainOne reached its agent, without building a real one."""
    calls = {"build": [], "load": []}

    def fakeBuild(name, env, seed=0, **overrides):
        calls["build"].append({"name": name, "seed": seed, "overrides": overrides})
        return calls.setdefault("agent", FakeAgent())

    def fakeLoad(name, path, env=None):
        calls["load"].append({"name": name, "path": str(path), "env": env})
        return calls.setdefault("agent", FakeAgent())

    monkeypatch.setattr(algos, "build", fakeBuild)
    monkeypatch.setattr(algos, "load", fakeLoad)
    return calls


def runTrain(prices, battery, tmp_path, **kwargs):
    kwargs.setdefault("scaler", PriceScaler(CAP_2025))
    kwargs.setdefault("horizon", 12)
    kwargs.setdefault("timesteps", 10)
    return train.trainOne("maskableppo", prices, battery, outDir=str(tmp_path), **kwargs)


def parentCheckpoint(tmp_path, cap=CAP_2025):
    """A previous fold's directory, as trainOne would have left it."""
    directory = tmp_path / "fold00" / "maskableppo_seed0"
    directory.mkdir(parents=True)
    train.saveScaler(directory, PriceScaler(cap))
    (directory / "agent.zip").write_bytes(b"")
    return directory


class TestColdStart:
    # A run with no parent builds a fresh agent and never reaches for a checkpoint.
    def test_builds_rather_than_loads(self, prices, battery, tmp_path, spy):
        runTrain(prices, battery, tmp_path)
        assert len(spy["build"]) == 1 and spy["load"] == []

    # The checkpoint path returned is the one the agent was told to write.
    def test_returns_the_path_it_wrote(self, prices, battery, tmp_path, spy):
        agent, path = runTrain(prices, battery, tmp_path)
        assert agent.savedTo == str(path)
        assert path.name == "agent"

    # Every checkpoint records the transform it trained under, so the next fold can
    # check it and so evaluation cannot silently rescale the observation.
    def test_writes_its_own_scaler(self, prices, battery, tmp_path, spy):
        _, path = runTrain(prices, battery, tmp_path)
        assert train.loadScaler(path.parent) == PriceScaler(CAP_2025)

    def test_a_scaler_without_a_scale_reads_back_as_one(self, tmp_path):
        """Every agent published before config.price_scale trained under a cap that
        mapped to 1.0. Reading it back at today's default would silently restate every
        price it learned."""
        (tmp_path / "scaler.json").write_text('{"cap": 23200.0}')
        assert train.loadScaler(tmp_path) == PriceScaler(23200.0, scale=1.0)


class TestWarmStart:
    # A run with a parent loads that checkpoint instead of building a new agent.
    def test_loads_the_parent_checkpoint(self, prices, battery, tmp_path, spy):
        parent = parentCheckpoint(tmp_path)
        runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert spy["build"] == []
        assert spy["load"][0]["path"] == str(parent / "agent")
        assert spy["load"][0]["env"] is not None

    # The chain is reseeded per fold so two seeds do not collapse onto one trajectory.
    def test_reseeds_the_agent(self, prices, battery, tmp_path, spy):
        parent = parentCheckpoint(tmp_path)
        agent, _ = runTrain(prices, battery, tmp_path / "out", warmFrom=parent, seed=3)
        assert agent.seededWith == 3

    # A parent trained under a different market price cap means every observation value
    # denotes a different price, so the chain must break loudly rather than continue.
    def test_refuses_a_parent_with_a_different_price_cap(self, prices, battery,
                                                         tmp_path, spy):
        parent = parentCheckpoint(tmp_path, cap=CAP_2024)
        with pytest.raises(ValueError) as raised:
            runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert "17500" in str(raised.value) and "20300" in str(raised.value)

    # A missing parent is a broken chain, and the message has to say which directory.
    def test_refuses_a_missing_parent(self, prices, battery, tmp_path, spy):
        missing = tmp_path / "fold99" / "maskableppo_seed0"
        with pytest.raises(FileNotFoundError, match="fold99"):
            runTrain(prices, battery, tmp_path / "out", warmFrom=missing)

    # algos.load takes no hyperparameter channel, so an override passed here would be
    # dropped without a word and the fold would train under the wrong settings.
    def test_refuses_hyperparameter_overrides(self, prices, battery, tmp_path, spy):
        parent = parentCheckpoint(tmp_path)
        with pytest.raises(ValueError, match="override"):
            runTrain(prices, battery, tmp_path / "out", warmFrom=parent,
                     learning_rate=1e-4)


class TestHoldExploration:
    # An off-policy agent resumes at the epsilon its chain reached, rather than jumping
    # back to exploration_initial_eps and re-randomising itself every fold.
    def test_an_off_policy_agent_holds_its_epsilon(self):
        agent = FakeOffPolicyAgent()
        assert agent.exploration_schedule(1.0) == 1.0
        algos.holdExploration(agent)
        assert agent.exploration_schedule(1.0) == 0.05
        assert agent.exploration_schedule(0.0) == 0.05

    # The schedule is pickled into the checkpoint, so it must not reference the agent.
    # A closure over `agent` reaches its env and the open monitor file, and save() dies
    # with "Cannot pickle files that are not opened for reading".
    def test_the_held_schedule_does_not_capture_the_agent(self):
        agent = FakeOffPolicyAgent()
        algos.holdExploration(agent)
        restored = pickle.loads(pickle.dumps(agent.exploration_schedule))
        assert restored(1.0) == 0.05

    # An on-policy agent has no schedule to hold, and must be left alone.
    def test_an_on_policy_agent_is_untouched(self):
        agent = FakeAgent()
        algos.holdExploration(agent)
        assert not hasattr(agent, "exploration_schedule")

    # The hold is applied on the warm path, so a resumed DQN does not re-randomise.
    def test_a_warm_start_applies_the_hold(self, prices, battery, tmp_path, monkeypatch):
        parent = parentCheckpoint(tmp_path)
        agent = FakeOffPolicyAgent()
        monkeypatch.setattr(algos, "load", lambda name, path, env=None: agent)
        runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert agent.exploration_schedule(1.0) == 0.05

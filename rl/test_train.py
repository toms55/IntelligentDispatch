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
from pathlib import Path

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


class FakeBuffer:
    def __init__(self, size=10, filled=0):
        self.buffer_size = size
        self.pos = filled


class FakeOffPolicyAgent(FakeAgent):
    """An agent with DQN's exploration schedule, which resets on every learn() call,
    and its replay buffer, which save() leaves behind."""

    def __init__(self, bufferSize=10):
        super().__init__()
        self.exploration_final_eps = 0.05
        self.exploration_schedule = lambda remaining: 1.0 * remaining
        self.replay_buffer = FakeBuffer(bufferSize, filled=4)
        self.loadedBufferFrom = None

    def save_replay_buffer(self, path):
        Path(path).write_bytes(pickle.dumps(self.replay_buffer))

    def load_replay_buffer(self, path):
        self.replay_buffer = pickle.loads(Path(path).read_bytes())
        self.loadedBufferFrom = str(path)


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


def parentCheckpoint(tmp_path, cap=CAP_2025, buffer=None):
    """A previous fold's directory, as trainOne would have left it. `buffer` writes the
    replay buffer an off-policy parent leaves beside its weights."""
    directory = tmp_path / "fold00" / "maskableppo_seed0"
    directory.mkdir(parents=True)
    train.saveScaler(directory, PriceScaler(cap))
    (directory / "agent.zip").write_bytes(b"")
    if buffer is not None:
        (directory / train.replayFile).write_bytes(pickle.dumps(buffer))
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


class TestReplayBuffer:
    """SB3's save() excludes the replay buffer, so a warm-started DQN would keep its
    weights and optimiser but relearn from an empty buffer on every fold. MaskablePPO is
    on-policy and loses nothing, so without this the two algorithms do not mean the same
    thing by "warm"."""

    def test_an_off_policy_agent_writes_its_buffer(self, prices, battery, tmp_path,
                                                   monkeypatch):
        agent = FakeOffPolicyAgent()
        monkeypatch.setattr(algos, "build", lambda name, env, seed=0, **kw: agent)
        _, path = runTrain(prices, battery, tmp_path)
        assert (path.parent / train.replayFile).exists()

    def test_an_on_policy_agent_writes_none(self, prices, battery, tmp_path, spy):
        _, path = runTrain(prices, battery, tmp_path)
        assert not (path.parent / train.replayFile).exists()

    def test_a_warm_start_carries_the_parents_buffer(self, prices, battery, tmp_path,
                                                     monkeypatch):
        parent = parentCheckpoint(tmp_path, buffer=FakeBuffer(size=10, filled=7))
        agent = FakeOffPolicyAgent()
        monkeypatch.setattr(algos, "load", lambda name, path, env=None: agent)
        runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert agent.loadedBufferFrom == str(parent / train.replayFile)
        assert agent.replay_buffer.pos == 7

    # A chain from before the buffer travelled has none. Continuing it would silently
    # restart DQN's experience, which is the whole failure this exists to remove.
    def test_refuses_a_parent_with_no_buffer(self, prices, battery, tmp_path,
                                             monkeypatch):
        parent = parentCheckpoint(tmp_path, buffer=None)
        agent = FakeOffPolicyAgent()
        monkeypatch.setattr(algos, "load", lambda name, path, env=None: agent)
        with pytest.raises(FileNotFoundError) as raised:
            runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert train.replayFile in str(raised.value)

    # The size is pickled into the buffer, so an old one would override the configured
    # value without a word.
    def test_refuses_a_buffer_of_another_size(self, prices, battery, tmp_path,
                                              monkeypatch):
        parent = parentCheckpoint(tmp_path, buffer=FakeBuffer(size=99))
        agent = FakeOffPolicyAgent(bufferSize=10)
        monkeypatch.setattr(algos, "load", lambda name, path, env=None: agent)
        with pytest.raises(ValueError) as raised:
            runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert "99" in str(raised.value) and "10" in str(raised.value)

    # An on-policy parent never wrote one, so nothing should be looked for.
    def test_an_on_policy_warm_start_wants_no_buffer(self, prices, battery, tmp_path,
                                                     spy):
        parent = parentCheckpoint(tmp_path, buffer=None)
        runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert spy["load"][0]["path"] == str(parent / "agent")


class TestMonitorColumns:
    # The out-of-bounds count per episode, so a training run shows whether the penalty
    # bites while it runs rather than only at the end.
    def test_monitor_logs_the_out_of_bounds_count(self, prices, battery, tmp_path, spy):
        _, path = runTrain(prices, battery, tmp_path)
        header = (path.parent / "monitor.csv").read_text().splitlines()[1]
        assert "outOfBoundsTotal" in header


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
        # An off-policy parent always leaves its buffer, and the warm start reads it.
        parent = parentCheckpoint(tmp_path, buffer=FakeBuffer())
        agent = FakeOffPolicyAgent()
        monkeypatch.setattr(algos, "load", lambda name, path, env=None: agent)
        runTrain(prices, battery, tmp_path / "out", warmFrom=parent)
        assert agent.exploration_schedule(1.0) == 0.05

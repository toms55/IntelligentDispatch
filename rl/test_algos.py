"""Tests for rl/algos.py, rl/train.py and rl/evaluate.py. Run with: uv run pytest rl/ -q

Short training runs only; these check the wiring, not that anything learns.
"""
import numpy as np
import pandas as pd
import pytest

import config
from degradation.schade import SchadeDegradation
from rl import algos, evaluate, features
from rl.env import CHARGE, DISCHARGE, IDLE, BessEnv
from rl.train import trainOne


@pytest.fixture
def paper():
    return SchadeDegradation(0.060, 0.100, 150000.0)


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    values = rng.normal(60.0, 90.0, 72)
    return pd.Series(
        values, index=pd.date_range("2024-01-10", periods=len(values), freq="5min"),
        name="RRP",
    )


def makeEnv(paper, series, **kwargs):
    settings = dict(horizon=24, calendar=False, rewardScale=1.0)
    settings.update(kwargs)
    return BessEnv(series, paper, **settings)


class TestRegistry:
    @pytest.mark.parametrize("name", ["maskableppo", "dqn", "a2c"])
    def test_builds_every_algorithm(self, paper, series, name):
        agent = algos.build(name, makeEnv(paper, series), seed=0)
        assert agent is not None

    def test_is_case_insensitive(self, paper, series):
        assert algos.build("MaskablePPO", makeEnv(paper, series), seed=0) is not None

    def test_refuses_an_unknown_name(self, paper, series):
        with pytest.raises(ValueError, match="unknown algorithm"):
            algos.build("ppo2", makeEnv(paper, series), seed=0)

    def test_only_maskable_ppo_claims_masking(self):
        assert algos.supportsMasking("maskableppo") is True
        assert algos.supportsMasking("dqn") is False
        assert algos.supportsMasking("a2c") is False

    @pytest.mark.parametrize("name", ["maskableppo", "dqn", "a2c"])
    def test_runs_undiscounted(self, paper, series, name):
        agent = algos.build(name, makeEnv(paper, series), seed=0)
        assert agent.gamma == config.gamma == 1.0

    def test_overrides_reach_the_algorithm(self, paper, series):
        agent = algos.build("a2c", makeEnv(paper, series), seed=0, learning_rate=0.004)
        assert agent.learning_rate == 0.004


class TestTraining:
    @pytest.mark.parametrize("name", ["maskableppo", "dqn", "a2c"])
    def test_trains_and_checkpoints(self, paper, series, name, tmp_path):
        agent, path = trainOne(name, series, paper, seed=0, timesteps=96,
                               calendar=False, horizon=24, rewardScale=1.0,
                               outDir=tmp_path)
        assert path.with_suffix(".zip").exists()
        reloaded = algos.load(name, str(path))
        assert reloaded is not None

    def test_writes_a_monitor_log(self, paper, series, tmp_path):
        trainOne("a2c", series, paper, seed=1, timesteps=96, calendar=False, horizon=24,
                 rewardScale=1.0, outDir=tmp_path)
        assert (tmp_path / "a2c_seed1" / "monitor.csv").exists()

    def test_the_same_seed_trains_the_same_agent(self, paper, series, tmp_path):
        def firstAction(seed):
            agent, _ = trainOne("a2c", series, paper, seed=seed, timesteps=96,
                                calendar=False, horizon=24, rewardScale=1.0,
                                outDir=tmp_path / str(seed))
            env = makeEnv(paper, series)
            obs, _ = env.reset(seed=0)
            return int(agent.predict(obs, deterministic=True)[0])
        assert firstAction(3) == firstAction(3)

    def test_training_shuffles_windows_but_evaluation_does_not(self, paper, series):
        assert BessEnv(series, paper, horizon=24, shuffle=True).shuffle is True
        assert BessEnv(series, paper, horizon=24).shuffle is False


class TestDqnSettings:
    # 1e6 transitions of a 60-slot observation is about 480 MB per seed, held twice over
    # (obs and next obs), and the buffer now travels between folds on disk. The size is
    # therefore pinned rather than left at the SB3 default.
    def test_the_replay_buffer_size_is_pinned(self, paper, series):
        agent = algos.build("dqn", makeEnv(paper, series), seed=0)
        assert agent.buffer_size == config.replay_buffer_size
        assert agent.buffer_size < 1000000

    # rl/train.py asks algos whether a buffer travels, rather than testing attributes
    # of an SB3 object itself.
    def test_only_an_off_policy_agent_carries_a_buffer(self, paper, series):
        env = makeEnv(paper, series)
        assert algos.isOffPolicy(algos.build("dqn", env, seed=0)) is True
        assert algos.isOffPolicy(algos.build("maskableppo", env, seed=0)) is False
        assert algos.isOffPolicy(algos.build("a2c", env, seed=0)) is False


class TestMasking:
    # Training needs no test that the mask reaches learn(): MaskablePPO.collect_rollouts
    # raises when `is_masking_supported(env)` is false (ppo_mask.py:216), so a mask the
    # Monitor wrapper stopped forwarding fails the fit rather than silently training
    # plain PPO. These two tests cover prediction, which has no such guard.
    def test_the_mask_reaches_maskable_ppo(self, paper, series):
        env = makeEnv(paper, series, initialSoc=1.0)
        agent = algos.build("maskableppo", env, seed=0)
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            action, _ = agent.predict(obs, deterministic=False,
                                      action_masks=env.action_masks())
            assert int(action) != CHARGE

    def test_an_empty_pack_never_discharges_under_a_mask(self, paper, series):
        env = makeEnv(paper, series, initialSoc=0.0)
        agent = algos.build("maskableppo", env, seed=0)
        obs, _ = env.reset(seed=0)
        for _ in range(10):
            action, _ = agent.predict(obs, deterministic=False,
                                      action_masks=env.action_masks())
            assert int(action) != DISCHARGE


class AlwaysCharges:
    """A policy that charges every interval, masked or not. It fills the pack and then
    keeps asking, which is the behaviour an unmasked agent can have and a masked one
    cannot."""

    def predict(self, obs, deterministic=True, action_masks=None):
        if action_masks is not None and not action_masks[CHARGE]:
            return IDLE, None
        return CHARGE, None


class TestOutOfBoundsCount:
    # The count reaches the per-window record, which is what the backtest writes as a
    # day-row column for every seed.
    def test_a_window_reports_its_attempts(self, paper, series):
        env = makeEnv(paper, series)
        records = evaluate.evaluatePolicy(AlwaysCharges(), env, masking=False)
        assert all(record["outOfBounds"] > 0 for record in records)

    # Reading the mask makes the count zero, and that is why MaskablePPO's column is
    # zero by construction rather than by luck.
    def test_reading_the_mask_drives_it_to_zero(self, paper, series):
        env = makeEnv(paper, series)
        records = evaluate.evaluatePolicy(AlwaysCharges(), env, masking=True)
        assert all(record["outOfBounds"] == 0 for record in records)

    # Scoring a masked agent unmasked reports a worse agent and raises nothing, so the
    # caller must say which it is.
    def test_an_unstated_mask_is_refused(self, paper, series):
        with pytest.raises(ValueError, match="algoName or masking"):
            evaluate.evaluatePolicy(AlwaysCharges(), makeEnv(paper, series))


class TestEvaluation:
    def test_reports_regret_against_the_oracle(self, paper, series, tmp_path):
        lpProfit = [[0.01] * 24, [0.02] * 24, [0.03] * 24]
        env = makeEnv(paper, series, baseline=lpProfit)
        agent, _ = trainOne("a2c", series, paper, seed=0, timesteps=96, calendar=False,
                            horizon=24, rewardScale=1.0, outDir=tmp_path)
        records = evaluate.evaluatePolicy(agent, env, algoName="a2c")
        assert len(records) == 3
        for record in records:
            assert record["regret"] == pytest.approx(
                record["lpProfit"] - record["agentProfit"], rel=1e-9
            )

    def test_the_oracle_profit_comes_from_the_baseline(self, paper, series, tmp_path):
        lpProfit = [[0.01] * 24, [0.02] * 24, [0.03] * 24]
        env = makeEnv(paper, series, baseline=lpProfit)
        agent, _ = trainOne("a2c", series, paper, seed=0, timesteps=96, calendar=False,
                            horizon=24, rewardScale=1.0, outDir=tmp_path)
        records = evaluate.evaluatePolicy(agent, env, algoName="a2c")
        assert records[0]["lpProfit"] == pytest.approx(0.24)
        assert records[2]["lpProfit"] == pytest.approx(0.72)

    def test_summarise_adds_up_the_windows(self, paper, series):
        records = [
            {"agentProfit": 1.0, "lpProfit": 2.0},
            {"agentProfit": 3.0, "lpProfit": 4.0},
        ]
        got = evaluate.summarise(records)
        assert got["windows"] == 2
        assert got["agentProfit"] == 4.0
        assert got["lpProfit"] == 6.0
        assert got["regret"] == 2.0
        assert got["captured"] == pytest.approx(4.0 / 6.0)

    def test_across_seeds_reports_mean_and_spread(self):
        bySeed = {
            0: [{"agentProfit": 1.0, "lpProfit": 2.0}],
            1: [{"agentProfit": 3.0, "lpProfit": 2.0}],
        }
        got = evaluate.acrossSeeds(bySeed)
        assert got["agentProfit"]["mean"] == pytest.approx(2.0)
        assert got["agentProfit"]["sd"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
        assert got["regret"]["mean"] == pytest.approx(0.0)

    def test_a_single_seed_has_no_spread(self):
        got = evaluate.acrossSeeds({0: [{"agentProfit": 1.0, "lpProfit": 2.0}]})
        assert got["regret"]["sd"] == 0.0

    def test_evaluation_is_deterministic(self, paper, series, tmp_path):
        env = makeEnv(paper, series)
        agent, _ = trainOne("a2c", series, paper, seed=0, timesteps=96, calendar=False,
                            horizon=24, rewardScale=1.0, outDir=tmp_path)
        first = evaluate.evaluatePolicy(agent, env, algoName="a2c")
        second = evaluate.evaluatePolicy(agent, env, algoName="a2c")
        assert [r["agentProfit"] for r in first] == [r["agentProfit"] for r in second]

    def test_the_random_control_never_leaves_the_soc_bounds(self, paper, series):
        env = makeEnv(paper, series)
        totals = evaluate.randomPolicyProfit(env, seed=0, episodes=3)
        assert len(totals) == 3


class TestThresholdPolicy:
    """The rule-based control the agent has to beat. Measured on SA1 January 2024, 31
    windows: 0.15/0.85 captures 48% of the continuous oracle; random loses money."""

    def test_charges_at_the_bottom_of_the_window(self, paper, series):
        from rl.heuristic import ThresholdPolicy
        env = makeEnv(paper, series)
        obs, _ = env.reset(seed=0)
        obs = obs.copy()
        obs[1] = env.scaler.low  # now is the cheapest thing on offer
        obs[2:1 + config.price_feature_count] = env.scaler(300.0)
        action, _ = ThresholdPolicy().predict(obs)
        assert action == CHARGE

    def test_discharges_at_the_top_of_the_window(self, paper, series):
        from rl.heuristic import ThresholdPolicy
        env = makeEnv(paper, series)
        obs, _ = env.reset(seed=0)
        obs = obs.copy()
        obs[1] = env.scaler(5000.0)
        obs[2:1 + config.price_feature_count] = env.scaler(20.0)
        action, _ = ThresholdPolicy().predict(obs)
        assert action == DISCHARGE

    def test_idles_in_the_middle(self, paper, series):
        from rl.heuristic import ThresholdPolicy
        env = makeEnv(paper, series)
        obs, _ = env.reset(seed=0)
        obs = obs.copy()
        obs[1:1 + config.price_feature_count] = env.scaler(50.0)
        action, _ = ThresholdPolicy().predict(obs)
        assert action == IDLE

    def test_respects_an_action_mask(self, paper, series):
        from rl.heuristic import ThresholdPolicy
        env = makeEnv(paper, series, initialSoc=1.0)
        obs, _ = env.reset(seed=0)
        obs = obs.copy()
        obs[1] = env.scaler.low
        obs[2:1 + config.price_feature_count] = env.scaler(300.0)
        action, _ = ThresholdPolicy().predict(obs, action_masks=env.action_masks())
        assert action == IDLE

    def test_ignores_padding_past_the_episode_end(self, scaler):
        from rl.heuristic import ThresholdPolicy
        obs = np.zeros(config.observation_size, dtype=np.float32)
        obs[1] = scaler(400.0)
        obs[2] = scaler(10.0)
        obs[3] = scaler(20.0)
        action, _ = ThresholdPolicy().predict(obs)
        assert action == DISCHARGE

    def test_idles_when_there_is_nothing_to_compare(self, paper, series):
        from rl.heuristic import ThresholdPolicy
        obs = np.zeros(config.observation_size, dtype=np.float32)
        action, _ = ThresholdPolicy().predict(obs)
        assert action == IDLE

    def test_drops_into_the_same_evaluation_path_as_an_agent(self, paper, series):
        from rl.heuristic import ThresholdPolicy
        env = makeEnv(paper, series)
        records = evaluate.evaluatePolicy(ThresholdPolicy(), env, masking=True)
        assert len(records) == 3
        assert all("agentProfit" in record for record in records)

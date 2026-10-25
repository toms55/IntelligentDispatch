"""Tests for rl/env.py. Run with: uv run pytest rl/ -q

TestMatchesTheOracle is the one to read first: everything downstream assumes it holds.
"""
import numpy as np
import pandas as pd
import pytest

import config
from degradation.schade import SchadeDegradation
from optimisation.lp import solve
from rl import features
from rl.env import CHARGE, DISCHARGE, IDLE, BessEnv


PAPER_MW = 0.060
PAPER_MWH = 0.100
PAPER_REPLACEMENT_COST = 150000.0

MIXED_PRICES = [20.0, 15.0, -50.0, 10.0, 300.0, 25.0, 18.0, 400.0, 12.0, -80.0, 9.0, 250.0]


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


def priceSeries(values, start="2024-01-10"):
    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="5min"), name="RRP"
    )


def makeEnv(paper, values=None, **kwargs):
    values = MIXED_PRICES if values is None else values
    settings = dict(horizon=len(values), calendar=False, rewardScale=1.0)
    settings.update(kwargs)
    return BessEnv(priceSeries(values), paper, **settings)


def actionsFrom(schedule, tolerance=1e-12):
    """The LP's schedule read back as agent actions."""
    return [
        CHARGE if boughtMWh > tolerance else (DISCHARGE if soldMWh > tolerance else IDLE)
        for boughtMWh, soldMWh in zip(schedule.bought, schedule.sold)
    ]


def rollout(env, actions):
    infos = []
    for action in actions:
        _, _, _, _, info = env.step(action)
        infos.append(info)
    return infos


class TestMatchesTheOracle:
    """Replay the oracle's own schedule; the environment must reproduce it exactly."""

    def test_soc_path_is_identical(self, paper):
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False, discretePower=True)
        env = makeEnv(paper)
        env.reset(seed=0)
        infos = rollout(env, actionsFrom(schedule))
        for got, want in zip([info["soc"] for info in infos], schedule.soc):
            assert got == pytest.approx(want, abs=1e-9)

    def test_degradation_is_identical(self, paper):
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False, discretePower=True)
        env = makeEnv(paper)
        env.reset(seed=0)
        infos = rollout(env, actionsFrom(schedule))
        assert sum(info["degradation"] for info in infos) == pytest.approx(
            schedule.degradationCost, rel=1e-9
        )

    def test_profit_is_identical(self, paper):
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False, discretePower=True)
        env = makeEnv(paper)
        env.reset(seed=0)
        infos = rollout(env, actionsFrom(schedule))
        assert infos[-1]["episode_totals"]["profit"] == pytest.approx(
            schedule.profit, rel=1e-9
        )

    def test_matches_with_calendar_aging_too(self, paper):
        # Looser tolerance: the LP's SOS2 blend agrees with np.interp only to the MIP gap.
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         discretePower=True)
        env = makeEnv(paper, calendar=True)
        env.reset(seed=0)
        infos = rollout(env, actionsFrom(schedule))
        assert sum(info["degradation"] for info in infos) == pytest.approx(
            schedule.degradationCost, rel=1e-4
        )

    def test_no_policy_beats_the_oracle(self, paper):
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False, discretePower=True)
        env = makeEnv(paper)
        rng = np.random.default_rng(0)
        for _ in range(20):
            env.reset(seed=0)
            info = {}
            for _ in range(len(MIXED_PRICES)):
                allowed = np.flatnonzero(env.action_masks())
                _, _, _, _, info = env.step(int(rng.choice(allowed)))
            assert info["episode_totals"]["profit"] <= schedule.profit + 1e-6


class TestEnergyMoves:
    def test_a_charge_step_is_the_rated_throughput(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        _, _, _, _, info = env.step(CHARGE)
        assert info["bought"] == pytest.approx(env.fullStep, rel=1e-12)
        assert info["sold"] == 0.0

    def test_a_discharge_step_is_the_rated_throughput(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        _, _, _, _, info = env.step(DISCHARGE)
        assert info["sold"] == pytest.approx(env.fullStep, rel=1e-12)
        assert info["bought"] == 0.0

    def test_the_rate_does_not_depend_on_soc(self, paper):
        moved = []
        for soc in (0.2, 0.5, 0.9):
            env = makeEnv(paper, initialSoc=soc)
            env.reset(seed=0)
            _, _, _, _, info = env.step(DISCHARGE)
            moved.append(info["sold"])
        assert moved[0] == pytest.approx(moved[1], rel=1e-12)
        assert moved[1] == pytest.approx(moved[2], rel=1e-12)

    def test_a_step_is_asymmetric_in_soc(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        _, _, _, _, up = env.step(CHARGE)
        env.reset(seed=0)
        _, _, _, _, down = env.step(DISCHARGE)
        gained = up["soc"] - config.initial_soc
        lost = config.initial_soc - down["soc"]
        assert gained == pytest.approx(
            env.fullStep * paper.chargeEfficiency / paper.MWh, rel=1e-12
        )
        assert lost == pytest.approx(
            env.fullStep / paper.dischargeEfficiency / paper.MWh, rel=1e-12
        )
        assert gained < lost

    def test_idle_moves_nothing(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        _, _, _, _, info = env.step(IDLE)
        assert info["bought"] == 0.0 and info["sold"] == 0.0
        assert info["soc"] == pytest.approx(config.initial_soc)


class TestBoundary:
    def test_a_part_step_fills_the_pack_exactly(self, paper):
        env = makeEnv(paper)
        start = 1.0 - (0.3 * env.fullStep * paper.chargeEfficiency) / paper.MWh
        env = makeEnv(paper, initialSoc=start)
        env.reset(seed=0)
        _, _, _, _, info = env.step(CHARGE)
        assert info["bought"] == pytest.approx(0.3 * env.fullStep, rel=1e-9)
        assert info["soc"] == pytest.approx(1.0, abs=1e-12)

    def test_a_part_step_empties_the_pack_exactly(self, paper):
        env = makeEnv(paper)
        start = (0.3 * env.fullStep / paper.dischargeEfficiency) / paper.MWh
        env = makeEnv(paper, initialSoc=start)
        env.reset(seed=0)
        _, _, _, _, info = env.step(DISCHARGE)
        assert info["sold"] == pytest.approx(0.3 * env.fullStep, rel=1e-9)
        assert info["soc"] == pytest.approx(0.0, abs=1e-12)

    def test_charge_is_masked_only_when_the_pack_is_full(self, paper):
        env = makeEnv(paper, initialSoc=1.0)
        env.reset(seed=0)
        assert env.action_masks()[CHARGE] == False  # noqa: E712
        env = makeEnv(paper, initialSoc=0.999)
        env.reset(seed=0)
        assert env.action_masks()[CHARGE] == True  # noqa: E712

    def test_discharge_is_masked_only_when_the_pack_is_empty(self, paper):
        env = makeEnv(paper, initialSoc=0.0)
        env.reset(seed=0)
        assert env.action_masks()[DISCHARGE] == False  # noqa: E712
        env = makeEnv(paper, initialSoc=0.001)
        env.reset(seed=0)
        assert env.action_masks()[DISCHARGE] == True  # noqa: E712

    def test_idle_is_never_masked(self, paper):
        for soc in (0.0, 0.5, 1.0):
            env = makeEnv(paper, initialSoc=soc)
            env.reset(seed=0)
            assert env.action_masks()[IDLE] == True  # noqa: E712

    def test_soc_never_leaves_zero_to_one(self, paper):
        env = makeEnv(paper, values=[-1000.0] * 40 + [10000.0] * 40, horizon=80)
        env.reset(seed=0)
        rng = np.random.default_rng(1)
        for _ in range(80):
            _, _, _, _, info = env.step(int(rng.integers(3)))
            assert -1e-12 <= info["soc"] <= 1.0 + 1e-12

    def test_an_unmasked_action_into_a_full_pack_is_a_no_op(self, paper):
        env = makeEnv(paper, initialSoc=1.0)
        env.reset(seed=0)
        _, _, _, _, info = env.step(CHARGE)
        assert info["bought"] == pytest.approx(0.0, abs=1e-12)
        assert info["soc"] == pytest.approx(1.0)


class TestReward:
    def test_the_reward_is_raw_profit(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        rng = np.random.default_rng(3)
        for _ in range(len(MIXED_PRICES)):
            _, reward, _, _, info = env.step(int(rng.integers(3)))
            assert reward == pytest.approx(info["profit"], rel=1e-12, abs=1e-15)

    def test_the_baseline_is_reporting_only(self, paper):
        """An oracle baseline fills lpProfit and regret, and leaves the reward alone."""
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False, discretePower=True)
        actions = [CHARGE, IDLE, CHARGE, DISCHARGE, DISCHARGE, IDLE,
                   CHARGE, DISCHARGE, IDLE, CHARGE, IDLE, DISCHARGE]
        plain = makeEnv(paper)
        plain.reset(seed=0)
        withBaseline = makeEnv(paper, baseline=[schedule.profitPerInterval])
        withBaseline.reset(seed=0)
        for action in actions:
            _, bare, _, _, _ = plain.step(action)
            _, reported, _, _, info = withBaseline.step(action)
            assert reported == bare
        assert info["regret"] == pytest.approx(
            schedule.profit - info["episode_totals"]["profit"], rel=1e-9)

    def test_idling_is_not_free_once_calendar_aging_is_on(self, paper):
        env = makeEnv(paper, values=[0.0] * 6, horizon=6, calendar=True)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(IDLE)
        assert info["calendarDegradation"] > 0
        assert reward < 0

    def test_reward_scale_divides_the_reward(self, paper):
        one = makeEnv(paper, rewardScale=1.0)
        one.reset(seed=0)
        _, bare, _, _, _ = one.step(DISCHARGE)
        ten = makeEnv(paper, rewardScale=10.0)
        ten.reset(seed=0)
        _, scaled, _, _, _ = ten.step(DISCHARGE)
        assert scaled == pytest.approx(bare / 10.0, rel=1e-12)


class TestObservation:
    def test_has_the_configured_length(self, paper):
        env = makeEnv(paper)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (config.observation_size,)
        assert obs.dtype == np.float32

    def test_stays_inside_the_declared_space(self, paper):
        env = makeEnv(paper, values=[-1000.0, 17500.0, 0.0] * 4)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        for action in (CHARGE, DISCHARGE, IDLE, CHARGE):
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)

    def test_soc_is_the_first_slot(self, paper):
        env = makeEnv(paper, initialSoc=0.37)
        obs, _ = env.reset(seed=0)
        assert obs[0] == pytest.approx(0.37, abs=1e-6)

    def test_time_remaining_is_the_last_slot_and_counts_down(self, paper):
        env = makeEnv(paper)
        obs, _ = env.reset(seed=0)
        assert obs[-1] == pytest.approx(1.0)
        seen = [obs[-1]]
        for _ in range(len(MIXED_PRICES)):
            obs, _, _, _, _ = env.step(IDLE)
            seen.append(obs[-1])
        assert all(b < a for a, b in zip(seen, seen[1:]))
        assert seen[-1] == pytest.approx(0.0)

    def test_never_reads_a_price_past_the_episode(self, paper):
        inside = MIXED_PRICES
        first = makeEnv(paper, values=inside + [9999.0] * 12, horizon=len(inside))
        second = makeEnv(paper, values=inside + [-999.0] * 12, horizon=len(inside))
        a, _ = first.reset(seed=0)
        b, _ = second.reset(seed=0)
        assert np.allclose(a, b)
        for _ in range(len(inside) - 1):
            a, _, _, _, _ = first.step(IDLE)
            b, _, _, _, _ = second.step(IDLE)
            assert np.allclose(a, b)

    def test_is_deterministic_for_a_given_state(self, paper):
        first = makeEnv(paper)
        second = makeEnv(paper)
        a, _ = first.reset(seed=0)
        b, _ = second.reset(seed=7)
        assert np.allclose(a, b)


class TestEpisode:
    def test_runs_for_exactly_the_horizon(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        for step in range(len(MIXED_PRICES)):
            _, _, terminated, truncated, _ = env.step(IDLE)
            assert terminated is False
            assert truncated == (step == len(MIXED_PRICES) - 1)

    def test_ends_truncated_not_terminated(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        for _ in range(len(MIXED_PRICES) - 1):
            env.step(IDLE)
        _, _, terminated, truncated, _ = env.step(IDLE)
        assert truncated is True and terminated is False

    def test_resets_to_the_configured_soc(self, paper):
        env = makeEnv(paper)
        env.reset(seed=0)
        for _ in range(4):
            env.step(DISCHARGE)
        obs, _ = env.reset(seed=0)
        assert obs[0] == pytest.approx(config.initial_soc, abs=1e-6)

    def test_windows_are_non_overlapping_and_horizon_aligned(self, paper):
        env = BessEnv(priceSeries([10.0] * 30), paper, horizon=12, calendar=False)
        assert env.windows == [0, 12]

    def test_a_series_shorter_than_one_episode_is_refused(self, paper):
        with pytest.raises(ValueError):
            BessEnv(priceSeries([10.0] * 5), paper, horizon=12)

    def test_the_same_seed_gives_the_same_rollout(self, paper):
        values = MIXED_PRICES * 4
        def run(seed):
            env = BessEnv(priceSeries(values), paper, horizon=12, calendar=False,
                          shuffle=True, rewardScale=1.0)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            return [env.step(int(rng.integers(3)))[1] for _ in range(12)]
        assert run(5) == run(5)

    def test_a_requested_window_is_honoured(self, paper):
        values = [10.0] * 12 + [500.0] * 12
        env = BessEnv(priceSeries(values), paper, horizon=12, calendar=False)
        env.reset(seed=0, options={"window": 1})
        _, _, _, _, info = env.step(DISCHARGE)
        assert info["price"] == 500.0


class TestForecastObservations:
    """At inference the agent SEES the forecast but is PAID the realised price."""

    def test_the_price_slots_come_from_the_forecast(self, paper):
        forecast = [999.0] * len(MIXED_PRICES)
        plain = makeEnv(paper)
        withForecast = makeEnv(paper, forecasts=[forecast])
        a, _ = plain.reset(seed=0)
        b, _ = withForecast.reset(seed=0)
        priceEnd = 1 + config.price_feature_count
        assert not np.allclose(a[1:priceEnd], b[1:priceEnd])
        assert b[1] == pytest.approx(withForecast.scaler(999.0), abs=1e-6)

    def test_settlement_still_uses_the_realised_price(self, paper):
        forecast = [999.0] * len(MIXED_PRICES)
        env = makeEnv(paper, forecasts=[forecast])
        env.reset(seed=0)
        _, _, _, _, info = env.step(DISCHARGE)
        assert info["price"] == MIXED_PRICES[0]
        assert info["revenue"] == pytest.approx(MIXED_PRICES[0] * info["sold"])

    def test_the_realised_past_is_not_leaked_into_the_forecast_view(self, paper):
        forecast = list(MIXED_PRICES)
        env = makeEnv(paper, forecasts=[forecast])
        env.reset(seed=0)
        env.step(IDLE)
        observed = env._observedPrices(0, len(MIXED_PRICES))
        assert observed[0] == 0.0
        assert observed[1] == pytest.approx(MIXED_PRICES[1])

    def test_a_forecast_equal_to_the_realised_path_scores_the_same(self, paper):
        actions = [CHARGE, IDLE, DISCHARGE, CHARGE, DISCHARGE, IDLE,
                   CHARGE, DISCHARGE, IDLE, CHARGE, IDLE, DISCHARGE]
        plain = makeEnv(paper)
        plain.reset(seed=0)
        mirrored = makeEnv(paper, forecasts=[list(MIXED_PRICES)])
        mirrored.reset(seed=0)
        for action in actions:
            _, a, _, _, first = plain.step(action)
            _, b, _, _, second = mirrored.step(action)
            assert a == pytest.approx(b)
            assert first["profit"] == pytest.approx(second["profit"])

    def test_nan_past_the_run_becomes_padding(self, paper):
        forecast = [20.0] * 4 + [float("nan")] * (len(MIXED_PRICES) - 4)
        env = makeEnv(paper, forecasts=[forecast])
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert not np.isnan(obs).any()


class TestForecastRefresh:
    """A refresh stack: the agent reads the latest run published by each step."""

    @staticmethod
    def stack(levels, every=4):
        horizon = len(MIXED_PRICES)
        grid = np.full((len(levels), horizon), np.nan)
        for refresh, level in enumerate(levels):
            grid[refresh, refresh * every:] = level
        return grid

    def test_the_view_switches_at_each_refresh(self, paper):
        env = makeEnv(paper, forecasts=[self.stack([100.0, 200.0, 300.0])])
        obs, _ = env.reset(seed=0)
        seen = [obs[1]]
        for step in range(len(MIXED_PRICES) - 1):
            obs, *_ = env.step(IDLE)
            if step + 1 in (3, 4, 8):
                seen.append(obs[1])
        assert seen == pytest.approx([env.scaler(level)
                                      for level in (100.0, 100.0, 200.0, 300.0)], abs=1e-6)

    def test_settlement_still_uses_the_realised_price(self, paper):
        env = makeEnv(paper, forecasts=[self.stack([999.0, 999.0, 999.0])])
        env.reset(seed=0)
        for step in range(5):
            _, _, _, _, info = env.step(DISCHARGE)
        assert info["price"] == MIXED_PRICES[4]

    def test_a_stack_of_the_realised_path_scores_the_same(self, paper):
        mirrored = np.tile(MIXED_PRICES, (3, 1))
        actions = [CHARGE, IDLE, DISCHARGE, CHARGE, DISCHARGE, IDLE,
                   CHARGE, DISCHARGE, IDLE, CHARGE, IDLE, DISCHARGE]
        plain, stacked = makeEnv(paper), makeEnv(paper, forecasts=[mirrored])
        a, _ = plain.reset(seed=0)
        b, _ = stacked.reset(seed=0)
        np.testing.assert_allclose(a, b)
        for action in actions:
            a, ra, *_ = plain.step(action)
            b, rb, *_ = stacked.step(action)
            np.testing.assert_allclose(a, b)
            assert ra == pytest.approx(rb)

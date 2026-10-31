"""Tests for rl/env.py. Run with: uv run pytest rl/ -q

TestMatchesTheOracle is the one to read first: everything downstream assumes it holds.
"""
import numpy as np
import pandas as pd
import pytest

import config
from degradation.schade import SchadeDegradation
from optimisation import dp
from optimisation.lp import solve
from rl import features
from rl.env import CHARGE, DISCHARGE, IDLE, BessEnv


PAPER_MW = 0.060
PAPER_MWH = 0.100
PAPER_REPLACEMENT_COST = 150000.0

MIXED_PRICES = [20.0, 15.0, -50.0, 10.0, 300.0, 25.0, 18.0, 400.0, 12.0, -80.0, 9.0, 250.0]

# `config.initial_soc` is zero, because an episode may not start with energy it never
# bought. A test of the throughput rate needs a pack that can move in both directions, so
# it states its own mid-pack start.
MID_SOC = 0.5

# An out-of-bounds penalty, in reward units. Stated here rather than taken from
# config.invalid_action_penalty, so a change to the experiment's value cannot quietly
# weaken these tests.
PENALTY = 1.0


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


def priceSeries(values, start="2024-01-10 04:05"):
    """Stamped from the first interval of a trading day. The clock slot reads the
    04:00-to-04:00 day, so an unaligned series would put every test off the boundary."""
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

    def test_profit_is_identical_under_the_penalty(self, paper):
        """The oracle's schedule is feasible, so it never goes out of bounds and the
        penalty never fires along it. This is what keeps the bound valid once the
        reward carries a training term."""
        schedule = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False, discretePower=True)
        actions = actionsFrom(schedule)
        env = makeEnv(paper, invalidPenalty=PENALTY)
        env.reset(seed=0)
        infos = rollout(env, actions)
        assert not any(info["outOfBounds"] for info in infos)
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
        env = makeEnv(paper, initialSoc=MID_SOC)
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
        env = makeEnv(paper, initialSoc=MID_SOC)
        env.reset(seed=0)
        _, _, _, _, up = env.step(CHARGE)
        env.reset(seed=0)
        _, _, _, _, down = env.step(DISCHARGE)
        gained = up["soc"] - MID_SOC
        lost = MID_SOC - down["soc"]
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


class TestOutOfBounds:
    """Charging a full pack or discharging an empty one. MaskablePPO cannot choose
    either, because the mask removes it. DQN reads no mask, so it can, and the penalty
    is what teaches it the boundary. `action_masks` is the only definition of the
    condition, so these tests follow it rather than restating the arithmetic."""

    def test_charging_a_full_pack_is_out_of_bounds(self, paper):
        env = makeEnv(paper, initialSoc=1.0, invalidPenalty=PENALTY)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(CHARGE)
        assert info["outOfBounds"] is True
        assert reward == pytest.approx(info["profit"] - PENALTY, rel=1e-12, abs=1e-15)

    def test_discharging_an_empty_pack_is_out_of_bounds(self, paper):
        env = makeEnv(paper, initialSoc=0.0, invalidPenalty=PENALTY)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(DISCHARGE)
        assert info["outOfBounds"] is True
        assert reward == pytest.approx(info["profit"] - PENALTY, rel=1e-12, abs=1e-15)

    def test_a_partial_top_up_is_never_out_of_bounds(self, paper):
        """The one that keeps the environment matched to solve(discretePower=True).
        That mode allows a part step where physics forces one, so penalising it would
        make the oracle solve a different problem."""
        env = makeEnv(paper)
        start = 1.0 - (0.3 * env.fullStep * paper.chargeEfficiency) / paper.MWh
        env = makeEnv(paper, initialSoc=start, invalidPenalty=PENALTY)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(CHARGE)
        assert 0.0 < info["bought"] < env.fullStep
        assert info["outOfBounds"] is False
        assert reward == pytest.approx(info["profit"], rel=1e-12, abs=1e-15)

    def test_a_part_step_that_empties_the_pack_is_never_out_of_bounds(self, paper):
        env = makeEnv(paper)
        start = (0.3 * env.fullStep / paper.dischargeEfficiency) / paper.MWh
        env = makeEnv(paper, initialSoc=start, invalidPenalty=PENALTY)
        env.reset(seed=0)
        _, reward, _, _, info = env.step(DISCHARGE)
        assert 0.0 < info["sold"] < env.fullStep
        assert info["outOfBounds"] is False
        assert reward == pytest.approx(info["profit"], rel=1e-12, abs=1e-15)

    def test_idle_is_never_out_of_bounds(self, paper):
        for soc in (0.0, 0.5, 1.0):
            env = makeEnv(paper, initialSoc=soc, invalidPenalty=PENALTY)
            env.reset(seed=0)
            _, reward, _, _, info = env.step(IDLE)
            assert info["outOfBounds"] is False
            assert reward == pytest.approx(info["profit"], rel=1e-12, abs=1e-15)

    def test_the_attempted_action_is_recorded_whatever_moved(self, paper):
        """The actions export derives its own column from the energy moved, so an
        out-of-bounds charge is filed there as "idle". This is the only record of what
        the agent actually chose."""
        env = makeEnv(paper, initialSoc=1.0, invalidPenalty=PENALTY)
        env.reset(seed=0)
        _, _, _, _, info = env.step(CHARGE)
        assert info["attemptedAction"] == CHARGE
        assert info["bought"] == pytest.approx(0.0, abs=1e-12)

    def test_the_penalty_moves_the_reward_and_nothing_else(self, paper):
        """Every reported figure stays raw dollars, so the oracle keeps bounding the
        agent and a profit column never carries a training term."""
        actions = [DISCHARGE, CHARGE, DISCHARGE, IDLE, CHARGE, CHARGE,
                   DISCHARGE, DISCHARGE, DISCHARGE, DISCHARGE, IDLE, CHARGE]
        free = makeEnv(paper, invalidPenalty=0.0)
        free.reset(seed=0)
        penalised = makeEnv(paper, invalidPenalty=PENALTY)
        penalised.reset(seed=0)
        seen = 0
        for action in actions:
            _, bare, _, _, plain = free.step(action)
            _, charged, _, _, info = penalised.step(action)
            for key in ("profit", "revenue", "degradation", "soc", "bought", "sold"):
                assert info[key] == pytest.approx(plain[key], rel=1e-12, abs=1e-15)
            seen += info["outOfBounds"]
            assert charged == pytest.approx(
                bare - PENALTY * info["outOfBounds"], rel=1e-12, abs=1e-15)
        assert seen > 0, "the action sequence must hit a boundary or this proves nothing"
        # Counting does not depend on the penalty, so both episodes report the same.
        assert info["episode_totals"] == plain["episode_totals"]
        assert info["episode_totals"]["outOfBounds"] == seen

    def test_the_episode_counts_every_attempt(self, paper):
        env = makeEnv(paper, invalidPenalty=PENALTY)
        env.reset(seed=0)
        info = {}
        for _ in range(len(MIXED_PRICES)):
            _, _, _, _, info = env.step(DISCHARGE)
        # The pack starts empty and nothing ever charges it, so every step is an attempt.
        assert info["episode_totals"]["outOfBounds"] == len(MIXED_PRICES)
        # Flat, because SB3's Monitor copies one key out of the final step's info and
        # cannot reach a nested total. See rl/train.py.
        assert info["outOfBoundsTotal"] == len(MIXED_PRICES)


class TestReward:
    def test_the_reward_is_raw_profit(self, paper):
        # No penalty, so the reward is profit and nothing else. The penalty term has its
        # own tests in TestOutOfBounds.
        env = makeEnv(paper, invalidPenalty=0.0)
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
        # Mid-pack, so the discharge actually moves energy. From an empty pack it moves
        # nothing, and the test would compare zero against zero.
        one = makeEnv(paper, initialSoc=MID_SOC, rewardScale=1.0)
        one.reset(seed=0)
        _, bare, _, _, _ = one.step(DISCHARGE)
        ten = makeEnv(paper, initialSoc=MID_SOC, rewardScale=10.0)
        ten.reset(seed=0)
        _, scaled, _, _, _ = ten.step(DISCHARGE)
        assert bare != 0.0
        assert scaled == pytest.approx(bare / 10.0, rel=1e-12)

    def test_the_penalty_is_in_reward_units_not_dollars(self, paper):
        """So its size means the same whatever reward_scale is set to."""
        for scale in (1.0, 20.0):
            env = makeEnv(paper, initialSoc=1.0, rewardScale=scale,
                          invalidPenalty=PENALTY)
            env.reset(seed=0)
            _, reward, _, _, info = env.step(CHARGE)
            assert reward == pytest.approx(info["profit"] / scale - PENALTY,
                                           rel=1e-12, abs=1e-15)


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

    def test_time_remaining_is_the_clock_slot_and_counts_down(self, paper):
        # Index by name, never by -1: a negative index silently follows the end of the
        # vector when a slot is appended, and episodeLeft also counts down from 1.0.
        env = makeEnv(paper)
        obs, _ = env.reset(seed=0)
        assert obs[features.clockIndex] == pytest.approx(1.0)
        seen = [obs[features.clockIndex]]
        for _ in range(len(MIXED_PRICES)):
            obs, _, _, _, _ = env.step(IDLE)
            seen.append(obs[features.clockIndex])
        assert all(b < a for a, b in zip(seen, seen[1:]))
        assert seen[-1] == pytest.approx(0.0)

    def test_the_episode_slot_counts_down_over_the_episode(self, paper):
        env = makeEnv(paper)
        obs, _ = env.reset(seed=0)
        assert obs[features.episodeIndex] == pytest.approx(1.0)
        for step in range(len(MIXED_PRICES)):
            obs, _, _, _, _ = env.step(IDLE)
            left = 1.0 - (step + 1) / len(MIXED_PRICES)
            assert obs[features.episodeIndex] == pytest.approx(left, abs=1e-6)

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


class TestCycleBudget:
    """`cycleBudget` caps the equivalent full cycles one episode may spend.

    One EFC is the nominal usable capacity removed cell-side. The cap is enforced by the
    clip in `_applyAction`, in the same `min` as the SoC floor, so it binds under every
    algorithm and not only the one that reads the mask.
    """

    def spent(self, paper, infos):
        """The EFC a rollout spent, read off `sold` rather than the env's bookkeeping."""
        return sum(info["sold"] for info in infos) / paper.dischargeEfficiency / paper.MWh

    def test_an_absent_budget_is_the_old_environment(self, paper):
        free = makeEnv(paper, initialSoc=MID_SOC, cycleBudget=None)
        free.reset(seed=0)
        infos = rollout(free, [DISCHARGE] * len(MIXED_PRICES))
        assert self.spent(paper, infos) == pytest.approx(MID_SOC, abs=1e-12)
        assert free._cyclesLeft() == 1.0

    def test_reset_restores_the_budget(self, paper):
        env = makeEnv(paper, initialSoc=MID_SOC, cycleBudget=0.25)
        env.reset(seed=0)
        rollout(env, [DISCHARGE] * len(MIXED_PRICES))
        assert env._cyclesLeft() == pytest.approx(0.0, abs=1e-12)
        env.reset(seed=0)
        assert env._cyclesLeft() == pytest.approx(1.0, abs=1e-12)
        assert env.spentMWh == 0.0

    def test_a_discharge_spends_cell_side_energy(self, paper):
        """Meter-side would understate the spend by nu_D, which is a 5% error."""
        env = makeEnv(paper, initialSoc=MID_SOC, cycleBudget=1.0)
        env.reset(seed=0)
        _, _, _, _, info = env.step(DISCHARGE)
        assert env.spentMWh == pytest.approx(info["sold"] / paper.dischargeEfficiency,
                                             abs=1e-15)
        assert info["cycles"] == pytest.approx(env.spentMWh / paper.MWh, abs=1e-15)

    def test_a_charge_and_an_idle_spend_nothing(self, paper):
        env = makeEnv(paper, initialSoc=MID_SOC, cycleBudget=1.0)
        env.reset(seed=0)
        rollout(env, [CHARGE, IDLE, CHARGE, IDLE])
        assert env.spentMWh == 0.0
        assert env._cyclesLeft() == pytest.approx(1.0, abs=1e-12)

    def test_discharge_is_masked_once_the_budget_is_spent(self, paper):
        env = makeEnv(paper, initialSoc=1.0, cycleBudget=0.25)
        env.reset(seed=0)
        while env.action_masks()[DISCHARGE]:
            env.step(DISCHARGE)
        assert env.spentMWh == pytest.approx(0.25 * paper.MWh, abs=1e-12)
        assert env.stored > 0.0  # masked by the budget, not by an empty pack
        assert env.action_masks()[IDLE]

    def test_a_discharge_is_clipped_to_the_remaining_budget(self, paper):
        """The same partial top-up rule the SoC boundary already has."""
        env = makeEnv(paper, initialSoc=1.0, cycleBudget=0.08)
        env.reset(seed=0)
        total = 0.0
        for _ in range(4):
            _, _, _, _, info = env.step(DISCHARGE)
            total += info["sold"] / paper.dischargeEfficiency
        assert total == pytest.approx(0.08 * paper.MWh, abs=1e-12)

    def test_the_tighter_of_the_two_clips_wins(self, paper):
        """Both clips apply at once, so a pack with less energy than budget is limited by
        the pack, and a pack with less budget than energy by the budget."""
        byPack = makeEnv(paper, initialSoc=0.05, cycleBudget=1.0)
        byPack.reset(seed=0)
        _, _, _, _, info = byPack.step(DISCHARGE)
        assert info["sold"] == pytest.approx(0.05 * paper.MWh * paper.dischargeEfficiency,
                                             abs=1e-12)

        byBudget = makeEnv(paper, initialSoc=1.0, cycleBudget=0.05)
        byBudget.reset(seed=0)
        _, _, _, _, info = byBudget.step(DISCHARGE)
        assert info["sold"] == pytest.approx(0.05 * paper.MWh * paper.dischargeEfficiency,
                                             abs=1e-12)

    def test_a_zero_budget_never_discharges(self, paper):
        env = makeEnv(paper, initialSoc=1.0, cycleBudget=0.0)
        env.reset(seed=0)
        infos = rollout(env, [DISCHARGE] * len(MIXED_PRICES))
        assert self.spent(paper, infos) == 0.0
        assert not env.action_masks()[DISCHARGE]
        assert env._cyclesLeft() == 0.0

    def test_the_budget_is_never_overspent_over_a_long_rollout(self, paper):
        """A masked random walk over a full weekly episode, which is where the budget is
        meant to bind. Nothing the agent can do may exceed it."""
        values = list(np.random.default_rng(0).normal(80, 200, 2016))
        env = makeEnv(paper, values, horizon=2016, initialSoc=MID_SOC, cycleBudget=7.0)
        rng = np.random.default_rng(1)
        for _ in range(3):
            env.reset(seed=0)
            info = {}
            for _ in range(2016):
                allowed = np.flatnonzero(env.action_masks())
                _, _, _, _, info = env.step(int(rng.choice(allowed)))
            assert info["episode_totals"]["cycles"] <= 7.0 + 1e-12

    def test_the_budget_does_not_reset_at_a_day_boundary_inside_the_episode(self, paper):
        """The budget is the episode's, so day 2 inherits what day 1 left."""
        values = [100.0] * 576
        env = makeEnv(paper, values, horizon=576, initialSoc=1.0, cycleBudget=0.25)
        env.reset(seed=0)
        rollout(env, [DISCHARGE] * 300)
        assert env.spentMWh == pytest.approx(0.25 * paper.MWh, abs=1e-12)
        assert not env.action_masks()[DISCHARGE]
        rollout(env, [DISCHARGE] * 276)
        assert env.spentMWh == pytest.approx(0.25 * paper.MWh, abs=1e-12)

    def test_the_budget_slot_falls_as_the_budget_is_spent(self, paper):
        env = makeEnv(paper, initialSoc=1.0, cycleBudget=0.5)
        obs, _ = env.reset(seed=0)
        assert obs[features.cycleIndex] == pytest.approx(1.0, abs=1e-6)
        seen = [obs[features.cycleIndex]]
        for _ in range(4):
            obs, _, _, _, _ = env.step(DISCHARGE)
            seen.append(obs[features.cycleIndex])
        assert all(b <= a + 1e-9 for a, b in zip(seen, seen[1:]))
        assert seen[-1] < seen[0]


class TestMatchesTheBudgetedOracle:
    """The gate. The environment and `dp.solve` must still be one problem with a budget
    and a terminal value in force, or the oracle stops bounding the agent."""

    @pytest.mark.parametrize("calendar", [False, True])
    @pytest.mark.parametrize("cycles,terminalPrice", [(0.05, 0.0), (0.5, 0.0),
                                                      (None, 100.0), (0.05, 100.0)])
    def test_a_budgeted_schedule_replays_identically(self, paper, calendar, cycles,
                                                     terminalPrice):
        schedule = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                            calendar=calendar, cycles=cycles,
                            terminalPrice=terminalPrice)
        env = makeEnv(paper, calendar=calendar, cycleBudget=cycles,
                      terminalPrice=terminalPrice)
        env.reset(seed=0)
        infos = rollout(env, actionsFrom(schedule))
        for got, want in zip([info["soc"] for info in infos], schedule.soc):
            assert got == pytest.approx(want, abs=1e-9)
        assert infos[-1]["episode_totals"]["profit"] == pytest.approx(
            schedule.profit, abs=1e-9)
        assert infos[-1]["episode_totals"]["cycles"] == pytest.approx(
            schedule.cyclesUsed, abs=1e-9)
        assert infos[-1]["episode_totals"]["terminalValue"] == pytest.approx(
            schedule.terminalValue, abs=1e-9)

    def test_no_policy_beats_the_budgeted_bound(self, paper):
        schedule = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                            calendar=False, cycles=0.05)
        env = makeEnv(paper, cycleBudget=0.05)
        rng = np.random.default_rng(0)
        for _ in range(20):
            env.reset(seed=0)
            info = {}
            for _ in range(len(MIXED_PRICES)):
                allowed = np.flatnonzero(env.action_masks())
                _, _, _, _, info = env.step(int(rng.choice(allowed)))
            assert info["episode_totals"]["profit"] <= schedule.dualBound + 1e-9

    @pytest.mark.parametrize("stored", [0.0, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("spent", [0.0, 0.25, 0.5])
    def test_the_discharge_mask_agrees_with_the_move(self, paper, stored, spent):
        """Over a (stored, spent) grid: a masked discharge moves nothing, and an unmasked
        one moves something. If these part company the agent can waste a step it was told
        was available."""
        env = makeEnv(paper, initialSoc=stored, cycleBudget=0.5)
        env.reset(seed=0)
        env.spentMWh = spent * paper.MWh
        allowed = env.action_masks()[DISCHARGE]
        _, _, _, _, info = env.step(DISCHARGE)
        assert (info["sold"] > env.tolerance) == bool(allowed)


class TestTerminalValue:
    def test_a_zero_price_is_the_old_environment(self, paper):
        old = makeEnv(paper, initialSoc=MID_SOC, terminalPrice=0.0)
        old.reset(seed=0)
        infos = rollout(old, [IDLE] * len(MIXED_PRICES))
        assert infos[-1]["episode_totals"]["terminalValue"] == 0.0

    def test_it_is_paid_once_on_the_closing_step(self, paper):
        env = makeEnv(paper, initialSoc=MID_SOC, terminalPrice=100.0)
        env.reset(seed=0)
        infos = rollout(env, [IDLE] * len(MIXED_PRICES))
        assert [info["terminalValue"] for info in infos[:-1]] == [0.0] * (len(infos) - 1)
        assert infos[-1]["terminalValue"] == pytest.approx(
            paper.terminalValue(MID_SOC * paper.MWh, 100.0), abs=1e-12)

    def test_it_enters_the_reward_of_the_closing_step(self, paper):
        env = makeEnv(paper, initialSoc=MID_SOC, terminalPrice=100.0)
        env.reset(seed=0)
        reward = 0.0
        for action in [IDLE] * len(MIXED_PRICES):
            _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(info["profit"] + info["terminalValue"], abs=1e-12)


class TestTradingDayLookahead:
    def test_the_lookahead_never_crosses_the_trading_day(self, paper):
        """AEMO publishes no RRP past the end of the trading day, so a weekly episode
        still reads one day ahead. Two envs whose day-2 prices differ wildly must give
        identical observations throughout day 1."""
        day = config.observation_intervals
        first = [50.0 + i % 7 for i in range(day)]
        quiet = BessEnv(priceSeries(first + [0.0] * day), paper, horizon=2 * day,
                        calendar=False, rewardScale=1.0)
        wild = BessEnv(priceSeries(first + [9000.0] * day), paper, horizon=2 * day,
                       calendar=False, rewardScale=1.0)
        a, _ = quiet.reset(seed=0)
        b, _ = wild.reset(seed=0)
        for _ in range(day):
            assert a == pytest.approx(b, abs=0.0)
            a, _, _, _, _ = quiet.step(IDLE)
            b, _, _, _, _ = wild.step(IDLE)
        assert a != pytest.approx(b, abs=0.0)


# Ends on prices that pay the pack to charge, as optimisation/test_dp.py states it.
CLOSING_PRICES = [20.0, -50.0, 10.0, 300.0, -80.0, 9.0, -200.0, -150.0]


class TestEndEmpty:
    """`endEmpty=True` closes every episode with the pack empty, whatever the policy.

    The override is the rule, and the mask only reports it, so a policy that reads no
    mask (DQN, A2C) must end empty as surely as MaskablePPO does.
    """

    @pytest.mark.parametrize("calendar", [False, True])
    @pytest.mark.parametrize("cycles", [None, 0.05, 0.5])
    @pytest.mark.parametrize("prices", [CLOSING_PRICES, MIXED_PRICES])
    def test_the_oracle_schedule_replays_identically(self, paper, calendar, cycles,
                                                      prices):
        schedule = dp.solve(prices, paper, initialSoc=config.initial_soc,
                            calendar=calendar, cycles=cycles, endEmpty=True)
        env = makeEnv(paper, values=prices, calendar=calendar, cycleBudget=cycles,
                      endEmpty=True)
        env.reset(seed=0)
        infos = rollout(env, actionsFrom(schedule))
        for got, want in zip([info["soc"] for info in infos], schedule.soc):
            assert got == pytest.approx(want, abs=1e-9)
        assert infos[-1]["episode_totals"]["profit"] == pytest.approx(
            schedule.profit, abs=1e-9)
        assert infos[-1]["episode_totals"]["cycles"] == pytest.approx(
            schedule.cyclesUsed, abs=1e-9)

    @pytest.mark.parametrize("cycles", [None, 0.05, 0.3])
    def test_any_policy_ends_empty_inside_the_budget(self, paper, cycles):
        """Random actions with the mask ignored, as DQN acts."""
        prices = list(np.random.default_rng(1).normal(50, 120, 60))
        schedule = dp.solve(prices, paper, initialSoc=config.initial_soc,
                            calendar=False, cycles=cycles, endEmpty=True)
        env = makeEnv(paper, values=prices, cycleBudget=cycles, endEmpty=True)
        rng = np.random.default_rng(0)
        for _ in range(50):
            env.reset(seed=0)
            infos = rollout(env, [int(a) for a in rng.integers(0, 3, len(prices))])
            totals = infos[-1]["episode_totals"]
            assert infos[-1]["soc"] == pytest.approx(0.0, abs=1e-9)
            if cycles is not None:
                assert totals["cycles"] <= cycles + 1e-9
            assert totals["profit"] <= schedule.dualBound + 1e-9

    def test_a_charge_that_cannot_be_sold_in_time_idles(self, paper):
        env = makeEnv(paper, values=CLOSING_PRICES, endEmpty=True)
        env.reset(seed=0)
        _, _, _, _, info = env.step(CHARGE)
        assert info["bought"] > 0.0
        rollout(env, [IDLE] * (len(CLOSING_PRICES) - 3))
        # Two intervals left. A second charge would leave more than the one discharge
        # after it can sell, so it idles.
        _, _, _, _, info = env.step(CHARGE)
        assert info["bought"] == 0.0
        assert info["sold"] == 0.0
        # The last interval. An idle would close holding energy, so it discharges.
        _, _, _, _, info = env.step(IDLE)
        assert info["sold"] > 0.0
        assert info["soc"] == pytest.approx(0.0, abs=1e-12)

    def test_the_mask_agrees_with_the_override(self, paper):
        """Every action the mask allows is the move made, and every one it forbids is
        changed or moves nothing."""
        prices = list(np.random.default_rng(3).normal(50, 120, 40))
        env = makeEnv(paper, values=prices, cycleBudget=0.2, endEmpty=True)
        rng = np.random.default_rng(4)
        for _ in range(20):
            env.reset(seed=0)
            for _ in range(len(prices)):
                mask = env.action_masks()
                assert mask.any()
                for action in (CHARGE, IDLE, DISCHARGE):
                    if mask[action]:
                        assert env._closingAction(action) == action
                env.step(int(rng.integers(0, 3)))

    def test_off_changes_nothing(self, paper):
        actions = [CHARGE, CHARGE, IDLE, DISCHARGE, CHARGE, IDLE, CHARGE, CHARGE]
        old = makeEnv(paper, values=CLOSING_PRICES)
        off = makeEnv(paper, values=CLOSING_PRICES, endEmpty=False)
        old.reset(seed=0)
        off.reset(seed=0)
        assert [i["soc"] for i in rollout(old, actions)] == [
            i["soc"] for i in rollout(off, actions)]

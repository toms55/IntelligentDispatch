"""
Tests for optimisation/dp.py, the exact discrete-power oracle.

Run with:   nix develop -c pytest optimisation/test_dp.py -q

The DP replaces `lp.solve(discretePower=True)`, so the tests have three jobs.

TestMatchesTheMilp shows the two agree on the same problem. The MILP is the formulation
the project has trusted so far, so it is the reference, and any disagreement is a defect
in one of them.

TestIsOptimal shows the DP is exact, by brute force over every policy on a horizon short
enough to enumerate. This is what the MILP cannot give: no gap, no time limit, and a
dual bound that equals the profit.

TestMatchesTheEnvironment shows the DP and `rl.env.BessEnv` are the same problem. That
is the central invariant of the repository. It holds more tightly here than for the
MILP, because the DP calls the same `schade.py` functions the environment does rather
than approximating the calendar curve with an SOS2 blend.
"""
import itertools
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import config
from degradation.schade import SchadeDegradation
from optimisation import dp
from optimisation.lp import solve
from rl.env import CHARGE, DISCHARGE, IDLE, BessEnv
from utils import nem_interval_hours

PAPER_MW = 0.060
PAPER_MWH = 0.100
PAPER_REPLACEMENT_COST = 150000.0

# The series rl/test_env.py uses, so a failure here is comparable with one there.
MIXED_PRICES = [20.0, 15.0, -50.0, 10.0, 300.0, 25.0, 18.0, 400.0, 12.0, -80.0, 9.0, 250.0]
FLAT_PRICES = [50.0] * 12
FALLING_PRICES = [200.0, 180.0, 160.0, 140.0, 120.0, 100.0, 80.0, 60.0]
NEGATIVE_PRICES = [-30.0, -40.0, -10.0, 90.0, 120.0, -60.0, 15.0, 200.0]

priceCache = Path(__file__).resolve().parent.parent / "data_cache"
needsPrices = pytest.mark.skipif(
    not priceCache.exists(),
    reason="needs the nemosis price cache in data_cache/",
)


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


def priceSeries(values, start="2024-01-10"):
    import pandas as pd

    return pd.Series(
        values, index=pd.date_range(start, periods=len(values), freq="5min"), name="RRP"
    )


def actionsFrom(schedule, tolerance=1e-12):
    """The schedule read back as agent actions, as rl/test_env.py does it."""
    return [
        CHARGE if bought > tolerance else (DISCHARGE if sold > tolerance else IDLE)
        for bought, sold in zip(schedule.bought, schedule.sold)
    ]


def rollProfit(prices, model, actions, initialSoc=config.initial_soc, calendar=False,
               dtHours=nem_interval_hours):
    """Score an action sequence with schade.py directly, independent of the DP."""
    stored = initialSoc * model.MWh
    total = 0.0
    for price, action in zip(prices, actions):
        new, bought, sold = dp.move(stored, action, model, dtHours)
        qCd, qCal, both = model.step(stored / model.MWh, new / model.MWh, dtHours)
        cost = both if calendar else model.cost(qCd)
        total += price * (sold - bought) - cost
        stored = new
    return total


# --- the DP against the MILP -------------------------------------------------


class TestMatchesTheMilp:
    """The MILP is the reference. The two state the same problem."""

    @pytest.mark.parametrize("prices", [MIXED_PRICES, FLAT_PRICES, FALLING_PRICES,
                                        NEGATIVE_PRICES])
    def test_profit_agrees_without_calendar(self, paper, prices):
        exact = dp.solve(prices, paper, initialSoc=config.initial_soc, calendar=False)
        milp = solve(prices, paper, initialSoc=config.initial_soc, calendar=False,
                     discretePower=True)
        assert exact.profit == pytest.approx(milp.profit, abs=1e-9)

    def test_soc_path_agrees_without_calendar(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        milp = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc, calendar=False,
                     discretePower=True)
        assert exact.soc == pytest.approx(milp.soc, abs=1e-9)

    def test_profit_agrees_with_calendar(self, paper):
        # Looser: the MILP blends the calendar curve with SOS2 and only agrees with
        # np.interp to around 1e-4. rl/test_env.py records the same limit.
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=True)
        milp = solve(MIXED_PRICES, paper, initialSoc=config.initial_soc, calendar=True,
                     discretePower=True)
        assert exact.profit == pytest.approx(milp.profit, rel=1e-4)

    @pytest.mark.parametrize("initialSoc", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_profit_agrees_from_any_starting_soc(self, paper, initialSoc):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=initialSoc, calendar=False)
        milp = solve(MIXED_PRICES, paper, initialSoc=initialSoc, calendar=False,
                     discretePower=True)
        assert exact.profit == pytest.approx(milp.profit, abs=1e-9)

    def test_never_below_the_milp_incumbent(self, paper):
        """The DP is exact, so it can only match or beat a gap-limited incumbent."""
        rng = np.random.default_rng(0)
        for _ in range(8):
            prices = list(rng.normal(80, 120, 24))
            exact = dp.solve(prices, paper, initialSoc=config.initial_soc,
                             calendar=False)
            milp = solve(prices, paper, initialSoc=config.initial_soc, calendar=False,
                         discretePower=True, gap=0.02)
            assert exact.profit >= milp.profit - 1e-9

    def test_never_above_the_milp_dual_bound(self, paper):
        """And it must sit under the MILP's bound, or one of them is wrong."""
        rng = np.random.default_rng(1)
        for _ in range(8):
            prices = list(rng.normal(80, 120, 24))
            exact = dp.solve(prices, paper, initialSoc=config.initial_soc,
                             calendar=False)
            milp = solve(prices, paper, initialSoc=config.initial_soc, calendar=False,
                         discretePower=True, gap=0.02)
            assert exact.profit <= milp.dualBound + 1e-6

    @needsPrices
    @pytest.mark.parametrize("day", [datetime(2024, 8, 5, 4), datetime(2024, 8, 6, 4)])
    def test_agrees_on_a_real_solved_day(self, paper, day):
        """A full 288-interval NSW1 trading day, at the backtest's own settings.

        Slow: the MILP side is minutes. This is the comparison the backtest rests on.
        """
        from data.prices import price_series

        prices = list(
            price_series(day, day + timedelta(days=1), "NSW1")
            .to_numpy(dtype=float)[:config.episode_intervals]
        )
        assert len(prices) == config.episode_intervals
        exact = dp.solve(prices, paper, initialSoc=config.initial_soc, calendar=False)
        milp = solve(prices, paper, initialSoc=config.initial_soc, calendar=False,
                     discretePower=True, gap=config.backtest_gap,
                     timeLimit=config.backtest_time_limit)
        assert exact.profit >= milp.profit - 1e-9
        assert exact.profit <= milp.dualBound + 1e-6


# --- the DP is exact ---------------------------------------------------------


class TestIsOptimal:
    """Brute force. The reason for the whole change is that this is provable."""

    @pytest.mark.parametrize("prices", [MIXED_PRICES[:8], FALLING_PRICES,
                                        NEGATIVE_PRICES, FLAT_PRICES[:8]])
    def test_equals_the_best_of_every_policy(self, paper, prices):
        exact = dp.solve(prices, paper, initialSoc=config.initial_soc, calendar=False)
        best = max(
            rollProfit(prices, paper, actions)
            for actions in itertools.product((CHARGE, IDLE, DISCHARGE),
                                             repeat=len(prices))
        )
        assert exact.profit == pytest.approx(best, abs=1e-12)

    def test_equals_the_best_of_every_policy_with_calendar(self, paper):
        prices = MIXED_PRICES[:8]
        exact = dp.solve(prices, paper, initialSoc=config.initial_soc, calendar=True)
        best = max(
            rollProfit(prices, paper, actions, calendar=True)
            for actions in itertools.product((CHARGE, IDLE, DISCHARGE), repeat=8)
        )
        assert exact.profit == pytest.approx(best, abs=1e-12)

    def test_no_random_policy_beats_it_over_a_full_episode(self, paper):
        rng = np.random.default_rng(2)
        prices = list(rng.normal(80, 150, config.episode_intervals))
        exact = dp.solve(prices, paper, initialSoc=config.initial_soc, calendar=False)
        for _ in range(50):
            actions = rng.integers(0, 3, len(prices))
            assert rollProfit(prices, paper, actions) <= exact.profit + 1e-9

    def test_reported_schedule_scores_its_own_profit(self, paper):
        """Replaying the returned actions must reproduce the returned profit."""
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        replay = rollProfit(MIXED_PRICES, paper, actionsFrom(exact))
        assert replay == pytest.approx(exact.profit, abs=1e-12)

    def test_the_bound_is_the_profit(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert exact.dualBound == exact.profit
        assert exact.mipGap == 0.0
        assert exact.status == "Optimal"

    def test_flat_prices_sell_the_free_charge_and_buy_nothing(self, paper):
        """No spread, so a round trip always loses to efficiency and degradation. But
        the pack starts half full, and that energy is free, so the oracle sells it."""
        exact = dp.solve(FLAT_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert exact.bought == pytest.approx([0.0] * len(FLAT_PRICES), abs=1e-12)
        assert sum(exact.sold) > 0.0
        assert exact.profit > 0.0
        assert exact.soc[-1] < config.initial_soc


# --- the DP against the environment ------------------------------------------


class TestMatchesTheEnvironment:
    """The invariant in CLAUDE.md: the oracle and the environment are one problem."""

    def test_move_is_the_environment_action(self, paper):
        """dp.move must be BessEnv._applyAction, at every SoC and every action."""
        env = BessEnv(priceSeries(MIXED_PRICES), paper, horizon=len(MIXED_PRICES),
                      calendar=False, rewardScale=1.0)
        for soc in np.linspace(0.0, 1.0, 41):
            for action in (CHARGE, IDLE, DISCHARGE):
                env.stored = soc * paper.MWh
                bought, sold = env._applyAction(action)
                new, dpBought, dpSold = dp.move(soc * paper.MWh, action, paper,
                                                nem_interval_hours)
                assert dpBought == pytest.approx(bought, abs=1e-15)
                assert dpSold == pytest.approx(sold, abs=1e-15)
                assert new == pytest.approx(env.stored, abs=1e-15)

    def test_soc_path_is_identical(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        env = BessEnv(priceSeries(MIXED_PRICES), paper, horizon=len(MIXED_PRICES),
                      calendar=False, rewardScale=1.0)
        env.reset(seed=0)
        got = [env.step(a)[4]["soc"] for a in actionsFrom(exact)]
        assert got == pytest.approx(exact.soc, abs=1e-9)

    @pytest.mark.parametrize("calendar", [False, True])
    def test_profit_is_identical(self, paper, calendar):
        """Exact in both modes, unlike the MILP, which needs rel=1e-4 with calendar on."""
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=calendar)
        env = BessEnv(priceSeries(MIXED_PRICES), paper, horizon=len(MIXED_PRICES),
                      calendar=calendar, rewardScale=1.0)
        env.reset(seed=0)
        info = {}
        for action in actionsFrom(exact):
            info = env.step(action)[4]
        assert info["episode_totals"]["profit"] == pytest.approx(exact.profit, abs=1e-9)

    def test_degradation_is_identical(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        env = BessEnv(priceSeries(MIXED_PRICES), paper, horizon=len(MIXED_PRICES),
                      calendar=False, rewardScale=1.0)
        env.reset(seed=0)
        total = sum(env.step(a)[4]["degradation"] for a in actionsFrom(exact))
        assert total == pytest.approx(exact.degradationCost, abs=1e-12)

    def test_no_agent_can_beat_it_in_the_environment(self, paper):
        """The point of the bound. Random masked rollouts must stay under it."""
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        env = BessEnv(priceSeries(MIXED_PRICES), paper, horizon=len(MIXED_PRICES),
                      calendar=False, rewardScale=1.0)
        rng = np.random.default_rng(0)
        for _ in range(30):
            env.reset(seed=0)
            info = {}
            for _ in range(len(MIXED_PRICES)):
                allowed = np.flatnonzero(env.action_masks())
                info = env.step(int(rng.choice(allowed)))[4]
            assert info["episode_totals"]["profit"] <= exact.profit + 1e-9


# --- the state space ---------------------------------------------------------


class TestStateSpace:
    def test_the_starting_energy_is_state_zero(self, paper):
        space = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, 24)
        assert space.energy[0] == pytest.approx(config.initial_soc * paper.MWh, abs=1e-15)

    def test_stays_small_over_a_full_episode(self, paper):
        """Guards the reason the DP is cheap. A change to the MW-to-MWh ratio that
        multiplies this would make the DP slow without failing anything else."""
        space = dp.stateSpace(paper, config.initial_soc, nem_interval_hours,
                              config.episode_intervals)
        assert len(space.energy) < 8000

    def test_grows_about_linearly(self, paper):
        """Two intervals of growth, not an exponential tree."""
        small = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, 48)
        large = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, 96)
        assert len(large.energy) < 3 * len(small.energy)

    def test_every_energy_is_inside_the_pack(self, paper):
        space = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, 96)
        assert space.energy.min() >= -1e-15
        assert space.energy.max() <= paper.MWh + 1e-15

    def test_successors_of_a_reachable_state_are_reachable_next(self, paper):
        """What lets the DP skip a per-layer mask. If this breaks, values from states
        the episode cannot reach could leak into ones it can."""
        horizon = 48
        space = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, horizon)
        for t in range(horizon):
            for i in space.reachable[t]:
                for action in (CHARGE, IDLE, DISCHARGE):
                    assert int(space.successor[action][i]) in space.reachable[t + 1]

    def test_is_cached_per_settings(self, paper):
        first = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, 24)
        again = dp.stateSpace(paper, config.initial_soc, nem_interval_hours, 24)
        other = dp.stateSpace(paper, 0.25, nem_interval_hours, 24)
        assert first is again
        assert first is not other

    def test_a_full_pack_cannot_charge_and_an_empty_one_cannot_discharge(self, paper):
        full, bought, _ = dp.move(paper.MWh, CHARGE, paper, nem_interval_hours)
        assert bought == 0.0
        assert full == pytest.approx(paper.MWh, abs=1e-15)
        empty, _, sold = dp.move(0.0, DISCHARGE, paper, nem_interval_hours)
        assert sold == 0.0
        assert empty == 0.0

    def test_a_partial_top_up_happens_only_at_the_boundary(self, paper):
        """The rule the MILP needs atFull/atEmpty binaries to express."""
        full = paper.chargePower * nem_interval_hours
        nearlyFull = paper.MWh - 0.5 * paper.chargeEfficiency * full
        _, bought, _ = dp.move(nearlyFull, CHARGE, paper, nem_interval_hours)
        assert 0.0 < bought < full
        _, bought, _ = dp.move(0.5 * paper.MWh, CHARGE, paper, nem_interval_hours)
        assert bought == pytest.approx(full, abs=1e-15)


# --- the value function ------------------------------------------------------


class TestValueFunction:
    def test_the_last_layer_is_worth_nothing(self, paper):
        value, _, _ = dp.valueFunction(MIXED_PRICES, paper, config.initial_soc,
                                       calendar=False)
        assert value[len(MIXED_PRICES)] == pytest.approx(0.0, abs=0.0)

    def test_the_first_layer_at_the_start_state_is_the_profit(self, paper):
        value, _, _ = dp.valueFunction(MIXED_PRICES, paper, config.initial_soc,
                                       calendar=False)
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert value[0][0] == pytest.approx(exact.profit, abs=1e-12)

    def test_every_reachable_state_holds_its_own_optimum(self, paper):
        """The free extra: the oracle at every state, not only along one path. Checked
        against a DP restarted from that state."""
        prices = MIXED_PRICES
        value, _, space = dp.valueFunction(prices, paper, config.initial_soc,
                                           calendar=False)
        t = 4
        for i in sorted(space.reachable[t])[:6]:
            soc = space.energy[i] / paper.MWh
            restarted = dp.solve(prices[t:], paper, initialSoc=soc, calendar=False)
            assert value[t][i] == pytest.approx(restarted.profit, abs=1e-9)


# --- the Schedule it returns -------------------------------------------------


class TestSchedule:
    def test_the_totals_are_the_sums_of_the_parts(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=True)
        assert exact.revenue == pytest.approx(sum(exact.revenuePerInterval), abs=1e-12)
        assert exact.profit == pytest.approx(sum(exact.profitPerInterval), abs=1e-12)
        assert exact.degradationCost == pytest.approx(
            sum(exact.degradationPerInterval), abs=1e-12
        )
        assert exact.profit == pytest.approx(
            exact.revenue - exact.degradationCost, abs=1e-12
        )

    def test_the_calendar_term_is_dropped_when_asked(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert exact.calendarPerInterval == [0.0] * len(MIXED_PRICES)

    def test_the_calendar_term_is_present_and_costs_money_otherwise(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=True)
        assert all(k > 0.0 for k in exact.calendarPerInterval)

    def test_every_list_is_one_entry_per_interval(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=True)
        for field in (exact.soc, exact.bought, exact.sold, exact.revenuePerInterval,
                      exact.cyclePerInterval, exact.calendarPerInterval,
                      exact.degradationPerInterval, exact.profitPerInterval):
            assert len(field) == len(MIXED_PRICES)

    def test_the_soc_path_stays_inside_the_pack(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert min(exact.soc) >= -1e-15
        assert max(exact.soc) <= 1.0 + 1e-15

    def test_charging_and_discharging_never_share_an_interval(self, paper):
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert all(b == 0.0 or s == 0.0 for b, s in zip(exact.bought, exact.sold))

    def test_throughput_is_the_rated_rate_or_a_boundary_top_up(self, paper):
        """The fairness condition: the oracle may only use the agent's own action set."""
        full = paper.chargePower * nem_interval_hours
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        socs = [config.initial_soc] + list(exact.soc)
        for prev, bought, sold in zip(socs, exact.bought, exact.sold):
            moved = max(bought, sold)
            atBoundary = prev > 1.0 - 1e-9 or prev < 1e-9 or moved < full - 1e-12
            assert moved <= full + 1e-12
            assert moved in (0.0,) or moved == pytest.approx(full, abs=1e-12) or atBoundary

    def test_a_single_interval_episode_works(self, paper):
        exact = dp.solve([100.0], paper, initialSoc=config.initial_soc, calendar=False)
        assert len(exact.soc) == 1
        assert exact.profit >= 0.0

    def test_is_deterministic(self, paper):
        first = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=True)
        again = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=True)
        assert first.profit == again.profit
        assert first.soc == again.soc

    def test_accepts_a_numpy_array_of_prices(self, paper):
        fromList = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                            calendar=False)
        fromArray = dp.solve(np.array(MIXED_PRICES), paper,
                             initialSoc=config.initial_soc, calendar=False)
        assert fromList.profit == pytest.approx(fromArray.profit, abs=1e-15)


# --- the wiring in rl/baseline.py --------------------------------------------


class TestBaselineUsesTheDp:
    def test_a_discrete_window_comes_back_exact(self, paper, tmp_path):
        from rl import baseline

        prices = priceSeries(MIXED_PRICES)
        record = baseline.solveWindow(
            list(prices.to_numpy(dtype=float)), list(prices.index), "SA1",
            horizon=len(MIXED_PRICES), calendar=False, discretePower=True, model=paper,
            cacheDir=tmp_path,
        )
        assert record["mipGap"] == 0.0
        assert record["dualBound"] == pytest.approx(record["profit"], abs=1e-12)
        assert not baseline.timedOut([record], config.backtest_gap)

    def test_the_cache_round_trips(self, paper, tmp_path):
        from rl import baseline

        prices = priceSeries(MIXED_PRICES)
        args = (list(prices.to_numpy(dtype=float)), list(prices.index), "SA1")
        kwargs = dict(horizon=len(MIXED_PRICES), calendar=False, discretePower=True,
                      model=paper, cacheDir=tmp_path)
        first = baseline.solveWindow(*args, **kwargs)
        assert len(list(Path(tmp_path).glob("*.json"))) == 1
        again = baseline.solveWindow(*args, **kwargs)
        assert first == again

    def test_the_key_separates_the_dp_from_the_milp(self, paper):
        from rl import baseline

        start = priceSeries(MIXED_PRICES).index[0]
        discrete = baseline.cacheKey("SA1", start, 288, False, True, paper, 0.5, 0.02)
        continuous = baseline.cacheKey("SA1", start, 288, False, False, paper, 0.5, 0.02)
        assert discrete != continuous

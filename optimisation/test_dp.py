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
            .to_numpy(dtype=float)[:config.observation_intervals]
        )
        assert len(prices) == config.observation_intervals
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

    def test_flat_prices_trade_nothing_from_an_empty_pack(self, paper):
        """No spread, so a round trip always loses to efficiency and degradation, and an
        empty pack has nothing to sell either. `config.initial_soc` is zero, so this is
        the case the backtest solves."""
        exact = dp.solve(FLAT_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert exact.bought == pytest.approx([0.0] * len(FLAT_PRICES), abs=1e-12)
        assert exact.sold == pytest.approx([0.0] * len(FLAT_PRICES), abs=1e-12)
        assert exact.profit == pytest.approx(0.0, abs=1e-12)

    def test_flat_prices_sell_a_charge_the_pack_was_given(self, paper):
        """Why `config.initial_soc` is zero. Start the pack half full and the oracle
        sells that energy into a market with no spread at all, because it never paid for
        it. That profit comes from the episode boundary, not from trading."""
        exact = dp.solve(FLAT_PRICES, paper, initialSoc=0.5, calendar=False)
        assert exact.bought == pytest.approx([0.0] * len(FLAT_PRICES), abs=1e-12)
        assert sum(exact.sold) > 0.0
        assert exact.profit > 0.0
        assert exact.soc[-1] < 0.5


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

    def test_saturates_over_a_long_episode(self, paper):
        """Guards the reason the DP is cheap, and the reason a week costs no more than a
        day. The reachable energies stop growing well before a weekly horizon, so the
        state count is the same at 1008 intervals and at 2016."""
        counts = [len(dp.stateSpace(paper, config.initial_soc, nem_interval_hours, h)
                      .energy) for h in (288, 1008, 2016)]
        assert counts[1] == counts[2]
        assert counts[0] < counts[1]
        assert counts[2] < 8000

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

    def test_the_last_layer_holds_the_terminal_value(self, paper):
        value, _, space = dp.valueFunction(MIXED_PRICES, paper, config.initial_soc,
                                           calendar=False, terminalPrice=100.0)
        expected = [paper.terminalValue(e, 100.0) for e in space.energy]
        assert value[len(MIXED_PRICES)] == pytest.approx(expected, abs=1e-12)

    def test_the_first_layer_at_the_start_state_is_the_profit(self, paper):
        value, _, _ = dp.valueFunction(MIXED_PRICES, paper, config.initial_soc,
                                       calendar=False)
        exact = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                         calendar=False)
        assert value[0][0] == pytest.approx(exact.profit, abs=1e-12)

    @pytest.mark.parametrize("cyclePrice", [0.0, 10.0, 60.0])
    def test_every_reachable_state_holds_its_own_optimum(self, paper, cyclePrice):
        """The free extra: the oracle at every state, not only along one path. Checked
        against a DP restarted from that state. The rent changes which action wins but
        not that the value is the optimum of the problem it states, so this holds at any
        price."""
        prices = MIXED_PRICES
        value, _, space = dp.valueFunction(prices, paper, config.initial_soc,
                                           calendar=False, cyclePrice=cyclePrice)
        t = 4
        for i in sorted(space.reachable[t])[:6]:
            soc = space.energy[i] / paper.MWh
            restarted, _, tail = dp.valueFunction(prices[t:], paper, soc, calendar=False,
                                                  cyclePrice=cyclePrice)
            assert value[t][i] == pytest.approx(restarted[0][0], abs=1e-9)
            assert tail.energy[0] == pytest.approx(space.energy[i], abs=1e-15)


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


# --- the cycle budget --------------------------------------------------------


def efcOf(schedule, model):
    """The equivalent full cycles a schedule spends: cell-side energy removed over the
    nominal usable capacity. Read off `sold`, independent of what the solver reported."""
    return sum(schedule.sold) / model.dischargeEfficiency / model.MWh


def rollUsage(prices, model, actions, initialSoc=config.initial_soc,
              dtHours=nem_interval_hours):
    """The EFC an action sequence spends, walked independently of the DP's own bookkeeping."""
    stored = initialSoc * model.MWh
    used = 0.0
    for _, action in zip(prices, actions):
        new, _, sold = dp.move(stored, action, model, dtHours)
        used += sold / model.dischargeEfficiency
        stored = new
    return used / model.MWh


def bruteForce(prices, model, cycles, initialSoc=config.initial_soc, calendar=False,
               dtHours=nem_interval_hours):
    """The best profit over every action sequence that obeys the budget. Exponential, so
    only for horizons of about 8."""
    import itertools

    budget = None if cycles is None else cycles * model.MWh
    best = None
    for actions in itertools.product((CHARGE, IDLE, DISCHARGE), repeat=len(prices)):
        stored, used, total = initialSoc * model.MWh, 0.0, 0.0
        for price, action in zip(prices, actions):
            new, bought, sold = dp.move(stored, action, model, dtHours)
            used += sold / model.dischargeEfficiency
            if budget is not None and used > budget + 1e-12:
                break
            qCd, qCal, both = model.step(stored / model.MWh, new / model.MWh, dtHours)
            total += price * (sold - bought) - (both if calendar else model.cost(qCd))
            stored = new
        else:
            best = total if best is None else max(best, total)
    return best


def exactBudgeted(prices, model, cycles, initialSoc=config.initial_soc, calendar=False,
                  dtHours=nem_interval_hours):
    """An exact DP over the joint (energy, energy removed) state, as the second reference.

    This is what `dp.solve(cycles=...)` deliberately does not do: the usage dimension does
    not saturate, so the state count grows without bound and a weekly horizon is out of
    reach. At 24 to 48 intervals it is milliseconds, and it is the only reference that
    exercises a discharge clipped by the remaining budget rather than by the SoC floor.
    """
    budget = cycles * model.MWh
    start = initialSoc * model.MWh
    # Keyed on the rounded (energy, usage) pair so two paths to one physical state merge,
    # but the state itself travels as the exact floats the walk reached. Reconstructing
    # energy from its key instead leaks rounding error into the energy and the reference
    # then creeps above the true optimum.
    states = {(dp._StateSpace._key(start), 0): (0.0, start, 0.0)}
    for price in prices:
        nxt = {}
        for key, (total, stored, used) in states.items():
            for action in (CHARGE, IDLE, DISCHARGE):
                new, bought, sold = dp.move(stored, action, model, dtHours)
                headroom = max(0.0, budget - used)
                removed = sold / model.dischargeEfficiency
                if removed > headroom:
                    sold = headroom * model.dischargeEfficiency
                    removed = headroom
                    new = stored - removed
                qCd, qCal, both = model.step(stored / model.MWh, new / model.MWh, dtHours)
                gained = price * (sold - bought) - (both if calendar else model.cost(qCd))
                nextUsed = used + removed
                nextKey = (dp._StateSpace._key(new), dp._StateSpace._key(nextUsed))
                if nextKey not in nxt or nxt[nextKey][0] < total + gained:
                    nxt[nextKey] = (total + gained, new, nextUsed)
        states = nxt
    return max(total for total, _, _ in states.values())


class TestCycleBudget:
    """`solve(cycles=...)` caps the equivalent full cycles a week of dispatch may spend.

    The cap is not a state dimension. It is priced: every MWh removed cell-side pays an
    extra `cyclePrice`, and the price is bisected until usage lands on the budget. So the
    answer arrives as a sandwich rather than a single number, and these tests pin both
    ends of it.
    """

    def test_an_unbinding_budget_changes_nothing(self, paper):
        free = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False)
        capped = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=100.0)
        assert capped.profit == pytest.approx(free.profit, abs=1e-12)
        assert capped.soc == pytest.approx(free.soc, abs=1e-12)
        assert capped.sold == pytest.approx(free.sold, abs=1e-12)
        assert capped.mipGap == 0.0
        assert capped.dualBound == pytest.approx(capped.profit, abs=1e-12)

    def test_an_unbinding_budget_does_not_search(self, paper):
        """No bisection runs when the free optimum already fits, so nothing is priced."""
        capped = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=100.0)
        assert capped.cyclePrice == 0.0

    def test_a_zero_budget_never_discharges(self, paper):
        capped = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=0.0)
        assert capped.sold == pytest.approx([0.0] * len(MIXED_PRICES), abs=1e-12)
        assert capped.cyclesUsed == pytest.approx(0.0, abs=1e-12)

    def test_a_full_discharge_is_exactly_one_cycle(self, paper):
        """The definition: one EFC is the nominal usable capacity removed cell-side."""
        stored, removed = paper.MWh, 0.0
        while stored > 0:
            new, _, sold = dp.move(stored, DISCHARGE, paper, nem_interval_hours)
            removed += sold / paper.dischargeEfficiency
            stored = new
        assert removed / paper.MWh == pytest.approx(1.0, abs=1e-12)

    def test_a_partial_discharge_counts_proportionally(self, paper):
        stored, removed = 0.5 * paper.MWh, 0.0
        while stored > 0:
            new, _, sold = dp.move(stored, DISCHARGE, paper, nem_interval_hours)
            removed += sold / paper.dischargeEfficiency
            stored = new
        assert removed / paper.MWh == pytest.approx(0.5, abs=1e-12)

    def test_the_budget_is_cell_side_not_meter_side(self, paper):
        """A full pack emptied at one cycle sells nu_D times the capacity, not the
        capacity. Getting this the wrong way round is a 5% error in the budget."""
        capped = dp.solve([500.0] * 40, paper, initialSoc=1.0, calendar=False, cycles=1.0)
        assert sum(capped.sold) == pytest.approx(paper.MWh * paper.dischargeEfficiency,
                                                 abs=1e-9)
        assert capped.cyclesUsed == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("cycles", [0.25, 0.5, 1.0, 2.0])
    def test_the_reported_schedule_respects_the_budget(self, paper, cycles):
        capped = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=cycles)
        assert efcOf(capped, paper) <= cycles + 1e-9
        assert rollUsage(MIXED_PRICES, paper, actionsFrom(capped)) <= cycles + 1e-9

    def test_cycles_used_is_what_the_schedule_spent(self, paper):
        capped = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=0.5)
        assert capped.cyclesUsed == pytest.approx(efcOf(capped, paper), abs=1e-12)

    def test_profit_is_non_increasing_as_the_budget_tightens(self, paper):
        ladder = [4.0, 2.0, 1.0, 0.5, 0.25, 0.0]
        profits = [dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=c).profit
                   for c in ladder]
        assert all(b <= a + 1e-9 for a, b in zip(profits, profits[1:]))

    def test_usage_is_non_increasing_in_the_cycle_price(self, paper):
        """What makes the bisection valid. Rent only ever discourages discharging, so
        usage falls as the price rises, and a predicate search on it is sound."""
        space = dp.stateSpace(paper, config.initial_soc, nem_interval_hours,
                              len(MIXED_PRICES))
        used = []
        for price in (0.0, 25.0, 50.0, 100.0, 200.0, 400.0, 1000.0):
            _, action, _ = dp.valueFunction(np.asarray(MIXED_PRICES, dtype=float), paper,
                                            config.initial_soc, nem_interval_hours,
                                            calendar=False, cyclePrice=price, space=space)
            index, total = 0, 0.0
            for step in range(len(MIXED_PRICES)):
                chosen = int(action[step][index])
                total += float(space.cellSide[chosen][index])
                index = int(space.successor[chosen][index])
            used.append(total)
        assert all(b <= a + 1e-12 for a, b in zip(used, used[1:]))
        assert used[0] > used[-1]
        assert used[-1] == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("prices", [MIXED_PRICES[:8], NEGATIVE_PRICES, FALLING_PRICES])
    @pytest.mark.parametrize("cycles", [0.25, 0.5, 1.0])
    def test_the_sandwich_holds_around_the_brute_force_optimum(self, paper, prices, cycles):
        """profit is achievable, dualBound is a ceiling, and the truth is between them."""
        capped = dp.solve(prices, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=cycles)
        best = bruteForce(prices, paper, cycles)
        assert capped.profit <= best + 1e-12
        assert best <= capped.dualBound + 1e-12

    @pytest.mark.parametrize("cycles", [0.25, 0.5, 1.0])
    def test_it_is_exact_wherever_the_gap_closes(self, paper, cycles):
        capped = dp.solve(MIXED_PRICES[:8], paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=cycles)
        if capped.mipGap == 0.0:
            assert capped.profit == pytest.approx(
                bruteForce(MIXED_PRICES[:8], paper, cycles), abs=1e-12)

    @pytest.mark.parametrize("length", [24, 48])
    def test_the_sandwich_holds_around_the_exact_joint_dp(self, paper, length):
        prices = (MIXED_PRICES * 4)[:length]
        capped = dp.solve(prices, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=1.0)
        exact = exactBudgeted(prices, paper, 1.0)
        assert capped.profit <= exact + 1e-9
        assert exact <= capped.dualBound + 1e-9

    def test_the_two_exact_references_agree(self, paper):
        """Brute force cannot express a discharge clipped by the budget, so it can only
        sit at or below the joint DP. If it ever sat above, one of them is wrong."""
        prices = MIXED_PRICES[:8]
        assert bruteForce(prices, paper, 0.5) <= exactBudgeted(prices, paper, 0.5) + 1e-12

    @pytest.mark.parametrize("length", [8, 12])
    @pytest.mark.parametrize("cycles", [0.05, 0.5])
    def test_it_matches_the_milp_cycle_row(self, paper, length, cycles):
        """The MILP states the budget as one exact row, so it is the second opinion on
        what the priced search is allowed to return. Tolerances are the solver's own:
        HiGHS satisfies the row to its feasibility tolerance, not exactly."""
        prices = MIXED_PRICES[:length]
        milp = solve(prices, paper, initialSoc=config.initial_soc, calendar=False,
                     discretePower=True, cycles=cycles)
        capped = dp.solve(prices, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=cycles)
        assert sum(milp.sold) / paper.dischargeEfficiency <= cycles * paper.MWh + 1e-6
        assert capped.profit <= milp.profit + 1e-6
        assert milp.profit <= capped.dualBound + 1e-5

    def test_the_bisection_terminates_on_flat_prices(self, paper):
        capped = dp.solve(FLAT_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=1.0)
        assert capped.mipGap == 0.0
        assert capped.cyclesUsed == pytest.approx(0.0, abs=1e-12)

    def test_the_state_space_is_shared_across_cycle_prices(self, paper):
        """The budget adds no state, so one build serves every price the bisection tries."""
        before = dp.stateSpace(paper, config.initial_soc, nem_interval_hours,
                               len(MIXED_PRICES))
        dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=0.5)
        after = dp.stateSpace(paper, config.initial_soc, nem_interval_hours,
                              len(MIXED_PRICES))
        assert before is after

    def test_the_cycle_price_reaches_no_reported_number(self, paper):
        """The rent is a search device. If it leaked into the costs the oracle and the
        environment would stop being the same problem, and the replay would drift."""
        capped = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=True, cycles=0.05)
        assert capped.cyclePrice > 0.0
        assert capped.degradationCost == pytest.approx(
            sum(capped.cyclePerInterval) + sum(capped.calendarPerInterval), abs=1e-12)
        assert capped.profit == pytest.approx(capped.revenue - capped.degradationCost,
                                              abs=1e-12)
        assert capped.profit == pytest.approx(
            rollProfit(MIXED_PRICES, paper, actionsFrom(capped), calendar=True),
            abs=1e-9)


class TestTerminalValue:
    """Closing stock is worth what selling it would net.

    Without this nothing values energy held at the close, so the last interval's best move
    is always to sell whatever is left at whatever it pays. On a one-day episode that is a
    few cents. On a week it is a Sunday-night dump.
    """

    def test_a_zero_price_is_the_old_problem(self, paper):
        """Bit-identical, so every oracle record solved before this still means itself."""
        old = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False)
        same = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, terminalPrice=0.0)
        assert same.profit == old.profit
        assert same.soc == old.soc

    def test_closing_stock_is_never_worth_more_than_selling_it(self, paper):
        for soc in (0.0, 0.25, 0.5, 1.0):
            stored = soc * paper.MWh
            value = paper.terminalValue(stored, 100.0)
            assert value <= 100.0 * stored * paper.dischargeEfficiency + 1e-12
            assert value >= 0.0 or soc == 0.0

    def test_an_empty_pack_is_worth_nothing(self, paper):
        assert paper.terminalValue(0.0, 500.0) == pytest.approx(0.0, abs=1e-12)

    def test_it_nets_off_the_cost_of_releasing_the_energy(self, paper):
        stored = paper.MWh
        gross = 100.0 * stored * paper.dischargeEfficiency
        release = paper.cost(paper.totalCycleDegradation(1.0, 0.0))
        assert paper.terminalValue(stored, 100.0) == pytest.approx(gross - release,
                                                                   abs=1e-12)

    def test_a_held_pack_is_not_dumped_into_the_last_interval(self, paper):
        """Prices fall through the window, so with no terminal value the oracle still
        sells the tail at the worst price of the day rather than hold it."""
        dumped = dp.solve(FALLING_PRICES, paper, initialSoc=1.0, calendar=False)
        held = dp.solve(FALLING_PRICES, paper, initialSoc=1.0, calendar=False,
                        terminalPrice=200.0)
        assert held.soc[-1] > dumped.soc[-1]

    def test_the_value_is_reported_apart_from_the_trading(self, paper):
        held = dp.solve(FALLING_PRICES, paper, initialSoc=1.0, calendar=False,
                        terminalPrice=200.0)
        closing = held.soc[-1] * paper.MWh
        assert held.terminalValue == pytest.approx(
            paper.terminalValue(closing, 200.0), abs=1e-9)
        assert held.profit == pytest.approx(
            held.revenue - held.degradationCost + held.terminalValue, abs=1e-9)

    def test_it_travels_with_the_budget(self, paper):
        both = dp.solve(MIXED_PRICES, paper, initialSoc=config.initial_soc,
                          calendar=False, cycles=0.5,
                        terminalPrice=100.0)
        assert efcOf(both, paper) <= 0.5 + 1e-9
        assert both.profit <= both.dualBound + 1e-9

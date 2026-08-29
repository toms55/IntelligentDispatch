"""
Tests for optimisation/lp.py, the perfect-foresight oracle (TODO.txt Step 1).

Run with:   nix develop -c pytest optimisation/ -q

The oracle sees the true realised price series and returns the best possible
dispatch against it. Its $ figure is the bound the RL agent is measured against,
so the tests here are mostly about the bound being *honest*: the LP must not be
able to book a cheaper degradation cost than the causal model in schade.py would
charge for the same SoC trajectory (TestDegradationIsHonest).
"""
import pytest

from degradation.schade import SchadeDegradation
from optimisation.lp import fillBottomUp, solve
from utils import nem_interval_hours


PAPER_MW = 0.060
PAPER_MWH = 0.100
PAPER_REPLACEMENT_COST = 150_000.0

# Eqs. 6-7: MW is a rate, the LP's variables are energy.
INTERVAL_LIMIT = PAPER_MW * nem_interval_hours


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


def recomputedDegradation(model, schedule, calendar):
    """Score the schedule with schade.py directly, independent of the LP."""
    total = 0.0
    socs = [schedule.initialSoc] + list(schedule.soc)
    for prev, new in zip(socs, socs[1:]):
        q = model.totalCycleDegradation(prev, new)
        if calendar:
            q += model.calendarAging(new)
        total += model.cost(q)
    return total


class TestBounds:
    """Never charge past capacity, never discharge below zero, never exceed MW."""

    def test_soc_stays_within_zero_and_one(self, paper):
        prices = [0.0, 0.0, 0.0, 500.0, 500.0, 500.0, 0.0, 0.0]
        s = solve(prices, paper, initialSoc=0.5, calendar=False)
        for soc in s.soc:
            assert -1e-9 <= soc <= 1.0 + 1e-9

    def test_cannot_charge_beyond_full(self, paper):
        # Free energy for far longer than it takes to fill: must stop at 1.0.
        s = solve([-1000.0] * 60, paper, initialSoc=0.5, calendar=False)
        assert max(s.soc) <= 1.0 + 1e-9
        assert s.soc[-1] == pytest.approx(1.0, abs=1e-6)

    def test_cannot_discharge_beyond_empty(self, paper):
        s = solve([10_000.0] * 60, paper, initialSoc=0.5, calendar=False)
        assert min(s.soc) >= -1e-9
        assert s.soc[-1] == pytest.approx(0.0, abs=1e-6)

    def test_respects_power_limit(self, paper):
        prices = [-1000.0] * 5 + [10_000.0] * 5
        s = solve(prices, paper, initialSoc=0.5, calendar=False)
        for boughtMWh, soldMWh in zip(s.bought, s.sold):
            assert boughtMWh <= INTERVAL_LIMIT + 1e-9
            assert soldMWh <= INTERVAL_LIMIT + 1e-9


class TestDegradationIsHonest:
    """The staircase fill: a segment may hold energy only if the one below is full.

    Eq. 5 caps each segment but never forces them to fill from the bottom, so
    without this the solver books discharges against whichever segments are
    cheapest rather than the ones physically drained. Mere ordering
    (stored[s] <= stored[s-1]) does not fix it -- the solver just drains all 16 segments
    uniformly and still bills the cheap bottom ones, understating a full-pack
    discharge by ~2x. An oracle that under-reports its own costs is not a bound
    on the agent at all.
    """

    def test_matches_schade_recompute(self, paper):
        prices = [0.0] * 4 + [5_000.0] * 8 + [0.0] * 4
        s = solve(prices, paper, initialSoc=0.8, calendar=False)
        assert s.degradationCost == pytest.approx(
            recomputedDegradation(paper, s, calendar=False), rel=1e-6
        )

    def test_deep_discharge_costs_the_top_segments(self, paper):
        # From full, price high enough that discharging is always worth it.
        s = solve([10_000.0] * 3, paper, initialSoc=1.0, calendar=False)
        expected = paper.cost(paper.totalCycleDegradation(1.0, s.soc[-1]))
        assert s.degradationCost == pytest.approx(expected, rel=1e-6)

    def test_costs_more_than_the_cheapest_segments_would(self, paper):
        """The failure mode, stated as a number: draining the top of the pack
        must not be priced as if it came off the bottom."""
        s = solve([10_000.0] * 3, paper, initialSoc=1.0, calendar=False)
        drained = 1.0 - s.soc[-1]
        cheapest = paper.cost(paper.totalCycleDegradation(drained, 0.0))
        assert s.degradationCost > 5 * cheapest


class TestNoSimultaneousChargeDischarge:
    """The NEM price floor of -$1000/MWh makes round-tripping at a loss look
    profitable to a naive formulation. If this fires, exclusivity binaries are
    needed on the flagged intervals."""

    def test_never_both_in_one_interval(self, paper):
        prices = [-1000.0, -1000.0, -900.0, 300.0, -1000.0, 50.0]
        s = solve(prices, paper, initialSoc=0.5, calendar=False)
        for boughtMWh, soldMWh in zip(s.bought, s.sold):
            assert min(boughtMWh, soldMWh) == pytest.approx(0.0, abs=1e-9)


class TestObjective:
    def test_zero_prices_mean_no_trading(self, paper):
        """Nothing to earn, and cycling costs money, so the battery sits.

        Note a *flat non-zero* price is not the same test: marginal cost rises
        with SoC, so at a flat $50 the battery profitably sells its cheap lower
        segments down to where marginal cost meets the price."""
        s = solve([0.0] * 12, paper, initialSoc=0.5, calendar=False)
        assert sum(s.sold) == pytest.approx(0.0, abs=1e-9)
        assert s.profit == pytest.approx(0.0, abs=1e-9)

    def test_buys_low_and_sells_high(self, paper):
        # From empty, so selling in the spike requires buying in the trough first.
        prices = [10.0] * 6 + [1_000.0] * 6
        s = solve(prices, paper, initialSoc=0.0, calendar=False)
        assert sum(s.bought[:6]) > 0
        assert sum(s.sold[6:]) > 0
        assert s.profit > 0

    def test_profit_is_revenue_minus_degradation(self, paper):
        prices = [10.0] * 6 + [1_000.0] * 6
        s = solve(prices, paper, initialSoc=0.5, calendar=False)
        assert s.profit == pytest.approx(s.revenue - s.degradationCost, rel=1e-9)

    def test_longer_horizon_is_never_worse(self, paper):
        """Perfect foresight over more intervals cannot reduce the optimum."""
        prices = [10.0] * 6 + [1_000.0] * 6
        short = solve(prices[:6], paper, initialSoc=0.5, calendar=False)
        full = solve(prices, paper, initialSoc=0.5, calendar=False)
        assert full.profit >= short.profit - 1e-9


class TestCalendarAging:
    """Q^CAL is piecewise linear but NON-convex -- its slopes go
    16.7, 4.2, 84, 13.1 (x1e-7 per hour) -- so an LP relaxation would interpolate
    across the kinks and understate the cost. Needs the SOS2 encoding."""

    def test_costs_something_even_when_idle(self, paper):
        s = solve([0.0] * 12, paper, initialSoc=0.5, calendar=True)
        assert s.degradationCost > 0

    def test_matches_schade_recompute(self, paper):
        prices = [0.0] * 4 + [5_000.0] * 8
        s = solve(prices, paper, initialSoc=0.8, calendar=True)
        assert s.degradationCost == pytest.approx(
            recomputedDegradation(paper, s, calendar=True), rel=1e-4
        )

    def test_sos2_does_not_undercut_the_curve(self, paper):
        """At a fixed idle SoC the LP's calendar cost must equal np.interp, not
        the cheaper straight line the kinks would otherwise allow."""
        s = solve([0.0] * 12, paper, initialSoc=0.65, calendar=True)
        perInterval = paper.cost(paper.calendarAging(0.65))
        assert s.degradationCost == pytest.approx(12 * perInterval, rel=1e-4)

    def test_off_switch_drops_the_calendar_term(self, paper):
        s = solve([0.0] * 12, paper, initialSoc=0.5, calendar=False)
        assert s.degradationCost == pytest.approx(0.0, abs=1e-9)


class TestFillBottomUp:
    """Where the pack sits at t=0. Must be a staircase, or the first interval
    starts in a state the constraints forbid."""

    def test_full_pack_fills_every_segment(self, paper):
        assert fillBottomUp(1.0, paper.segmentCapacities) == pytest.approx(
            paper.segmentCapacities
        )

    def test_empty_pack_fills_nothing(self, paper):
        assert sum(fillBottomUp(0.0, paper.segmentCapacities)) == 0.0

    def test_conserves_energy(self, paper):
        levels = fillBottomUp(0.3, paper.segmentCapacities)
        assert sum(levels) == pytest.approx(0.3 * paper.MWh)

    def test_is_a_staircase(self, paper):
        """Full below, empty above, at most one partial segment between."""
        levels = fillBottomUp(0.3, paper.segmentCapacities)
        partial = [
            n for n, (got, cap) in enumerate(zip(levels, paper.segmentCapacities))
            if 0 < got < cap
        ]
        assert len(partial) <= 1
        for n, (got, cap) in enumerate(zip(levels, paper.segmentCapacities)):
            if partial and n < partial[0]:
                assert got == pytest.approx(cap)
            elif partial and n > partial[0]:
                assert got == pytest.approx(0.0)

"""
Tests for degradation/schade.py. Equation numbers are the paper's own
(Schade & Egging-Bratseth 2024, doi:10.1002/est2.588).

Run with:   nix develop -c pytest degradation/ -q

Classes marked @notYet cover terms schade.py documents but has not implemented.
An AttributeError is expected; once the method exists and the test passes,
pytest reports XPASS as a failure, which is the cue to drop the marker. The
method names in those classes are proposals, not an agreed API.

@notYet only works for a test that reaches for a missing attribute. A test that
asserts on *behaviour* of code that already runs (e.g. a constructor that ought
to warn) fails some other way, so it carries @notYetBehaviour instead, which is
strict but does not pin the exception type.
"""
import math

import pytest

from degradation.schade import SchadeDegradation


# Table 2 case study: 60 kW / 100 kWh at 150 EUR/kWh, in MW / MWh / per-MWh.
PAPER_MW = 0.060
PAPER_MWH = 0.100
PAPER_REPLACEMENT_COST = 150_000.0


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


@pytest.fixture
def bess():
    return SchadeDegradation(50.0, 100.0, 200_000.0)


def stressRef(model, cd):
    """Eq. 1, recomputed independently of the implementation under test."""
    return model.maxCapacityLoss * cd ** (1.0 / model.fatigueStrengthExponent)


def refined(model, segments):
    """Re-cut the cycle-depth curve into `segments` pieces, in place.

    Every cached array hangs off SoC_breakpointVal, so moving the breakpoints
    without rebuilding all of them leaves the model quietly inconsistent --
    e.g. 256 segmentCosts against 16 segmentCapacities.
    """
    model.SoC_breakpointVal = [j / segments for j in range(segments + 1)]
    model.segmentCosts = model._segmentCosts()
    model.segmentCapacities = model._segmentCapacities()
    model.segmentMarginalCosts = [
        cost / cap for cost, cap in zip(model.segmentCosts, model.segmentCapacities)
    ]
    return model


def pwlRef(model, soc):
    """Piecewise-linear interpolant of Eq. 1 over the segment edges."""
    edges = model.SoC_breakpointVal
    for lo, hi in zip(edges, edges[1:]):
        if lo <= soc <= hi:
            t = (soc - lo) / (hi - lo)
            return stressRef(model, lo) + t * (stressRef(model, hi) - stressRef(model, lo))
    raise AssertionError(f"soc {soc} outside the breakpoint range")


class TestParameters:

    def test_table_2_values(self, paper):
        assert paper.maxCapacityLoss == pytest.approx(0.04519 / 100)
        assert paper.SoC_stress == pytest.approx(0.0085 / 100)
        assert paper.fatigueStrengthExponent == pytest.approx(0.4926)
        assert paper.chargeEfficiency == pytest.approx(0.95)
        assert paper.dischargeEfficiency == pytest.approx(0.95)

    def test_sizing_is_stored_verbatim(self, bess):
        assert bess.chargePower == 50.0
        assert bess.dischargePower == 50.0
        assert bess.MWh == 100.0
        assert bess.replacementCost == 200_000.0

    def test_cycle_depth_breakpoints_are_16_uniform_segments(self, paper):
        edges = paper.SoC_breakpointVal
        assert len(edges) == 17  # lower edges plus one final upper edge
        assert edges[0] == 0.0
        assert edges[-1] == 1.0
        widths = [hi - lo for lo, hi in zip(edges, edges[1:])]
        assert all(w == pytest.approx(1 / 16) for w in widths)

    def test_calendar_curve_is_keil_table_3(self, paper):
        assert paper.calendarAgingBreakpointVal == [0.00, 0.30, 0.60, 0.70, 1.00]
        assert paper.y == [3.75e-7, 8.76e-7, 10.01e-7, 18.41e-7, 22.34e-7]
        assert len(paper.y) == len(paper.calendarAgingBreakpointVal)

    def test_calendar_curve_is_monotone_with_a_plateau_then_a_step(self, paper):
        socs = paper.calendarAgingBreakpointVal
        rates = paper.y
        assert all(b > a for a, b in zip(rates, rates[1:]))
        slopes = [
            (r1 - r0) / (s1 - s0)
            for s0, s1, r0, r1 in zip(socs, socs[1:], rates, rates[1:])
        ]
        assert slopes[1] < slopes[0]
        assert slopes[2] > 5 * slopes[1]

    def test_calendar_rate_at_full_soc_is_two_percent_per_year(self, paper):
        # §5.1, and the check that y is a fraction per hour, not a percent.
        assert paper.y[-1] * 8760 == pytest.approx(0.0196, abs=1e-4)


class TestCycleDegradation:
    """Eq. 1."""

    def test_full_cycle_loses_max_capacity_loss(self, paper):
        assert paper.cycleDegradation(1.0) == pytest.approx(paper.maxCapacityLoss)

    def test_zero_depth_costs_nothing(self, paper):
        assert paper.cycleDegradation(0.0) == 0.0

    def test_exponent_is_the_reciprocal_of_the_stored_parameter(self, paper):
        for cd in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert paper.cycleDegradation(cd) == pytest.approx(stressRef(paper, cd))
        assert 1 / paper.fatigueStrengthExponent == pytest.approx(2.0300, abs=1e-4)
        assert paper.cycleDegradation(0.5) / paper.cycleDegradation(1.0) < 0.25

    def test_is_increasing(self, paper):
        vals = [paper.cycleDegradation(i / 200) for i in range(201)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_is_convex(self, paper):
        # Convexity licenses the segment construction and the LP embedding.
        vals = [paper.cycleDegradation(i / 200) for i in range(201)]
        seconds = [lo - 2 * mid + hi for lo, mid, hi in zip(vals, vals[1:], vals[2:])]
        assert all(s >= -1e-18 for s in seconds)

    def test_is_independent_of_system_size(self, paper, bess):
        for cd in (0.2, 0.6, 1.0):
            assert paper.cycleDegradation(cd) == pytest.approx(bess.cycleDegradation(cd))


class TestSegmentCosts:
    """Eq. 2."""

    def test_there_is_one_cost_per_segment(self, paper):
        assert len(paper.segmentCosts) == len(paper.SoC_breakpointVal) - 1 == 16

    def test_matches_equation_2(self, paper):
        edges = paper.SoC_breakpointVal
        expected = [
            paper.MWh * paper.replacementCost
            * (stressRef(paper, hi) - stressRef(paper, lo))
            for lo, hi in zip(edges, edges[1:])
        ]
        assert paper.segmentCosts == pytest.approx(expected)

    def test_costs_increase_with_segment_index(self, paper):
        costs = paper.segmentCosts
        assert all(b > a for a, b in zip(costs, costs[1:]))
        assert all(c > 0 for c in costs)

    def test_costs_telescope_to_the_cost_of_one_full_cycle(self, paper):
        total = paper.MWh * paper.replacementCost * paper.maxCapacityLoss
        assert sum(paper.segmentCosts) == pytest.approx(total)

    def test_scales_linearly_with_size_and_replacement_cost(self):
        base = SchadeDegradation(1.0, 10.0, 100_000.0)
        bigger = SchadeDegradation(1.0, 20.0, 100_000.0)
        dearer = SchadeDegradation(1.0, 10.0, 300_000.0)
        for b, big, dear in zip(base.segmentCosts, bigger.segmentCosts, dearer.segmentCosts):
            assert big == pytest.approx(2 * b)
            assert dear == pytest.approx(3 * b)

    def test_top_segment_costs_far_more_than_the_bottom_one(self, paper):
        assert paper.segmentCosts[-1] / paper.segmentCosts[0] > 25


class TestTotalCycleDegradation:
    """Eq. 11 under reading (i): Q^CD = PWL(prev) - PWL(new)."""

    def test_charging_and_idling_cost_nothing(self, paper):
        assert paper.totalCycleDegradation(0.4, 0.4) == 0.0
        assert paper.totalCycleDegradation(0.4, 0.9) == 0.0
        assert paper.totalCycleDegradation(0.0, 1.0) == 0.0

    def test_full_discharge_equals_one_full_cycle(self, paper):
        assert paper.totalCycleDegradation(1.0, 0.0) == pytest.approx(
            paper.cycleDegradation(1.0)
        )

    def test_exact_at_the_breakpoints(self, paper):
        edges = paper.SoC_breakpointVal
        for lo, hi in zip(edges, edges[1:]):
            expected = stressRef(paper, hi) - stressRef(paper, lo)
            assert paper.totalCycleDegradation(hi, lo) == pytest.approx(expected)

    def test_matches_the_piecewise_linear_interpolant(self, paper):
        for prev, new in [(1.0, 0.0), (0.9, 0.1), (0.55, 0.45), (0.3, 0.28), (0.07, 0.02)]:
            assert paper.totalCycleDegradation(prev, new) == pytest.approx(
                pwlRef(paper, prev) - pwlRef(paper, new)
            )

    def test_telescopes_under_subdivision(self, paper):
        # What makes Eq. 11 safe at 5-minute resolution.
        prev, new = 0.93, 0.11
        whole = paper.totalCycleDegradation(prev, new)
        for n in (2, 3, 7, 60, 288):
            socs = [prev + (new - prev) * i / n for i in range(n + 1)]
            pieces = sum(
                paper.totalCycleDegradation(a, b) for a, b in zip(socs, socs[1:])
            )
            assert pieces == pytest.approx(whole, rel=1e-12)

    def test_is_monotone_in_depth(self, paper):
        deeper = [paper.totalCycleDegradation(1.0, 1.0 - i / 100) for i in range(101)]
        assert all(b >= a for a, b in zip(deeper, deeper[1:]))

    def test_the_same_energy_costs_more_taken_from_the_top(self, paper):
        # 7.4x, not the 25x that separates the top and bottom SEGMENTS: a
        # quarter spans four segments, which averages the ratio down.
        top = paper.totalCycleDegradation(1.0, 0.75)
        bottom = paper.totalCycleDegradation(0.25, 0.0)
        assert top > 7 * bottom

    def test_contains_neither_replacement_cost_nor_rated_energy(self, paper, bess):
        for prev, new in [(1.0, 0.0), (0.8, 0.3), (0.45, 0.4)]:
            assert paper.totalCycleDegradation(prev, new) == pytest.approx(
                bess.totalCycleDegradation(prev, new)
            )

    def test_never_underestimates_the_true_curve(self, paper):
        # §5.2: a chord above a convex curve, so the error is one-sided.
        for i in range(1, 401):
            depth = i / 400
            assert paper.totalCycleDegradation(depth, 0.0) >= stressRef(paper, depth) - 1e-18

    def test_pwl_error_is_small_for_depths_worth_cycling(self, paper):
        # Max 1.31% at CD >= 0.25. Shallow cycles are over-estimated far more
        # (2x at 1/32 depth), which is a property of the linearisation.
        worst = max(
            (paper.totalCycleDegradation(i / 400, 0.0) - stressRef(paper, i / 400))
            / stressRef(paper, i / 400)
            for i in range(100, 401)
        )
        assert worst < 0.02

    def test_16_segment_grid_is_a_free_discretisation_choice(self):
        # §1.1: refining the cycle-depth grid is legitimate, and must converge
        # onto Eq. 1 from above.
        fine = refined(SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST), 256)
        coarse = SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)
        for depth in (0.3, 0.55, 0.77):
            true = stressRef(coarse, depth)
            assert true <= fine.totalCycleDegradation(depth, 0.0) + 1e-18
            assert fine.totalCycleDegradation(depth, 0.0) <= coarse.totalCycleDegradation(depth, 0.0)


class TestSegmentCapacities:
    """Eq. 5. Cell-side energy held by each segment -- the cap on the LP's
    SOC-segment state variable S[j][h]."""

    def test_there_is_one_capacity_per_segment(self, paper):
        assert len(paper.segmentCapacities) == len(paper.SoC_breakpointVal) - 1 == 16

    def test_matches_equation_5(self, paper):
        edges = paper.SoC_breakpointVal
        expected = [paper.MWh * (hi - lo) for lo, hi in zip(edges, edges[1:])]
        assert paper.segmentCapacities == pytest.approx(expected)

    def test_capacities_sum_to_rated_energy(self, paper, bess):
        assert sum(paper.segmentCapacities) == pytest.approx(paper.MWh)
        assert sum(bess.segmentCapacities) == pytest.approx(bess.MWh)

    def test_uniform_grid_gives_uniform_capacities(self, paper):
        assert all(c == pytest.approx(paper.MWh / 16) for c in paper.segmentCapacities)

    def test_scales_with_rated_energy_only(self):
        base = SchadeDegradation(1.0, 10.0, 100_000.0)
        bigger = SchadeDegradation(1.0, 20.0, 100_000.0)
        dearer = SchadeDegradation(1.0, 10.0, 300_000.0)
        stronger = SchadeDegradation(5.0, 10.0, 100_000.0)
        for b, big, dear, strong in zip(
            base.segmentCapacities, bigger.segmentCapacities,
            dearer.segmentCapacities, stronger.segmentCapacities,
        ):
            assert big == pytest.approx(2 * b)
            assert dear == pytest.approx(b)
            assert strong == pytest.approx(b)

    def test_still_sums_to_rated_energy_under_refinement(self):
        fine = refined(SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST), 256)
        assert len(fine.segmentCapacities) == 256
        assert sum(fine.segmentCapacities) == pytest.approx(fine.MWh)


class TestLpEmbedding:
    """Eq. 3. The objective row the perfect-foresight LP is built from.

    Two sides in play, and mixing them is a silent 5% error:
      * cell-side  -- energy stored, what segmentCapacities and
        segmentMarginalCosts are denominated in;
      * meter-side -- energy sold, what lpObjectiveCoefficients prices and what
        the LP's D[j][h] variable measures.
    """

    def test_there_is_one_coefficient_per_segment(self, paper):
        assert len(paper.segmentMarginalCosts) == 16
        assert len(paper.lpObjectiveCoefficients()) == 16

    def test_marginal_cost_is_cost_over_capacity(self, paper):
        expected = [
            c / cap for c, cap in zip(paper.segmentCosts, paper.segmentCapacities)
        ]
        assert paper.segmentMarginalCosts == pytest.approx(expected)

    def test_objective_row_is_the_marginal_cost_grossed_up_by_discharge_losses(self, paper):
        expected = [c / paper.dischargeEfficiency for c in paper.segmentMarginalCosts]
        assert paper.lpObjectiveCoefficients() == pytest.approx(expected)
        # Selling a MWh costs more than storing one, never less.
        for meter, cell in zip(paper.lpObjectiveCoefficients(), paper.segmentMarginalCosts):
            assert meter > cell

    def test_coefficients_are_positive_and_strictly_increasing(self, paper):
        coeffs = paper.lpObjectiveCoefficients()
        assert all(c > 0 for c in coeffs)
        assert all(b > a for a, b in zip(coeffs, coeffs[1:]))

    def test_meter_side_capacity_carries_the_efficiency_factor(self, paper):
        # The trap: pairing a meter-side coefficient with a raw cell-side cap.
        # D[j][h] is bounded by cap * nu_D, and a full cycle sells MWh * nu_D.
        meterCaps = [c * paper.dischargeEfficiency for c in paper.segmentCapacities]
        assert sum(meterCaps) == pytest.approx(paper.MWh * paper.dischargeEfficiency)
        assert sum(meterCaps) < sum(paper.segmentCapacities)

    def test_objective_row_prices_a_full_cycle_exactly(self, paper):
        # Draining every segment must cost one full cycle, no more, no less.
        booked = sum(
            coeff * cap * paper.dischargeEfficiency
            for coeff, cap in zip(paper.lpObjectiveCoefficients(), paper.segmentCapacities)
        )
        assert booked == pytest.approx(sum(paper.segmentCosts))
        assert booked == pytest.approx(
            paper.MWh * paper.replacementCost * paper.maxCapacityLoss
        )

    def test_objective_row_agrees_with_equation_11(self, paper):
        # The bridge that makes the oracle a real bound: pricing a discharge
        # segment by segment through the LP row must equal the money Eq. 11
        # charges for the same move. Exact, not approximate.
        edges = paper.SoC_breakpointVal
        coeffs = paper.lpObjectiveCoefficients()
        for prev, new in [(1.0, 0.0), (0.9, 0.1), (1.0, 0.75), (0.25, 0.0), (0.53, 0.47)]:
            booked = 0.0
            for coeff, lo, hi in zip(coeffs, edges, edges[1:]):
                overlap = max(0.0, min(prev, hi) - max(new, lo))
                booked += coeff * overlap * paper.MWh * paper.dischargeEfficiency
            eq11 = (
                paper.replacementCost * paper.MWh
                * paper.totalCycleDegradation(prev, new)
            )
            assert booked == pytest.approx(eq11)

    def test_top_segment_costs_far_more_to_sell_than_the_bottom_one(self, paper):
        # Why TODO.txt's S[j][h] <= S[j-1][h] ordering constraint is mandatory:
        # the row is cheapest at the bottom, so an unordered LP books every
        # discharge against segment 0 and under-reports degradation wildly.
        coeffs = paper.lpObjectiveCoefficients()
        assert coeffs[-1] / coeffs[0] > 25
        honest = sum(coeffs[13:])   # physically draining the top three
        cheated = sum(coeffs[:3])   # what an unordered LP would book instead
        assert honest / cheated > 10

    def test_scales_linearly_with_replacement_cost_and_not_with_size(self):
        base = SchadeDegradation(1.0, 10.0, 100_000.0)
        bigger = SchadeDegradation(1.0, 20.0, 100_000.0)
        dearer = SchadeDegradation(1.0, 10.0, 300_000.0)
        stronger = SchadeDegradation(5.0, 10.0, 100_000.0)
        for b, big, dear, strong in zip(
            base.lpObjectiveCoefficients(), bigger.lpObjectiveCoefficients(),
            dearer.lpObjectiveCoefficients(), stronger.lpObjectiveCoefficients(),
        ):
            # $/MWh is intensive: a battery twice the size costs the same per MWh.
            assert big == pytest.approx(b)
            assert dear == pytest.approx(3 * b)
            assert strong == pytest.approx(b)

    def test_is_in_the_paper_s_money_range(self, paper):
        # §4: ~EUR 68/MWh averaged over a full 100% discharge. The top segment
        # is roughly twice that, the bottom near zero.
        coeffs = paper.lpObjectiveCoefficients()
        average = sum(
            c * cap for c, cap in zip(coeffs, paper.segmentCapacities)
        ) / paper.MWh
        assert average == pytest.approx(71, abs=3)
        assert coeffs[0] == pytest.approx(4, abs=1)
        assert coeffs[-1] == pytest.approx(140, abs=5)

    def test_refining_the_grid_keeps_the_objective_row_consistent(self):
        fine = refined(SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST), 256)
        coeffs = fine.lpObjectiveCoefficients()
        assert len(coeffs) == 256
        assert all(b > a for a, b in zip(coeffs, coeffs[1:]))
        booked = sum(
            c * cap * fine.dischargeEfficiency
            for c, cap in zip(coeffs, fine.segmentCapacities)
        )
        assert booked == pytest.approx(
            fine.MWh * fine.replacementCost * fine.maxCapacityLoss
        )


# --------------------------------------------------------------------------
# Not implemented yet. See the module docstring for what @notYet means.
# --------------------------------------------------------------------------

notYet = pytest.mark.xfail(raises=AttributeError, strict=True)
notYetBehaviour = pytest.mark.xfail(strict=True)


class TestCalendarDegradation:
    """Q^CAL, Eqs. 23-27.  calendarDegradation(soc, dt_hours=1.0) -> fraction."""

    def test_equals_table_3_at_every_breakpoint(self, paper):
        for soc, rate in zip(paper.calendarAgingBreakpointVal, paper.y):
            assert paper.calendarDegradation(soc) == pytest.approx(rate)

    def test_interpolates_linearly_between_breakpoints(self, paper):
        assert paper.calendarDegradation(0.45) == pytest.approx((paper.y[1] + paper.y[2]) / 2)
        assert paper.calendarDegradation(0.65) == pytest.approx((paper.y[2] + paper.y[3]) / 2)
        assert paper.calendarDegradation(0.15) == pytest.approx((paper.y[0] + paper.y[1]) / 2)

    def test_is_increasing_in_soc(self, paper):
        vals = [paper.calendarDegradation(i / 100) for i in range(101)]
        assert all(b >= a for a, b in zip(vals, vals[1:]))

    def test_sitting_full_for_a_year_loses_two_percent(self, paper):
        assert paper.calendarDegradation(1.0) * 8760 == pytest.approx(0.0196, abs=1e-4)

    def test_an_empty_battery_still_ages(self, paper):
        assert paper.calendarDegradation(0.0) == pytest.approx(paper.y[0])
        assert paper.calendarDegradation(0.0) > 0

    def test_scales_with_the_timestep(self, paper):
        # y is a rate per hour; Eq. 27 only reads per-interval at dt = 1 h.
        hourly = paper.calendarDegradation(0.8)
        assert paper.calendarDegradation(0.8, dt_hours=1 / 12) == pytest.approx(hourly / 12)
        assert paper.calendarDegradation(0.8, dt_hours=24) == pytest.approx(24 * hourly)
        assert paper.calendarDegradation(0.8, dt_hours=0.0) == 0.0

    def test_holding_high_is_worse_than_holding_mid(self, paper):
        assert paper.calendarDegradation(0.6) / paper.calendarDegradation(0.3) < 1.2
        assert paper.calendarDegradation(0.7) / paper.calendarDegradation(0.6) > 1.7

    def test_is_independent_of_system_size(self, paper, bess):
        for soc in (0.0, 0.35, 0.65, 1.0):
            assert paper.calendarDegradation(soc) == pytest.approx(bess.calendarDegradation(soc))


@notYet
class TestSocDegradation:
    """Q^SOC, Eqs. 12-22 collapsed to the two endpoints of a discharge run:

        Q^SOC = SoC_stress * |(socRunStart + socRunEnd)/2 - 1/2|

    charged once per run.  socDegradation(socRunStart, socRunEnd) -> fraction.
    """

    def test_a_cycle_centred_on_half_costs_nothing(self, paper):
        assert paper.socDegradation(1.0, 0.0) == pytest.approx(0.0, abs=1e-18)
        assert paper.socDegradation(0.75, 0.25) == pytest.approx(0.0, abs=1e-18)
        assert paper.socDegradation(0.52, 0.48) == pytest.approx(0.0, abs=1e-18)

    def test_matches_equation_22(self, paper):
        for start, end in [(1.0, 0.9), (0.9, 0.4), (0.3, 0.1), (0.6, 0.0)]:
            expected = paper.SoC_stress * abs((start + end) / 2 - 0.5)
            assert paper.socDegradation(start, end) == pytest.approx(expected)

    def test_is_symmetric_about_half(self, paper):
        # A §5.1 rescaling artefact, not physics; Q^CAL carries the real
        # high-SOC asymmetry.
        assert paper.socDegradation(1.0, 0.9) == pytest.approx(paper.socDegradation(0.1, 0.0))
        assert paper.socDegradation(0.8, 0.7) == pytest.approx(paper.socDegradation(0.3, 0.2))

    def test_is_bounded_by_half_the_stress_parameter(self, paper):
        for start in (0.0, 0.25, 0.5, 0.75, 1.0):
            for end in (0.0, 0.25, 0.5, 0.75, 1.0):
                if end <= start:
                    assert 0.0 <= paper.socDegradation(start, end) <= paper.SoC_stress * 0.5

    def test_worst_case_is_nine_percent_of_a_full_cycle_not_nineteen(self, paper):
        # SoC_stress / maxCapacityLoss = 18.8% overstates it 2x, since A^CYC
        # caps at 0.5.
        worst = paper.socDegradation(1.0, 1.0)
        assert worst == pytest.approx(4.25e-5)
        assert worst / paper.cycleDegradation(1.0) == pytest.approx(0.094, abs=0.001)

    def test_is_depth_blind(self, paper):
        assert paper.socDegradation(1.0, 0.8) == pytest.approx(paper.socDegradation(0.95, 0.85))

    def test_dwarfs_cycle_degradation_for_a_shallow_cycle_at_the_top(self, paper):
        qSoc = paper.socDegradation(1.0, 0.9)
        qCd = paper.totalCycleDegradation(1.0, 0.9)
        assert qSoc / qCd > 4
        assert paper.socDegradation(1.0, 0.0) < 0.01 * paper.totalCycleDegradation(1.0, 0.0)

    def test_charging_costs_nothing(self, paper):
        assert paper.socDegradation(0.2, 0.9) == 0.0

    def test_splitting_a_discharge_into_two_runs_doubles_the_charge(self, paper):
        # Q^SOC scales with the number of runs, not with energy or depth. At
        # 5 minutes one idle interval closes a run, so the term tracks policy
        # choppiness rather than physics. Main reason use_q_soc defaults False.
        oneRun = paper.socDegradation(1.0, 0.5)
        twoRuns = paper.socDegradation(1.0, 0.75) + paper.socDegradation(0.75, 0.5)
        assert twoRuns == pytest.approx(2 * oneRun)
        fourRuns = sum(
            paper.socDegradation(1.0 - i * 0.125, 1.0 - (i + 1) * 0.125) for i in range(4)
        )
        assert fourRuns == pytest.approx(4 * oneRun)


@notYet
class TestDegradationCost:
    """Eq. 10, K_h = r * e_RAT * Q_h.  degradationCost(q) -> currency."""

    def test_matches_equation_10(self, bess):
        assert bess.degradationCost(1e-4) == pytest.approx(200_000.0 * 100.0 * 1e-4)

    def test_has_no_hidden_factor_of_a_hundred(self, paper):
        # a, f and y are stored as fractions, so Eq. 10 drops the /100 it needs
        # in the paper's percent units.
        assert paper.degradationCost(paper.maxCapacityLoss) == pytest.approx(
            PAPER_REPLACEMENT_COST * PAPER_MWH * paper.maxCapacityLoss
        )

    def test_reproduces_the_figure_3_two_day_result(self, paper):
        # Table 5 annualised Q = 7.55 %/yr, annualised from 2 days by x182.
        qTwoDay = 0.0755 / 182
        assert paper.degradationCost(qTwoDay) == pytest.approx(6.21, abs=0.02)

    def test_is_linear_and_zero_at_zero(self, bess):
        assert bess.degradationCost(0.0) == 0.0
        assert bess.degradationCost(2e-4) == pytest.approx(2 * bess.degradationCost(1e-4))

    def test_replacement_cost_is_per_mwh(self, paper):
        assert paper.replacementCost == 150_000.0
        assert paper.degradationCost(paper.cycleDegradation(1.0)) == pytest.approx(0.678, abs=0.001)

    def test_cost_per_mwh_discharged_scales_slightly_superlinearly(self, paper):
        # §4: ~EUR 68/MWh at 100% depth, 33 at 50%, 6 at 10%, i.e. CD^1.03.
        def perMwh(depth):
            q = paper.totalCycleDegradation(depth, 0.0)
            return paper.degradationCost(q) / (depth * paper.MWh)

        assert perMwh(1.0) == pytest.approx(68, abs=3)
        assert perMwh(0.5) == pytest.approx(33, abs=3)
        assert perMwh(0.1) == pytest.approx(6, abs=3)
        exponent = math.log(perMwh(1.0) / perMwh(0.1)) / math.log(10)
        assert exponent == pytest.approx(1.03, abs=0.05)


@notYet
class TestStepDegradation:
    """Per-step Q_h for the RL reward and the LP objective:

        stepDegradation(socPrev, socNew, dt_hours=1.0, use_q_soc=False)
    """

    def test_is_the_sum_of_the_mechanisms(self, paper):
        prev, new = 0.9, 0.4
        expected = (
            paper.totalCycleDegradation(prev, new)
            + paper.calendarDegradation(new, dt_hours=1.0)
        )
        assert paper.stepDegradation(prev, new) == pytest.approx(expected)

    def test_omits_q_soc_by_default(self, paper):
        prev, new = 1.0, 0.9
        withSoc = paper.stepDegradation(prev, new, use_q_soc=True)
        without = paper.stepDegradation(prev, new, use_q_soc=False)
        assert paper.stepDegradation(prev, new) == pytest.approx(without)
        assert withSoc > without
        assert withSoc - without == pytest.approx(paper.socDegradation(prev, new))

    def test_an_idle_step_still_costs_calendar_aging(self, paper):
        idle = paper.stepDegradation(0.7, 0.7)
        assert idle == pytest.approx(paper.calendarDegradation(0.7))
        assert idle > 0

    def test_a_charging_step_costs_calendar_aging_only(self, paper):
        assert paper.stepDegradation(0.2, 0.6) == pytest.approx(paper.calendarDegradation(0.6))

    def test_only_the_calendar_term_scales_with_dt(self, paper):
        # Q^CD and Q^SOC are per-event; Q^CAL is per-time.
        prev, new = 0.8, 0.5
        hourly = paper.stepDegradation(prev, new, dt_hours=1.0)
        fiveMin = paper.stepDegradation(prev, new, dt_hours=1 / 12)
        cycle = paper.totalCycleDegradation(prev, new)
        assert fiveMin == pytest.approx(cycle + paper.calendarDegradation(new) / 12)
        assert hourly - fiveMin == pytest.approx(paper.calendarDegradation(new) * (1 - 1 / 12))

    def test_telescopes_under_subdivision_at_fixed_wall_clock_time(self, paper):
        # One hour of discharge as 1 step or as 12 must cost the same, up to the
        # curvature of the calendar PWL. With use_q_soc=True it would not.
        prev, new = 0.9, 0.3
        whole = paper.stepDegradation(prev, new, dt_hours=1.0)
        socs = [prev + (new - prev) * i / 12 for i in range(13)]
        parts = sum(
            paper.stepDegradation(a, b, dt_hours=1 / 12) for a, b in zip(socs, socs[1:])
        )
        assert parts == pytest.approx(whole, rel=0.02)

    def test_mechanism_shares_match_section_5_5(self, paper):
        # §5.5 at r = 150: CD 83.1%, CAL 11.1%, SOC 5.8%. The exact split needs
        # the paper's dispatch; the ordering is checkable here.
        prev, new = 1.0, 0.1
        qCd = paper.totalCycleDegradation(prev, new)
        qCal = paper.calendarDegradation(new, dt_hours=1.0)
        qSoc = paper.socDegradation(prev, new)
        assert qCd / (qCd + qCal + qSoc) > 0.75
        assert qSoc < qCal < qCd

    def test_rejects_a_step_that_leaves_the_soc_range(self, paper):
        with pytest.raises(ValueError):
            paper.stepDegradation(1.2, 0.5)
        with pytest.raises(ValueError):
            paper.stepDegradation(0.5, -0.1)


class TestValidityBounds:
    """§2.2: no C-rate term, calibrated below ~1C. Proposed: a cRate property
    and a warning at construction above 1C."""

    @notYet
    def test_reports_its_c_rate(self, paper, bess):
        assert paper.cRate == pytest.approx(0.6)
        assert bess.cRate == pytest.approx(0.5)

    @notYetBehaviour
    def test_warns_when_sized_outside_the_calibrated_range(self):
        # The constructor runs fine today and simply stays silent, so this one
        # fails without ever raising AttributeError -- hence @notYetBehaviour.
        with pytest.warns(UserWarning, match="C-rate"):
            SchadeDegradation(200.0, 100.0, 200_000.0)

"""
Perfect-foresight oracle over a known price series (TODO.txt Step 1).

The battery is 16 stacked segments. Emptying a high segment costs more than a low
one, and those per-segment costs are constants out of schade.py, so the real
degradation objective goes into the model exactly -- no flat-cost approximation.

A MILP, not an LP: the staircase fill needs a binary per segment, and calendar
aging is piecewise linear but non-convex so it needs SOS2 binaries on top.

Energy is tracked on two sides of the meter. `stored` is cell-side (what the
battery actually holds); `bought` and `sold` are meter-side (what the market
sees). Charging loses nu_C on the way in, discharging loses nu_D on the way out.
"""
from dataclasses import dataclass

import pulp

from utils import nem_interval_hours


@dataclass
class Schedule:
    initialSoc: float
    soc: list  # fraction, at the end of each interval
    bought: list  # meter-side MWh bought per interval
    sold: list  # meter-side MWh sold per interval
    revenue: float
    degradationCost: float
    profit: float


def fillBottomUp(soc, segmentCapacities):
    """Starting energy per segment, filling the lowest segment first."""
    remaining = soc * sum(segmentCapacities)
    levels = []
    for capacity in segmentCapacities:
        levels.append(min(capacity, max(0.0, remaining)))
        remaining -= capacity
    return levels


def defaultSolver(verbose=False):
    """HiGHS if available, else CBC.

    verbose lets the solver's own branch-and-bound log through, which is the only
    real progress signal available while a solve is running.
    """
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        # msg=False alone does not silence HiGHS; it needs output_flag too.
        return pulp.HiGHS(msg=verbose, output_flag=verbose, log_to_console=verbose)
    return pulp.PULP_CBC_CMD(msg=verbose)


def solve(prices, model, initialSoc=0.5, dtHours=nem_interval_hours, calendar=True,
          solver=None):
    """Best possible dispatch against a known price series.

    prices: $/MWh per interval.  model: a SchadeDegradation.
    """
    intervals = range(len(prices))
    segmentCapacity = model.segmentCapacities
    segments = range(len(segmentCapacity))
    segmentCost = model.lpObjectiveCoefficients()  # $/MWh sold, meter-side
    # Eqs. 6-7: MW is a rate, the variables are energy.
    energyLimit = model.chargePower * dtHours
    chargeEff, dischargeEff = model.chargeEfficiency, model.dischargeEfficiency
    startLevels = fillBottomUp(initialSoc, segmentCapacity)

    def grid(name, **kwargs):
        return [
            [pulp.LpVariable(f"{name}_{s}_{t}", **kwargs) for t in intervals]
            for s in segments
        ]

    problem = pulp.LpProblem("oracle", pulp.LpMaximize)
    stored = [
        [pulp.LpVariable(f"stored_{s}_{t}", 0, segmentCapacity[s]) for t in intervals]
        for s in segments
    ]
    bought = grid("bought", lowBound=0)
    sold = grid("sold", lowBound=0)
    # Staircase fill. Eq. 5 caps each segment but never forces the pack to drain
    # from the top, and mere ordering (stored[s] <= stored[s-1]) is not enough --
    # the solver just drains every segment uniformly and still bills the cheap
    # bottom ones. inUse[s][t] means "segment s may hold energy", allowed only
    # once the segment below it is full.
    inUse = grid("inUse", cat="Binary")

    for t in intervals:
        for s in segments:
            previous = stored[s][t - 1] if t else startLevels[s]
            problem += stored[s][t] == (
                previous + chargeEff * bought[s][t] - (1 / dischargeEff) * sold[s][t]
            )
            problem += stored[s][t] <= segmentCapacity[s] * inUse[s][t]
            if s:
                problem += stored[s - 1][t] >= segmentCapacity[s - 1] * inUse[s][t]
                # Implied by the two above, but stating them explicitly tightens
                # the relaxation enough to matter -- without them CBC stalls.
                problem += stored[s][t] <= stored[s - 1][t]
                problem += inUse[s][t] <= inUse[s - 1][t]
        problem += pulp.lpSum(bought[s][t] for s in segments) <= energyLimit
        problem += pulp.lpSum(sold[s][t] for s in segments) <= energyLimit

    storedEnergy = [pulp.lpSum(stored[s][t] for s in segments) for t in intervals]

    revenueExpr = pulp.lpSum(
        prices[t] * pulp.lpSum(sold[s][t] - bought[s][t] for s in segments)
        for t in intervals
    )
    cycleCost = pulp.lpSum(
        segmentCost[s] * sold[s][t] for t in intervals for s in segments
    )
    calendarCost = _calendarCost(problem, storedEnergy, model, dtHours) if calendar else 0.0
    problem += revenueExpr - cycleCost - calendarCost

    problem.solve(solver or defaultSolver())
    assert pulp.LpStatus[problem.status] == "Optimal", pulp.LpStatus[problem.status]

    def perInterval(variables):
        return [sum(variables[s][t].value() for s in segments) for t in intervals]

    boughtPerInterval = perInterval(bought)
    soldPerInterval = perInterval(sold)
    soc = [level / model.MWh for level in perInterval(stored)]
    revenue = sum(
        price * (out - inn)
        for price, out, inn in zip(prices, soldPerInterval, boughtPerInterval)
    )
    profit = pulp.value(problem.objective)
    return Schedule(
        initialSoc,
        soc,
        boughtPerInterval,
        soldPerInterval,
        revenue,
        revenue - profit,
        profit,
    )


def _calendarCost(problem, storedEnergy, model, dtHours):
    """Q^CAL via SOS2: express SoC as a blend of two *adjacent* breakpoints.

    The curve's slopes are not monotone, so a plain LP would interpolate across
    the kinks and understate the cost. The binaries force adjacency.
    """
    breakpoints = model.calendarAgingBreakpointVal
    costAt = [model.cost(rate * dtHours) for rate in model.y]
    corners = range(len(breakpoints))
    pieces = range(len(breakpoints) - 1)
    total = []
    for t, energy in enumerate(storedEnergy):
        weight = [pulp.LpVariable(f"weight_{k}_{t}", 0, 1) for k in corners]
        onPiece = [pulp.LpVariable(f"onPiece_{k}_{t}", cat="Binary") for k in pieces]
        problem += pulp.lpSum(weight) == 1
        problem += pulp.lpSum(onPiece) == 1
        problem += pulp.lpSum(
            soc * w for soc, w in zip(breakpoints, weight)
        ) == (1 / model.MWh) * energy
        for k in corners:
            # A corner carries weight only if the piece on one of its sides is on.
            problem += weight[k] <= pulp.lpSum(
                onPiece[j] for j in (k - 1, k) if j in pieces
            )
        total.append(pulp.lpSum(c * w for c, w in zip(costAt, weight)))
    return pulp.lpSum(total)


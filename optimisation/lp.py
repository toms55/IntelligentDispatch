"""
Perfect-foresight oracle over a known price series (TODO.txt Step 1).

The battery is 16 stacked segments; emptying a high segment costs more than a low one,
and those per-segment costs are constants out of schade.py.

`stored` is cell-side, `bought`/`sold` are meter-side. Charging loses nu_C going in,
discharging loses nu_D going out.

`discretePower=True` restricts every interval to full rate or nothing, matching the RL
agent's action set. See `_discretePower`.
"""
from dataclasses import dataclass

import pulp

from utils import nem_interval_hours


@dataclass
class Schedule:
    initialSoc: float
    soc: list
    bought: list
    sold: list
    revenue: float
    degradationCost: float
    profit: float
    # Appended with defaults: test_cli.py builds a Schedule from seven positional args.
    revenuePerInterval: list = None
    cyclePerInterval: list = None
    calendarPerInterval: list = None
    degradationPerInterval: list = None
    profitPerInterval: list = None
    status: str = "Optimal"
    mipGap: float = None
    # Use this as the oracle bound, never `profit`: a gap/time-limited solve returns an
    # incumbent that can sit below the true optimum.
    dualBound: float = None
    # The cycle budget's shadow price, in $ per MWh removed cell-side, and the equivalent
    # full cycles the schedule spent. cyclePrice is 0.0 when no budget binds. It is a
    # search device and reaches none of the figures above: see optimisation/dp.py.
    cyclePrice: float = None
    cyclesUsed: float = None
    # What the closing stock is worth, already included in `profit` but reported apart
    # from it so it is never mistaken for trading revenue.
    terminalValue: float = 0.0


def fillBottomUp(soc, segmentCapacities):
    """Starting energy per segment, filling the lowest segment first."""
    remaining = soc * sum(segmentCapacities)
    levels = []
    for capacity in segmentCapacities:
        levels.append(min(capacity, max(0.0, remaining)))
        remaining -= capacity
    return levels


def defaultSolver(verbose=False, timeLimit=None, gap=None, threads=None, options=None):
    """HiGHS if available, else CBC.

    threads should be 1 when solving windows in parallel processes, or HiGHS takes
    every core in every worker.
    """
    if "HiGHS" in pulp.listSolvers(onlyAvailable=True):
        settings = dict(msg=verbose, output_flag=verbose, log_to_console=verbose,
                        timeLimit=timeLimit, gapRel=gap)
        if threads is not None:
            settings["threads"] = threads
        return pulp.HiGHS(**settings, **(options or {}))
    return pulp.PULP_CBC_CMD(msg=verbose, timeLimit=timeLimit, gapRel=gap,
                             threads=threads)


def solve(prices, model, initialSoc=0.5, dtHours=nem_interval_hours, calendar=True,
          solver=None, discretePower=False, timeLimit=None, gap=None, threads=None,
          options=None, cycles=None, terminalPrice=0.0):
    """Best possible dispatch against a known price series.

    prices: $/MWh per interval. model: a SchadeDegradation. discretePower restricts
    every interval to full rate or nothing. options are extra HiGHS settings.

    `cycles` caps the equivalent full cycles the schedule may spend, as one exact
    constraint row. `terminalPrice` values whatever the pack still holds at the close.
    Both are here to cross-check `optimisation.dp` on short horizons. Do not wire this
    path into the backtest: the cycle row couples every interval, and a 288-interval
    discrete solve already took 1555 s to prove.
    """
    intervals = range(len(prices))
    segmentCapacity = model.segmentCapacities
    segments = range(len(segmentCapacity))
    segmentCost = model.lpObjectiveCoefficients()
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
    switches = _discreteSwitches(intervals) if discretePower else None
    inUse = grid("inUse", cat="Binary")

    storedEnergy = []
    for t in intervals:
        for s in segments:
            previous = stored[s][t - 1] if t else startLevels[s]
            problem += stored[s][t] == (
                previous + chargeEff * bought[s][t] - (1 / dischargeEff) * sold[s][t]
            )
            problem += stored[s][t] <= segmentCapacity[s] * inUse[s][t]
            if s:
                problem += stored[s - 1][t] >= segmentCapacity[s - 1] * inUse[s][t]
                problem += stored[s][t] <= stored[s - 1][t]
                problem += inUse[s][t] <= inUse[s - 1][t]
        totalBought = pulp.lpSum(bought[s][t] for s in segments)
        totalSold = pulp.lpSum(sold[s][t] for s in segments)
        storedEnergy.append(pulp.lpSum(stored[s][t] for s in segments))
        if discretePower:
            _discretePower(problem, switches, t, totalBought, totalSold,
                           storedEnergy[t], energyLimit, model.MWh, inUse)
        else:
            problem += totalBought <= energyLimit
            problem += totalSold <= energyLimit

    if cycles is not None:
        problem += pulp.lpSum(
            (1 / dischargeEff) * sold[s][t] for s in segments for t in intervals
        ) <= cycles * model.MWh

    # What the closing stock is worth: the energy it would deliver at `terminalPrice`,
    # less the cost of releasing it. segmentCost is already $ per MWh sold out of each
    # segment, so the release cost is linear in `stored` and no new binaries appear.
    last = len(prices) - 1
    terminalExpr = pulp.lpSum(
        dischargeEff * (terminalPrice - segmentCost[s]) * stored[s][last]
        for s in segments
    ) if terminalPrice else 0.0

    revenueExpr = pulp.lpSum(
        prices[t] * pulp.lpSum(sold[s][t] - bought[s][t] for s in segments)
        for t in intervals
    )
    cycleExpr = [
        pulp.lpSum(segmentCost[s] * sold[s][t] for s in segments) for t in intervals
    ]
    calendarExpr = (
        _calendarCost(problem, storedEnergy, model, dtHours) if calendar
        else [0.0] * len(prices)
    )
    problem += (revenueExpr - pulp.lpSum(cycleExpr) - pulp.lpSum(calendarExpr)
                + terminalExpr)

    problem.solve(solver or defaultSolver(timeLimit=timeLimit, gap=gap, threads=threads,
                                            options=options))
    status = pulp.LpStatus[problem.status]
    if status != "Optimal" and stored[0][0].value() is None:
        raise AssertionError(f"solver returned {status} with no feasible solution")

    def perInterval(variables):
        return [sum(variables[s][t].value() for s in segments) for t in intervals]

    boughtPerInterval = perInterval(bought)
    soldPerInterval = perInterval(sold)
    soc = [level / model.MWh for level in perInterval(stored)]
    revenuePerInterval = [
        price * (out - inn)
        for price, out, inn in zip(prices, soldPerInterval, boughtPerInterval)
    ]
    cyclePerInterval = [pulp.value(term) for term in cycleExpr]
    calendarPerInterval = [float(pulp.value(term)) for term in calendarExpr]
    degradationPerInterval = [c + k for c, k in zip(cyclePerInterval, calendarPerInterval)]
    profitPerInterval = [
        r - d for r, d in zip(revenuePerInterval, degradationPerInterval)
    ]
    revenue = sum(revenuePerInterval)
    degradationCost = sum(degradationPerInterval)
    terminalValue = float(pulp.value(terminalExpr)) if terminalPrice else 0.0
    cyclesUsed = sum(soldPerInterval) / dischargeEff / model.MWh
    profit = pulp.value(problem.objective)
    gapAchieved, dualBound = _solverInfo(problem, profit)
    return Schedule(
        initialSoc,
        soc,
        boughtPerInterval,
        soldPerInterval,
        revenue,
        degradationCost,
        profit,
        revenuePerInterval,
        cyclePerInterval,
        calendarPerInterval,
        degradationPerInterval,
        profitPerInterval,
        status,
        gapAchieved,
        dualBound,
        0.0,
        cyclesUsed,
        terminalValue,
    )


def _discreteSwitches(intervals):
    def column(name):
        return [pulp.LpVariable(f"{name}_{t}", cat="Binary") for t in intervals]

    return {
        "chargeOn": column("chargeOn"),
        "dischargeOn": column("dischargeOn"),
        "atFull": column("atFull"),
        "atEmpty": column("atEmpty"),
    }


def _discretePower(problem, switches, t, totalBought, totalSold, storedTotal,
                   energyLimit, capacity, inUse):
    """Restrict interval t to a full-rate charge, a full-rate discharge, or idle.

    A partial top-up is allowed only where physics forces one. atFull/atEmpty must stay
    BINARY: relaxed to [0,1] the pair admits a half-rate charge at mid SoC.
    """
    chargeOn = switches["chargeOn"][t]
    dischargeOn = switches["dischargeOn"][t]
    atFull = switches["atFull"][t]
    atEmpty = switches["atEmpty"][t]
    top = len(inUse) - 1

    problem += chargeOn + dischargeOn <= 1
    problem += totalBought <= energyLimit * chargeOn
    problem += energyLimit * chargeOn - totalBought <= energyLimit * atFull
    problem += storedTotal >= capacity * atFull
    problem += totalSold <= energyLimit * dischargeOn
    problem += energyLimit * dischargeOn - totalSold <= energyLimit * atEmpty
    problem += storedTotal <= capacity * (1 - atEmpty)

    # Without these, atFull/atEmpty are free whenever the pack happens to be full/empty.
    problem += atFull <= chargeOn
    problem += atEmpty <= dischargeOn
    problem += atFull <= inUse[top][t]
    problem += atEmpty + inUse[1][t] <= 1


def _solverInfo(problem, objective):
    """(relative MIP gap, dual bound), or (None, objective) if the solver is mute.

    HiGHS only minimises, so pulp negates a maximisation and mip_dual_bound comes back
    with the opposite sign; flip it back here.
    """
    try:
        info = problem.solverModel.getInfo()
        bound = float(info.mip_dual_bound)
        if problem.sense == pulp.LpMaximize:
            bound = -bound
        return float(info.mip_gap), bound
    except Exception:
        return None, objective


def _calendarCost(problem, storedEnergy, model, dtHours):
    """Q^CAL per interval via SOS2: SoC as a blend of two adjacent breakpoints."""
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
            problem += weight[k] <= pulp.lpSum(
                onPiece[j] for j in (k - 1, k) if j in pieces
            )
        total.append(pulp.lpSum(c * w for c, w in zip(costAt, weight)))
    return total

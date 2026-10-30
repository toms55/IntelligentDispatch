"""
Exact oracle for the discrete-power problem, by dynamic programming.

`optimisation.lp.solve(discretePower=True)` states this problem as a MILP. It does not
have to be one. With three actions the state is a single number, the energy in the pack.
The transition is deterministic, and the cost of an interval depends only on the energy
before it, the action, and that interval's price. So the problem is a shortest path over
a layered graph, and value iteration solves it exactly.

Two things make it cheap. The set of energies reachable from a fixed starting SoC grows
by about 20 states per interval, so a 288-interval episode has under 6000 states. And
the degradation cost of a move does not depend on the interval, so every cost is
computed once and then only gathered.

The result is exact. There is no MIP gap, no time limit, and no dual bound that can sit
above the optimum. `Schedule.dualBound` equals `Schedule.profit`.

The transition rule here is `rl.env.BessEnv._applyAction`, written out again. That is
deliberate: the oracle and the environment must be the same problem, and here they are
the same arithmetic. `optimisation/test_dp.py` holds this module to the MILP.
"""
import numpy as np

import config
from optimisation.lp import Schedule
from utils import nem_interval_hours

CHARGE, IDLE, DISCHARGE = 0, 1, 2
_actions = (CHARGE, IDLE, DISCHARGE)

# Energies are keyed on integer multiples of this, so that two paths to one physical SoC
# give one state. The band of safe values is wide. A charge step is nu_C * F and a
# discharge step is F / nu_D, whose ratio is nu_C * nu_D = 0.9025 = 361/400 exactly, so
# two different states are at least (F / nu_D) / 400 apart, around 1.3e-5 MWh. Float
# noise over 288 intervals stays near 1e-16. Anything between those two bounds works.
quantum = 1e-9


def move(stored, action, model, dtHours):
    """One interval. Returns (new stored, bought, sold), all meter-side but `stored`.

    The same arithmetic as `rl.env.BessEnv._applyAction`, including the clip at the SoC
    boundary that gives a partial top-up.
    """
    full = model.chargePower * dtHours
    if action == CHARGE:
        bought = min(full, max(0.0, (model.MWh - stored) / model.chargeEfficiency))
        return min(stored + model.chargeEfficiency * bought, model.MWh), bought, 0.0
    if action == DISCHARGE:
        sold = min(full, max(0.0, stored * model.dischargeEfficiency))
        return max(stored - sold / model.dischargeEfficiency, 0.0), 0.0, sold
    return stored, 0.0, 0.0


class _StateSpace:
    """The reachable energies, their transitions, and the cost of each move.

    `energy[i]` is the energy of state i. `successor[a][i]`, `bought[a][i]`,
    `sold[a][i]`, `cycle[a][i]` and `calendar[a][i]` describe taking action a from it.
    `reachable[t][i]` says whether state i can be reached after t intervals.

    A state reachable at t always has its successors reachable at t+1, so the value of a
    reachable state never reads the value of an unreachable one. Values outside the
    reachable set are computed anyway, because masking costs more than it saves, and
    they are never used.
    """

    def __init__(self, model, initialSoc, dtHours, horizon):
        start = initialSoc * model.MWh
        index = {self._key(start): 0}
        energies = [start]
        masks = [{0}]

        for _ in range(horizon):
            layer = set()
            for i in masks[-1]:
                for action in _actions:
                    new, _, _ = move(energies[i], action, model, dtHours)
                    key = self._key(new)
                    if key not in index:
                        index[key] = len(energies)
                        energies.append(new)
                    layer.add(index[key])
            masks.append(layer)

        count = len(energies)
        self.energy = np.array(energies, dtype=float)
        self.successor = np.zeros((3, count), dtype=np.int64)
        self.bought = np.zeros((3, count), dtype=float)
        self.sold = np.zeros((3, count), dtype=float)
        self.cycle = np.zeros((3, count), dtype=float)
        self.calendar = np.zeros((3, count), dtype=float)

        for action in _actions:
            for i, stored in enumerate(energies):
                new, bought, sold = move(stored, action, model, dtHours)
                qCd, qCal, _ = model.step(stored / model.MWh, new / model.MWh, dtHours)
                # A successor is absent only for a state whose one reachable layer is the
                # last, and the episode ends before any action is taken from there. Point
                # it at itself to keep the index valid.
                self.successor[action][i] = index.get(self._key(new), i)
                self.bought[action][i] = bought
                self.sold[action][i] = sold
                self.cycle[action][i] = model.cost(qCd)
                self.calendar[action][i] = model.cost(qCal)

        self.throughput = self.sold - self.bought
        # Energy removed cell-side, which is what an equivalent full cycle is measured in:
        # sum(cellSide) / MWh is the EFC a path spends.
        self.cellSide = self.sold / model.dischargeEfficiency
        self.reachable = masks

    @staticmethod
    def _key(energy):
        return int(round(energy / quantum))


_cache = {}


def stateSpace(model, initialSoc, dtHours, horizon):
    """The state space for these battery settings, built once per process."""
    key = (model.chargePower, model.MWh, model.replacementCost,
           model.chargeEfficiency, model.dischargeEfficiency, initialSoc, dtHours,
           horizon)
    if key not in _cache:
        _cache[key] = _StateSpace(model, initialSoc, dtHours, horizon)
    return _cache[key]


def valueFunction(prices, model, initialSoc=0.5, dtHours=nem_interval_hours,
                  calendar=True, space=None, cyclePrice=0.0, terminalPrice=0.0):
    """The optimal value of every state at every interval, and the action that gets it.

    Returns (value, action, space). `value[t][i]` is the most profit obtainable over
    intervals t to the end, starting that interval with `space.energy[i]` in the pack.
    This is the oracle evaluated at every state, not only along one path, so it costs the
    same as solving the one path.

    `cyclePrice` charges an extra `cyclePrice` dollars per MWh removed cell-side. It is
    the rent `solve` bisects on to meet a cycle budget, and it is not a real cost, so a
    value read out of here is a value after rent, not a profit. `terminalPrice` values
    whatever the pack still holds at the close, so `value[horizon]` stops being zeros.
    """
    space = space or stateSpace(model, initialSoc, dtHours, len(prices))
    horizon = len(prices)
    degradation = space.cycle + space.calendar if calendar else space.cycle
    if cyclePrice:
        degradation = degradation + cyclePrice * space.cellSide

    value = np.zeros((horizon + 1, len(space.energy)), dtype=float)
    if terminalPrice:
        value[horizon] = [model.terminalValue(e, terminalPrice) for e in space.energy]
    action = np.zeros((horizon, len(space.energy)), dtype=np.int8)
    for t in range(horizon - 1, -1, -1):
        candidates = (prices[t] * space.throughput - degradation
                      + value[t + 1][space.successor])
        action[t] = candidates.argmax(axis=0)
        value[t] = candidates.max(axis=0)
    return value, action, space



def _rollout(prices, model, action, space, initialSoc, calendar, terminalPrice):
    """Walk the chosen actions forward and score them with the true costs.

    The rent reaches nothing here. Every figure comes from `space.cycle` and
    `space.calendar`, so a rented solve still reports a real profit, and
    `cyclesUsed` is the equivalent full cycles the walk actually spent.
    """
    index = 0  # _StateSpace numbers the starting energy first.
    soc, bought, sold = [], [], []
    revenuePerInterval, cyclePerInterval, calendarPerInterval = [], [], []
    removed = 0.0
    for t in range(len(prices)):
        chosen = int(action[t][index])
        bought.append(float(space.bought[chosen][index]))
        sold.append(float(space.sold[chosen][index]))
        revenuePerInterval.append(float(prices[t] * space.throughput[chosen][index]))
        cyclePerInterval.append(float(space.cycle[chosen][index]))
        calendarPerInterval.append(
            float(space.calendar[chosen][index]) if calendar else 0.0
        )
        removed += float(space.cellSide[chosen][index])
        index = int(space.successor[chosen][index])
        soc.append(float(space.energy[index] / model.MWh))

    degradationPerInterval = [
        c + k for c, k in zip(cyclePerInterval, calendarPerInterval)
    ]
    profitPerInterval = [
        r - d for r, d in zip(revenuePerInterval, degradationPerInterval)
    ]
    revenue = sum(revenuePerInterval)
    degradationCost = sum(degradationPerInterval)
    terminal = (model.terminalValue(float(space.energy[index]), terminalPrice)
                if terminalPrice else 0.0)
    profit = sum(profitPerInterval) + terminal
    return Schedule(
        initialSoc,
        soc,
        bought,
        sold,
        revenue,
        degradationCost,
        profit,
        revenuePerInterval,
        cyclePerInterval,
        calendarPerInterval,
        degradationPerInterval,
        profitPerInterval,
        "Optimal",
        0.0,
        profit,
        0.0,
        removed / model.MWh,
        terminal,
    )


def solve(prices, model, initialSoc=0.5, dtHours=nem_interval_hours, calendar=True,
          cycles=None, terminalPrice=0.0, dualGap=config.cycle_dual_gap):
    """The exact discrete-power oracle, as a `Schedule`.

    Drop-in for `optimisation.lp.solve(..., discretePower=True)`. With no cycle budget
    `status` is "Optimal", `mipGap` is 0.0 and `dualBound` is the profit, because the
    answer is exact.

    `cycles` caps the equivalent full cycles the schedule may spend. A hard cap would
    need a second state dimension, and cumulative discharge is continuous, so instead
    every MWh removed cell-side pays rent and the rent is bisected until usage lands on
    the budget. Two numbers come out of that, and both are honest: `profit` is a schedule
    inside the budget, so it is achievable, and `dualBound` is
    min over rents of (value after rent + rent * budget), which no legal schedule can
    beat. `mipGap` is the relative width between them, and `cyclePrice` is the rent that
    bought the incumbent, in $ per MWh cell-side. The two never meet exactly: the battery
    discharges in whole intervals, so realised usage steps and no rent lands on the
    budget.
    """
    prices = np.asarray(prices, dtype=float)
    space = stateSpace(model, initialSoc, dtHours, len(prices))

    def at(cyclePrice):
        value, action, _ = valueFunction(prices, model, initialSoc, dtHours, calendar,
                                         space=space, cyclePrice=cyclePrice,
                                         terminalPrice=terminalPrice)
        schedule = _rollout(prices, model, action, space, initialSoc, calendar,
                            terminalPrice)
        return schedule, float(value[0][0])

    free, freeValue = at(0.0)
    if cycles is None or free.cyclesUsed <= cycles + 1e-9:
        return free

    budget = cycles * model.MWh
    # Above this rent a discharge can never pay for itself, so usage is zero and the
    # bracket's upper end is feasible by construction.
    ceiling = max(float(prices.max()), 0.0) * model.dischargeEfficiency + 1.0
    best, rented = at(ceiling)
    bestPrice = ceiling
    bound = min(freeValue, rented + ceiling * budget)

    lo, hi = 0.0, ceiling
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        schedule, rented = at(mid)
        bound = min(bound, rented + mid * budget)
        if schedule.cyclesUsed <= cycles + 1e-9:
            hi = mid
            if schedule.profit > best.profit:
                best, bestPrice = schedule, mid
        else:
            lo = mid
        if bound - best.profit <= abs(bound) * dualGap or hi - lo <= 1e-12:
            break

    best.cyclePrice = bestPrice
    best.dualBound = bound
    best.mipGap = (bound - best.profit) / abs(bound) if bound else 0.0
    return best

"""The battery dispatch environment (TODO.txt step 2).

Three actions: charge at full rate, discharge at full rate, idle. The rate is a power
throughput, `chargePower * dtHours` meter-side, and does not depend on SoC.

The environment and `optimisation.lp.solve(discretePower=True)` must be the same
problem: same action set, partial-top-up rule, episode length, starting SoC, degradation
terms. `rl/test_env.py::TestMatchesTheOracle` holds the two to 1e-9.
"""
import gymnasium
import numpy as np
from gymnasium import spaces

import config
from data.forecasts import activeRun
from rl import features
from rl.ablation import Ablation
from utils import nem_interval_hours

CHARGE, IDLE, DISCHARGE = 0, 1, 2


class BessEnv(gymnasium.Env):
    """One episode is one LP chunk: `horizon` intervals from a fixed starting SoC.

    prices: a pandas Series of RRP with a DatetimeIndex. baseline: per-window lists of
    the LP's own per-interval profit, keyed by window index. It is reporting only: it
    fills `lpProfit` and the episode's `regret`, and never enters the reward, which is
    the agent's own profit.

    scaler: the price transform the observation uses. It defaults to the cap in force at
    the end of `prices`, which is right for a fresh run. When you score a checkpoint,
    pass the scaler the checkpoint trained under instead.
    """

    metadata = {"render_modes": []}

    def __init__(self, prices, model, baseline=None, horizon=config.episode_intervals,
                 initialSoc=config.initial_soc, dtHours=nem_interval_hours,
                 calendar=True, rewardScale=config.reward_scale, shuffle=False,
                 forecasts=None, scaler=None, cycleBudget=config.cycle_budget,
                 terminalPrice=0.0, invalidPenalty=config.invalid_action_penalty,
                 endEmpty=False, ablation=None):
        self.ablation = (ablation if ablation is not None else Ablation()).validate()
        if (not self.ablation.timeEncoding == "clock" and horizon
                != config.observation_intervals):
            raise ValueError("a sine/cosine or absent time encoding replaces the episode "
                             "slot, which only equals the clock on a one-day episode")
        self.prices = prices.to_numpy(dtype=float)
        self.timestamps = prices.index
        # One cap for the whole run, from the last timestamp. Pass `scaler` to reuse the
        # one a checkpoint was trained under; see rl/train.py.
        self.scaler = scaler if scaler is not None else features.scalerFor(prices.index)

        self.model = model
        self.baseline = baseline
        self.horizon = horizon
        self.initialSoc = initialSoc
        self.dtHours = dtHours
        self.calendar = calendar
        self.rewardScale = rewardScale
        # Charging a full pack or discharging an empty one, in reward units. See
        # config.invalid_action_penalty.
        self.invalidPenalty = invalidPenalty
        self.shuffle = shuffle
        # None means realised prices (training). Otherwise one entry per window: a
        # single path, or a data.forecasts.refreshMatrix stack of runs, of which the
        # agent reads the latest published by each step. Settlement always uses the
        # realised price regardless.
        self.forecasts = forecasts
        # The most equivalent full cycles one episode may spend, standing in for a cycle
        # warranty. `None` is no budget, which is the environment every published agent
        # trained in. The budget is enforced by the clip in `_applyAction`, not by the
        # mask, so it binds under DQN and A2C as well, exactly as the SoC boundary does.
        self.cycleBudget = cycleBudget
        self.budgetMWh = None if cycleBudget is None else cycleBudget * model.MWh
        # What the pack still holds at the close is worth. Zero is the old environment,
        # where the last interval's best move is always to sell whatever is left.
        self.terminalPrice = terminalPrice
        # The pack must be empty at the close. Enforced in `_applyAction` by overriding
        # the move, as the budget is, so it binds under DQN and A2C, which read no mask:
        # a charge that would leave too much to sell in time idles, and an idle that
        # would do the same discharges.
        self.endEmpty = endEmpty

        self.windows = list(range(0, len(self.prices) - horizon + 1, horizon))
        if not self.windows:
            raise ValueError(
                f"{len(self.prices)} intervals is shorter than one {horizon}-interval episode"
            )

        # The trading day's cheap and dear quarters, for the shaping bonus. One pair per
        # episode, read from the realised prices, and used in the reward only.
        self.quarters = self._dayQuarters() if self.ablation.shapingBonus else None

        self.costing = self.ablation.costing(model, dtHours)
        # 2L + 1 actions: the first L charge at falling shares of rated power, the middle
        # one idles, the last L discharge at rising shares. One level is the three-action
        # problem every published agent faced, and the arithmetic is then unchanged.
        self.levels = self.ablation.actionLevels
        self.idleAction = self.levels
        self.dischargeAction = 2 * self.levels
        self.fullStep = model.chargePower * dtHours
        self.tolerance = 1e-9 * self.fullStep
        # What one full-rate discharge removes, cell-side.
        self.fullRemoval = self.fullStep / model.dischargeEfficiency

        low, high = features.observationBounds(self.scaler, self.ablation)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(2 * self.levels + 1)

        self.windowCursor = 0
        self.windowIndex = 0
        self.stepIndex = 0
        self.stored = initialSoc * model.MWh
        self.spentMWh = 0.0

    # --- Gymnasium API -------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        requested = (options or {}).get("window")
        if requested is not None:
            self.windowIndex = requested
        elif self.shuffle:
            self.windowIndex = int(self.np_random.integers(len(self.windows)))
        else:
            self.windowIndex = self.windowCursor % len(self.windows)
            self.windowCursor += 1
        self.stepIndex = 0
        self.stored = self.initialSoc * self.model.MWh
        self.spentMWh = 0.0
        self.totals = {"revenue": 0.0, "degradation": 0.0, "profit": 0.0, "baseline": 0.0,
                       "outOfBounds": 0, "cycles": 0.0, "terminalValue": 0.0}
        return self._observation(), {}

    def step(self, action):
        price = self.prices[self._offset()]
        socPrev = self.stored / self.model.MWh
        action = int(action)
        # Read before the move, because the move is what removes the headroom.
        outOfBounds = not self._legalMask()[action]
        boughtMWh, soldMWh = self._applyAction(self._closingAction(action))
        socNew = self.stored / self.model.MWh

        revenue = price * (soldMWh - boughtMWh)
        qCd, qCal, degradationCost = self._degradation(socPrev, socNew)
        profit = revenue - degradationCost
        baseline = self._baselineAt(self.stepIndex)

        self.stepIndex += 1
        truncated = self.stepIndex >= self.horizon
        # Paid once, on the step that closes the episode, so the agent is rewarded for
        # holding stock rather than dumping it into the last interval.
        terminal = (self.model.terminalValue(self.stored, self.terminalPrice)
                    if truncated and self.terminalPrice else 0.0)
        # The penalty is the one term in the reward that is not profit. It stays out of
        # `totals` and out of every info key below, so the figures the backtest reports
        # remain raw dollars and the oracle keeps bounding the agent.
        reward = (profit + terminal) / self.rewardScale - self.invalidPenalty * outOfBounds
        reward += self._shaping(price, boughtMWh, soldMWh)

        self.totals["revenue"] += revenue
        self.totals["degradation"] += degradationCost
        self.totals["profit"] += profit + terminal
        self.totals["baseline"] += baseline
        self.totals["outOfBounds"] += outOfBounds
        self.totals["cycles"] = self.spentMWh / self.model.MWh
        self.totals["terminalValue"] += terminal

        info = {
            "price": price,
            "soc": socNew,
            "bought": boughtMWh,
            "sold": soldMWh,
            "revenue": revenue,
            "degradation": degradationCost,
            "cycleDegradation": qCd,
            "calendarDegradation": qCal,
            "profit": profit,
            "lpProfit": baseline,
            # What the agent chose, which is not what moved: an out-of-bounds charge
            # moves nothing and so reads as idle anywhere the energy is the record.
            # The direction, as CHARGE, IDLE or DISCHARGE, whatever the action set: the
            # actions export and the out-of-bounds report read it as those three.
            "attemptedAction": self._direction(action),
            "outOfBounds": outOfBounds,
            "cycles": self.totals["cycles"],
            "terminalValue": terminal,
        }
        if truncated:
            info["episode_totals"] = dict(self.totals)
            info["regret"] = self.totals["baseline"] - self.totals["profit"]
            # Flat, because SB3's Monitor copies one key out of the final step's info
            # and cannot reach a nested total. rl/train.py asks it for this one.
            info["outOfBoundsTotal"] = self.totals["outOfBounds"]
        return self._observation(), reward, False, truncated, info

    def action_masks(self):
        """MaskablePPO reads this by name. Masked when zero energy would move, or when
        `_closingAction` would override the move.

        The ablation `hideMask` shows the algorithm every action as legal, which is a
        plain PPO. Only what the algorithm reads changes: `_legalMask` stays the
        definition of out of bounds, so the count and the penalty are the same."""
        if self.ablation.hideMask:
            return np.ones(2 * self.levels + 1, dtype=bool)
        return self._legalMask()

    def _legalMask(self):
        """The one definition of an action that moves energy. `step` counts an action
        outside it as out of bounds, whatever the algorithm was shown."""
        return np.array(
            [self._chargeHeadroom() > self.tolerance
             and self._closingAction(action) == action
             for action in range(self.levels)]
            + [self._closingAction(self.idleAction) == self.idleAction]
            + [self._dischargeRoom() > self.tolerance] * self.levels,
            dtype=bool,
        )

    def _direction(self, action):
        if action < self.idleAction:
            return CHARGE
        return IDLE if action == self.idleAction else DISCHARGE

    def _share(self, action):
        """The share of rated power an action asks for, always positive."""
        return abs(self.levels - action) / self.levels

    # --- Internals -----------------------------------------------------------

    def _dayQuarters(self):
        """The 25th and 75th percentile of each episode's realised prices."""
        shaped = self.prices[:len(self.windows) * self.horizon].reshape(
            len(self.windows), self.horizon)
        return np.percentile(shaped, 25, axis=1), np.percentile(shaped, 75, axis=1)

    def _shaping(self, price, boughtMWh, soldMWh):
        """The buy-low/sell-high bonus, in reward units. It moves nothing the backtest
        reports, exactly as the out-of-bounds penalty does not."""
        if not self.ablation.shapingBonus:
            return 0.0
        low, high = self.quarters[0][self.windowIndex], self.quarters[1][self.windowIndex]
        if boughtMWh > 0 and price <= low:
            return self.ablation.shapingBonus * boughtMWh / self.fullStep
        if soldMWh > 0 and price >= high:
            return self.ablation.shapingBonus * soldMWh / self.fullStep
        return 0.0

    def _offset(self):
        return self.windows[self.windowIndex] + self.stepIndex

    def _chargeHeadroom(self):
        return (self.model.MWh - self.stored) / self.model.chargeEfficiency

    def _dischargeHeadroom(self):
        return self.stored * self.model.dischargeEfficiency

    def _budgetHeadroom(self):
        """Energy, cell-side, the cycle budget still allows. Unbounded with no budget."""
        if self.budgetMWh is None:
            return float("inf")
        return self.budgetMWh - self.spentMWh

    def _dischargeRoom(self):
        """The most a discharge can sell, meter-side, before the rate limit.

        Both clips in one place, so the mask and the move cannot disagree: the SoC floor
        and the cycle budget, the budget converted to meter-side energy.
        """
        return min(self._dischargeHeadroom(),
                   self._budgetHeadroom() * self.model.dischargeEfficiency)

    def _emptiable(self, stored, intervals):
        """Whether `stored`, cell-side, can still be sold off in `intervals` full-rate
        discharges without overrunning the cycle budget. Charging spends no budget, so
        the headroom read here is the one the discharges would draw on."""
        return (stored <= intervals * self.fullRemoval + self.tolerance
                and stored <= self._budgetHeadroom() + self.tolerance)

    def _closingAction(self, action):
        """The move actually made under `endEmpty`. Unchanged without it.

        From an emptiable state a discharge always stays emptiable, so the override never
        runs out of moves. `optimisation.dp` reaches the same rule from the other end, by
        valuing every non-empty closing state at minus infinity.
        """
        if not self.endEmpty:
            return action
        after = max(self.horizon - self.stepIndex - 1, 0)
        direction = self._direction(action)
        if direction == CHARGE:
            charged = min(self.stored + self.model.chargeEfficiency
                          * min(self._share(action) * self.fullStep,
                                max(0.0, self._chargeHeadroom())),
                          self.model.MWh)
            if not self._emptiable(charged, after):
                action = self.idleAction
        elif direction == DISCHARGE and action != self.dischargeAction:
            # A part-rate discharge that leaves too much to sell in time is raised to
            # the full rate. With one level the only discharge is the full rate.
            sold = min(self._share(action) * self.fullStep,
                       max(0.0, self._dischargeRoom()))
            left = max(self.stored - sold / self.model.dischargeEfficiency, 0.0)
            if not self._emptiable(left, after):
                action = self.dischargeAction
        if action == self.idleAction and not self._emptiable(self.stored, after):
            action = self.dischargeAction
        return action

    def _applyAction(self, action):
        """Move energy, clipped to what fits. Returns meter-side (bought, sold)."""
        direction = self._direction(action)
        step = self._share(action) * self.fullStep
        if direction == CHARGE:
            boughtMWh = min(step, max(0.0, self._chargeHeadroom()))
            self.stored += self.model.chargeEfficiency * boughtMWh
            self.stored = min(self.stored, self.model.MWh)
            return boughtMWh, 0.0
        if direction == DISCHARGE:
            soldMWh = min(step, max(0.0, self._dischargeRoom()))
            removed = soldMWh / self.model.dischargeEfficiency
            self.stored = max(self.stored - removed, 0.0)
            self.spentMWh += removed
            return 0.0, soldMWh
        return 0.0, 0.0

    def _degradation(self, socPrev, socNew):
        if self.costing is not None:
            # The ablation's own wear cost, the same arithmetic as optimisation.dp.costed:
            # a rate on the energy that left the cells.
            removed = max(socPrev - socNew, 0.0) * self.model.MWh
            return 0.0, 0.0, self.costing[1] * removed
        qCd, qCal, costBoth = self.model.step(socPrev, socNew, self.dtHours)
        if self.calendar:
            return qCd, qCal, costBoth
        return qCd, qCal, self.model.cost(qCd)

    def _baselineAt(self, stepIndex):
        if self.baseline is None:
            return 0.0
        return self.baseline[self.windowIndex][stepIndex]

    def _observation(self):
        start = self.windows[self.windowIndex]
        # AEMO publishes no RRP past the end of the trading day, so the agent reads one
        # trading day ahead whatever the episode length, or a realised-price agent trains
        # on information its forecast-priced twin can never have. An episode opens at
        # 04:00, so the boundary is a whole number of trading days into it.
        day = config.observation_intervals
        horizonEnd = min(self.horizon, (self.stepIndex // day + 1) * day)
        # The episode is over once stepIndex reaches the horizon, and that observation
        # carries no prices, so its clock slot reads zero rather than the next day's.
        remaining = (
            0.0 if self.stepIndex >= self.horizon
            else features.tradingDayRemaining(self.timestamps[self._offset()])
        )
        return features.observation(
            self.stored / self.model.MWh,
            self._observedPrices(start, start + self.horizon),
            self.stepIndex,
            horizonEnd,
            self.scaler,
            remaining,
            self._cyclesLeft(),
            max(0.0, 1.0 - self.stepIndex / self.horizon),
            ablation=self.ablation,
            settled=self.prices[start:start + self.horizon],
            stepIndex=self.stepIndex,
        )

    def _cyclesLeft(self):
        """The unspent fraction of the cycle budget. 1.0 while no budget is in force.

        A fraction rather than the cycles themselves, so the slot stays in [0, 1] with
        every other fraction in the vector.
        """
        if self.budgetMWh is None:
            return 1.0
        if self.budgetMWh == 0.0:
            return 0.0
        return max(0.0, self._budgetHeadroom() / self.budgetMWh)

    def _observedPrices(self, start, end):
        """The price path the agent reads. Slots before stepIndex are zeroed rather than
        filled with realised prices, so a forecast run never leaks the true past."""
        if self.forecasts is None:
            return self.prices[start:end]
        forward = activeRun(self.forecasts[self.windowIndex], self.stepIndex)
        path = np.zeros(self.horizon, dtype=float)
        usable = min(self.horizon - self.stepIndex, len(forward) - self.stepIndex)
        if usable > 0:
            path[self.stepIndex:self.stepIndex + usable] = (
                forward[self.stepIndex:self.stepIndex + usable]
            )
        return np.nan_to_num(path, nan=0.0)

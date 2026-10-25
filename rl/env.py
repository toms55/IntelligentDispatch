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
                 forecasts=None, scaler=None):
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
        self.shuffle = shuffle
        # None means realised prices (training). Otherwise one entry per window: a
        # single path, or a data.forecasts.refreshMatrix stack of runs, of which the
        # agent reads the latest published by each step. Settlement always uses the
        # realised price regardless.
        self.forecasts = forecasts

        self.windows = list(range(0, len(self.prices) - horizon + 1, horizon))
        if not self.windows:
            raise ValueError(
                f"{len(self.prices)} intervals is shorter than one {horizon}-interval episode"
            )

        self.fullStep = model.chargePower * dtHours
        self.tolerance = 1e-9 * self.fullStep

        low, high = features.observationBounds(self.scaler)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.windowCursor = 0
        self.windowIndex = 0
        self.stepIndex = 0
        self.stored = initialSoc * model.MWh

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
        self.totals = {"revenue": 0.0, "degradation": 0.0, "profit": 0.0, "baseline": 0.0}
        return self._observation(), {}

    def step(self, action):
        price = self.prices[self._offset()]
        socPrev = self.stored / self.model.MWh
        boughtMWh, soldMWh = self._applyAction(int(action))
        socNew = self.stored / self.model.MWh

        revenue = price * (soldMWh - boughtMWh)
        qCd, qCal, degradationCost = self._degradation(socPrev, socNew)
        profit = revenue - degradationCost
        baseline = self._baselineAt(self.stepIndex)
        reward = profit / self.rewardScale

        self.totals["revenue"] += revenue
        self.totals["degradation"] += degradationCost
        self.totals["profit"] += profit
        self.totals["baseline"] += baseline

        self.stepIndex += 1
        truncated = self.stepIndex >= self.horizon
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
        }
        if truncated:
            info["episode_totals"] = dict(self.totals)
            info["regret"] = self.totals["baseline"] - self.totals["profit"]
        return self._observation(), reward, False, truncated, info

    def action_masks(self):
        """MaskablePPO reads this by name. Masked only when zero energy would move."""
        return np.array(
            [
                self._chargeHeadroom() > self.tolerance,
                True,
                self._dischargeHeadroom() > self.tolerance,
            ],
            dtype=bool,
        )

    # --- Internals -----------------------------------------------------------

    def _offset(self):
        return self.windows[self.windowIndex] + self.stepIndex

    def _chargeHeadroom(self):
        return (self.model.MWh - self.stored) / self.model.chargeEfficiency

    def _dischargeHeadroom(self):
        return self.stored * self.model.dischargeEfficiency

    def _applyAction(self, action):
        """Move energy, clipped to what fits. Returns meter-side (bought, sold)."""
        if action == CHARGE:
            boughtMWh = min(self.fullStep, max(0.0, self._chargeHeadroom()))
            self.stored += self.model.chargeEfficiency * boughtMWh
            self.stored = min(self.stored, self.model.MWh)
            return boughtMWh, 0.0
        if action == DISCHARGE:
            soldMWh = min(self.fullStep, max(0.0, self._dischargeHeadroom()))
            self.stored -= soldMWh / self.model.dischargeEfficiency
            self.stored = max(self.stored, 0.0)
            return 0.0, soldMWh
        return 0.0, 0.0

    def _degradation(self, socPrev, socNew):
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
        end = start + self.horizon
        # The episode is over once stepIndex reaches the horizon, and that observation
        # carries no prices, so its clock slot reads zero rather than the next day's.
        remaining = (
            0.0 if self.stepIndex >= self.horizon
            else features.tradingDayRemaining(self.timestamps[self._offset()])
        )
        return features.observation(
            self.stored / self.model.MWh,
            self._observedPrices(start, end),
            self.stepIndex,
            self.horizon,
            self.scaler,
            remaining,
        )

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

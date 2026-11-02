"""The observation vector, built the same way in training and at inference.

Training fills the forward price slots from realised history; inference fills them from
AEMO P5MIN and PREDISPATCH forecasts. Nothing here knows which.

Padding past the episode end must stay zero, never real prices past the boundary, or the
agent gets foresight the oracle lacks.
"""
import math

import numpy as np

import config
from utils import nem_interval
from utils.market import market_price_floor, marketPriceCap

# The layout, named once. Index by these rather than by a literal or by -1: a negative
# index silently follows the end of the vector when a slot is appended.
socIndex = 0
priceSlice = slice(1, 1 + config.price_feature_count)
clockIndex = 1 + config.price_feature_count
cycleIndex = clockIndex + 1
episodeIndex = cycleIndex + 1
# Half-hour history slots, when an ablation asks for them, follow the baseline layout.
historyStart = episodeIndex + 1


def observationSize(ablation=None):
    """The vector's length: the baseline's 62 slots plus any history slots."""
    return config.observation_size + (0 if ablation is None else ablation.history)


class PriceScaler:
    """A fixed price to observation transform, pinned to one market price cap.

    asinh is odd about zero, so the transform needs no sign handling. It is roughly
    linear below the knee and roughly logarithmic above it. The cap maps to exactly 1.0.

    Build one scaler per run and keep it for the life of the experiment. The divisor must
    not change between training and evaluation. If it changes, the same observation value
    means two different prices and the policy reads the wrong world. rl/train.py saves
    the scaler beside the checkpoint and rl/evaluate.py loads it back.
    """

    def __init__(self, cap, floor=market_price_floor, knee=config.price_knee,
                 transform="asinh", clipLow=None, clipHigh=None):
        if transform not in ("asinh", "linear", "clip"):
            raise ValueError(f"unknown price transform {transform!r}")
        if transform == "clip" and (clipLow is None or clipHigh is None
                                    or clipLow >= clipHigh):
            raise ValueError("a clip transform needs clipLow below clipHigh")
        if cap <= 0:
            raise ValueError(f"cap must be positive, got {cap}")
        if knee <= 0:
            raise ValueError(f"knee must be positive, got {knee}")
        if floor >= 0:
            raise ValueError(f"floor must be negative, got {floor}")
        self.cap = float(cap)
        self.floor = float(floor)
        self.knee = float(knee)
        # The ablation study's price transforms. "asinh" is the baseline. "linear" maps
        # the price to price / cap, so a $5000 spike is 0.2 and ordinary prices sit near
        # zero. "clip" clamps the price to [clipLow, clipHigh] first (Kortmann et al.'s
        # 10th to 90th percentile), then applies the asinh. The settled price the agent is
        # paid is never transformed; this is the observation only.
        self.transform = transform
        self.clipLow = None if clipLow is None else float(clipLow)
        self.clipHigh = None if clipHigh is None else float(clipHigh)
        self._divisor = math.asinh(self.cap / self.knee)

    def __call__(self, price):
        clipped = min(max(price, self.floor), self.cap)
        if self.transform == "linear":
            return clipped / self.cap
        if self.transform == "clip":
            clipped = min(max(clipped, self.clipLow), self.clipHigh)
        return math.asinh(clipped / self.knee) / self._divisor

    @property
    def low(self):
        """The normalised market price floor, the smallest value any slot can hold."""
        return self(self.floor)

    @property
    def high(self):
        """The normalised market price cap. Exactly 1.0 by construction."""
        return 1.0

    def asDict(self):
        out = {"cap": self.cap, "floor": self.floor, "knee": self.knee}
        # Only when it differs, so a scaler.json written before the study reads back equal.
        if self.transform != "asinh":
            out["transform"] = self.transform
        if self.clipLow is not None:
            out["clipLow"], out["clipHigh"] = self.clipLow, self.clipHigh
        return out

    def __eq__(self, other):
        if not isinstance(other, PriceScaler):
            return NotImplemented
        return self.asDict() == other.asDict()

    def __repr__(self):
        extra = "".join(f", {k}={v!r}" for k, v in self.asDict().items()
                        if k not in ("cap", "floor", "knee"))
        return (f"PriceScaler(cap={self.cap!r}, floor={self.floor!r}, "
                f"knee={self.knee!r}{extra})")


def scalerFor(timestamps, knee=config.price_knee, **transform):
    """The scaler for a run, pinned to the cap in force at the last timestamp.

    The last timestamp, not the first: the caps only rise, so the final cap is also the
    largest over the window and no earlier price clips.
    """
    if len(timestamps) == 0:
        raise ValueError("cannot pin a price scaler to an empty window")
    return PriceScaler(marketPriceCap(timestamps[-1]), knee=knee, **transform)


def priceFeatures(prices, start, horizonEnd, scaler,
                  lookahead=config.observation_intervals):
    """The forward price vector as seen at interval `start`. Nothing past `horizonEnd`
    or past the end of `prices` is read; those slots are zero.

    `lookahead` is the ablation's view length: a five-minute slot is read only if it falls
    inside it, and a half-hour bucket only if the whole bucket does, so a shorter view
    zeroes the far buckets and keeps the vector's size."""
    limit = min(horizonEnd, len(prices), start + lookahead)
    out = []
    for step in range(config.five_minute_interval_count):
        index = start + step
        out.append(scaler(prices[index]) if index < limit else 0.0)
    bucket = start + config.five_minute_interval_count
    for _ in range(config.price_bucket_count):
        end = bucket + config.bucket_window_intervals
        window = ([prices[i] for i in range(bucket, min(end, limit))]
                  if end <= start + lookahead else [])
        out.append(scaler(sum(window) / len(window)) if window else 0.0)
        bucket = end
    return out


def historyFeatures(settled, start, stepIndex, scaler, count):
    """`count` half-hour means of the prices settled in this episode, newest first.

    `settled` is the episode's realised price path and `start` its first index in it (0
    from `observation`, whose price path is already the episode's own), so the
    slots read only intervals `start` to `start + stepIndex`: nothing before the episode,
    which would cross a region join in the pooled training set, and nothing from now on.
    A bucket the episode has not reached yet is zero, and a partly filled one is the mean
    of what has settled."""
    out = []
    newest = start + stepIndex
    for _ in range(count):
        oldest = max(newest - config.bucket_window_intervals, start)
        window = [settled[i] for i in range(oldest, newest)]
        out.append(scaler(sum(window) / len(window)) if window else 0.0)
        newest -= config.bucket_window_intervals
    return out


def timeSlots(remaining, ablation):
    """The clock slot and the episode slot, as the ablation's time encoding has them.

    `remaining` is the fraction of the trading day left; the episode slot is the same
    number on a one-day episode, which is why the sine and cosine may take its place.
    Returns (clock, None) when the episode slot keeps its own value."""
    if ablation is None or ablation.timeEncoding == "clock":
        return remaining, None
    if ablation.timeEncoding == "none":
        return 0.0, 0.0
    angle = 2.0 * math.pi * (1.0 - remaining)
    return (math.sin(angle) + 1.0) / 2.0, (math.cos(angle) + 1.0) / 2.0


def tradingDayRemaining(stamp):
    """The fraction of the 04:00-to-04:00 NEM trading day left when `stamp`'s interval
    opens.

    Read from the clock, not from the episode. An episode-relative fraction is the same
    number only while an episode is a whole trading day that starts at 04:00, and it is
    wrong everywhere else: the current forecast screen starts its episode at the run
    time, so it read 1.0 at every hour of the day, and a horizon below 288 would do the
    same from its second window on.

    AEMO stamps an interval by its end, so the measurement runs from the interval's
    start. The 04:05 stamp therefore reads exactly 1.0 and the 04:00 stamp 1/288, which
    reproduces the old formula interval for interval on an aligned 288-interval episode.
    Every published agent trained under that, so this changes no weight's meaning.
    """
    opened = stamp - nem_interval
    elapsed = (opened.hour + opened.minute / 60 - 4) % 24
    return 1.0 - elapsed / 24


def observation(soc, prices, start, horizonEnd, scaler, remaining, cyclesLeft,
                episodeLeft, ablation=None, settled=None, stepIndex=0):
    """The full vector: SoC, forward prices, time left, budget left, episode left.

    `remaining` comes from `tradingDayRemaining`. The caller passes it rather than a
    stamp because only the caller knows whether an interval is still open: the
    observation returned on truncation has no interval left, and its price slots are all
    zero, so its clock slot is zero too.

    `cyclesLeft` is the unspent fraction of the cycle budget and `episodeLeft` the
    unspent fraction of the episode. Both are required, with no default. The policy is a
    function of this vector alone, so an agent that cannot see its remaining budget
    cannot ration it, and a default here would hand the live screen a budget nobody
    chose. `remaining` reads the clock and so repeats every trading day, which is why it
    cannot stand in for `episodeLeft` on an episode longer than a day.
    """
    lookahead = config.observation_intervals if ablation is None else ablation.lookahead
    clock, episode = timeSlots(remaining, ablation)
    values = (
        [soc]
        + priceFeatures(prices, start, horizonEnd, scaler, lookahead)
        + [clock, cyclesLeft, episodeLeft if episode is None else episode]
    )
    if ablation is not None and ablation.history:
        values += historyFeatures(settled if settled is not None else prices, 0,
                                  stepIndex, scaler, ablation.history)
    return np.asarray(values, dtype=np.float32)


def observationBounds(scaler, ablation=None):
    """Explicit finite low/high arrays for the Box space."""
    size = observationSize(ablation)
    low = np.empty(size, dtype=np.float32)
    high = np.empty(size, dtype=np.float32)
    low[socIndex], high[socIndex] = 0.0, 1.0
    low[priceSlice] = scaler.low
    high[priceSlice] = scaler.high
    # The trading-day clock, the cycle budget left and the episode left are all fractions.
    low[clockIndex:historyStart], high[clockIndex:historyStart] = 0.0, 1.0
    # History slots hold transformed prices, so they share the price slots' range.
    low[historyStart:], high[historyStart:] = scaler.low, scaler.high
    return low, high

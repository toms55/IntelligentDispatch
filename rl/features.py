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


class PriceScaler:
    """A fixed price to observation transform, pinned to one market price cap.

    asinh is odd about zero, so the transform needs no sign handling. It is roughly
    linear below the knee and roughly logarithmic above it. The cap maps to exactly 1.0.

    Build one scaler per run and keep it for the life of the experiment. The divisor must
    not change between training and evaluation. If it changes, the same observation value
    means two different prices and the policy reads the wrong world. rl/train.py saves
    the scaler beside the checkpoint and rl/evaluate.py loads it back.
    """

    def __init__(self, cap, floor=market_price_floor, knee=config.price_knee):
        if cap <= 0:
            raise ValueError(f"cap must be positive, got {cap}")
        if knee <= 0:
            raise ValueError(f"knee must be positive, got {knee}")
        if floor >= 0:
            raise ValueError(f"floor must be negative, got {floor}")
        self.cap = float(cap)
        self.floor = float(floor)
        self.knee = float(knee)
        self._divisor = math.asinh(self.cap / self.knee)

    def __call__(self, price):
        clipped = min(max(price, self.floor), self.cap)
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
        return {"cap": self.cap, "floor": self.floor, "knee": self.knee}

    def __eq__(self, other):
        if not isinstance(other, PriceScaler):
            return NotImplemented
        return self.asDict() == other.asDict()

    def __repr__(self):
        return f"PriceScaler(cap={self.cap!r}, floor={self.floor!r}, knee={self.knee!r})"


def scalerFor(timestamps, knee=config.price_knee):
    """The scaler for a run, pinned to the cap in force at the last timestamp.

    The last timestamp, not the first: the caps only rise, so the final cap is also the
    largest over the window and no earlier price clips.
    """
    if len(timestamps) == 0:
        raise ValueError("cannot pin a price scaler to an empty window")
    return PriceScaler(marketPriceCap(timestamps[-1]), knee=knee)


def priceFeatures(prices, start, horizonEnd, scaler):
    """The forward price vector as seen at interval `start`. Nothing past `horizonEnd`
    or past the end of `prices` is read; those slots are zero."""
    limit = min(horizonEnd, len(prices))
    out = []
    for step in range(config.five_minute_interval_count):
        index = start + step
        out.append(scaler(prices[index]) if index < limit else 0.0)
    bucket = start + config.five_minute_interval_count
    for _ in range(config.price_bucket_count):
        window = [prices[i] for i in range(bucket, min(bucket + config.bucket_window_intervals, limit))]
        out.append(scaler(sum(window) / len(window)) if window else 0.0)
        bucket += config.bucket_window_intervals
    return out


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
                episodeLeft):
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
    values = (
        [soc]
        + priceFeatures(prices, start, horizonEnd, scaler)
        + [remaining, cyclesLeft, episodeLeft]
    )
    return np.asarray(values, dtype=np.float32)


def observationBounds(scaler):
    """Explicit finite low/high arrays for the Box space."""
    low = np.empty(config.observation_size, dtype=np.float32)
    high = np.empty(config.observation_size, dtype=np.float32)
    low[socIndex], high[socIndex] = 0.0, 1.0
    low[priceSlice] = scaler.low
    high[priceSlice] = scaler.high
    # The trading-day clock, the cycle budget left and the episode left are all fractions.
    low[clockIndex:], high[clockIndex:] = 0.0, 1.0
    return low, high

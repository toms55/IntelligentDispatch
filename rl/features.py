"""The observation vector, built the same way in training and at inference.

Training fills the forward price slots from realised history; inference fills them from
AEMO P5MIN and PREDISPATCH forecasts. Nothing here knows which.

Padding past the episode end must stay zero, never real prices past the boundary, or the
agent gets foresight the oracle lacks.
"""
import math

import numpy as np

import config
from utils.market import market_price_floor, marketPriceCap


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


def observation(soc, prices, start, horizonEnd, scaler):
    """The full vector: SoC, forward prices, time remaining."""
    remaining = (horizonEnd - start) / (horizonEnd if horizonEnd else 1)
    values = [soc] + priceFeatures(prices, start, horizonEnd, scaler) + [remaining]
    return np.asarray(values, dtype=np.float32)


def observationBounds(scaler):
    """Explicit finite low/high arrays for the Box space."""
    low = np.empty(config.observation_size, dtype=np.float32)
    high = np.empty(config.observation_size, dtype=np.float32)
    low[0], high[0] = 0.0, 1.0  # soc
    priceEnd = 1 + config.price_feature_count
    low[1:priceEnd] = scaler.low
    high[1:priceEnd] = scaler.high
    low[-1], high[-1] = 0.0, 1.0  # intervals remaining
    return low, high

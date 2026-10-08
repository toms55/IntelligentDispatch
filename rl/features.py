"""The observation vector, built the same way in training and at inference.

Training fills the forward price slots from realised history; inference fills them from
AEMO P5MIN and PREDISPATCH forecasts. Nothing here knows which.

Padding past the episode end must stay zero, never real prices past the boundary, or the
agent gets foresight the oracle lacks.
"""
import math

import numpy as np

import config


def _rawNormalise(price):
    scaled = math.log1p(abs(price) / config.price_log_scale) / config.price_log_divisor
    return math.copysign(scaled, price)


_FLOOR = _rawNormalise(config.price_floor)
_CAP = _rawNormalise(config.price_cap)
priceFloorNormalised = _FLOOR
priceCapNormalised = _CAP


def normalisePrice(price):
    """A fixed transform, never a running statistic, to roughly [-0.52, 1.0]."""
    scaled = math.log1p(abs(price) / config.price_log_scale) / config.price_log_divisor
    signed = math.copysign(scaled, price)
    return min(max(signed, _FLOOR), _CAP)


def priceFeatures(prices, start, horizonEnd):
    """The forward price vector as seen at interval `start`. Nothing past `horizonEnd`
    or past the end of `prices` is read; those slots are zero."""
    limit = min(horizonEnd, len(prices))
    out = []
    for step in range(config.five_minute_interval_count):
        index = start + step
        out.append(normalisePrice(prices[index]) if index < limit else 0.0)
    bucket = start + config.five_minute_interval_count
    for _ in range(config.price_bucket_count):
        window = [prices[i] for i in range(bucket, min(bucket + config.price_bucket_intervals, limit))]
        out.append(normalisePrice(sum(window) / len(window)) if window else 0.0)
        bucket += config.price_bucket_intervals
    return out


def observation(soc, prices, start, horizonEnd):
    """The full vector: SoC, forward prices, time remaining."""
    remaining = (horizonEnd - start) / (horizonEnd if horizonEnd else 1)
    values = [soc] + priceFeatures(prices, start, horizonEnd) + [remaining]
    return np.asarray(values, dtype=np.float32)


def observationBounds():
    """Explicit finite low/high arrays for the Box space."""
    low = np.empty(config.observation_size, dtype=np.float32)
    high = np.empty(config.observation_size, dtype=np.float32)
    low[0], high[0] = 0.0, 1.0  # soc
    priceEnd = 1 + config.price_feature_count
    low[1:priceEnd] = priceFloorNormalised
    high[1:priceEnd] = priceCapNormalised
    low[-1], high[-1] = 0.0, 1.0  # intervals remaining
    return low, high

"""Per-trading-day price statistics, and the high-volatility label.

The label is the mean excess over a spike threshold across the day:

    spikeIntensity = sum(max(RRP - spike_threshold, 0)) / episode_intervals

Only the excess counts, so cheap intervals cannot offset a spike and a negative price
never makes a day high-volatility. That is deliberate: a standard deviation or a full
intraday range is moved by a cheap interval as much as by a dear one, and it is the dear
ones a battery is paid for.

Both values are in raw dollars from `data/prices.py`, never the asinh-transformed
observation, which compresses exactly the spikes this is measuring.
"""
import pandas as pd

import config
from backtest import folds


def dayVolatility(dayPrices, threshold=config.spike_threshold,
                  cutoff=config.high_vol_intensity):
    """Spike intensity for one trading day, and whether it reaches the cutoff."""
    values = dayPrices.to_numpy(dtype=float)
    intensity = float((values - threshold).clip(min=0).sum() / len(values))
    return {"spikeIntensity": intensity, "highVol": bool(intensity >= cutoff)}


def volatilityFrame(prices, days, **kwargs):
    """One row per trading day in `days`, indexed by its 04:00 start.

    Each day must be a complete 288 intervals. A short day would divide the excess by a
    smaller denominator and overstate the intensity.
    """
    rows = {}
    for day in days:
        end = day + folds.oneDay
        window = folds.sliceWindow(prices, day, end)
        folds.requireFullDays(window, day, end)
        rows[day] = dayVolatility(window, **kwargs)
    return pd.DataFrame.from_dict(rows, orient="index")

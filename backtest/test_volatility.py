"""Tests for backtest/volatility.py. Run with: uv run pytest backtest/ -q

The measure is the mean excess over a spike threshold across a trading day:

    spikeIntensity = sum(max(RRP - 300, 0)) / 288

Measured on real NSW1 data, a cutoff of 1.20 selects 62 of 365 days in 2025 (17%) which
carry 64.6% of the oracle's profit, and 82 of 366 in 2024 (22%) carrying 75.9%. That
cross-check belongs in the dry run against real prices, not here.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

import config
from backtest import volatility
from utils import nem_interval

DAY = datetime(2024, 3, 7, 4)


def day(values):
    """One trading day of 288 interval-ending stamps, 04:05 through next-day 04:00."""
    index = pd.date_range(DAY + nem_interval, DAY + timedelta(days=1), freq=nem_interval)
    assert len(values) == len(index) == 288
    return pd.Series(values, index=index, dtype=float, name="RRP")


def spiky(count, high, base=50.0):
    return day([high] * count + [base] * (288 - count))


class TestSpikeIntensity:
    # Twelve intervals $600 above the threshold average out to exactly $25 over the day.
    def test_is_the_mean_excess_over_the_threshold(self):
        assert volatility.dayVolatility(spiky(12, 900.0))["spikeIntensity"] == 25.0

    # A day that never reaches the threshold has no spike intensity at all.
    def test_a_day_below_the_threshold_is_zero(self):
        assert volatility.dayVolatility(day([120.0] * 288))["spikeIntensity"] == 0.0

    # Only the excess counts, so cheap intervals cannot offset a spike. This is what
    # separates the measure from a standard deviation or a full intraday range.
    def test_negative_prices_do_not_reduce_the_intensity(self):
        cheap = day([900.0] * 12 + [-1000.0] * 276)
        assert volatility.dayVolatility(cheap)["spikeIntensity"] == 25.0

    # Raising the threshold leaves less excess, by exactly the amount removed.
    def test_a_different_threshold_moves_the_intensity(self):
        got = volatility.dayVolatility(spiky(12, 900.0), threshold=600.0)
        assert got["spikeIntensity"] == pytest.approx(12 * 300.0 / 288)

    # The cutoff is inclusive: a day sitting exactly on it counts as high-volatility.
    def test_the_cutoff_is_inclusive(self):
        onTheLine = day([300.0 + 288.0] + [50.0] * 287)
        assert volatility.dayVolatility(onTheLine)["spikeIntensity"] == 1.0
        assert volatility.dayVolatility(onTheLine, cutoff=1.0)["highVol"] is True
        assert volatility.dayVolatility(day([300.0 + 287.0] + [50.0] * 287),
                                        cutoff=1.0)["highVol"] is False

    # The shipped defaults are the ones the NSW1 calibration was done against.
    def test_the_defaults_are_the_calibrated_pair(self):
        assert config.spike_threshold == 300.0
        assert config.high_vol_intensity == 1.20


class TestVolatilityFrame:
    # One row per requested trading day, keyed on the 04:00 start so it joins to the
    # backtest's testDay column.
    def test_one_row_per_day_indexed_by_the_0400_start(self):
        days = [DAY + timedelta(days=n) for n in range(3)]
        index = pd.date_range(DAY + nem_interval, DAY + timedelta(days=3),
                              freq=nem_interval)
        prices = pd.Series(range(len(index)), index=index, dtype=float)
        frame = volatility.volatilityFrame(prices, days)
        assert list(frame.index) == days
        assert "spikeIntensity" in frame and "highVol" in frame

    # A day that is not a complete 288 intervals is refused rather than averaged over a
    # short denominator, which would overstate the intensity.
    def test_an_incomplete_day_is_refused(self):
        index = pd.date_range(DAY + nem_interval, periods=200, freq=nem_interval)
        prices = pd.Series(1.0, index=index)
        with pytest.raises(ValueError):
            volatility.volatilityFrame(prices, [DAY])

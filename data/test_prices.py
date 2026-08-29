"""Tests for data/prices.py. No network: fetch_price_data is monkeypatched."""
from datetime import datetime

import pandas as pd
import pytest

from data import prices


@pytest.fixture
def raw():
    """Two regions, one intervened duplicate, deliberately out of order."""
    return pd.DataFrame(
        {
            "SETTLEMENTDATE": pd.to_datetime(
                [
                    "2024-01-01 00:10",
                    "2024-01-01 00:05",
                    "2024-01-01 00:05",
                    "2024-01-01 00:05",
                ]
            ),
            "REGIONID": ["SA1", "SA1", "SA1", "NSW1"],
            "INTERVENTION": [0, 0, 1, 0],
            "RRP": [120.0, 100.0, 999.0, 50.0],
        }
    )


@pytest.fixture(autouse=True)
def noNetwork(monkeypatch, raw):
    monkeypatch.setattr(prices, "fetch_price_data", lambda *a, **k: raw)


def test_keeps_only_the_requested_region():
    series = prices.price_series(datetime(2024, 1, 1), datetime(2024, 1, 2), "SA1")
    assert list(series.values) == [100.0, 120.0]


def test_drops_intervention_rows():
    """AEMO reruns dispatch on intervention, leaving two rows for one interval.
    Keeping both would double-count the interval and shift the whole series."""
    series = prices.price_series(datetime(2024, 1, 1), datetime(2024, 1, 2), "SA1")
    assert 999.0 not in series.values
    assert series.index.is_unique


def test_sorted_by_time():
    series = prices.price_series(datetime(2024, 1, 1), datetime(2024, 1, 2), "SA1")
    assert series.index.is_monotonic_increasing


def test_region_is_selectable():
    series = prices.price_series(datetime(2024, 1, 1), datetime(2024, 1, 2), "NSW1")
    assert list(series.values) == [50.0]

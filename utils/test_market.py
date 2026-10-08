"""Tests for utils/market.py. Run with: uv run pytest utils/ -q

The numbers here are the AEMC determinations, not a restatement of the module. Check a
change against the AEMC publication, never against market.py.
"""
from datetime import datetime

import pytest

from utils.market import (cumulativePriceThreshold, marketPriceCap,
                          market_price_floor, price_settings)

# (financial year start, MPC $/MWh, CPT $) exactly as the AEMC determined them.
determinations = [
    (datetime(2022, 7, 1, 4), 15500.0, 1398100.0),
    (datetime(2023, 7, 1, 4), 16600.0, 1490200.0),
    (datetime(2024, 7, 1, 4), 17500.0, 1573700.0),
    (datetime(2025, 7, 1, 4), 20300.0, 1823600.0),
    (datetime(2026, 7, 1, 4), 23200.0, 2225900.0),
]


class TestMarketPriceCap:
    @pytest.mark.parametrize("effective,cap,_cpt", determinations)
    def test_each_determination(self, effective, cap, _cpt):
        assert marketPriceCap(effective) == cap

    @pytest.mark.parametrize("effective,cap,_cpt", determinations)
    def test_holds_until_the_next_reset(self, effective, cap, _cpt):
        assert marketPriceCap(effective.replace(month=12, day=31)) == cap

    def test_changes_at_0400_not_midnight(self):
        """The NEM trading day starts at 04:00, so the first four hours of 1 July still
        carry the old financial year's cap."""
        assert marketPriceCap(datetime(2025, 7, 1, 3, 55)) == 17500.0
        assert marketPriceCap(datetime(2025, 7, 1, 4, 0)) == 20300.0

    def test_a_pandas_timestamp_works(self):
        pandas = pytest.importorskip("pandas")
        assert marketPriceCap(pandas.Timestamp("2025-08-17 14:30")) == 20300.0

    def test_before_the_table_starts_is_refused(self):
        with pytest.raises(ValueError, match="2022-07-01"):
            marketPriceCap(datetime(2022, 6, 30, 12))

    def test_rises_monotonically(self):
        """What makes one cap per run safe: the cap at the end of a window is also the
        largest cap over it, so nothing earlier in the window clips."""
        caps = [cap for _, cap, _ in price_settings]
        assert all(later > earlier for earlier, later in zip(caps, caps[1:]))


class TestCumulativePriceThreshold:
    @pytest.mark.parametrize("effective,_cap,cpt", determinations)
    def test_each_determination(self, effective, _cap, cpt):
        assert cumulativePriceThreshold(effective) == cpt

    def test_changes_at_0400_not_midnight(self):
        assert cumulativePriceThreshold(datetime(2025, 7, 1, 3, 55)) == 1573700.0
        assert cumulativePriceThreshold(datetime(2025, 7, 1, 4, 0)) == 1823600.0

    def test_before_the_table_starts_is_refused(self):
        with pytest.raises(ValueError, match="2022-07-01"):
            cumulativePriceThreshold(datetime(2021, 1, 1))


class TestPriceFloor:
    def test_is_the_administered_floor(self):
        assert market_price_floor == -1000.0


class TestTheTable:
    def test_matches_the_determinations(self):
        assert list(price_settings) == [tuple(row) for row in determinations]

    def test_is_sorted_by_effective_date(self):
        dates = [effective for effective, _, _ in price_settings]
        assert dates == sorted(dates)

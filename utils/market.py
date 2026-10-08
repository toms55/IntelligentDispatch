"""NEM administered price settings, by the trading day they take effect.

The AEMC resets the Market Price Cap (MPC) and the Cumulative Price Threshold (CPT) on
1 July. The NEM trading day starts at 04:00, so each row takes effect at 04:00 and the
first four hours of 1 July still carry the old financial year's settings.

The caps rise at every reset. That is what lets a run pin one cap: the cap in force at
the end of a window is also the largest cap over it.
"""
from bisect import bisect_right

from datetime import datetime

# (effective from, MPC $/MWh, CPT $)
price_settings = (
    (datetime(2022, 7, 1, 4), 15500.0, 1398100.0),
    (datetime(2023, 7, 1, 4), 16600.0, 1490200.0),
    (datetime(2024, 7, 1, 4), 17500.0, 1573700.0),
    (datetime(2025, 7, 1, 4), 20300.0, 1823600.0),
    (datetime(2026, 7, 1, 4), 23200.0, 2225900.0),
)

# The floor did not move across any period the table covers. Make it a column if the
# AEMC ever changes it.
market_price_floor = -1000.0

_effective = [row[0] for row in price_settings]


def _settingsAt(when):
    """The row in force at `when`. Accepts a datetime or a pandas Timestamp."""
    index = bisect_right(_effective, when) - 1
    if index < 0:
        raise ValueError(
            f"no NEM price settings before {_effective[0].date()}; got {when}"
        )
    return price_settings[index]


def marketPriceCap(when):
    """The Market Price Cap in force at `when`, in $/MWh."""
    return _settingsAt(when)[1]


def cumulativePriceThreshold(when):
    """The Cumulative Price Threshold in force at `when`, in $."""
    return _settingsAt(when)[2]

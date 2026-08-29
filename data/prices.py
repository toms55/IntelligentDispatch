"""Clean RRP series for one NEM region, ready to feed the oracle."""
from datetime import datetime

import pandas as pd

from data.NEMOSIS import fetch_price_data


def price_series(start: datetime, end: datetime, region: str = "SA1") -> pd.Series:
    """5-minute RRP for `region`, indexed by SETTLEMENTDATE.

    NEMOSIS returns one row per region per interval, plus a duplicate when AEMO
    intervened in dispatch.
    """
    raw = fetch_price_data(start, end, ["SETTLEMENTDATE", "REGIONID", "INTERVENTION", "RRP"])
    rows = raw[(raw["REGIONID"] == region) & (raw["INTERVENTION"] == 0)]
    return rows.set_index("SETTLEMENTDATE")["RRP"].astype(float).sort_index()

"""AEMO price forecasts, as published at a given run time.

Stitches two AEMO runs: P5MIN REGIONSOLUTION (5-minute, 11 intervals, 55 minutes ahead)
under PREDISPATCH PRICE (30-minute, to the end of the trading day). PDPASA and STPASA
carry no RRP column at all.

Selection is always by RUN TIME, never by forecasted time -- the latter would pick
whichever run was most accurate in hindsight.

A PREDISPATCH value is held flat across its six 5-minute intervals rather than
interpolated, since dispatch price is a step function.
"""
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils import nem_interval, nem_interval_hours

rawCache = Path(__file__).resolve().parent.parent / "data_cache" / "nemseer_raw"

p5minIntervals = 11
predispatchInterval = timedelta(minutes=30)


def _nemseer():
    import nemseer

    rawCache.mkdir(parents=True, exist_ok=True)
    return nemseer


def _stamp(moment):
    return pd.Timestamp(moment).strftime("%Y/%m/%d %H:%M")


def p5minForecast(runTime, region="SA1"):
    """RRP by interval end, as the P5MIN run at `runTime` saw it. 11 intervals."""
    nemseer = _nemseer()
    runTime = pd.Timestamp(runTime)
    first = runTime + nem_interval
    last = runTime + p5minIntervals * nem_interval
    tables = nemseer.compile_data(
        _stamp(runTime), _stamp(runTime), _stamp(first), _stamp(last),
        "P5MIN", "REGIONSOLUTION", raw_cache=str(rawCache),
    )
    frame = next(iter(tables.values()))
    rows = frame[(frame["REGIONID"] == region) & (frame["RUN_DATETIME"] == runTime)]
    return rows.set_index("INTERVAL_DATETIME")["RRP"].astype(float).sort_index()


def predispatchForecast(runTime, region="SA1", hours=24):
    """RRP by 30-minute interval end, as the PREDISPATCH run at or before `runTime` saw it.

    `hours` cannot exceed what the run reaches; that varies with time of day since
    PREDISPATCH always stops at the end of a trading day. Episodes align to 04:00
    because that is where 24h is available.
    """
    nemseer = _nemseer()
    runTime = pd.Timestamp(runTime)
    run = runTime.floor("30min")
    first = run + predispatchInterval
    last = run + timedelta(hours=hours)
    tables = nemseer.compile_data(
        _stamp(run), _stamp(run), _stamp(first), _stamp(last),
        "PREDISPATCH", "PRICE", raw_cache=str(rawCache),
    )
    frame = next(iter(tables.values()))
    rows = frame[
        (frame["REGIONID"] == region)
        & (frame["INTERVENTION"] == 0)
        & (frame["PREDISPATCH_RUN_DATETIME"] == run)
    ]
    return rows.set_index("DATETIME")["RRP"].astype(float).sort_index()


def _expandToFiveMinutes(halfHourly):
    """One 30-minute value becomes six 5-minute values, held flat."""
    values = {}
    for end, price in halfHourly.items():
        end = pd.Timestamp(end)
        for step in range(6):
            values[end - step * nem_interval] = price
    return pd.Series(values).sort_index()


def forecastSeries(runTime, region="SA1", horizon=288, allowPartial=False):
    """The forward price path the agent can see at `runTime`. P5MIN wins on the overlap.

    Raises if the two runs together do not reach the whole horizon, unless allowPartial.
    """
    runTime = pd.Timestamp(runTime)
    wanted = pd.DatetimeIndex(
        [runTime + (step + 1) * nem_interval for step in range(horizon)]
    )

    def _fetch(source):
        try:
            return source()
        except ValueError:
            # nemseer itself raises when a whole run has no archive at all (as opposed
            # to our own short-path check below); with allowPartial that is just more
            # missing coverage to pad, not a reason to give up on the other source.
            if not allowPartial:
                raise
            return pd.Series(dtype=float)

    coarse = _fetch(lambda: _expandToFiveMinutes(
        predispatchForecast(runTime, region, hours=horizon * nem_interval_hours)
    ))
    fine = _fetch(lambda: p5minForecast(runTime, region))
    merged = pd.concat([coarse, fine])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()

    path = merged.reindex(wanted)
    missing = int(path.isna().sum())
    if missing and not allowPartial:
        raise ValueError(
            f"forecast at {runTime} covers {horizon - missing} of {horizon} intervals; "
            f"the run reaches {merged.index.max()}"
        )
    return path


def forecastMatrix(runTimes, region="SA1", horizon=288, allowPartial=True):
    """One forward path per run time, as a 2-D array. Padded with NaN past run coverage."""
    rows = []
    for runTime in runTimes:
        path = forecastSeries(runTime, region, horizon, allowPartial=allowPartial)
        rows.append(path.to_numpy(dtype=float))
    return np.asarray(rows)

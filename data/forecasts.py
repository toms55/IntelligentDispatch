"""AEMO price forecasts, as published at a given run time.

Stitches two AEMO runs: P5MIN REGIONSOLUTION (5-minute, 11 intervals, 55 minutes ahead)
under PREDISPATCH PRICE (30-minute, to the end of the trading day). PDPASA and STPASA
carry no RRP column at all.

Selection is always by RUN TIME, never by forecasted time -- the latter would pick
whichever run was most accurate in hindsight.

A PREDISPATCH value is held flat across its six 5-minute intervals rather than
interpolated, since dispatch price is a step function.
"""
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from utils import nem_interval, nem_interval_hours

rawCache = Path(__file__).resolve().parent.parent / "data_cache" / "nemseer_raw"
forecastCacheRoot = Path(__file__).resolve().parent.parent / "data_cache" / "forecast_paths"
tradingDayStart = time(4, 0)

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


def _stitch(halfHourly, fiveMinute):
    """The one merge rule: PREDISPATCH held flat, then P5MIN overwrites the overlap."""
    # An empty part is dropped before the concat, not after: concatenating an empty
    # Series decides the result dtype from it, which pandas is about to change.
    parts = [part for part in (_expandToFiveMinutes(halfHourly), fiveMinute)
             if len(part)]
    if not parts:
        return pd.Series(dtype=float)
    merged = pd.concat(parts)
    return merged[~merged.index.duplicated(keep="last")].sort_index()


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

    coarse = _fetch(lambda: predispatchForecast(
        runTime, region, hours=horizon * nem_interval_hours))
    fine = _fetch(lambda: p5minForecast(runTime, region))
    merged = _stitch(coarse, fine)

    path = merged.reindex(wanted)
    missing = int(path.isna().sum())
    if missing and not allowPartial:
        raise ValueError(
            f"forecast at {runTime} covers {horizon - missing} of {horizon} intervals; "
            f"the run reaches {merged.index.max()}"
        )
    return path


def forecastMatrix(runTimes, region="SA1", horizon=288, allowPartial=True, source=None):
    """One forward path per run time, as a 2-D array. Padded with NaN past run coverage.

    `source` is a frame from `loadForecastCache`. Given one this does no I/O at all: a
    run time the cache never saw becomes a NaN row, which is what allowPartial means on
    the live path too.
    """
    if source is not None:
        wanted = pd.DatetimeIndex([pd.Timestamp(runTime) for runTime in runTimes])
        block = source.reindex(index=wanted, columns=_stepColumns(horizon))
        return block.to_numpy(dtype=float)
    rows = []
    for runTime in runTimes:
        path = forecastSeries(runTime, region, horizon, allowPartial=allowPartial)
        rows.append(path.to_numpy(dtype=float))
    return np.asarray(rows)


def _stepColumns(horizon):
    return [str(step) for step in range(1, horizon + 1)]


def _monthStarts(runStart, runEnd):
    months = pd.date_range(pd.Timestamp(runStart).normalize().replace(day=1),
                           pd.Timestamp(runEnd), freq="MS")
    return [(month, month + pd.offsets.MonthEnd(1)) for month in months]


def _monthTable(nemseer, table, runFrom, runTo, horizon, columns):
    """One compile_data call for a whole month, subset before anything else sees it.

    P5MIN REGIONSOLUTION is 112 columns and about 56 MB a month, of which four are
    wanted, and compile_data has no column selection. Subsetting here is what keeps a
    month in memory rather than half a gigabyte.
    """
    # nemseer validates the forecasted window against the run's own resolution:
    # PREDISPATCH rejects any minute outside {0, 30}, P5MIN any outside multiples of 5.
    # P5MIN only ever reaches 55 minutes past its run, and nemseer refuses a wider ask.
    reach = columns["reach"] or horizon * nem_interval
    last = runTo + reach
    tables = nemseer.compile_data(
        _stamp(runFrom), _stamp(runTo), _stamp(runFrom + columns["step"]),
        _stamp(last), table, columns["table"], raw_cache=str(rawCache),
    )
    frame = next(iter(tables.values()))
    keep = [columns["run"], columns["interval"], "REGIONID", "RRP"]
    if "INTERVENTION" in frame.columns:
        frame = frame[frame["INTERVENTION"] == 0]
    frame = frame[keep].rename(columns={columns["run"]: "run",
                                        columns["interval"]: "interval"})
    return frame[frame["run"].dt.time == tradingDayStart]


predispatchColumns = {"table": "PRICE", "run": "PREDISPATCH_RUN_DATETIME",
                      "interval": "DATETIME", "step": predispatchInterval, "reach": None}
p5minColumns = {"table": "REGIONSOLUTION", "run": "RUN_DATETIME",
                "interval": "INTERVAL_DATETIME", "step": nem_interval,
                "reach": p5minIntervals * nem_interval}


def prefetchForecasts(runStart, runEnd, region, horizon=288, cacheDir=None):
    """Stitch every 04:00 forward path in the span into one parquet per region.

    `forecastSeries` makes two compile_data calls per run time, each reading a whole
    monthly parquet to extract one run. This makes two per month instead, and a year of
    paths lands in under 2 MB.
    """
    directory = Path(cacheDir) if cacheDir is not None else forecastCacheRoot
    directory.mkdir(parents=True, exist_ok=True)
    nemseer = _nemseer()
    paths = {}
    for runFrom, runTo in _monthStarts(runStart, runEnd):
        runFrom = max(runFrom + timedelta(hours=4), pd.Timestamp(runStart))
        runTo = min(runTo + timedelta(hours=4), pd.Timestamp(runEnd))
        if runFrom > runTo:
            continue
        coarse = _monthTable(nemseer, "PREDISPATCH", runFrom, runTo, horizon,
                             predispatchColumns)
        fine = _monthTable(nemseer, "P5MIN", runFrom, runTo, horizon, p5minColumns)
        coarse = coarse[coarse["REGIONID"] == region]
        fine = fine[fine["REGIONID"] == region]
        for runTime in sorted(set(coarse["run"]) | set(fine["run"])):
            merged = _stitch(_runSeries(coarse, runTime), _runSeries(fine, runTime))
            wanted = pd.DatetimeIndex(
                [runTime + (step + 1) * nem_interval for step in range(horizon)])
            paths[runTime] = merged.reindex(wanted).to_numpy(dtype=float)

    frame = pd.DataFrame.from_dict(paths, orient="index",
                                   columns=_stepColumns(horizon))
    # Add to the cache rather than replace it, so prefetching a second span does not
    # throw away the first one. A new run wins on any run time both hold.
    stored = loadForecastCache(region, directory)
    if stored is not None and list(stored.columns) == list(frame.columns):
        frame = pd.concat([stored[~stored.index.isin(frame.index)], frame])
    frame = frame.sort_index()
    target = directory / f"{region}.parquet"
    frame.to_parquet(target)
    return target


def _runSeries(frame, runTime):
    rows = frame[frame["run"] == runTime]
    return rows.set_index("interval")["RRP"].astype(float).sort_index()


def loadForecastCache(region, cacheDir=None):
    """The prefetched paths for a region, or None when nothing has been prefetched."""
    directory = Path(cacheDir) if cacheDir is not None else forecastCacheRoot
    target = directory / f"{region}.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)

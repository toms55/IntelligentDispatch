"""AEMO price forecasts, as published at a given run time.

Stitches two AEMO runs: P5MIN REGIONSOLUTION (5-minute, 11 intervals, 55 minutes ahead)
under PREDISPATCH PRICE (30-minute, to the end of the trading day). PDPASA and STPASA
carry no RRP column at all.

Selection is always by RUN TIME, never by forecasted time -- the latter would pick
whichever run was most accurate in hindsight.

A PREDISPATCH value is held flat across its six 5-minute intervals rather than
interpolated, since dispatch price is a step function.

Runs come straight from NEMWeb, from one of three places by age: the monthly MMSDM
archive once AEMO has published the month, Reports/Archive for about a year before
that, and Reports/Current for the last few days. All three are the same AEMO CSV, so a
historical run and a live one go through the same code.
"""
import csv
import io
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from utils import nem_interval, nem_interval_hours

rawCache = Path(__file__).resolve().parent.parent / "data_cache" / "forecast_raw"
forecastCacheRoot = Path(__file__).resolve().parent.parent / "data_cache" / "forecast_paths"
tradingDayStart = time(4, 0)

p5minIntervals = 11
predispatchInterval = timedelta(minutes=30)

nemweb = "https://nemweb.com.au/"
mmsdmUrl = (nemweb + "Data_Archive/Wholesale_Electricity/MMSDM/{year}/"
            "MMSDM_{year}_{month:02d}/MMSDM_Historical_Data_SQLLoader/")
# AEMO renamed the MMSDM files from August 2024 on. The CSV inside did not change.
archiveRenamed = (2024, 8)

sources = {
    "PREDISPATCH": {
        "monthFolder": "PREDISP_ALL_DATA",  # every run, not just the last per interval
        "oldName": "PUBLIC_DVD_PREDISPATCHPRICE_{stamp}",
        "newName": "PUBLIC_ARCHIVE#PREDISPATCHPRICE#ALL#FILE01#{stamp}",
        "table": ("PREDISPATCH", "REGION_PRICES"),
        "reports": "PredispatchIS_Reports", "prefix": "PUBLIC_PREDISPATCHIS_",
        "columns": ("PREDISPATCHSEQNO", "DATETIME"),
    },
    "P5MIN": {
        "monthFolder": "DATA",
        "oldName": "PUBLIC_DVD_P5MIN_REGIONSOLUTION_{stamp}",
        "newName": "PUBLIC_ARCHIVE#P5MIN_REGIONSOLUTION#FILE01#{stamp}",
        "table": ("P5MIN", "REGIONSOLUTION"),
        "reports": "P5_Reports", "prefix": "PUBLIC_P5MIN_",
        "columns": ("RUN_DATETIME", "INTERVAL_DATETIME"),
    },
}
rowColumns = ["run", "interval", "REGIONID", "RRP"]

# A PREDISPATCH run is filed under its trading day, so the run labelled 04:00 on the 1st,
# the previous trading day's last, sits in the previous month's file.
fileLag = timedelta(hours=4, minutes=30)


def nemNow():
    """The current NEM time: AEST, naive, no daylight saving."""
    return pd.Timestamp.now("UTC").tz_localize(None) + pd.Timedelta(hours=10)


def _download(url, target):
    """Stream `url` to `target`. False on a 404, which is AEMO not having published it."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=600) as response, \
                open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return False
        raise
    partial.replace(target)
    return True


def _listing(folder):
    """The zip file names in a NEMWeb directory page."""
    request = urllib.request.Request(nemweb + folder, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        page = response.read().decode("utf-8", "replace")
    return re.findall(r'HREF="[^"]*/([^/"]+\.zip)"', page, flags=re.IGNORECASE)


def _readTable(kind, handle, table):
    """Normalised rows of one table in a zipped AEMO CSV. Interventions are dropped."""
    table = list(table)
    keyColumn, intervalColumn = sources[kind]["columns"]
    wanted = [keyColumn, intervalColumn, "REGIONID", "RRP", "INTERVENTION"]
    rows, positions = [], None
    with zipfile.ZipFile(handle) as archive:
        with archive.open(archive.namelist()[0]) as raw:
            for row in csv.reader(io.TextIOWrapper(raw, "utf-8")):
                if row[1:3] != table:
                    continue
                if row[0] == "I":
                    positions = [row.index(column) for column in wanted]
                elif row[0] == "D":
                    rows.append([row[at] for at in positions])
    frame = pd.DataFrame(rows, columns=wanted)
    frame = frame[frame.INTERVENTION.astype(int) == 0]
    stamp = "%Y/%m/%d %H:%M:%S"
    if kind == "PREDISPATCH":
        sequence = frame[keyColumn]
        run = (pd.to_datetime(sequence.str[:8], format="%Y%m%d") + timedelta(hours=4)
               + pd.to_timedelta(sequence.str[8:].astype(int) * 30, unit="min"))
    else:
        run = pd.to_datetime(frame[keyColumn], format=stamp)
    return pd.DataFrame({"run": run.to_numpy(),
                         "interval": pd.to_datetime(frame[intervalColumn],
                                                    format=stamp).to_numpy(),
                         "REGIONID": frame.REGIONID.to_numpy(),
                         "RRP": frame.RRP.astype(float).to_numpy()})


def _monthRows(kind, year, month):
    """Every run in one MMSDM month, or None while AEMO has not published it."""
    cached = rawCache / f"{kind}_{year}{month:02d}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    source = sources[kind]
    name = source["newName" if (year, month) >= archiveRenamed else "oldName"].format(
        stamp=f"{year}{month:02d}010000")
    url = (mmsdmUrl.format(year=year, month=month) + source["monthFolder"] + "/"
           + urllib.parse.quote(name) + ".zip")
    zipped = rawCache / f"{kind}_{year}{month:02d}.zip"
    if not _download(url, zipped):
        return None
    frame = _readTable(kind, zipped, source["table"])
    # A published month with no rows is a format change, and caching it would hide it.
    if frame.empty:
        raise ValueError(f"{url} holds no {'.'.join(source['table'])} rows")
    frame.to_parquet(cached)
    zipped.unlink()
    return frame


def _reportRows(kind, runTime):
    """One run from Reports/Current, else Reports/Archive. None when neither holds it."""
    stamp = pd.Timestamp(runTime).strftime("%Y%m%d%H%M")
    cached = rawCache / f"{kind}_{stamp}.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    source = sources[kind]
    prefix = f"{source['prefix']}{stamp}_"
    frame = None
    current = f"Reports/Current/{source['reports']}/"
    for name in _listing(current):
        if name.startswith(prefix):
            target = rawCache / name
            if _download(nemweb + current + name, target):
                frame = _readTable(kind, target, source["table"])
                target.unlink()
            break
    else:
        frame = _archivedReport(kind, runTime, prefix)
    if frame is not None:
        frame.to_parquet(cached)
    return frame


def _archivedReport(kind, runTime, prefix):
    """The run from a Reports/Archive bundle: daily for P5MIN, weekly for PREDISPATCH.

    Bundles are named by publication date, which can be the day before the run label,
    so both days are searched. The bundle is kept on disk: one serves many runs.
    """
    source = sources[kind]
    folder = f"Reports/Archive/{source['reports']}/"
    days = {(pd.Timestamp(runTime) - delta).strftime("%Y%m%d")
            for delta in (timedelta(0), timedelta(days=1))}
    for name in _listing(folder):
        covered = re.findall(r"_(\d{8})(?=[_.])", name)
        if not covered or not any(covered[0] <= day <= covered[-1] for day in days):
            continue
        bundle = rawCache / "archive" / name
        if not bundle.exists() and not _download(nemweb + folder + name, bundle):
            continue
        with zipfile.ZipFile(bundle) as outer:
            for inner in outer.namelist():
                if inner.startswith(prefix):
                    with outer.open(inner) as handle:
                        return _readTable(kind, io.BytesIO(handle.read()),
                                          source["table"])
    return None


def runRows(kind, runTimes):
    """Rows (run, interval, REGIONID, RRP) for exactly these runs, every region.

    A run is looked for in its MMSDM month first and in the NEMWeb reports only when
    the month is unpublished or lacks it. A run found nowhere is simply absent.
    """
    wanted = pd.DatetimeIndex(sorted({pd.Timestamp(runTime) for runTime in runTimes}))
    months = sorted({(moment.year, moment.month)
                     for runTime in wanted for moment in (runTime, runTime - fileLag)})
    parts = []
    for year, month in months:
        frame = _monthRows(kind, year, month)
        if frame is not None:
            parts.append(frame[frame.run.isin(wanted)])
    found = set(pd.concat(parts).run) if parts else set()
    for runTime in wanted:
        if runTime not in found:
            frame = _reportRows(kind, runTime)
            if frame is not None:
                parts.append(frame[frame.run == runTime])
    parts = [part for part in parts if len(part)]
    if not parts:
        return pd.DataFrame(columns=rowColumns)
    return pd.concat(parts, ignore_index=True)


def _series(rows, region, runTime, last):
    """One run's RRP for one region, by interval end, over (runTime, last]."""
    rows = rows[(rows.REGIONID == region) & (rows.run == runTime)
                & (rows.interval > runTime) & (rows.interval <= last)]
    return rows.set_index("interval")["RRP"].astype(float).sort_index()


def p5minForecast(runTime, region="SA1"):
    """RRP by interval end, as the P5MIN run at `runTime` saw it. 11 intervals."""
    runTime = pd.Timestamp(runTime)
    return _series(runRows("P5MIN", [runTime]), region, runTime,
                   runTime + p5minIntervals * nem_interval)


def predispatchForecast(runTime, region="SA1", hours=24):
    """RRP by 30-minute interval end, as the PREDISPATCH run at or before `runTime` saw it.

    `hours` cannot exceed what the run reaches; that varies with time of day since
    PREDISPATCH always stops at the end of a trading day. Episodes align to 04:00
    because that is where 24h is available.
    """
    run = pd.Timestamp(runTime).floor("30min")
    return _series(runRows("PREDISPATCH", [run]), region, run,
                   run + timedelta(hours=hours))


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

    coarse = predispatchForecast(runTime, region, hours=horizon * nem_interval_hours)
    fine = p5minForecast(runTime, region)
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


def _requireDivides(horizon, every):
    if every <= 0 or horizon % every:
        raise ValueError(
            f"a forecast refresh every {every} intervals does not divide a "
            f"{horizon}-interval episode"
        )


def refreshRunTimes(dayStart, horizon=config.episode_intervals,
                    every=config.forecast_refresh_intervals):
    """The runs an episode opening at `dayStart` reads, one every `every` intervals."""
    _requireDivides(horizon, every)
    dayStart = pd.Timestamp(dayStart)
    return [dayStart + offset * nem_interval for offset in range(0, horizon, every)]


def refreshMatrix(dayStarts, region="SA1", horizon=config.episode_intervals,
                  every=config.forecast_refresh_intervals, allowPartial=True, source=None):
    """Every refresh run of every episode, laid on the episode's own step grid.

    Shape (days, refreshes, horizon). Row r is the run published r * every intervals in,
    placed so column k is the episode's step k. Columns before the run are NaN, because
    it had not been published yet, and the run is cut at the episode end, which is the
    end of the trading day where PREDISPATCH stops anyway. every == horizon is one 04:00
    run per day.
    """
    _requireDivides(horizon, every)
    starts = [pd.Timestamp(day) for day in dayStarts]
    grid = np.full((len(starts), horizon // every, horizon), np.nan)
    if not starts:
        return grid
    for refresh, offset in enumerate(range(0, horizon, every)):
        runs = [day + offset * nem_interval for day in starts]
        grid[:, refresh, offset:] = forecastMatrix(
            runs, region, horizon - offset, allowPartial=allowPartial, source=source)
    return grid


def activeRun(forward, step):
    """The path the agent reads at `step`: the latest refresh published by then.

    `forward` is one episode's forecast, either a single path or a refreshMatrix stack.
    """
    forward = np.asarray(forward, dtype=float)
    if forward.ndim == 1:
        return forward
    every = forward.shape[1] // forward.shape[0]
    return forward[min(step // every, forward.shape[0] - 1)]


def decisionPath(forward):
    """The forecast price of each step as the agent read it when deciding that step."""
    forward = np.asarray(forward, dtype=float)
    if forward.ndim == 1:
        return forward
    return np.array([activeRun(forward, step)[step] for step in range(forward.shape[1])])


def liveForecast(region="SA1", horizon=288):
    """The forward path from the latest runs right now. NaN past the trading day's end,
    where PREDISPATCH stops."""
    return forecastSeries(nemNow().floor("5min"), region, horizon, allowPartial=True)


def prefetchForecasts(runStart, runEnd, region, horizon=288, cacheDir=None,
                      every=config.forecast_refresh_intervals):
    """Stitch every refresh run of every trading day in the span into one parquet per
    region: the 04:00 run that opens the day, then one every `every` intervals.

    Runs are fetched a month at a time, so a month of paths costs two archive reads
    rather than two per run. A run AEMO never published is left out. Each path is
    stored from its own run time, uncut; refreshMatrix cuts it at the episode end.
    """
    directory = Path(cacheDir) if cacheDir is not None else forecastCacheRoot
    directory.mkdir(parents=True, exist_ok=True)
    runStart, runEnd = pd.Timestamp(runStart), pd.Timestamp(runEnd)
    days = pd.date_range(runStart.normalize() + timedelta(hours=4), runEnd, freq="D")
    days = days[days >= runStart]
    runTimes = pd.DatetimeIndex(
        [run for day in days for run in refreshRunTimes(day, horizon, every)])
    paths = {}
    for _, chunk in pd.Series(runTimes, index=runTimes).groupby(
            [runTimes.year, runTimes.month]):
        # PREDISPATCH runs only on the half-hour; floor as predispatchForecast does, or a
        # 5-minute refresh misses every run and falls through to the NEMWeb reports.
        coarse = runRows("PREDISPATCH", chunk.index.floor("30min"))
        fine = runRows("P5MIN", chunk)
        coarse, fine = (rows[rows.REGIONID == region] for rows in (coarse, fine))
        for runTime in chunk:
            merged = _stitch(
                _series(coarse, region, runTime.floor("30min"),
                        runTime + horizon * nem_interval),
                _series(fine, region, runTime, runTime + p5minIntervals * nem_interval))
            if len(merged):
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


def loadForecastCache(region, cacheDir=None):
    """The prefetched paths for a region, or None when nothing has been prefetched."""
    directory = Path(cacheDir) if cacheDir is not None else forecastCacheRoot
    target = directory / f"{region}.parquet"
    if not target.exists():
        return None
    return pd.read_parquet(target)

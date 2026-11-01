"""Reading a rolling backtest off disk, and narrowing it to a scope.

Pure functions over the frame `backtest.records` writes. Nothing here imports Dash, so
the whole layer is testable without a server. Every ratio still comes from
`backtest.metrics` -- this module only selects rows.
"""
import functools
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from backtest import records
from backtest.actions import exportFile, exportName

resultsRoot = Path("results")
# The runs the dashboard draws, one CSV per run and region, <run>_<region>.csv, beside
# results/actions/ and named as its exports are. Git tracks it; `backtest.publish` is
# the only code that writes it. A run is named by its path without the region,
# results/backtestResults/<run>, though no such directory exists.
backtestRoot = resultsRoot / "backtestResults"
weekDays = 7

# A dashboard entry is one trained model scored one way.
modelFields = ("algo", "trainMode", "seed", "priceSource")

# Every episode starts at 04:00, the NEM trading day boundary, so a calendar date must
# be offset by this before it matches a testDay stamp.
dayStartHour = 4
interval = timedelta(minutes=5)


class Run(NamedTuple):
    region: str
    days: pd.DataFrame
    # How often a test day pulled a new AEMO forecast; None where the run never says.
    forecastEvery: object
    resultsDir: Path


def publishedFile(resultsDir, region):
    """The CSV of one region of the published run `resultsDir`."""
    resultsDir = Path(resultsDir)
    return resultsDir.parent / exportName(resultsDir.name, region)


def loadRun(region, resultsDir):
    """One region's finished rows and the forecast pull they were produced under."""
    resultsDir = Path(resultsDir)
    path = publishedFile(resultsDir, region)
    if not path.exists():
        raise FileNotFoundError(
            f"no published backtest for {region} at {path}; run "
            f"`cli rolling --region {region} ...` and `python -m backtest.publish` first"
        )
    days = records.readDayFile(path)
    every = days.pop("forecastEvery") if "forecastEvery" in days else None
    every = None if every is None or every.empty else int(every.iloc[0])
    days = records.sortDays(days, path)
    if "forecast" in set(days.priceSource):
        verifyRefresh(resultsDir, region, every)
    return Run(region, days, every, resultsDir)


def verifyRefresh(resultsDir, region, every):
    """Refuse a forecast-priced run unless its test-day forecast pull is stated and held.

    `forecastEvery` is how often a test day pulled a new AEMO forecast, the only refresh
    the backtest screen describes; training reads realised prices and has none. The
    published CSV must state it, and every forecast decision in the actions export must
    have read the run that schedule implies: day start + (step // every * every)
    intervals. Returns the verified `every`.
    """
    where = publishedFile(resultsDir, region)
    if every is None:
        raise ValueError(
            f"{where} does not state forecastEvery, so the backtest's forecast pull "
            "frequency is unknown. A result set from before the refresh read one 04:00 "
            "run per day: stamp its manifest with `python -m backtest.stamp` and "
            "publish it again")
    path = actionsFile(resultsDir, region)
    if path is None:
        raise FileNotFoundError(
            f"{resultsDir} has forecast rows for {region} but no actions export, so its "
            "forecast pull frequency cannot be checked; run `python -m backtest.actions`")
    rows = _readActions(str(path), path.stat().st_mtime)
    rows = rows[rows.priceSource == "forecast"]
    opened = rows.intervalTime - interval
    day = (opened - timedelta(hours=dayStartHour)).dt.floor("D") + timedelta(hours=dayStartHour)
    step = (opened - day) // interval
    expected = day + (step // every * every) * interval
    wrong = rows.forecastRunTime != expected
    if wrong.any():
        first = rows[wrong].iloc[0]
        raise ValueError(
            f"{path} disagrees with forecastEvery={every} in {where} on {int(wrong.sum())} "
            f"decision(s); the first, at {first.intervalTime}, read the "
            f"{first.forecastRunTime} run, not {expected[wrong].iloc[0]}")
    return every


def selectors(frame):
    """The control values the data actually holds, not the ones config.py declares."""
    return {column: sorted(frame[column].unique().tolist())
            for column in ("algo", "trainMode", "priceSource", "seed")}


def scope(frame, day=None, week=None):
    """The rows for one day, one week from `week`, or the whole job."""
    if day is not None and week is not None:
        raise ValueError("scope takes one of day or week, not both")
    if day is not None:
        return frame[frame.testDay == pd.Timestamp(day)]
    if week is not None:
        start = pd.Timestamp(week)
        return frame[(frame.testDay >= start)
                     & (frame.testDay < start + timedelta(days=weekDays))]
    return frame


# What a dispatch chart reads: one row per interval, in the layout the forecast screen
# also returns, so `live.blocks` turns either one into the same blocks.
dispatchColumns = ("settlementdate", "price", "action", "soc")


def decisions(run, day):
    """Every agent decision of `day` in the run's actions export, any model or source."""
    path = actionsFile(run.resultsDir, run.region)
    if path is None:
        return pd.DataFrame(columns=list(actionColumns))
    rows = _readActions(str(path), path.stat().st_mtime)
    day = pd.Timestamp(day)
    return rows[(rows.intervalTime > day) & (rows.intervalTime <= day + timedelta(days=1))]


def dispatchPath(rows, algo=None, trainMode=None, seed=None, priceSource=None):
    """One model's intervals for a day: the realised price, and the action it took.

    `rows` are actions-export rows (`decisions`). `price` is the realised price the
    agent was paid, whichever price source it observed.

    The seed and the price source are part of the match. An export from before every
    seed was logged holds seed 0 only, so a seed that is not in it has to come back
    empty rather than draw another seed's dispatch under its own name.
    """
    for column, wanted in (("algo", algo), ("trainMode", trainMode), ("seed", seed),
                           ("priceSource", priceSource)):
        if wanted is not None and not rows.empty:
            rows = rows[rows[column] == wanted]
    if rows.empty:
        return pd.DataFrame(columns=list(dispatchColumns))
    rows = rows.sort_values("intervalTime")
    return pd.DataFrame({
        "settlementdate": pd.to_datetime(rows.intervalTime).to_numpy(),
        "price": rows.realisedPrice.to_numpy(dtype=float),
        "action": rows.action.to_numpy(),
        "soc": rows.soc.to_numpy(dtype=float),
    }, columns=list(dispatchColumns))


def availableRegions(resultsDir):
    """Every region published for the run `resultsDir`."""
    resultsDir = Path(resultsDir)
    prefix = f"{resultsDir.name}_"
    return sorted(path.stem[len(prefix):] for path in resultsDir.parent.glob(f"{prefix}*.csv")
                  # A longer run name that starts with this one is another run.
                  if "_" not in path.stem[len(prefix):])


def actionsPath(resultsDir, region):
    """Where the actions export of the run `resultsDir` lives, written or not.

    `backtest.publish` names it after the run, as it names the run's own CSV, so the
    export of results/backtestResults/<run> is results/actions/<run>_<region>.parquet,
    and pointing the dashboard at another run changes the forecast numbers with every
    other number. The live CSV beside it is a backtest still writing, and is not read.
    """
    resultsDir = Path(resultsDir)
    return resultsDir.parent.parent / "actions" / exportFile(resultsDir.name, region)


def actionsFile(resultsDir, region):
    """The actions export of the run `resultsDir`, or None when it was not exported."""
    path = actionsPath(resultsDir, region)
    return path if path.exists() else None


actionColumns = ("forecastRunTime", "intervalTime", "action", "soc", "forecastPrice",
                 "realisedPrice", "seed", "algo", "trainMode", "priceSource")


@functools.lru_cache(maxsize=4)
def _readActions(path, modified):
    # `modified` is only in the key, so a re-export is read again rather than served
    # from the cache.
    return pd.read_parquet(path, columns=list(actionColumns))


forecastColumns = ("settlementdate", "forecastPrice", "forecastRunTime")


def forecastPath(resultsDir, region, day, algo=None, trainMode=None, seed=None):
    """The forecast one model acted on over `day`, and the AEMO run each price came from.

    None when the run has no actions export, which is not the same as an empty frame:
    an export without this model or seed holds no forecast for it, and must not lend
    another seed's.
    """
    path = actionsFile(resultsDir, region)
    if path is None:
        return None
    rows = _readActions(str(path), path.stat().st_mtime)
    rows = rows[rows.priceSource == "forecast"]
    for column, wanted in (("algo", algo), ("trainMode", trainMode), ("seed", seed)):
        if wanted is not None:
            rows = rows[rows[column] == wanted]
    day = pd.Timestamp(day)
    rows = rows[(rows.intervalTime > day) & (rows.intervalTime <= day + timedelta(days=1))]
    rows = rows.sort_values("intervalTime")
    return pd.DataFrame({"settlementdate": rows.intervalTime.to_numpy(),
                         "forecastPrice": rows.forecastPrice.to_numpy(dtype=float),
                         "forecastRunTime": rows.forecastRunTime.to_numpy()},
                        columns=list(forecastColumns))


def findResults(given=None, root=backtestRoot):
    """The run to read, when the command line does not name one.

    The dashboard draws the runs published under `backtestRoot`, so it looks there for
    the runs its CSVs belong to. One candidate is used. Several candidates raise,
    because picking one of two experiments for the reader would show numbers from a
    run they did not ask for.
    """
    if given is not None:
        return Path(given)
    root = Path(root)
    found = sorted({root / path.stem.rsplit("_", 1)[0] for path in root.glob("*_*.csv")}
                   if root.is_dir() else [])
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(
            f"no published backtest under {root}; run `cli rolling` and then "
            "`python -m backtest.publish`, or pass --results")
    names = ", ".join(str(item) for item in found)
    raise FileNotFoundError(f"several runs under {root}: {names}. Pass --results to "
                            "name one")


def modelLabel(algo, trainMode, seed, priceSource):
    return f"{algo}-{trainMode}-s{int(seed)}-{priceSource}"


def models(frame):
    """One entry per trained model scored one way.

    Seed and price source belong in the name because each is a separate run on disk,
    not a separate view of one run.
    """
    unique = (frame[list(modelFields)].drop_duplicates()
              .sort_values(list(modelFields)))
    return [modelLabel(*row) for row in unique.to_numpy().tolist()]


def modelRows(frame, label):
    """The rows of one model entry, found by the label `models` built."""
    if frame.empty:
        return frame
    keys = frame[list(modelFields)].to_numpy().tolist()
    return frame[[modelLabel(*key) == label for key in keys]]


def monthScope(frame, month):
    """The rows whose test day falls in the calendar month holding `month`."""
    if frame.empty:
        return frame
    return frame[frame.testDay.dt.to_period("M") == pd.Period(pd.Timestamp(month), "M")]

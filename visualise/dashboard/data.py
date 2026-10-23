"""Reading a rolling backtest off disk, and narrowing it to a scope.

Pure functions over the frame `backtest.records` writes. Nothing here imports Dash, so
the whole layer is testable without a server. Every ratio still comes from
`backtest.metrics` -- this module only selects rows.
"""
import functools
import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from backtest import records
from backtest.actions import exportName

resultsRoot = Path("results")
# Where `cli rolling` writes when it is not told otherwise. A run of its own name sits
# beside it, so `findResults` looks for one when this is not there.
defaultResults = resultsRoot / "rolling"
weekDays = 7

# A dashboard entry is one trained model scored one way.
modelFields = ("algo", "trainMode", "seed", "priceSource")

# Every episode starts at 04:00, the NEM trading day boundary, so a calendar date must
# be offset by this before it matches a testDay stamp.
dayStartHour = 4


class Run(NamedTuple):
    region: str
    days: pd.DataFrame
    manifest: dict
    resultsDir: Path


def loadRun(region, resultsDir=defaultResults):
    """One region's finished rows and the configuration that produced them."""
    resultsDir = Path(resultsDir)
    days = records.readDays(resultsDir, region)
    if days.empty:
        raise FileNotFoundError(
            f"no rolling results for {region} in {resultsDir}; run "
            f"`cli rolling --region {region} ...` first"
        )
    path = records.manifestPath(resultsDir, region)
    manifest = json.loads(path.read_text()) if path.exists() else {}
    return Run(region, days, manifest, resultsDir)


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


def foldOfDay(frame, day):
    """Which fold tested `day`, so the day view knows which trajectory file to open."""
    rows = frame[frame.testDay == pd.Timestamp(day)]
    return None if rows.empty else int(rows.fold.iloc[0])


def trajectories(run, day):
    """Every per-interval series written for `day`, across train modes and price sources."""
    fold = foldOfDay(run.days, day)
    if fold is None:
        return pd.DataFrame(columns=list(records.trajectoryColumns))
    path = records.regionDir(run.resultsDir, run.region) / f"fold{fold:02d}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=list(records.trajectoryColumns))
    frame = pd.read_parquet(path)
    return frame[pd.to_datetime(frame.testDay) == pd.Timestamp(day)]



# What a dispatch chart reads: one row per interval, in the layout the forecast screen
# also returns, so `live.blocks` turns either one into the same blocks.
dispatchColumns = ("settlementdate", "price", "action", "soc")


def dispatchPath(trajectory, algo=None, trainMode=None, seed=None, priceSource=None):
    """One model's intervals for a day: the realised price, and the action it took.

    The action comes from `bought` and `sold`, the same two numbers the forecast screen
    reads, so both screens draw a dispatch block the same way. `price` is the realised
    price the agent was paid, whichever price source it observed.

    The seed is part of the match. A fold scores every seed but writes the per-interval
    trajectory of one of them (`backtest.runner`'s `trajectorySeed`), so a seed whose
    trajectory is not on disk has to come back empty rather than draw another seed's
    dispatch under its own name.
    """
    rows = trajectory
    for column, wanted in (("algo", algo), ("trainMode", trainMode), ("seed", seed),
                           ("priceSource", priceSource)):
        if wanted is not None and not rows.empty:
            rows = rows[rows[column] == wanted]
    if rows.empty:
        return pd.DataFrame(columns=list(dispatchColumns))
    rows = rows.sort_values("interval")
    bought = rows.bought.to_numpy(dtype=float)
    sold = rows.sold.to_numpy(dtype=float)
    return pd.DataFrame({
        "settlementdate": pd.to_datetime(rows.settlementdate).to_numpy(),
        "price": rows.price.to_numpy(dtype=float),
        "action": np.select([bought > 0, sold > 0], ["charge", "discharge"], "idle"),
        "soc": rows.soc.to_numpy(dtype=float),
    }, columns=list(dispatchColumns))


def availableRegions(resultsDir=defaultResults):
    """Every region with at least one finished fold under `resultsDir`."""
    root = Path(resultsDir)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir()
                  if child.is_dir() and any(child.glob("fold*.csv")))


def actionsFile(resultsDir, region):
    """The actions export of the run in `resultsDir`, or None when it was not exported.

    `backtest.actions` names its CSV after the run, and a merged run directory is the
    run's name with `_all` appended, so the file follows `--results`: pointing the
    dashboard at another run changes the forecast numbers with every other number. The
    Space carries it gzipped, which pandas reads the same way.
    """
    resultsDir = Path(resultsDir)
    name = exportName(resultsDir.name.removesuffix("_all"), region)
    for path in (resultsDir.parent / "actions" / name,
                 resultsDir.parent / "actions" / f"{name}.gz"):
        if path.exists():
            return path
    return None


@functools.lru_cache(maxsize=4)
def _readActions(path, modified):
    # `modified` is only in the key, so a re-export is read again rather than served
    # from the cache.
    return pd.read_csv(path, parse_dates=["forecastRunTime", "intervalTime"],
                       usecols=["forecastRunTime", "intervalTime", "forecastPrice",
                                "seed", "algo", "trainMode"])


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


def findResults(given=None, root=resultsRoot):
    """The records directory to read, when the command line does not name one.

    `cli rolling --results` names each experiment, so a machine holds its runs under
    names of their own (`results/rolling2023_profit_all`) and the fixed default misses.
    The dashboard therefore looks under `root` for a directory that holds finished
    folds. One candidate is used. Several candidates raise, because picking one of two
    experiments for the reader would show numbers from a run they did not ask for.
    """
    if given is not None:
        return Path(given)
    if availableRegions(defaultResults):
        return Path(defaultResults)
    root = Path(root)
    found = sorted(child for child in root.iterdir()
                   if availableRegions(child)) if root.is_dir() else []
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(
            f"no rolling results under {root}; run `cli rolling` first, or pass "
            "--results")
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

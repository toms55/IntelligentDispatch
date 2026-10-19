"""Reading a rolling backtest off disk, and narrowing it to a scope.

Pure functions over the frame `backtest.records` writes. Nothing here imports Dash, so
the whole layer is testable without a server. Every ratio still comes from
`backtest.metrics` -- this module only selects rows.
"""
import json
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from backtest import records

defaultResults = Path("results/rolling")
weekDays = 7


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
            for column in ("algo", "arm", "priceSource", "seed")}


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
    """Every per-interval series written for `day`, across arms and price sources."""
    fold = foldOfDay(run.days, day)
    if fold is None:
        return pd.DataFrame(columns=list(records.trajectoryColumns))
    path = records.regionDir(run.resultsDir, run.region) / f"fold{fold:02d}.parquet"
    if not path.exists():
        return pd.DataFrame(columns=list(records.trajectoryColumns))
    frame = pd.read_parquet(path)
    return frame[pd.to_datetime(frame.testDay) == pd.Timestamp(day)]


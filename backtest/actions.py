"""The agent's dispatch, one CSV row per decision, under both price sources.

A backtest writes its own CSV as it runs: `cli rolling` appends each fold's rows, every
seed's, once the fold is scored (see `appendRows`). The export below is for result sets
from before that, which hold the trajectories but no CSV.

An episode reads the 04:00 run that opens its trading day, then a newer run every
`forecastEvery` intervals (see data/forecasts.refreshMatrix). The backtest records the
run each decision read in the trajectory, and the export copies it; a trajectory from
before that has its run derived from the manifest instead. A result set whose manifest predates the refresh read one 04:00
run all day. The fold trajectories in results/<run>/all/ already hold everything needed;
this only reshapes them.

    uv run python -m backtest.actions results/rolling2023 [more runs...]

writes one CSV per run and region into results/actions/, named after the run, so every
backtest's dispatch sits in one folder: results/actions/rolling2023_NSW1.csv,
results/actions/rolling2023_profit_fc4h_SA1.csv. A backtest that writes its CSV as it
runs needs none of this.

`priceSource` says what the agent saw: "forecast" is the AEMO forecast, "realised" the
price it was paid. The label is copied from the row the runner wrote, never inferred,
and a realised row is refused unless its observed price is its paid price on every
interval. A realised row has no forecastRunTime and no forecastPrice.
"""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from rl.env import CHARGE, DISCHARGE
from utils import nem_interval

columns = (
    "forecastRunTime", "intervalTime", "action", "soc",
    "forecastPrice", "realisedPrice", "region", "fold", "seed", "algo", "trainMode",
    "priceSource", "attemptedAction", "outOfBounds",
)

# What `action` says when nothing moved.
IDLE_ACTION = "idle"

priceSources = ("forecast", "realised")


def actionRows(trajectory, region, every=None):
    """The agent rows of one fold's trajectory, both price sources, in the export layout.

    intervalTime is the interval's end, as AEMO stamps it, and soc is the state of
    charge at that end, after the action. `every` is the refresh interval the run used;
    None is one 04:00 run per day.
    """
    rows = trajectory[trajectory.algo != "oracle"]
    unknown = set(rows.priceSource) - set(priceSources)
    if unknown:
        raise ValueError(f"unknown priceSource {sorted(unknown)}; expected {priceSources}")
    forecast = rows.priceSource == "forecast"
    # A realised agent observed what it was paid. One that did not is mislabelled.
    wrong = ~forecast & (rows.observedPrice != rows.price)
    if wrong.any():
        first = rows[wrong].iloc[0]
        raise ValueError(
            f"{int(wrong.sum())} realised row(s) observed a price other than the one "
            f"paid, so they are not realised; the first is fold {first.fold}, seed "
            f"{first.seed}, {first.settlementdate}: saw {first.observedPrice}, paid "
            f"{first.price}")
    action = np.select([rows.bought > 0, rows.sold > 0], ["charge", "discharge"],
                       IDLE_ACTION)
    # What the agent chose, which is not always what moved: charging a full pack moves
    # nothing and so reads as idle above. A trajectory from before the backtest recorded
    # the choice carries no such column, and its rows say nothing rather than guessing.
    if "attemptedAction" in rows:
        attempted = pd.Series(np.select(
            [rows.attemptedAction == CHARGE, rows.attemptedAction == DISCHARGE],
            ["charge", "discharge"], IDLE_ACTION), index=rows.index)
        attempted = attempted.where(rows.attemptedAction.notna())
    else:
        attempted = pd.Series(None, index=rows.index, dtype=object)
    # The env's own flag, copied. Never re-derived from `action`: the mask is a tolerance
    # on the headroom, so an out-of-bounds step can still move an atom of energy and read
    # as a move here. backtest/outofbounds.py selects on this column.
    flagged = (rows.outOfBounds if "outOfBounds" in rows
               else pd.Series(None, index=rows.index, dtype=object))
    if "forecastRunTime" in rows and rows.forecastRunTime[forecast].notna().all():
        # Recorded by the backtest as the run the agent actually read.
        runTime = pd.to_datetime(rows.forecastRunTime)
    else:
        # A trajectory from before the backtest recorded it: derived from the manifest.
        runTime = pd.to_datetime(rows.testDay)
        if every is not None:
            runTime = runTime + (rows.interval // every * every) * nem_interval
    return pd.DataFrame({
        "forecastRunTime": runTime.where(forecast),
        "intervalTime": rows.settlementdate,
        "action": action,
        "soc": rows.soc,
        "forecastPrice": rows.observedPrice.where(forecast),
        "realisedPrice": rows.price,
        "region": region,
        "fold": rows.fold,
        "seed": rows.seed,
        "algo": rows.algo,
        "trainMode": rows.trainMode,
        "priceSource": rows.priceSource,
        "attemptedAction": attempted,
        "outOfBounds": flagged,
    }, columns=columns)


def exportName(runDir, region):
    """The CSV's name in the actions folder: the run's name, with the region appended
    unless the run is already named for it."""
    name = Path(runDir).name
    return f"{name}.csv" if name.endswith(f"_{region}") else f"{name}_{region}.csv"


def _runDir(resultsDir):
    """The run a results directory belongs to. The folds job gives each seed its own
    results/<run>/seed<N>, and every seed of a region shares the run's one CSV."""
    resultsDir = Path(resultsDir)
    return resultsDir.parent if re.fullmatch(r"seed\d+", resultsDir.name) else resultsDir


def liveFile(resultsDir, region):
    """The CSV a running backtest appends to: the one `exportRun` would write for it."""
    runDir = _runDir(resultsDir)
    return runDir.parent / "actions" / exportName(runDir, region)


def _requireSameColumns(handle, path):
    """Refuse to append to a CSV whose header is not this layout.

    A --resume run against a file written before a column was added would otherwise put
    a wider block under a narrower header, and every row of it would parse into the
    wrong fields. The header is only written once, by whichever fold arrives first.

    Reads through the caller's own locked handle, and never opens the file again. A
    POSIX record lock is dropped as soon as the process closes any descriptor for the
    file, whichever descriptor took the lock, so a second open() here would release the
    lock and let two seeds interleave one fold's rows.
    """
    handle.seek(0)
    header = handle.readline().strip().split(",")
    if header != list(columns):
        missing = [name for name in columns if name not in header]
        extra = [name for name in header if name not in columns]
        # Name the real difference. Equality is on the order too, so a header holding
        # every column in another order has nothing missing and nothing extra, and
        # reporting only the missing names would say "missing []".
        reason = (f"it is missing {missing}" if missing else
                  f"it holds unknown columns {extra}" if extra else
                  f"its columns are in another order: {header}")
        raise ValueError(
            f"{path} was written under a different layout and cannot be appended to: "
            f"{reason}. Re-run under a new name, or export the run again with "
            f"`python -m backtest.actions`."
        )


def appendRows(path, frame):
    """Append one fold's rows to the CSV at `path`, the header first if it is new.

    Every seed of a region runs as its own process and appends to the same file, so the
    block is written under an exclusive lock, whole, and flushed before the lock goes:
    one fold's rows never interleave with another's. The rows land in the order folds
    finish, not sorted; readers sort.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "a+" rather than "a": the header check reads through this same handle, because a
    # second open() would drop the lock. Appending still writes at the end whatever the
    # read left the file position at, which is what O_APPEND guarantees.
    with open(path, "a+", newline="") as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX)
        try:
            fresh = os.fstat(handle.fileno()).st_size == 0
            if not fresh:
                _requireSameColumns(handle, path)
            frame.to_csv(handle, header=fresh,
                         index=False, columns=list(columns))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.lockf(handle, fcntl.LOCK_UN)


unit = ("fold", "seed", "algo", "trainMode", "priceSource")


def writtenUnits(path):
    """The (fold, seed, algo, trainMode, priceSource) blocks already in the CSV, so a resumed run
    appends only the ones it is missing. A fold is appended before its rows reach the
    fold CSV, so one killed in between is scored again but not written twice."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=list(unit)).drop_duplicates()
    return set(frame.itertuples(index=False, name=None))


def refuseExisting(resultsDir, region, backtestRoot=Path("results/backtestResults")):
    """Stop a fresh backtest whose name is taken, by its actions CSV or by a published
    region. Redo a backtest under a new name; continue one with --resume."""
    published = Path(backtestRoot) / exportName(runName(resultsDir, region), region)
    taken = [path for path in (liveFile(resultsDir, region), published) if path.exists()]
    if taken:
        raise ValueError(
            f"a backtest with this name already exists for {region}: "
            + ", ".join(map(str, taken))
            + "; pick a new --results name, or pass --resume to continue it")


def runName(resultsDir, region):
    """The run's name in results/backtestResults/, as backtest.publish.runName gives it. Not
    imported from there, because publish imports the dashboard, which imports this."""
    return _runDir(resultsDir).name.removesuffix("_all").removesuffix(f"_{region}")


def exportRun(runDir, outDir=None):
    """One CSV per region under runDir/all/, written into outDir (by default the
    actions folder beside runDir). Returns the paths written."""
    outDir = Path(runDir).parent / "actions" if outDir is None else Path(outDir)
    outDir.mkdir(parents=True, exist_ok=True)
    written = []
    for regionDir in sorted(Path(runDir, "all").iterdir()):
        manifest = regionDir / "manifest.json"
        every = (json.loads(manifest.read_text()).get("forecastEvery")
                 if manifest.exists() else None)
        # Result sets from before the rename call trainMode "arm".
        parts = [actionRows(pd.read_parquet(path).rename(columns={"arm": "trainMode"}),
                            regionDir.name, every)
                 for path in sorted(regionDir.glob("fold*.parquet"))]
        if not parts:
            continue
        frame = (pd.concat(parts, ignore_index=True)
                 .sort_values(["priceSource", "seed", "algo", "trainMode",
                               "intervalTime"]))
        out = outDir / exportName(runDir, regionDir.name)
        frame.to_csv(out, index=False)
        written.append(out)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="results/<run> directories")
    parser.add_argument("--out", help="folder for the CSVs; default <run>/../actions")
    args = parser.parse_args(argv)
    for runDir in args.runs:
        for path in exportRun(runDir, args.out):
            print(path)


if __name__ == "__main__":
    main()

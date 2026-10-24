"""The agent's dispatch under forecast prices, one CSV row per decision.

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
"""
import argparse
import fcntl
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils import nem_interval

columns = (
    "forecastRunTime", "intervalTime", "action", "soc",
    "forecastPrice", "realisedPrice", "region", "fold", "seed", "algo", "trainMode",
)


def actionRows(trajectory, region, every=None):
    """Forecast-priced agent rows of one fold's trajectory, in the export layout.

    intervalTime is the interval's end, as AEMO stamps it, and soc is the state of
    charge at that end, after the action. `every` is the refresh interval the run used;
    None is one 04:00 run per day.
    """
    rows = trajectory[(trajectory.priceSource == "forecast")
                      & (trajectory.algo != "oracle")]
    action = np.select([rows.bought > 0, rows.sold > 0], ["charge", "discharge"], "idle")
    if "forecastRunTime" in rows and rows.forecastRunTime.notna().all():
        # Recorded by the backtest as the run the agent actually read.
        runTime = pd.to_datetime(rows.forecastRunTime)
    else:
        # A trajectory from before the backtest recorded it: derived from the manifest.
        runTime = pd.to_datetime(rows.testDay)
        if every is not None:
            runTime = runTime + (rows.interval // every * every) * nem_interval
    return pd.DataFrame({
        "forecastRunTime": runTime,
        "intervalTime": rows.settlementdate,
        "action": action,
        "soc": rows.soc,
        "forecastPrice": rows.observedPrice,
        "realisedPrice": rows.price,
        "region": region,
        "fold": rows.fold,
        "seed": rows.seed,
        "algo": rows.algo,
        "trainMode": rows.trainMode,
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


def appendRows(path, frame):
    """Append one fold's rows to the CSV at `path`, the header first if it is new.

    Every seed of a region runs as its own process and appends to the same file, so the
    block is written under an exclusive lock, whole, and flushed before the lock goes:
    one fold's rows never interleave with another's. The rows land in the order folds
    finish, not sorted; readers sort.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX)
        try:
            frame.to_csv(handle, header=os.fstat(handle.fileno()).st_size == 0,
                         index=False, columns=list(columns))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.lockf(handle, fcntl.LOCK_UN)


unit = ("fold", "seed", "algo", "trainMode")


def writtenUnits(path):
    """The (fold, seed, algo, trainMode) blocks already in the CSV, so a resumed run
    appends only the ones it is missing. A fold is appended before its rows reach the
    fold CSV, so one killed in between is scored again but not written twice."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, usecols=list(unit)).drop_duplicates()
    return set(frame.itertuples(index=False, name=None))


def refuseExisting(resultsDir, region, backtestRoot=Path("backtestResults")):
    """Stop a fresh backtest whose name is taken, by its actions CSV or by a published
    region. Redo a backtest under a new name; continue one with --resume."""
    taken = [path for path in (liveFile(resultsDir, region),
                               Path(backtestRoot) / runName(resultsDir, region) / region)
             if path.exists()]
    if taken:
        raise ValueError(
            f"a backtest with this name already exists for {region}: "
            + ", ".join(map(str, taken))
            + "; pick a new --results name, or pass --resume to continue it")


def runName(resultsDir, region):
    """The run's name under backtestResults/, as backtest.publish.runName gives it. Not
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
                 .sort_values(["seed", "algo", "trainMode", "intervalTime"]))
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

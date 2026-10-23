"""The agent's dispatch under forecast prices, one CSV row per decision.

An episode reads the 04:00 run that opens its trading day, then a newer run every
`forecastEvery` intervals (see data/forecasts.refreshMatrix). The backtest records the
run each decision read in the trajectory, and the export copies it; a trajectory from
before that has its run derived from the manifest instead. A result set whose manifest predates the refresh read one 04:00
run all day. The fold trajectories in results/<run>/all/ already hold everything needed;
this only reshapes them.

    uv run python -m backtest.actions results/rolling2023 [more runs...]

writes one CSV per run and region into results/actions/, named after the run, so every
backtest's dispatch sits in one folder: results/actions/rolling2023_NSW1.csv,
results/actions/rolling2023_profit_fc4h_SA1.csv. The folds job calls this once it has
merged the seeds.
"""
import argparse
import json
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

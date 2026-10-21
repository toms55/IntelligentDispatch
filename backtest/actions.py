"""The agent's dispatch under forecast prices, one CSV row per decision.

An episode reads the 04:00 run that opens its trading day, then a newer run every
`forecastEvery` intervals (see data/forecasts.refreshMatrix), so forecastRunTime is the
run each decision read. A result set whose manifest predates the refresh read one 04:00
run all day. The fold trajectories in results/<run>/all/ already hold everything needed;
this only reshapes them.

    uv run python -m backtest.actions results/rolling2023 [more runs...]

writes <run>/actions_<region>.csv beside each run's fold files.
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


def exportRun(runDir):
    """One CSV per region under runDir/all/. Returns the paths written."""
    written = []
    for regionDir in sorted(Path(runDir, "all").iterdir()):
        manifest = regionDir / "manifest.json"
        every = (json.loads(manifest.read_text()).get("forecastEvery")
                 if manifest.exists() else None)
        parts = [actionRows(pd.read_parquet(path), regionDir.name, every)
                 for path in sorted(regionDir.glob("fold*.parquet"))]
        if not parts:
            continue
        frame = (pd.concat(parts, ignore_index=True)
                 .sort_values(["seed", "algo", "trainMode", "intervalTime"]))
        out = Path(runDir, f"actions_{regionDir.name}.csv")
        frame.to_csv(out, index=False)
        written.append(out)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="results/<run> directories")
    args = parser.parse_args(argv)
    for runDir in args.runs:
        for path in exportRun(runDir):
            print(path)


if __name__ == "__main__":
    main()

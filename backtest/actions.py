"""The agent's dispatch under forecast prices, one CSV row per decision.

Every episode reads one forward path, the 04:00 run that opens its trading day (see
data/forecasts.forecastMatrix), so a day's 288 rows share one forecastRunTime. The
fold trajectories in results/<run>/all/ already hold everything needed; this only
reshapes them.

    uv run python -m backtest.actions results/rolling2023 [more runs...]

writes <run>/actions_<region>.csv beside each run's fold files.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

columns = (
    "forecastRunTime", "intervalTime", "action", "soc",
    "forecastPrice", "realisedPrice", "region", "fold", "seed", "algo", "arm",
)


def actionRows(trajectory, region):
    """Forecast-priced agent rows of one fold's trajectory, in the export layout.

    intervalTime is the interval's end, as AEMO stamps it, and soc is the state of
    charge at that end, after the action.
    """
    rows = trajectory[(trajectory.priceSource == "forecast")
                      & (trajectory.algo != "oracle")]
    action = np.select([rows.bought > 0, rows.sold > 0], ["charge", "discharge"], "idle")
    return pd.DataFrame({
        "forecastRunTime": rows.testDay,
        "intervalTime": rows.settlementdate,
        "action": action,
        "soc": rows.soc,
        "forecastPrice": rows.observedPrice,
        "realisedPrice": rows.price,
        "region": region,
        "fold": rows.fold,
        "seed": rows.seed,
        "algo": rows.algo,
        "arm": rows.arm,
    }, columns=columns)


def exportRun(runDir):
    """One CSV per region under runDir/all/. Returns the paths written."""
    written = []
    for regionDir in sorted(Path(runDir, "all").iterdir()):
        parts = [actionRows(pd.read_parquet(path), regionDir.name)
                 for path in sorted(regionDir.glob("fold*.parquet"))]
        if not parts:
            continue
        frame = (pd.concat(parts, ignore_index=True)
                 .sort_values(["seed", "algo", "arm", "intervalTime"]))
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

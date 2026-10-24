"""State the forecast pull frequency of a result set written before manifests held it.

A manifest's `forecastEvery` is how often a test day pulled a new AEMO forecast, and the
dashboard refuses a forecast-priced run that does not state it. Result sets from before
the intra-day refresh carry no such key. Before it, every test day read one 04:00 run,
so a set whose every row was written earlier can be stamped once a day: `forecastEvery`
equal to its horizon. A set with any row written later is refused, because its pull
cannot be told from its manifest; rerun it instead.

    uv run python -m backtest.stamp results/rolling2023_profit_all

This is the only code that edits a finished manifest. Never edit one by hand.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtest import records

# bb5e073, "forecast: refresh the agent's view every 4 hours within an episode", authored
# 2026-10-21 20:25:48 +1100. writtenAt is naive local time on the same clock.
refreshIntroduced = datetime(2026, 10, 21, 20, 25, 48)


def stampRegion(resultsDir, region):
    """Stamp one region's manifest once a day, or raise. Returns the value it states."""
    path = records.manifestPath(resultsDir, region)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    manifest = json.loads(path.read_text())
    if "forecastEvery" in manifest:
        return manifest["forecastEvery"]
    days = records.readDays(resultsDir, region)
    if days.empty:
        raise ValueError(f"{resultsDir} holds no rows for {region}, so nothing dates it")
    latest = pd.to_datetime(days.writtenAt).max()
    if latest >= refreshIntroduced:
        raise ValueError(
            f"{path} states no forecastEvery and a row was written at {latest}, after the "
            f"intra-day refresh existed ({refreshIntroduced}), so its pull frequency "
            "cannot be known. Rerun it")
    manifest["forecastEvery"] = int(manifest["horizon"])
    records.writeManifest(resultsDir, region, manifest)
    return manifest["forecastEvery"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", nargs="+", help="results/<run>_all directories")
    args = parser.parse_args(argv)
    for resultsDir in map(Path, args.results):
        for region in sorted(child.name for child in resultsDir.iterdir()
                             if child.is_dir()):
            print(f"{resultsDir}/{region}: forecastEvery "
                  f"{stampRegion(resultsDir, region)}")


if __name__ == "__main__":
    main()

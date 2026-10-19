"""The rolling backtest's output: one CSV and one parquet per fold, plus a manifest.

Each fold file is written whole and moved into place with os.replace, so it is either
the previous complete version or the new complete version and never half of one. The
runner rewrites a fold after every unit it finishes, which costs nothing at forty rows
and makes resume unit-granular without an append log.

The CSV stores numerators and denominators only. Every ratio is formed in
backtest/metrics.py, so the dashboard and the backtest cannot disagree about what
"captured" means.
"""
import json
from pathlib import Path

import pandas as pd

dayColumns = (
    "fold", "algo", "arm", "seed", "priceSource", "testDay",
    "agentProfit", "agentRevenue", "agentDegradation", "oracleDualBound",
    "spikeIntensity", "highVol", "timesteps", "warmFrom", "writtenAt",
)

trajectoryColumns = (
    "fold", "algo", "arm", "seed", "priceSource", "testDay",
    "interval", "settlementdate", "price", "observedPrice",
    "soc", "bought", "sold", "profit",
)

# What makes a row's work already done, for resume. One unit covers a whole test window.
unitColumns = ("fold", "algo", "arm", "seed", "priceSource")

_dayTypes = {"fold": "int64", "seed": "int64", "timesteps": "int64",
             "highVol": "bool", "agentProfit": "float64", "agentRevenue": "float64",
             "agentDegradation": "float64", "oracleDualBound": "float64",
             "spikeIntensity": "float64"}


def regionDir(resultsDir, region):
    return Path(resultsDir) / region


def _write(path, frame, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    writer(frame, temporary)
    temporary.replace(path)
    return path


def _requireColumns(rows, columns, label):
    wanted = set(columns)
    for row in rows:
        missing, extra = wanted - set(row), set(row) - wanted
        if missing or extra:
            raise ValueError(
                f"{label} row does not match the schema; missing {sorted(missing)}, "
                f"unexpected {sorted(extra)}"
            )


def writeFold(resultsDir, region, fold, rows):
    """Replace fold `fold`'s rows with `rows`."""
    _requireColumns(rows, dayColumns, "day")
    frame = pd.DataFrame(list(rows), columns=list(dayColumns))
    return _write(regionDir(resultsDir, region) / f"fold{fold:02d}.csv", frame,
                  lambda f, path: f.to_csv(path, index=False))


def writeTrajectories(resultsDir, region, fold, rows):
    """Replace fold `fold`'s per-interval trajectories with `rows`."""
    _requireColumns(rows, trajectoryColumns, "trajectory")
    frame = pd.DataFrame(list(rows), columns=list(trajectoryColumns))
    return _write(regionDir(resultsDir, region) / f"fold{fold:02d}.parquet", frame,
                  lambda f, path: f.to_parquet(path, index=False))


def readTrajectories(resultsDir, region, fold):
    """Fold `fold`'s per-interval rows already on disk, as dicts. Empty when absent."""
    path = regionDir(resultsDir, region) / f"fold{fold:02d}.parquet"
    if not path.exists():
        return []
    return pd.read_parquet(path).to_dict("records")


def readDays(resultsDir, region):
    """Every fold's rows, in fold order. Empty frame when nothing has run yet."""
    paths = sorted(regionDir(resultsDir, region).glob("fold*.csv"))
    if not paths:
        return pd.DataFrame(columns=list(dayColumns))
    frame = pd.concat(
        # keep_default_na so an absent warmFrom stays "" rather than becoming NaN
        [pd.read_csv(path, parse_dates=["testDay"], dtype=_dayTypes,
                     keep_default_na=False) for path in paths],
        ignore_index=True,
    ).sort_values(["fold", "algo", "arm", "seed", "priceSource", "testDay"],
                  ignore_index=True)
    keys = list(unitColumns) + ["testDay"]
    duplicated = frame[frame.duplicated(keys, keep=False)]
    if not duplicated.empty:
        raise ValueError(
            f"{len(duplicated)} duplicate rows in {regionDir(resultsDir, region)}; "
            f"first is {duplicated.iloc[0][keys].to_dict()}"
        )
    return frame


def completedUnits(frame):
    """The (fold, algo, arm, seed, priceSource) tuples already on disk."""
    if frame.empty:
        return set()
    return set(map(tuple, frame[list(unitColumns)].drop_duplicates().to_numpy().tolist()))


def manifestPath(resultsDir, region):
    return regionDir(resultsDir, region) / "manifest.json"


def writeManifest(resultsDir, region, spec):
    path = manifestPath(resultsDir, region)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True, default=str))
    return path


def requireSameManifest(resultsDir, region, spec):
    """Refuse to add to a result set that was produced under a different configuration."""
    path = manifestPath(resultsDir, region)
    if not path.exists():
        return writeManifest(resultsDir, region, spec)
    stored = json.loads(path.read_text())
    wanted = json.loads(json.dumps(spec, default=str))
    moved = sorted(key for key in set(stored) | set(wanted)
                   if stored.get(key) != wanted.get(key))
    if moved:
        raise ValueError(
            f"{path} was written under a different configuration; these differ: "
            + ", ".join(f"{key} {stored.get(key)!r} -> {wanted.get(key)!r}"
                        for key in moved)
        )
    return path

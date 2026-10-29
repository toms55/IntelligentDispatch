"""How often the agent tried to leave the bounds, and in which intervals.

Out of bounds means the agent chose to charge a full pack or to discharge an empty one.
`rl/env.py:action_masks` is the only definition. MaskablePPO cannot choose either,
because the mask removes the action, so the count is zero for it by construction and
this is the figure that prices running DQN or A2C without a mask.

    uv run python -m backtest.outofbounds rolling2023_profit_dqn NSW1
    uv run python -m backtest.outofbounds rolling2023_profit_dqn NSW1 --seed 0 --limit 40

The counts come from the published day rows, so every seed is covered. The intervals
come from the actions export beside them, which holds one row per decision. Nothing here
solves or trains: it reads two CSVs.
"""
import argparse
from pathlib import Path

import pandas as pd

import config
from visualise.dashboard import data

modelFields = ["algo", "trainMode", "seed", "priceSource"]


def counts(days):
    """One row per model: attempts, intervals scored, and the rate between them."""
    if "outOfBounds" not in days:
        raise ValueError(
            "these rows carry no outOfBounds column, so they predate the count and "
            "cannot be summarised. Printing zero would read as 'the agent never did "
            "it', which is a different claim. Re-run the backtest."
        )
    # One test day is one episode, so the row count gives the days and the horizon gives
    # the intervals. The published rows do not carry the horizon, so this reads
    # config.episode_intervals: a run under a non-default --horizon states its own in the
    # manifest, which stays with the raw results.
    grouped = days.groupby(modelFields, dropna=False)
    out = grouped.agg(outOfBounds=("outOfBounds", "sum"), days=("outOfBounds", "size"))
    out["intervals"] = out.days * config.episode_intervals
    out["rate"] = out.outOfBounds / out.intervals
    return out.reset_index()


def intervals(exported):
    """The rows of an actions export the environment flagged as out of bounds.

    Selected on the env's own flag, which `rl/env.py:action_masks` defines and the
    backtest carries through. Never re-derived from `action`: the mask is a tolerance on
    the headroom, so a flagged step can still move an atom of energy and read as a move,
    and a count from the energy would quietly lose those intervals. A partial top-up is
    not out of bounds at all, because the discrete oracle allows one too.
    """
    if "outOfBounds" not in exported:
        raise ValueError(
            "this actions export carries no outOfBounds column, so the environment's "
            "own flag was never recorded. Deriving it from the energy moved would "
            "disagree with the published day rows at the SoC boundary. Re-run the "
            "backtest."
        )
    return exported[exported.outOfBounds.fillna(False).astype(bool)]


def report(run, region, seed=None, limit=20, root=None):
    """Print the counts, then the intervals. Returns the two frames."""
    runDir = (data.backtestRoot if root is None else Path(root)) / run
    days = data.loadRun(region, runDir).days
    print(f"{run} {region}: out-of-bounds attempts per model")
    # The horizon is stated, because the published rows do not carry it and the rate
    # divides by it. A run made under another --horizon needs its manifest read.
    print(f"rate is per interval, at {config.episode_intervals} intervals a test day\n")
    print(counts(days).to_string(index=False))

    path = data.actionsFile(runDir, region)
    if path is None:
        print("\nno actions export beside this run, so no interval list")
        return counts(days), None
    exported = pd.read_csv(path, parse_dates=["intervalTime", "forecastRunTime"])
    if seed is not None:
        exported = exported[exported.seed == seed]
    found = intervals(exported)
    print(f"\n{len(found)} intervals, showing up to {limit}\n")
    if found.empty:
        print("none: the agent never attempted a move it could not make")
    else:
        shown = ["intervalTime", "attemptedAction", "soc", "realisedPrice",
                 "algo", "seed", "priceSource"]
        print(found[shown].head(limit).to_string(index=False))
    return counts(days), found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="published run name, without the region")
    parser.add_argument("region")
    parser.add_argument("--seed", type=int, help="one seed's intervals only")
    parser.add_argument("--limit", type=int, default=20,
                        help="intervals to print (default: 20)")
    parser.add_argument("--root", help=f"where published runs live "
                                       f"(default: {data.backtestRoot})")
    args = parser.parse_args(argv)
    report(args.run, args.region, seed=args.seed, limit=args.limit, root=args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

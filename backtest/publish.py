"""A finished backtest, copied where the dashboard reads it and git tracks it.

The folds job writes each region's merged rows to results/<run>_<region>/all/<region>/,
one CSV per fold, and git ignores everything under results/ but the actions exports and
the published runs. This writes each region as one file, laid out as the actions
exports are:

    results/backtestResults/<run>_<region>.csv   every fold's day rows, + forecastEvery
    results/actions/<run>_<region>.parquet       every decision, from the live CSV the
                                                 backtest appended to as it ran

The run's name drops the region, so the five regional jobs of one experiment share it.
`forecastEvery` is the one manifest value the dashboard reads, so it travels as a
column; the rest of the manifest stays with the raw results. The day view draws its
dispatch from the actions export, so the fold trajectories are not published either.

    uv run python -m backtest.publish results/rolling2023_profit_fc5m_NSW1 [more runs...]

The folds job calls this once it has merged the seeds. A region is published only
if the dashboard can then load it, so a run it cannot draw is refused here rather than
on the screen. Publishing replaces the region's files whole, and commits nothing: `git
add results/backtestResults results/actions` is still the reader's.
"""
import argparse
import json
from pathlib import Path

from backtest import records
from backtest.actions import exportName, liveFile, readLive, writeExport
from visualise.dashboard import data


def runName(runDir, region):
    """The published run's name: the results directory's, without the region."""
    return Path(runDir).name.removesuffix("_all").removesuffix(f"_{region}")


def regionDirs(runDir):
    """The region folders of a run: under all/ where the folds job merged the seeds,
    else directly in runDir, as one `cli rolling` process or a merged `_all` run
    writes them."""
    merged = Path(runDir, "all")
    return sorted((merged if merged.is_dir() else Path(runDir)).iterdir())


def requireWithinBudget(days, region):
    """Refuse a region where any episode spent more cycles than its budget allowed.

    The environment clips every discharge to the remaining budget and masks the action
    once it is gone, so this cannot fire. It runs anyway, because it is the only check
    that sees the whole run on real prices rather than one episode of a test, and it runs
    before anything is written or committed. A run with no budget has nothing to check.
    """
    if "cycleBudget" not in days or days.cycleBudget.isna().all():
        return
    agent = days[days.algo != "oracle"]
    spent = agent.groupby(["seed", "priceSource", "episodeStart"]).agg(
        cycles=("agentCycles", "sum"), budget=("cycleBudget", "max"))
    over = spent[spent.cycles > spent.budget + 1e-9]
    if len(over):
        seed, source, episode = over.index[0]
        worst = over.iloc[0]
        raise ValueError(
            f"{region} has {len(over)} episode(s) over the cycle budget; the first is "
            f"seed {seed}, {source} prices, the episode starting {episode}, which spent "
            f"{worst.cycles:.6f} of {worst.budget} cycles"
        )


def publish(runDir, root=data.backtestRoot, name=None):
    """Write every finished region of runDir to root/<run>_<region>.csv.

    Returns the files written. A region the dashboard cannot load is removed again and
    the error raised.
    """
    written = []
    for source in regionDirs(runDir):
        if not source.is_dir() or not any(source.glob("fold*.csv")):
            continue
        region = source.name
        run = Path(root) / (name or runName(runDir, region))
        days = records.readDays(source.parent, region)
        manifest = records.manifestPath(source.parent, region)
        every = (json.loads(manifest.read_text()).get("forecastEvery")
                 if manifest.exists() else None)
        if every is not None:
            days["forecastEvery"] = int(every)
        requireWithinBudget(days, region)
        target = run.parent / exportName(run.name, region)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".csv.tmp")
        days.to_csv(temporary, index=False)
        temporary.replace(target)
        # The backtest's live CSV, as the tracked Parquet the dashboard reads. A run
        # exported by `python -m backtest.actions` has no live CSV and is already one.
        live = liveFile(runDir, region)
        if live.exists():
            writeExport(readLive(live), data.actionsPath(run, region))
        try:
            data.loadRun(region, run)
        except Exception:
            target.unlink()
            raise
        written.append(target)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="results/<run> directories")
    parser.add_argument("--root", default=str(data.backtestRoot),
                        help=f"where the dashboard reads (default: {data.backtestRoot})")
    parser.add_argument("--name", help="the published run's name; default the "
                                       "results directory's, without the region")
    args = parser.parse_args(argv)
    for runDir in args.runs:
        for path in publish(runDir, args.root, args.name):
            print(path)


if __name__ == "__main__":
    main()

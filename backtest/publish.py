"""A finished backtest, copied where the dashboard reads it and git tracks it.

The folds job writes each region's merged rows to results/<run>_<region>/all/<region>/,
and git ignores everything under results/ but the actions exports. The dashboard reads
one directory per run with a folder per region, so this copies each region in:

    backtestResults/<run>/<region>/   fold*.csv, fold*.parquet, manifest.json

The run's name drops the region, so the five regional jobs of one experiment land in
one directory. The actions export is not copied: it stays in results/actions/, which
git already tracks, and data.actionsFile finds it there.

    uv run python -m backtest.publish results/rolling2023_profit_fc5m_NSW1 [more runs...]

The folds job calls this once it has exported the actions. A region is published only
if the dashboard can then load it, so a run it cannot draw is refused here rather than
on the screen. Publishing replaces the region whole, and commits nothing: `git add
backtestResults` is still the reader's.
"""
import argparse
import shutil
from pathlib import Path

from backtest import records
from visualise.dashboard import data

published = ("fold*.csv", "fold*.parquet", "manifest.json")


def runName(runDir, region):
    """The published run's name: the results directory's, without the region."""
    return Path(runDir).name.removesuffix("_all").removesuffix(f"_{region}")


def regionDirs(runDir):
    """The region folders of a run: under all/ where the folds job merged the seeds,
    else directly in runDir, as one `cli rolling` process or a merged `_all` run
    writes them."""
    merged = Path(runDir, "all")
    return sorted((merged if merged.is_dir() else Path(runDir)).iterdir())


def publish(runDir, root=data.backtestRoot, name=None):
    """Copy every finished region of runDir into root/<run>/<region>/.

    Returns the region directories written. A region the dashboard cannot load is
    removed again and the error raised.
    """
    written = []
    for source in regionDirs(runDir):
        if not source.is_dir() or not any(source.glob("fold*.csv")):
            continue
        region = source.name
        run = Path(root) / (name or runName(runDir, region))
        target = records.regionDir(run, region)
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        for pattern in published:
            for path in sorted(source.glob(pattern)):
                shutil.copy2(path, target / path.name)
        try:
            data.loadRun(region, run)
        except Exception:
            shutil.rmtree(target)
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

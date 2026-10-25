"""A finished backtest, copied where the dashboard reads it and git tracks it.

The folds job writes each region's merged rows to results/<run>_<region>/all/<region>/,
one CSV per fold, and git ignores everything under results/ but the actions exports and
the published runs. This writes each region as one CSV, laid out as the actions
exports are:

    results/backtestResults/<run>_<region>.csv   every fold's day rows, + forecastEvery
    results/actions/<run>_<region>.csv           every decision, already there

The run's name drops the region, so the five regional jobs of one experiment share it.
`forecastEvery` is the one manifest value the dashboard reads, so it travels as a
column; the rest of the manifest stays with the raw results. The day view draws its
dispatch from the actions export, so the fold trajectories are not published either.

    uv run python -m backtest.publish results/rolling2023_profit_fc5m_NSW1 [more runs...]

It also restamps the oracle columns of a published run from a solved corpus:

    uv run python -m backtest.publish --restamp-oracle results/backtestResults/<run>_NSW1.csv

`oracleDualBound` is the bound every ratio divides by. A gap-limited MILP wrote an
incumbent and a loose bound, and oracleResults/dp holds the same windows solved exactly
by optimisation/dp.py, so restamping replaces the bound with the optimum and the gap
with zero. The agent rows are untouched: the run is rescored against a tighter bound,
not rerun. A day the corpus does not hold, or a bound below a row's agentProfit, refuses
the whole file, because either one means the corpus solved a different problem.

The folds job calls this once it has merged the seeds. A region is published only
if the dashboard can then load it, so a run it cannot draw is refused here rather than
on the screen. Publishing replaces the region's CSV whole, and commits nothing: `git add
results/backtestResults` is still the reader's.
"""
import argparse
import functools
import json
import os
from pathlib import Path

from backtest import records
from backtest.actions import exportName
from rl import baseline
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
        target = run.parent / exportName(run.name, region)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".csv.tmp")
        days.to_csv(temporary, index=False)
        temporary.replace(target)
        try:
            data.loadRun(region, run)
        except Exception:
            target.unlink()
            raise
        written.append(target)
    return written


@functools.lru_cache(maxsize=8)
def corpusDays(corpus, region):
    """One region's records in `corpus`, by trading date. A corpus holds thousands of
    files, and a run asks for a thousand days, so the directory is read once."""
    index = {}
    for path in sorted(Path(corpus).glob(f"{region}_*.json")):
        index.setdefault(path.name.split("_")[1], []).append(path)
    return index


def corpusRecord(corpus, region, day):
    """The one record `corpus` holds for a region and a trading day, as a dict.

    A record is named <region>_<date>_<time>_<key>.json, and the key states the
    parameters of its solve. One corpus is one setting, so the region and the date name
    one record in it. Two would mean two settings, which no caller can tell apart.
    """
    found = corpusDays(str(corpus), region).get(f"{day:%Y-%m-%d}", [])
    if not found:
        raise FileNotFoundError(
            f"{corpus} holds no oracle record for {region} on {day:%Y-%m-%d}, so that "
            f"day cannot be restamped; solve the span with `cli baseline --cache-dir "
            f"{corpus}` first")
    if len(found) > 1:
        raise ValueError(
            f"{corpus} holds {len(found)} oracle records for {region} on "
            f"{day:%Y-%m-%d}: {[path.name for path in found]}. They were solved at "
            "different settings, and the file does not say which one it was scored "
            "against")
    return json.loads(found[0].read_text())


def restampOracle(path, corpus=baseline.dpRoot):
    """Rewrite one published CSV's oracle columns from `corpus`. Returns the two sums.

    `path` is results/backtestResults/<run>_<region>.csv, which names its region.
    """
    path = Path(path)
    region = path.stem.split("_")[-1]
    days = records.readDayFile(path)
    bounds, gaps = {}, {}
    for day in sorted(set(days.testDay)):
        record = corpusRecord(corpus, region, day)
        bounds[day], gaps[day] = record["dualBound"], record["mipGap"]
    stamped = days.assign(oracleDualBound=days.testDay.map(bounds),
                          oracleGap=days.testDay.map(gaps))
    over = stamped[stamped.agentProfit > stamped.oracleDualBound + 1e-9]
    if not over.empty:
        first = over.iloc[0]
        raise ValueError(
            f"{len(over)} rows of {path} beat the bound in {corpus}; the first, "
            f"{first.algo} seed {int(first.seed)} on {first.testDay:%Y-%m-%d}, made "
            f"{first.agentProfit:.4f} against {first.oracleDualBound:.4f}. A bound below "
            "an agent is not a bound, so the corpus solved another problem")
    before = days.groupby("testDay").oracleDualBound.first().sum()
    temporary = path.with_suffix(".csv.tmp")
    stamped.to_csv(temporary, index=False)
    original = path.read_bytes()
    temporary.replace(path)
    try:
        # As publishing does: a file the dashboard cannot draw is not left on disk.
        data.loadRun(region, path.parent / path.stem.removesuffix(f"_{region}"))
    except Exception:
        path.write_bytes(original)
        raise
    return float(before), float(stamped.groupby("testDay").oracleDualBound.first().sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="results/<run> directories, or, with "
                                                "--restamp-oracle, published CSVs")
    parser.add_argument("--restamp-oracle", action="store_true",
                        help="rewrite the named published CSVs' oracle columns from a "
                             "solved corpus instead of publishing a run")
    # Every command runs from the repository root, so the relative path reads better.
    corpus = os.path.relpath(baseline.dpRoot)
    parser.add_argument("--corpus", default=corpus, metavar="DIR",
                        help=f"with --restamp-oracle, the records to read "
                             f"(default: {corpus})")
    parser.add_argument("--root", default=str(data.backtestRoot),
                        help=f"where the dashboard reads (default: {data.backtestRoot})")
    parser.add_argument("--name", help="the published run's name; default the "
                                       "results directory's, without the region")
    args = parser.parse_args(argv)
    if args.restamp_oracle:
        for path in args.runs:
            before, after = restampOracle(path, args.corpus)
            change = (after - before) / before * 100 if before else float("nan")
            print(f"{path}  bound {before:.2f} -> {after:.2f}  ({change:+.3f}%)")
        return
    for runDir in args.runs:
        for path in publish(runDir, args.root, args.name):
            print(path)


if __name__ == "__main__":
    main()

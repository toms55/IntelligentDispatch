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
column. The whole manifest travels beside it as the run's context, with a one-line
description of what the experiment tested, because the raw results never leave this
machine and a clone would otherwise hold numbers with no settings:

    results/backtestResults/<run>_<region>.json  the manifest, the description, and
                                                 the raw results it came from
    results/backtestResults/README.md            one line per published run

The day view draws its dispatch from the actions export, so the fold trajectories are not
published.

    uv run python -m backtest.publish results/rolling2023_profit_fc5m_NSW1 [more runs...] \
        --description "what this run tested, and against which run"

`--context-only` writes the context of a run already published, without copying it again.
A republish without `--description` keeps the description already there.

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
on the screen. Publishing replaces the region's files whole, and commits nothing: `git
add results/backtestResults results/actions` is still the reader's.
"""
import argparse
import functools
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backtest import records
from backtest.actions import exportName, liveFile, readLive, writeExport
from rl import baseline, oracleArchive
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
    # One episode per algorithm and train mode too: a pooled run scores the shared agent
    # and its fine-tuned copy on the same episode, and their cycles must not add up.
    keys = [key for key in ("algo", "trainMode", "seed", "priceSource", "episodeStart")
            if key in agent]
    spent = agent.groupby(keys).agg(
        cycles=("agentCycles", "sum"), budget=("cycleBudget", "max"))
    over = spent[spent.cycles > spent.budget + 1e-9]
    if len(over):
        first = dict(zip(keys, over.index[0]))
        worst = over.iloc[0]
        raise ValueError(
            f"{region} has {len(over)} episode(s) over the cycle budget; the first is "
            + ", ".join(f"{key} {value}" for key, value in first.items())
            + f", which spent {worst.cycles:.6f} of {worst.budget} cycles"
        )


def contextPath(root, run, region):
    """The JSON beside a published region's CSV."""
    return Path(root) / f"{run}_{region}.json"


def writeContext(runDir, root=data.backtestRoot, name=None, description=None,
                 regions=None):
    """Each region's manifest and description beside its published CSV, then the index.

    Returns the files written. `regions` limits it to those, so `publish` describes only
    what it published.
    """
    written = []
    for source in regionDirs(runDir):
        if not source.is_dir() or not any(source.glob("fold*.csv")):
            continue
        region = source.name
        if regions is not None and region not in regions:
            continue
        run = name or runName(runDir, region)
        if not (Path(root) / exportName(run, region)).exists():
            raise ValueError(f"{run} has no published {region} CSV under {root}; "
                             f"publish it before writing its context")
        target = contextPath(root, run, region)
        old = json.loads(target.read_text()) if target.exists() else {}
        manifest = records.manifestPath(source.parent, region)
        context = {
            "run": run,
            "region": region,
            "description": description or old.get("description"),
            "rawResults": str(Path(runDir)),
            "contextWrittenAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        }
        target.write_text(json.dumps(context, indent=2, default=str) + "\n")
        written.append(target)
    writeIndex(root)
    return written


def writeIndex(root=data.backtestRoot):
    """README.md: every published run, its regions and its description. A run published
    before contexts existed is listed with none, so the index never omits a run."""
    root = Path(root)
    runs = {}
    for csv in sorted(root.glob("*_*.csv")):
        run, region = csv.stem.rsplit("_", 1)
        runs.setdefault(run, {"regions": [], "description": None})
        runs[run]["regions"].append(region)
        context = contextPath(root, run, region)
        if context.exists():
            described = json.loads(context.read_text()).get("description")
            runs[run]["description"] = runs[run]["description"] or described
    lines = [
        "# Published backtests",
        "",
        "Written by `python -m backtest.publish`; do not edit by hand. Each",
        "`<run>_<region>.csv` has a `<run>_<region>.json` beside it holding the run's full",
        "manifest (episode, budget, end-empty, refresh, timesteps, pooling) and where its",
        "raw results live.",
        "",
        "| run | regions | what it tested |",
        "|---|---|---|",
    ]
    for run, entry in runs.items():
        about = (entry["description"] or "no description recorded").replace("|", "\\|")
        lines.append(f"| `{run}` | {' '.join(entry['regions'])} | {about} |")
    (root / "README.md").write_text("\n".join(lines) + "\n")


def publish(runDir, root=data.backtestRoot, name=None, description=None):
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
    if written:
        writeContext(runDir, root, name, description,
                     regions={path.stem.rsplit("_", 1)[1] for path in written})
    return written


@functools.lru_cache(maxsize=8)
def corpusDays(corpus, region):
    """One region's records in `corpus`, by trading date. A corpus holds thousands of
    files, and a run asks for a thousand days, so the directory is read once."""
    names = {path.name for path in Path(corpus).glob(f"{region}_*.json")}
    names |= set(oracleArchive.regionRecords(corpus, region))
    index = {}
    for name in sorted(names):
        index.setdefault(name.split("_")[1], []).append(name)
    return index


def corpusRecord(corpus, region, day):
    """The one record `corpus` holds for a region and a trading day, as a dict.

    A record is named <region>_<date>_<time>_<key>.json, as a loose file or as a row
    of the region's Parquet (rl/oracleArchive.py), and the key states the
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
            f"{day:%Y-%m-%d}: {found}. They were solved at "
            "different settings, and the file does not say which one it was scored "
            "against")
    return json.loads(baseline.findRecord(found[0], cacheDir=corpus))


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
    parser.add_argument("--description",
                        help="one line: what the run tested, and against which run")
    parser.add_argument("--context-only", action="store_true",
                        help="write the context of a run already published, and nothing "
                             "else")
    args = parser.parse_args(argv)
    if args.restamp_oracle:
        for path in args.runs:
            before, after = restampOracle(path, args.corpus)
            change = (after - before) / before * 100 if before else float("nan")
            print(f"{path}  bound {before:.2f} -> {after:.2f}  ({change:+.3f}%)")
        return
    for runDir in args.runs:
        if args.context_only:
            paths = writeContext(runDir, args.root, args.name, args.description)
        else:
            paths = publish(runDir, args.root, args.name, args.description)
        for path in paths:
            print(path)


if __name__ == "__main__":
    main()

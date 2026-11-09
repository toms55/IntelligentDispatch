"""One agent trained on every region at once, then a fine-tuned copy per region.

Each fold trains one shared agent on the regions' 52-week training windows laid end to
end (1,820 trading days for five regions), warm from the previous fold's shared agent.
Each region then fine-tunes a copy of that fold's shared agent on its own window and
nothing else. The copy branches afresh from the shared agent every fold, so a regional
copy never drifts away from the shared chain. Both are scored on the region's test week:

    trainMode "pooled"     the shared agent as it stands
    trainMode "finetune"   the region's fine-tuned copy

The environment indexes prices by position and reads timestamps only for the trading-day
clock, so the regions join end to end. Every region's window is whole trading days from
04:00, so no episode straddles two regions. `pooledPrices` refuses a window that would.
The price transform is pinned to the market price cap, which is NEM-wide, so one scaler
serves every region.

One process per seed, as scripts/hpc/rolling2023Folds.pbs runs them. Each seed writes its own
results/<run>_<region>/seed<N>/ for every region, and appends its decisions to the
region's one actions CSV under the lock in backtest/actions.py. `merge` joins the seeds
into results/<run>_<region>/all/, which is where backtest.publish reads them.

    uv run python -m backtest.pooled --run pooled_daily_empty_efc1 --seed 0 --resume
    uv run python -m backtest.pooled --run pooled_daily_empty_efc1 --merge --seeds 0 1 2 3 4
"""
import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from backtest import actions, folds, records, runner
from data.forecasts import refreshMatrix
from degradation.schade import SchadeDegradation
from rl import policies
from rl.evaluate import evaluateSeeds
from rl.train import trainOne

regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
# The shared agent's checkpoints sit under runs/<run>_POOLED/POOLED/, as a region's would.
pooledName = "POOLED"
trainModes = ("pooled", "finetune")


class PooledConfig(NamedTuple):
    # The experiment every region shares: span, episode, budget, refresh, oracle. Its
    # region, results, out, algos, seeds and train modes are set per region below.
    base: runner.RollingConfig
    run: str
    seed: int
    algo: str
    regions: tuple = regions
    pooledWarm: int = 150_000
    pooledCold: int = 1_500_000
    fineTune: int = 30_000
    resultsRoot: Path = Path("results")
    runsRoot: Path = Path("runs")


def runDir(cfg, region):
    """results/<run>_<region>, the name backtest.publish strips the region from."""
    return Path(cfg.resultsRoot) / f"{cfg.run}_{region}"


def regionConfig(cfg, region):
    """The runner's config for one region and this seed, so its helpers apply unchanged."""
    results = runDir(cfg, region) / f"seed{cfg.seed}"
    return cfg.base._replace(
        region=region, algos=(cfg.algo,), seeds=(cfg.seed,), trainModes=trainModes,
        warmTimesteps=cfg.fineTune, coldTimesteps=cfg.pooledCold,
        resultsDir=results, outDir=Path(cfg.runsRoot) / f"{cfg.run}_{region}",
        actionsFile=actions.liveFile(results, region))


def pooledConfig(cfg):
    """Where the shared agent's checkpoints go."""
    return cfg.base._replace(region=pooledName,
                             outDir=Path(cfg.runsRoot) / f"{cfg.run}_{pooledName}")


def manifestOf(cfg, regionCfg):
    """The runner's manifest plus what pooling adds, so a resume under other pooling
    settings is refused."""
    return {**runner.manifestOf(regionCfg),
            "pooled": {"regions": list(cfg.regions), "pooledWarm": cfg.pooledWarm,
                       "pooledCold": cfg.pooledCold, "fineTune": cfg.fineTune}}


def pooledPrices(windows, horizon):
    """The regions' training windows end to end, in the order given.

    BessEnv cuts episodes every `horizon` intervals from the first, so each window must
    be a whole number of episodes or every later region's episodes would straddle two
    regions and open at the wrong hour.
    """
    for region, window in windows.items():
        if len(window) % horizon:
            raise ValueError(
                f"{region}'s training window is {len(window)} intervals, not a whole "
                f"number of {horizon}-interval episodes, so the next region's episodes "
                f"would straddle the join")
    return pd.concat(list(windows.values()))


def _train(rollingCfg, foldIndex, trainMode, algo, seed, prices, scaler, warmFrom,
           timesteps):
    """Train one checkpoint unless it is already on disk, as runner._ensureTrained does,
    with the timesteps given rather than the warm or cold budget."""
    directory = runner.checkpointDir(rollingCfg, foldIndex, trainMode, algo, seed)
    if (directory / "agent.zip").exists():
        return directory
    if warmFrom is not None and not (Path(warmFrom) / "agent.zip").exists():
        raise ValueError(f"fold {foldIndex} {trainMode} starts from {warmFrom}, which "
                         f"holds no checkpoint")
    trainOne(
        algo, prices, SchadeDegradation(*rollingCfg.battery), seed=seed,
        timesteps=timesteps, calendar=rollingCfg.calendar, horizon=rollingCfg.horizon,
        rewardScale=rollingCfg.rewardScale, outDir=directory.parent, scaler=scaler,
        warmFrom=warmFrom, initialSoc=rollingCfg.initialSoc,
        cycleBudget=rollingCfg.cycleBudget, terminalPrice=rollingCfg.terminalPrice,
        endEmpty=rollingCfg.endEmpty,
    )
    return directory


def runPooled(cfg, onProgress=None):
    """Every fold for one seed: train the shared agent, fine-tune per region, score
    both. Returns the number of (region, train mode, price source) units written."""
    regional = {region: regionConfig(cfg, region) for region in cfg.regions}
    # Both checks before any plan, because a plan solves whatever the cache lacks.
    for region, regionCfg in regional.items():
        records.requireSameManifest(regionCfg.resultsDir, region,
                                    manifestOf(cfg, regionCfg))
        missing = runner.auditPlan(regionCfg)["missing"]
        if missing:
            raise ValueError(
                f"{missing} {region} episodes are not in the oracle cache; build it with "
                f"STAGE=oracle scripts/rolling.sh under the same settings first")

    plans = {region: runner._plan(regionCfg) for region, regionCfg in regional.items()}
    first = plans[cfg.regions[0]]
    for region, plan in plans.items():
        if [fold.index for fold in plan.selected] != [f.index for f in first.selected]:
            raise ValueError(f"{region}'s fold table differs from {cfg.regions[0]}'s")
        if plan.scaler != first.scaler:
            raise ValueError(f"{region}'s price transform differs from "
                             f"{cfg.regions[0]}'s; the shared agent needs one")

    shared = pooledConfig(cfg)
    existing = {region: records.readDays(regionCfg.resultsDir, region)
                for region, regionCfg in regional.items()}
    done = {region: records.completedUnits(frame) for region, frame in existing.items()}
    logged = {region: actions.writtenUnits(regionCfg.actionsFile)
              for region, regionCfg in regional.items()}
    sources = cfg.base.priceSources
    written = 0

    for position, fold in enumerate(first.selected):
        units = [(fold.index, cfg.algo, mode, cfg.seed, source)
                 for mode in trainModes for source in sources]
        if all(unit in done[region] for region in cfg.regions for unit in units):
            if onProgress:
                onProgress(position + 1, len(first.selected), fold)
            continue

        windows = {}
        for region, plan in plans.items():
            window = folds.sliceWindow(plan.prices, fold.trainStart, fold.trainEnd)
            folds.requireFullDays(window, fold.trainStart, fold.trainEnd)
            windows[region] = window

        parent = (None if position == 0 else runner.checkpointDir(
            shared, first.selected[position - 1].index, "pooled", cfg.algo, cfg.seed))
        pooledSteps = cfg.pooledWarm if parent else cfg.pooledCold
        pooledDir = _train(shared, fold.index, "pooled", cfg.algo, cfg.seed,
                           pooledPrices(windows, cfg.base.horizon), first.scaler,
                           parent, pooledSteps)
        provenance = {"pooled": (pooledSteps, "" if parent is None else str(parent)),
                      "finetune": (cfg.fineTune, str(pooledDir))}

        for region, regionCfg in regional.items():
            pending = [(mode, source) for mode in trainModes for source in sources
                       if (fold.index, cfg.algo, mode, cfg.seed, source)
                       not in done[region]]
            if not pending:
                continue
            plan = plans[region]
            checkpoints = {
                "pooled": pooledDir,
                "finetune": _train(regionCfg, fold.index, "finetune", cfg.algo,
                                   cfg.seed, windows[region], plan.scaler, pooledDir,
                                   cfg.fineTune),
            }
            testPrices = folds.sliceWindow(plan.prices, fold.testStart, fold.testEnd)
            folds.requireFullDays(testPrices, fold.testStart, fold.testEnd)
            episodes = fold.testEpisodes(regionCfg.horizon)
            testRecords = runner._foldRecords(regionCfg, plan, fold.testStart,
                                              len(episodes))
            rows = existing[region][existing[region].fold == fold.index].to_dict("records")
            decisions = []
            forecasts = None
            for mode, source in pending:
                if source == "forecast" and forecasts is None:
                    # allowPartial=False: a NaN row reads as a flat $0 day.
                    forecasts = refreshMatrix(
                        episodes, region, regionCfg.horizon, regionCfg.forecastEvery,
                        allowPartial=False, source=plan.forecastSource)
                observed = forecasts if source == "forecast" else None
                results = evaluateSeeds(
                    {cfg.seed: checkpoints[mode] / "agent"}, testPrices, plan.model,
                    baselines=[record["profitPerInterval"] for record in testRecords],
                    algoName=cfg.algo, calendar=regionCfg.calendar,
                    horizon=regionCfg.horizon, rewardScale=regionCfg.rewardScale,
                    forecasts=observed,
                    bounds=[record["dualBound"] for record in testRecords],
                    initialSoc=regionCfg.initialSoc, cycleBudget=regionCfg.cycleBudget,
                    terminalPrice=regionCfg.terminalPrice, endEmpty=regionCfg.endEmpty,
                )
                steps, warmFrom = provenance[mode]
                # _resultRows reads the warm/cold budget off the config, which says
                # nothing true about a pooled or fine-tuned checkpoint.
                rows += [{**row, "timesteps": steps, "warmFrom": warmFrom}
                         for row in runner._resultRows(
                             regionCfg, plan, fold, cfg.algo, mode, source, results,
                             position, testRecords, episodes)]
                if (fold.index, cfg.seed, cfg.algo, mode, source) not in logged[region]:
                    decisions += runner._resultTrajectories(
                        regionCfg, fold, cfg.algo, mode, source, results[cfg.seed],
                        testPrices, observed, seed=cfg.seed, episodes=episodes)
                written += 1
            if decisions:
                # Before the fold CSV, as in the runner: a fold killed between the two is
                # scored again on resume, and `logged` keeps its rows from doubling.
                actions.appendRows(regionCfg.actionsFile,
                                   actions.actionRows(pd.DataFrame(decisions), region))
            records.writeFold(regionCfg.resultsDir, region, fold.index, rows)
        gc.collect()
        if onProgress:
            onProgress(position + 1, len(first.selected), fold)

    publishFinal(cfg, first.selected[-1])
    return written


def publishFinal(cfg, fold):
    """Each region's last-fold agents into trainedPolicies/: its fine-tuned copy, and the
    shared agent under its own train mode, so the forecast screen can run either."""
    shared = pooledConfig(cfg)
    published = []
    for region in cfg.regions:
        for mode in trainModes:
            owner = shared if mode == "pooled" else regionConfig(cfg, region)
            checkpoint = runner.checkpointDir(owner, fold.index, mode, cfg.algo, cfg.seed)
            published.append(policies.publish(
                checkpoint, region, cfg.algo, mode, cfg.seed,
                root=cfg.base.policyRoot, tag=cfg.base.policyTag, command="pooled",
                # The regional run directory, not the pooled one, so both train modes
                # of a fold name the one experiment that scored them.
                rawRun=Path(regionConfig(cfg, region).outDir).name,
                fold=fold.index, pooledRegions=list(cfg.regions),
                trainWindow=[fold.trainStart.isoformat(), fold.trainEnd.isoformat()],
                testWindow=[fold.testStart.isoformat(), fold.testEnd.isoformat()],
                capDate=cfg.base.capDate.isoformat(),
            ))
    return published


def merge(run, seeds, regionList=regions, resultsRoot=Path("results")):
    """Join every seed's rows into results/<run>_<region>/all/, as the folds job does."""
    for region in regionList:
        root = Path(resultsRoot) / f"{run}_{region}"
        frame = pd.concat([records.readDays(root / f"seed{seed}", region)
                           for seed in seeds], ignore_index=True)
        records.sortDays(frame, root)
        for fold, rows in frame.groupby("fold"):
            records.writeFold(root / "all", region, fold, rows.to_dict("records"))
        manifest = json.loads(records.manifestPath(root / f"seed{seeds[0]}", region)
                              .read_text())
        records.writeManifest(root / "all", region,
                              {**manifest, "seeds": [int(seed) for seed in seeds]})
        print(f"merged         {len(frame)} rows, seeds {list(seeds)} -> {root / 'all'}")


def _config(args):
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    every, rest = divmod(args.forecast_refresh, 5)
    if rest or every <= 0 or args.horizon % every:
        raise SystemExit(f"--forecast-refresh {args.forecast_refresh} must be whole "
                         f"5-minute intervals dividing the {args.horizon}-interval episode")
    base = runner.RollingConfig(
        region=args.regions[0], start=start, end=end,
        capDate=datetime.fromisoformat(args.cap_date), trainWeeks=args.train_weeks,
        priceSources=(("realised", "forecast") if args.price_source == "both"
                      else (args.price_source,)),
        forecastEvery=every, horizon=args.horizon,
        cycleBudget=None if args.cycles == "none" else float(args.cycles),
        terminalPrice=args.terminal_price, endEmpty=args.end_empty,
        calendar=False, discretePower=True, gap=args.gap, timeLimit=args.time_limit,
        trajectorySeed=None,
        foldFilter=None if args.folds is None else frozenset(
            range(int(args.folds.split("-")[0]), int(args.folds.split("-")[-1]) + 1)),
        policyRoot=Path(args.policies), policyTag=args.policy_tag,
    )
    return PooledConfig(base=base, run=args.run, seed=args.seed, algo=args.algo,
                        regions=tuple(args.regions), pooledWarm=args.pooled_warm,
                        pooledCold=args.pooled_cold, fineTune=args.fine_tune,
                        resultsRoot=Path(args.results_root),
                        runsRoot=Path(args.runs_root))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m backtest.pooled",
                                     description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True,
                        help="results/<run>_<region>; the published run's name")
    parser.add_argument("--regions", nargs="+", default=list(regions))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--algo", default="maskableppo")
    parser.add_argument("--pooled-warm", type=int, default=150_000, metavar="N",
                        help="shared agent's steps per warm fold (default: 150000)")
    parser.add_argument("--pooled-cold", type=int, default=1_500_000, metavar="N",
                        help="shared agent's steps on the first fold (default: 1500000)")
    parser.add_argument("--fine-tune", type=int, default=30_000, metavar="N",
                        help="each region's fine-tune steps per fold (default: 30000)")
    # The daily end-empty setup of scripts/rolling.sh, so the runs compare directly.
    parser.add_argument("--start", default="2023-01-01T04:00")
    parser.add_argument("--end", default="2026-09-27T04:00")
    parser.add_argument("--cap-date", default="2026-10-01T04:00")
    parser.add_argument("--train-weeks", type=int, default=52)
    parser.add_argument("--horizon", type=int, default=288)
    parser.add_argument("--cycles", default="1.0", help="per episode, or none")
    parser.add_argument("--terminal-price", type=float, default=0.0)
    parser.add_argument("--end-empty", action="store_true")
    parser.add_argument("--forecast-refresh", type=int, default=5, metavar="MINUTES")
    parser.add_argument("--price-source", choices=["realised", "forecast", "both"],
                        default="both")
    parser.add_argument("--gap", type=float, default=0.02)
    parser.add_argument("--time-limit", type=float, default=1800)
    parser.add_argument("--folds", metavar="N-M", help="a contiguous subset of folds")
    parser.add_argument("--policies", default=str(policies.defaultRoot))
    parser.add_argument("--policy-tag")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--resume", action="store_true",
                        help="continue a run of this name; without it a name already "
                             "in use is refused")
    parser.add_argument("--merge", action="store_true",
                        help="join --seeds into all/ instead of running")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)

    if args.merge:
        merge(args.run, args.seeds, args.regions, Path(args.results_root))
        return 0
    cfg = _config(args)
    if not args.resume:
        for region in cfg.regions:
            actions.refuseExisting(runDir(cfg, region), region)
    try:
        written = runPooled(cfg, onProgress=lambda done, total, fold: print(
            f"  [{done:3d}/{total}] fold {fold.index:2d} test {fold.testStart:%Y-%m-%d}",
            flush=True))
    except ValueError as error:
        raise SystemExit(str(error))
    print(f"wrote          {written} new unit(s) over {len(cfg.regions)} regions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

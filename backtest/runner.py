"""The rolling-origin backtest driver.

Span-wide work happens once: one price fetch, one oracle build, one volatility pass, one
price transform. The reward is the agent's own profit, so the oracle is only the bound the
test days are scored against, and it is built from the first test day rather than from the
start of the span. Per-fold bounds are then slices of it, found by whole-day offset,
which is only safe because every boundary is a trading day and requireFullDays has
proved the index complete. The start-stamp check in _foldRecords is what makes the
arithmetic trustworthy: a day-shifted slice gives wrong regret on every row and raises
nothing on its own.

The checkpoint on disk is the training record and the fold CSV is the completion record,
so resume needs no separate log.
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

import config
from backtest import folds, records, volatility
from data.forecasts import (decisionPath, decisionRuns, loadForecastCache, refreshMatrix,
                            refreshRunTimes)
from data.prices import price_series
from degradation.schade import SchadeDegradation
from rl.baseline import buildBaseline, cacheKey, cacheRoot
from rl import policies
from rl.evaluate import evaluateSeeds
from rl.features import PriceScaler
from rl.train import trainOne
from utils import nem_interval
from utils.market import marketPriceCap


class RollingConfig(NamedTuple):
    region: str
    start: datetime
    end: datetime
    capDate: datetime
    trainWeeks: int
    testWeeks: int = 1
    stepWeeks: int = 1
    algos: tuple = ("maskableppo",)
    seeds: tuple = config.seeds
    trainModes: tuple = ("warm",)
    warmTimesteps: int = 30_000
    coldTimesteps: int = 300_000
    priceSources: tuple = ("realised",)
    # What fills the observation's price slots in training. Settlement is realised either
    # way; "forecast" trains on the same refresh runs a forecast-priced test reads.
    trainSource: str = "realised"
    forecastEvery: int = config.forecast_refresh_intervals
    horizon: int = config.episode_intervals
    rewardScale: float = config.reward_scale
    initialSoc: float = config.initial_soc
    calendar: bool = False
    # The oracle supplies the test-day bound only; training never consults it. `captured`
    # divides by this bound, and a continuous oracle may ramp to any power between zero
    # and rated, which the agent's three actions cannot, so a continuous bound is one the
    # agent can never reach. The existing data_cache/lp_baseline entries are continuous,
    # so this key misses every one of them and the discrete cache must be built first.
    # See config.backtest_time_limit for why the limit is 1800s.
    discretePower: bool = True
    gap: float = config.backtest_gap
    timeLimit: Optional[float] = config.backtest_time_limit
    workers: int = 1
    battery: tuple = (config.battery_mw, config.battery_mwh,
                      config.battery_replacement_cost)
    trajectorySeed: Optional[int] = 0
    foldFilter: Optional[frozenset] = None
    allowSolve: bool = False
    outDir: Path = Path("runs/rolling")
    resultsDir: Path = Path("results/rolling")
    # Where the last fold's agent is published when the run finishes. The per-fold
    # checkpoints under outDir are working state; this is the one a reader looks for.
    policyRoot: Path = policies.defaultRoot


class Plan(NamedTuple):
    folds: list
    selected: list
    prices: object
    oracle: list
    oracleStart: datetime
    volatility: object
    scaler: PriceScaler
    model: SchadeDegradation
    forecastSource: object


def manifestOf(cfg):
    """What two runs must agree on before their rows may share a result set."""
    # forecastEvery is recorded for every run, realised-only included: it is how often a
    # test day pulls a new AEMO forecast (and, under --train-source forecast, each
    # training day too), and the dashboard refuses a result set that does not state it.
    skip = {"foldFilter", "allowSolve", "outDir", "resultsDir", "workers", "timeLimit"}
    # Result sets from before training could read forecasts carry no such key.
    if cfg.trainSource == "realised":
        skip.add("trainSource")
    # Checkpoints trained against the old oracle-baselined reward carry no such key, so
    # this keeps a resumed run from mixing the two rewards in one result set.
    return {**{key: value for key, value in cfg._asdict().items() if key not in skip},
            "reward": "profit"}


def _foldTable(cfg):
    """Every fold in the span, and the ones this run covers. Raises on either empty."""
    everyFold = folds.enumerateFolds(cfg.start, cfg.end, cfg.trainWeeks,
                                     cfg.testWeeks, cfg.stepWeeks)
    if not everyFold:
        raise ValueError(
            f"{cfg.start:%Y-%m-%d} to {cfg.end:%Y-%m-%d} has no room for a "
            f"{cfg.trainWeeks}-week train plus a {cfg.testWeeks}-week test"
        )
    selected = [fold for fold in everyFold
                if cfg.foldFilter is None or fold.index in cfg.foldFilter]
    if not selected:
        raise ValueError(
            f"the fold filter {sorted(cfg.foldFilter)} selected none of the "
            f"{len(everyFold)} folds in the span, which are 0 to {len(everyFold) - 1}"
        )
    return everyFold, selected


def _plan(cfg):
    everyFold, selected = _foldTable(cfg)
    spanEnd = everyFold[-1].testEnd
    prices = price_series(cfg.start, spanEnd, cfg.region)
    folds.requireFullDays(prices, cfg.start, spanEnd)
    model = SchadeDegradation(*cfg.battery)
    oracleStart = everyFold[0].testStart
    oracle = buildBaseline(
        folds.sliceWindow(prices, oracleStart, spanEnd), region=cfg.region, horizon=cfg.horizon, calendar=cfg.calendar,
        discretePower=cfg.discretePower, model=model, initialSoc=cfg.initialSoc,
        gap=cfg.gap, timeLimit=cfg.timeLimit, workers=cfg.workers,
    )
    days = sorted({day for fold in selected for day in fold.testDays})
    forecastDays = days
    if cfg.trainSource == "forecast":
        forecastDays = sorted(set(days).union(
            *(_trainDays(fold) for fold in selected)))
    return Plan(everyFold, selected, prices, oracle, oracleStart,
                volatility.volatilityFrame(prices, days),
                PriceScaler(marketPriceCap(cfg.capDate)), model,
                _forecastSource(cfg, forecastDays))


def _trainDays(fold):
    """The 04:00 start of each trading day in the training window."""
    count = (fold.trainEnd - fold.trainStart) // folds.oneDay
    return [fold.trainStart + step * folds.oneDay for step in range(count)]


def _forecastSource(cfg, days):
    """The prefetched forecast paths, checked to cover every refresh run of every day
    given, test days and, when the agent trains on forecasts, training days, each from its
    run time to the episode end.

    Without the check a run the prefetch missed becomes a silent NaN row, and the agent
    scores as if it idled. An absent cache falls back to the live per-run path.
    """
    if "forecast" not in cfg.priceSources and cfg.trainSource != "forecast":
        return None
    source = loadForecastCache(cfg.region)
    if source is None:
        return None
    # Each run must reach the episode end, and no further is needed.
    needed, opener = {}, {}
    for day in days:
        for offset, run in zip(range(0, cfg.horizon, cfg.forecastEvery),
                               refreshRunTimes(day, cfg.horizon, cfg.forecastEvery)):
            needed[run], opener[run] = cfg.horizon - offset, pd.Timestamp(day)
    runs = list(needed)
    want = np.array([needed[run] for run in runs])

    def coverage(frame):
        present = frame.reindex(runs).iloc[:, :cfg.horizon].notna().to_numpy()
        return (present & (np.arange(cfg.horizon) < want[:, None])).sum(axis=1)

    # A run AEMO never published, or one whose PREDISPATCH half it skipped (both on
    # 2024-09-05 from 14:00 in every region), is filled from the latest earlier run of the
    # same episode, which is all a live agent would have had. Runs are in time order, so
    # an earlier run is already filled when a later one reads it. The episode's opening
    # run has nothing to hold, so a gap there is still refused below.
    filled = {}
    for index in np.flatnonzero(coverage(source) < want):
        run = runs[index]
        before = source.index.searchsorted(run) - 1
        if before < 0 or source.index[before] < opener[run]:
            continue
        earlier = source.index[before]
        lag = (run - earlier) // nem_interval
        prior = filled.get(earlier)
        if prior is None:
            prior = source.loc[earlier].to_numpy(dtype=float)
        held = np.concatenate([prior[lag:], np.full(lag, np.nan)])
        own = (source.loc[run].to_numpy(dtype=float) if run in source.index
               else np.full(len(source.columns), np.nan))
        filled[run] = np.where(np.isnan(own), held, own)
        print(f"forecast run {run:%Y-%m-%d %H:%M} incomplete, holding {earlier:%H:%M} "
              f"for {int(np.isnan(own[:needed[run]]).sum())} step(s)",
              file=sys.stderr, flush=True)
    if filled:
        source = pd.concat([
            source.drop(index=[run for run in filled if run in source.index]),
            pd.DataFrame.from_dict(filled, orient="index", columns=source.columns),
        ]).sort_index()

    gaps = [run for run in needed if run not in source.index]
    if gaps:
        raise ValueError(
            f"the forecast cache for {cfg.region} is missing {len(gaps)} run(s), "
            f"first {gaps[0]:%Y-%m-%d %H:%M}; run "
            f"`cli prefetch --region {cfg.region}` over the whole span"
        )
    # A thin row is worse than a missing one: the environment zeroes the NaN padding, so
    # the agent reads a flat $0 stretch and idles through it, and the loss lands in the
    # results looking like a decision.
    covered = coverage(source)
    thin = np.flatnonzero(covered < want)
    if len(thin):
        first = thin[0]
        raise ValueError(
            f"{len(thin)} forecast run(s) are partial, first "
            f"{runs[first]:%Y-%m-%d %H:%M} with {int(covered[first])} of "
            f"{int(want[first])} intervals; either the prefetch did not cover the whole "
            f"span, or AEMO published no PREDISPATCH run at {runs[first]:%H:%M} that "
            f"day. Drop the fold with --folds if it is the archive"
        )
    return source


def _spanDays(cfg, plan):
    span = plan.folds[-1].testEnd - plan.folds[0].trainStart
    return span // folds.oneDay


def auditPlan(cfg):
    """The --dry-run report: the fold table, and how much oracle solving it would need."""
    everyFold, selected = _foldTable(cfg)
    model = SchadeDegradation(*cfg.battery)
    # The span is every fold, not the selection: one oracle build covers every test day
    # whichever folds run. Training days need no oracle at all.
    spanStart, spanEnd = everyFold[0].testStart, everyFold[-1].testEnd
    days = [spanStart + n * folds.oneDay
            for n in range((spanEnd - spanStart) // folds.oneDay)]
    missing = [day for day in days if not (cacheRoot / cacheKey(
        cfg.region, day + nem_interval, cfg.horizon, cfg.calendar, cfg.discretePower,
        model, cfg.initialSoc, cfg.gap)).exists()]
    return {"folds": len(everyFold), "selected": len(selected),
            "days": len(days), "missing": len(missing),
            "leftoverDays": folds.leftoverDays(everyFold, cfg.end),
            "first": selected[0], "last": selected[-1]}


def _foldRecords(cfg, plan, start, dayCount):
    """The oracle records for `dayCount` days from `start`, checked against their stamps."""
    first = folds.dayOffset(plan.oracleStart, start)
    if first + dayCount > len(plan.oracle):
        raise ValueError(
            f"the oracle record set holds {len(plan.oracle)} days, too few to reach day "
            f"{first + dayCount} for the window starting {start:%Y-%m-%d}"
        )
    chunk = plan.oracle[first:first + dayCount]
    for step, record in enumerate(chunk):
        wanted = (start + step * folds.oneDay + nem_interval).isoformat()
        if record["start"] != wanted:
            raise ValueError(
                f"oracle record {first + step} starts {record['start']}, expected "
                f"{wanted}; the record set is shifted against the price index"
            )
    return chunk


def checkpointDir(cfg, fold, trainMode, algo, seed):
    return (Path(cfg.outDir) / cfg.region / f"fold{fold:02d}" / trainMode
            / f"{algo}_seed{seed}")


def _ensureTrained(cfg, fold, trainMode, algo, seed, parent, trainPrices, scaler,
                   trainForecasts=None):
    """Train this fold's agent unless its checkpoint is already on disk.

    `trainForecasts` is a callable, so a resumed fold never builds the matrix."""
    directory = checkpointDir(cfg, fold, trainMode, algo, seed)
    if (directory / "agent.zip").exists():
        return directory
    warmFrom = parent if trainMode == "warm" else None
    if warmFrom is not None and not (Path(warmFrom) / "agent.zip").exists():
        raise ValueError(
            f"fold {fold} warm starts from {warmFrom}, which holds no checkpoint. Run "
            f"the earlier folds first, or pass trainModes=('cold',) to fit each fold fresh"
        )
    trainOne(
        algo, trainPrices, SchadeDegradation(*cfg.battery), seed=seed, timesteps=cfg.warmTimesteps if warmFrom else cfg.coldTimesteps,
        calendar=cfg.calendar, horizon=cfg.horizon, rewardScale=cfg.rewardScale,
        outDir=directory.parent, scaler=scaler, warmFrom=warmFrom,
        forecasts=None if trainForecasts is None else trainForecasts(),
    )
    return directory


def _trajectoryRows(cfg, fold, algo, trainMode, seed, source, day, steps, observed,
                    runs=None):
    for step, info in enumerate(steps):
        yield {
            "fold": fold, "algo": algo, "trainMode": trainMode, "seed": seed,
            "priceSource": source, "testDay": day, "interval": step,
            "settlementdate": day + (step + 1) * nem_interval,
            "price": info["price"],
            "observedPrice": info["price"] if observed is None else observed[step],
            "forecastRunTime": None if runs is None else runs[step],
            "soc": info["soc"], "bought": info["bought"], "sold": info["sold"],
            "profit": info["profit"],
        }


def _oracleRows(cfg, fold, day, record, prices):
    for step in range(cfg.horizon):
        yield {
            "fold": fold, "algo": "oracle", "trainMode": "oracle",
            "seed": cfg.trajectorySeed, "priceSource": "realised", "testDay": day,
            "interval": step, "settlementdate": day + (step + 1) * nem_interval,
            "price": prices[step], "observedPrice": prices[step], "forecastRunTime": None,
            "soc": record["soc"][step], "bought": record["bought"][step],
            "sold": record["sold"][step], "profit": record["profitPerInterval"][step],
        }


def runRolling(cfg, onProgress=None):
    # Both checks come before _plan, because _plan solves the oracle cache. A guard that
    # fires after the solve it forbids has already spent the hours it exists to save.
    records.requireSameManifest(cfg.resultsDir, cfg.region, manifestOf(cfg))

    audit = auditPlan(cfg)
    if audit["missing"] and not cfg.allowSolve:
        raise ValueError(
            f"{audit['missing']} of {audit['days']} days are not in the oracle cache. "
            f"Solving them is hours of MILP, so pass allowSolve to go ahead."
        )

    plan = _plan(cfg)
    existing = records.readDays(cfg.resultsDir, cfg.region)
    done = records.completedUnits(existing)
    testDays = cfg.testWeeks * 7
    written = 0

    for position, fold in enumerate(plan.selected):
        rows = existing[existing.fold == fold.index].to_dict("records")
        trajectories = []

        trainPrices = folds.sliceWindow(plan.prices, fold.trainStart, fold.trainEnd)
        folds.requireFullDays(trainPrices, fold.trainStart, fold.trainEnd)
        trainForecasts = None
        if cfg.trainSource == "forecast":
            # Built once per fold, on first need, and shared by every seed.
            cache = {}

            def trainForecasts(fold=fold, cache=cache):
                if "grid" not in cache:
                    cache["grid"] = refreshMatrix(
                        _trainDays(fold), cfg.region, cfg.horizon, cfg.forecastEvery,
                        allowPartial=False, source=plan.forecastSource)
                return cache["grid"]
        testPrices = folds.sliceWindow(plan.prices, fold.testStart, fold.testEnd)
        folds.requireFullDays(testPrices, fold.testStart, fold.testEnd)

        testRecords = _foldRecords(cfg, plan, fold.testStart, testDays)
        testBaselines = [record["profitPerInterval"] for record in testRecords]
        bounds = [record["dualBound"] for record in testRecords]
        gaps = [record["mipGap"] for record in testRecords]

        for algo in cfg.algos:
            for trainMode in cfg.trainModes:
                checkpoints = {}
                for seed in cfg.seeds:
                    parent = (None if position == 0 else
                              checkpointDir(cfg, plan.selected[position - 1].index,
                                            trainMode, algo, seed))
                    checkpoints[seed] = _ensureTrained(
                        cfg, fold.index, trainMode, algo, seed, parent, trainPrices,
                        plan.scaler, trainForecasts)

                for source in cfg.priceSources:
                    pending = [seed for seed in cfg.seeds
                               if (fold.index, algo, trainMode, seed, source) not in done]
                    if not pending:
                        continue
                    # allowPartial=False: a NaN row reads as a flat $0 day and the
                    # agent idles through it, so a missing run must stop the run.
                    forecasts = (refreshMatrix(
                        fold.testDays, cfg.region, cfg.horizon, cfg.forecastEvery,
                        allowPartial=False, source=plan.forecastSource)
                        if source == "forecast" else None)
                    results = evaluateSeeds(
                        {seed: checkpoints[seed] / "agent" for seed in pending},
                        testPrices, plan.model, baselines=testBaselines,
                        algoName=algo, calendar=cfg.calendar, horizon=cfg.horizon,
                        rewardScale=cfg.rewardScale, forecasts=forecasts, bounds=bounds,
                    )
                    rows += _resultRows(cfg, plan, fold, algo, trainMode, source, results,
                                        position, gaps)
                    if cfg.trajectorySeed in results:
                        trajectories += _resultTrajectories(
                            cfg, fold, algo, trainMode, source, results[cfg.trajectorySeed],
                            testPrices, forecasts)
                    written += len(pending)

        if trajectories:
            # The parquet is replaced whole, so carry over the series this pass did not
            # regenerate. Without it a resumed run drops every other train mode and source.
            fresh = {(row["algo"], row["trainMode"], row["priceSource"])
                     for row in trajectories} | {("oracle", "oracle", "realised")}
            kept = [row for row in records.readTrajectories(
                        cfg.resultsDir, cfg.region, fold.index)
                    if (row["algo"], row["trainMode"], row["priceSource"]) not in fresh]
            trajectories += [row for day, record in zip(fold.testDays, testRecords)
                             for row in _oracleRows(
                                 cfg, fold.index, day, record,
                                 folds.sliceWindow(plan.prices, day,
                                                   day + folds.oneDay).to_numpy())]
            records.writeTrajectories(cfg.resultsDir, cfg.region, fold.index,
                                      kept + trajectories)
        records.writeFold(cfg.resultsDir, cfg.region, fold.index, rows)
        if onProgress:
            onProgress(position + 1, len(plan.selected), fold)
    publishFinal(cfg, plan.selected[-1])
    return written


def publishFinal(cfg, fold):
    """Copy the last fold's agents into `trainedPolicies/`.

    A warm chain ends on the fold that saw the most recent prices, so that fold is the
    trained model of the whole run. The dashboard reads only this directory, which is
    what lets one machine train and another draw the result.
    """
    published = []
    for algo in cfg.algos:
        for trainMode in cfg.trainModes:
            for seed in cfg.seeds:
                checkpoint = checkpointDir(cfg, fold.index, trainMode, algo, seed)
                if not (checkpoint / "agent.zip").exists():
                    continue
                published.append(policies.publish(
                    checkpoint, cfg.region, algo, trainMode, seed,
                    root=cfg.policyRoot, command="rolling", fold=fold.index,
                    trainWindow=[fold.trainStart.isoformat(), fold.trainEnd.isoformat()],
                    testWindow=[fold.testStart.isoformat(), fold.testEnd.isoformat()],
                    capDate=cfg.capDate.isoformat(),
                ))
    return published


def _resultRows(cfg, plan, fold, algo, trainMode, source, results, position, gaps):
    stamp = datetime.now().isoformat(timespec="seconds")

    def parentOf(seed):
        """The checkpoint this seed warm started from, which is its own, not seed 0's."""
        return ("" if position == 0 else
                str(checkpointDir(cfg, plan.selected[position - 1].index,
                                  trainMode, algo, seed)))

    out = []
    for seed, perWindow in sorted(results.items()):
        parent = parentOf(seed)
        for record in perWindow:
            day = fold.testDays[record["window"]]
            stats = plan.volatility.loc[day]
            out.append({
                "fold": fold.index, "algo": algo, "trainMode": trainMode, "seed": seed,
                "priceSource": source, "testDay": day,
                "agentProfit": record["agentProfit"],
                "agentRevenue": record["agentRevenue"],
                "agentDegradation": record["agentDegradation"],
                "oracleDualBound": record["lpProfit"],
                # None when the solve stopped with no incumbent; the CSV holds NaN.
                "oracleGap": (float("nan") if gaps[record["window"]] is None
                              else gaps[record["window"]]),
                "spikeIntensity": float(stats.spikeIntensity),
                "highVol": bool(stats.highVol),
                "timesteps": cfg.warmTimesteps if (trainMode == "warm" and position)
                             else cfg.coldTimesteps,
                "warmFrom": parent if trainMode == "warm" else "",
                "writtenAt": stamp,
            })
    return out


def _resultTrajectories(cfg, fold, algo, trainMode, source, perWindow, testPrices, forecasts):
    out = []
    for record in perWindow:
        day = fold.testDays[record["window"]]
        observed, runs = ((None, None) if forecasts is None
                          else (decisionPath(forecasts[record["window"]]),
                                decisionRuns(day, forecasts[record["window"]])))
        out += list(_trajectoryRows(cfg, fold.index, algo, trainMode, cfg.trajectorySeed,
                                    source, day, record["trajectory"], observed, runs))
    return out

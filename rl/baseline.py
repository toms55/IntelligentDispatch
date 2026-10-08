"""Per-window oracle solutions, solved once and cached (TODO.txt step 1 chunking)."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import config
from degradation.schade import SchadeDegradation
from optimisation.lp import solve
from utils import nem_interval_hours

cacheRoot = Path(__file__).resolve().parent.parent / "data_cache" / "lp_baseline"


def chunkWindows(length, horizon=config.episode_intervals):
    """Start offsets of the non-overlapping windows inside a series of `length`."""
    return list(range(0, length - horizon + 1, horizon))


def cacheKey(region, start, horizon, calendar, discretePower, model, initialSoc,
             gap=None):
    parts = [
        region,
        start.isoformat(),
        str(horizon),
        f"cal={int(bool(calendar))}",
        f"disc={int(bool(discretePower))}",
        f"mw={model.chargePower!r}",
        f"mwh={model.MWh!r}",
        f"rc={model.replacementCost!r}",
        f"soc={initialSoc!r}",
        f"gap={gap!r}",
    ]
    digest = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return f"{region}_{start.date()}_{start.strftime('%H%M')}_{digest}.json"


def solveWindow(prices, timestamps, region, horizon=config.episode_intervals,
                calendar=True, discretePower=True, model=None,
                initialSoc=config.initial_soc, dtHours=nem_interval_hours,
                timeLimit=config.oracle_time_limit, gap=config.oracle_gap,
                threads=None, cacheDir=None):
    """One window's oracle solution, from cache when it is there."""
    model = model or _defaultModel()
    directory = Path(cacheDir) if cacheDir else cacheRoot
    path = directory / cacheKey(region, timestamps[0], horizon, calendar,
                                discretePower, model, initialSoc, gap)
    if path.exists():
        return json.loads(path.read_text())

    schedule = solve(list(prices), model, initialSoc=initialSoc, dtHours=dtHours,
                     calendar=calendar, discretePower=discretePower,
                     timeLimit=timeLimit, gap=gap, threads=threads)
    record = {
        "start": timestamps[0].isoformat(),
        "profitPerInterval": list(schedule.profitPerInterval),
        "revenuePerInterval": list(schedule.revenuePerInterval),
        "degradationPerInterval": list(schedule.degradationPerInterval),
        "soc": list(schedule.soc),
        "bought": list(schedule.bought),
        "sold": list(schedule.sold),
        "profit": schedule.profit,
        "dualBound": schedule.dualBound,
        "revenue": schedule.revenue,
        "degradationCost": schedule.degradationCost,
        "status": schedule.status,
        "mipGap": schedule.mipGap,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return record


def buildBaseline(prices, region=config.default_region,
                  horizon=config.episode_intervals, calendar=True, discretePower=True,
                  model=None, initialSoc=config.initial_soc,
                  timeLimit=config.oracle_time_limit, gap=config.oracle_gap,
                  workers=1, cacheDir=None, onDone=None):
    """Solve every window of `prices` and return the records in window order."""
    model = model or _defaultModel()
    starts = chunkWindows(len(prices), horizon)
    jobs = [
        (
            list(prices.iloc[start:start + horizon].to_numpy(dtype=float)),
            list(prices.index[start:start + horizon]),
        )
        for start in starts
    ]
    common = dict(region=region, horizon=horizon, calendar=calendar,
                  discretePower=discretePower, model=model, initialSoc=initialSoc,
                  timeLimit=timeLimit, gap=gap, cacheDir=cacheDir,
                  threads=1 if workers > 1 else None)

    if workers <= 1:
        records = []
        for windowPrices, windowTimes in jobs:
            records.append(solveWindow(windowPrices, windowTimes, **common))
            if onDone:
                onDone(len(records), len(jobs), records[-1])
        return records

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(solveWindow, windowPrices, windowTimes, **common)
            for windowPrices, windowTimes in jobs
        ]
        records = []
        for future in futures:
            records.append(future.result())
            if onDone:
                onDone(len(records), len(jobs), records[-1])
        return records


def profitPerInterval(records):
    return [record["profitPerInterval"] for record in records]


def dualBounds(records):
    """The per-window upper bound. Never sum profitPerInterval for a quoted comparison:
    a timed-out window can come back with only the do-nothing schedule."""
    return [record["dualBound"] for record in records]


def timedOut(records, requestedGap):
    return [
        record for record in records
        if record["mipGap"] is None or record["mipGap"] > requestedGap + 1e-9
    ]


def _defaultModel():
    return SchadeDegradation(config.battery_mw, config.battery_mwh,
                             config.battery_replacement_cost)

"""Per-window oracle solutions, solved once and cached (TODO.txt step 1 chunking)."""
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import config
from degradation.schade import SchadeDegradation
from optimisation import dp
from optimisation.lp import solve
from utils import nem_interval_hours

_repo = Path(__file__).resolve().parent.parent
# New records are written here.
cacheRoot = _repo / "data_cache" / "lp_baseline"
# The solved corpora git tracks: oracleResults/dp holds the discrete oracle solved
# exactly by optimisation/dp.py, every trading day of every region, and oracleResults/
# itself holds the MILP records from before it.
oracleRoot = _repo / "oracleResults"
dpRoot = oracleRoot / "dp"
# Where a record is looked up, in order, and the first entry is where a new one lands.
# A cache key states every parameter of its solve, the solver included, so a key only
# ever matches a record of the same problem whichever directory holds it.
readRoots = (cacheRoot, dpRoot, oracleRoot)


def findRecord(name, cacheDir=None):
    """The path of the cached record `name`, or None when no directory holds it."""
    for directory in ((Path(cacheDir),) if cacheDir else readRoots):
        path = directory / name
        if path.exists():
            return path
    return None


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
        # Discrete windows are solved exactly by optimisation/dp.py now. Without this
        # part the key would serve an older MILP record, whose dual bound can sit above
        # the true optimum.
        f"solver={'dp' if discretePower else 'milp'}",
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
    name = cacheKey(region, timestamps[0], horizon, calendar, discretePower, model,
                    initialSoc, gap)
    found = findRecord(name, cacheDir)
    if found is not None:
        return json.loads(found.read_text())
    directory = Path(cacheDir) if cacheDir else cacheRoot
    path = directory / name

    if discretePower:
        # Exact, and milliseconds rather than minutes. timeLimit, gap and threads do not
        # apply: there is no search to cut short, so no fallback either.
        schedule = dp.solve(list(prices), model, initialSoc=initialSoc,
                            dtHours=dtHours, calendar=calendar)
        fallback = None
    else:
        common = dict(initialSoc=initialSoc, dtHours=dtHours, calendar=calendar,
                      timeLimit=timeLimit, gap=gap, threads=threads)
        schedule, fallback = _solveWithFallback(list(prices), model, False, common)
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
        "fallback": fallback,
    }
    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return record


# HiGHS 1.14 can stop checking its time limit after a MIP restart: NSW1 2023-11-02 ran
# 19 hours against an 1800s limit, holding up a whole oracle job. So every limited solve
# runs in a child that is killed past the limit plus this grace, and a killed solve is
# retried with restarts off, then as the continuous oracle. The continuous optimum is
# never below the discrete one, so its dual bound is still a valid, looser, bound.
killGrace = 300
# Memory a solve may add on top of what the worker already holds. A normal discrete day
# peaks near 1.2GB; some VIC1 days ran a worker past 8GB in minutes, and fifteen of them
# took a 64GB-129GB job down. Hitting the cap kills that child alone, and the fallback
# treats it like an overrun.
memoryCap = 8 * 2**30


def _solveWithFallback(prices, model, discretePower, common):
    """(Schedule, None) normally, else (Schedule, the fallback that produced it)."""
    attempts = [(None, dict(discretePower=discretePower))]
    if discretePower:
        attempts += [
            ("noRestart", dict(discretePower=True,
                               options={"mip_allow_restart": False})),
            ("continuous", dict(discretePower=False)),
        ]
    timeLimit = common["timeLimit"]
    failure = None
    for name, settings in attempts:
        if timeLimit is None:
            return solve(prices, model, **common, **settings), name
        try:
            return _withDeadline(timeLimit + killGrace, solve, prices, model,
                                 **common, **settings), name
        except (TimeoutError, RuntimeError) as error:
            failure = error
    raise TimeoutError(f"every oracle attempt overran {timeLimit}s + {killGrace}s "
                       f"or died (last: {failure})")


def _capMemory(extra):
    """Limit this process's address space to what it holds now plus `extra` bytes."""
    import resource
    with open("/proc/self/statm") as statm:
        held = int(statm.read().split()[0]) * resource.getpagesize()
    resource.setrlimit(resource.RLIMIT_AS, (held + extra, resource.RLIM_INFINITY))


def _runInChild(connection, function, args, kwargs):
    if memoryCap:
        _capMemory(memoryCap)
    connection.send(function(*args, **kwargs))
    connection.close()


def _withDeadline(seconds, function, *args, **kwargs):
    """function(*args, **kwargs) in a forked child, killed after `seconds`."""
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    child = context.Process(target=_runInChild, args=(sender, function, args, kwargs))
    child.start()
    sender.close()
    try:
        if not receiver.poll(seconds):
            raise TimeoutError(f"{function.__name__} overran {seconds}s")
        return receiver.recv()
    except EOFError:
        child.join()
        raise RuntimeError(f"{function.__name__} died with exit code "
                           f"{child.exitcode}") from None
    finally:
        if child.is_alive():
            child.kill()
        child.join()
        receiver.close()


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
        or record.get("fallback")
    ]


def _defaultModel():
    return SchadeDegradation(config.battery_mw, config.battery_mwh,
                             config.battery_replacement_cost)

"""Run an optimisation or RL module over a slice of NEM prices, until the dashboard exists.

    python -m cli lp 2024-01-10 2024-01-11 --no-calendar --csv day.csv
    python -m cli baseline --split train --workers 8
    python -m cli train --split train --algo maskableppo --seed 0
    python -m cli eval  --split val   --algo maskableppo

`lp` is the oracle. `baseline` solves and caches one oracle schedule per episode window,
which `eval` scores against. `train` and `eval` are the agent.  The reward is the agent's
own profit, so training needs no oracle.

A window is either two explicit dates or a named `--split` out of config.py.
"""
import argparse
import contextlib
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from data.prices import price_series
from utils import nem_interval
from degradation.schade import SchadeDegradation
from optimisation.lp import defaultSolver, solve
from rl import policies


DEFAULT_PATH = ""  # --csv / --plot given without a PATH; filled in by _outputPath


def _formatter(prog):
    return argparse.HelpFormatter(prog, max_help_position=28)


@contextlib.contextmanager
def _stage(label, quiet):
    """Announce a step on stderr and tick the elapsed seconds while it runs."""
    if quiet:
        yield
        return

    live = sys.stderr.isatty()
    start = time.monotonic()
    done = threading.Event()

    def tick():
        while not done.wait(0.25):
            print(f"\r  {label}... {time.monotonic() - start:4.0f}s",
                  end="", file=sys.stderr, flush=True)

    if live:
        ticker = threading.Thread(target=tick, daemon=True)
        ticker.start()
    else:
        print(f"  {label}...", file=sys.stderr, flush=True)

    try:
        yield
    finally:
        done.set()
        elapsed = time.monotonic() - start
        prefix = "\r" if live else ""
        print(f"{prefix}  {label}... {elapsed:.1f}s{' ' * 8}", file=sys.stderr, flush=True)


def _sharedParser():
    """Arguments common to every module: the window, the market, the battery."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("start", type=datetime.fromisoformat, nargs="?",
                        help="start of the window, ISO date or datetime "
                             "(e.g. 2024-01-10 or 2024-01-10T06:00); omit and use --split")
    parser.add_argument("end", type=datetime.fromisoformat, nargs="?",
                        help="end of the window, exclusive, same format as start")
    parser.add_argument("--split", action="append", metavar="NAME=START..END",
                        help="a named window instead of two dates, e.g. "
                             "train=2024-01-01T04:00..2024-11-01T04:00. Use '..' "
                             "between the dates, because a single ':' is ambiguous "
                             "against an ISO time. Give one per run")
    parser.add_argument("--region", default=config.default_region, metavar="ID",
                        help=f"NEM region: SA1, NSW1, VIC1, QLD1, TAS1 "
                             f"(default: {config.default_region})")
    parser.add_argument("--mw", type=float, default=config.battery_mw,
                        help=f"charge/discharge power limit, in MW (default: {config.battery_mw})")
    parser.add_argument("--mwh", type=float, default=config.battery_mwh,
                        help=f"usable energy capacity, in MWh (default: {config.battery_mwh})")
    parser.add_argument("--replacement-cost", type=float,
                        default=config.battery_replacement_cost, metavar="COST",
                        help=f"pack replacement, in $/MWh of capacity "
                             f"(default: {config.battery_replacement_cost:.0f})")
    parser.add_argument("--initial-soc", type=float, default=config.initial_soc,
                        metavar="SOC",
                        help=f"state of charge at the start of every episode and every "
                             f"oracle window, 0-1 (default: {config.initial_soc})")
    parser.add_argument("--csv", nargs="?", const=DEFAULT_PATH, metavar="PATH",
                        help="write the per-interval trajectory as CSV; "
                             "PATH defaults to <module>_<region>_<start>.csv in the cwd")
    parser.add_argument("--plot", nargs="?", const=DEFAULT_PATH, metavar="PATH",
                        help="save the price/SoC/dispatch figure; "
                             "PATH defaults to <module>_<region>_<start>.png in the cwd")
    parser.add_argument("--show", action="store_true",
                        help="display the figure in a window instead of saving it")
    parser.set_defaults(splitName=None)
    return parser


def _battery(args):
    return SchadeDegradation(args.mw, args.mwh, args.replacement_cost)


def _parseSplit(value):
    """One --split value, as (name, start, end). Raises SystemExit on any malformed part.

    The separator is '..' rather than ':' because an ISO datetime already contains ':'.
    """
    name, equals, window = value.partition("=")
    if not equals or not name.strip():
        raise SystemExit(f"--split wants NAME=START..END, got {value!r}")
    start, dots, end = window.partition("..")
    if not dots:
        raise SystemExit(f"--split {name.strip()} wants START..END, got {window!r}")
    try:
        return name.strip(), datetime.fromisoformat(start), datetime.fromisoformat(end)
    except ValueError as error:
        raise SystemExit(f"--split {name.strip()}: {error}")


def _window(args):
    """Resolve the window to (start, end). Explicit dates win over --split."""
    splits = args.split or []
    if len(splits) > 1:
        raise SystemExit(f"give one --split per run, got {len(splits)}")
    if args.start is None:
        if not splits:
            raise SystemExit("give two dates, or --split NAME=START..END")
        args.splitName, args.start, args.end = _parseSplit(splits[0])
    elif args.end is None:
        raise SystemExit("give an end date as well as a start, or use --split alone")
    if args.end <= args.start:
        raise SystemExit(f"the window ends before it starts: {args.start} to {args.end}")
    return args.start, args.end


def _prices(args):
    start, end = _window(args)
    return price_series(start, end, args.region)


def _outputPath(args, suffix):
    label = args.splitName or args.start.date()
    return f"{args.module}_{args.region}_{label}.{suffix}"


def _frame(prices, schedule):
    return pd.DataFrame(
        {
            "settlementdate": prices.index,
            "price": prices.to_list(),
            "soc": schedule.soc,
            "bought": schedule.bought,
            "sold": schedule.sold,
        }
    )


def _plot(frame, destination):
    """Price above, SoC and net dispatch below. destination=None opens a window."""
    import matplotlib
    if destination is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, sharex=True, figsize=(12, 7))
    time = frame["settlementdate"]

    top.plot(time, frame["price"], linewidth=0.9)
    top.set_ylabel("$/MWh")
    top.axhline(0, color="grey", linewidth=0.5)

    bottom.bar(time, frame["sold"] - frame["bought"], width=0.003, color="tab:orange",
               label="net MWh (meter side)")
    bottom.set_ylabel("MWh")
    soc = bottom.twinx()
    soc.plot(time, frame["soc"], color="tab:blue", linewidth=1.0, label="SoC")
    soc.set_ylabel("SoC")
    soc.set_ylim(0, 1)
    bottom.legend(loc="upper left", fontsize="small")
    soc.legend(loc="upper right", fontsize="small")
    figure.autofmt_xdate()
    figure.tight_layout()

    if destination is None:
        plt.show()
    else:
        figure.savefig(destination, dpi=120)
    plt.close(figure)


def _report(prices, schedule, args):
    print(f"{len(prices)} intervals of {args.region}, {prices.index[0]} to {prices.index[-1]}")
    print(f"price      min {prices.min():9.2f}  max {prices.max():9.2f}  mean {prices.mean():8.2f}")
    print(f"revenue        {schedule.revenue:12.4f}")
    print(f"degradation    {schedule.degradationCost:12.4f}")
    print(f"profit         {schedule.profit:12.4f}")
    print(f"soc            min {min(schedule.soc):.3f}  max {max(schedule.soc):.3f}"
          f"  start {schedule.initialSoc:.3f}")
    print(f"energy sold    {sum(schedule.sold):12.4f} MWh")

    wantsPlot = args.plot is not None or args.show
    if args.csv is None and not wantsPlot:
        return

    frame = _frame(prices, schedule)
    if args.csv is not None:
        path = args.csv or _outputPath(args, "csv")
        frame.to_csv(path, index=False)
        print(f"wrote          {path}")
    if wantsPlot:
        path = None if args.show else (args.plot or _outputPath(args, "png"))
        _plot(frame, path)
        if path is not None:
            print(f"wrote          {path}")


def _runLp(args):
    prices = _prices(args)
    schedule = solve(
        prices.to_list(),
        _battery(args),
        initialSoc=args.initial_soc,
        calendar=not args.no_calendar,
        discretePower=args.discrete_power,
        gap=args.gap,
        timeLimit=args.time_limit,
    )
    _report(prices, schedule, args)
    if args.discrete_power:
        bound = "n/a" if schedule.dualBound is None else f"{schedule.dualBound:12.4f}"
        gap = "n/a" if schedule.mipGap is None else f"{schedule.mipGap:.2%}"
        print(f"dual bound     {bound}   (mip gap {gap}, status {schedule.status})")
    return 0


def _runBaseline(args):
    from rl import baseline

    prices = _prices(args)
    windows = baseline.chunkWindows(len(prices), args.horizon)
    print(f"{len(prices)} intervals of {args.region} -> {len(windows)} windows "
          f"of {args.horizon}")

    def progress(done, total, record):
        print(f"  [{done:4d}/{total}] {record['start'][:10]} "
              f"profit {record['profit']:8.4f}  bound {record['dualBound']:8.4f}  "
              f"gap {record['mipGap']:.2%}", flush=True)

    records = baseline.buildBaseline(
        prices, region=args.region, horizon=args.horizon,
        calendar=not args.no_calendar, discretePower=not args.continuous,
        model=_battery(args), initialSoc=args.initial_soc,
        gap=args.gap, timeLimit=args.time_limit, workers=args.workers,
        onDone=progress if args.verbose else None,
    )
    profit = sum(record["profit"] for record in records)
    bound = sum(record["dualBound"] for record in records)
    print(f"windows        {len(records):12d}")
    print(f"oracle profit  {profit:12.4f}")
    print(f"dual bound     {bound:12.4f}")
    print(f"median/window  {statistics.median(r['profit'] for r in records):12.4f}"
          f"   <- a sane reward_scale")
    return 0


def _runTrain(args):
    from rl.features import PriceScaler, scalerFor
    from rl.train import trainOne
    from utils.market import marketPriceCap

    prices = _prices(args)
    model = _battery(args)
    scaler = (PriceScaler(marketPriceCap(args.cap_date)) if args.cap_date
              else scalerFor(prices.index))
    print(f"price scale     cap {scaler.cap:.0f}, knee {scaler.knee:.0f}")
    for seed in args.seeds:
        with _stage(f"train {args.algo} seed {seed}", quiet=not args.verbose):
            _, path = trainOne(
                args.algo, prices, model, seed=seed,
                timesteps=args.timesteps, calendar=not args.no_calendar,
                horizon=args.horizon, rewardScale=args.reward_scale, outDir=args.out,
                scaler=scaler,
            )
        print(f"wrote          {path}.zip")
        # One fit from fresh weights is a cold agent, which is the train mode the
        # rolling backtest gives the same thing. The dashboard reads this directory.
        published = policies.publish(
            path.parent, args.region, args.algo, "cold", seed,
            root=args.policies, command="train", timesteps=args.timesteps,
            window=[args.start.isoformat(), args.end.isoformat()],
            priceCap=scaler.cap,
        )
        print(f"published      {published}")
    return 0


def _runEval(args):
    from rl import baseline, evaluate

    prices = _prices(args)
    model = _battery(args)
    records = baseline.buildBaseline(
        prices, region=args.region, horizon=args.horizon,
        calendar=not args.no_calendar, model=model, initialSoc=args.initial_soc,
        discretePower=args.baseline_oracle == "discrete",
        gap=args.gap, timeLimit=args.time_limit, workers=args.workers,
    )
    baselines = baseline.profitPerInterval(records)
    # Never the summed per-interval profit: a timed-out window can contribute zeros.
    bounds = baseline.dualBounds(records)
    unproven = baseline.timedOut(records, args.gap or 0.0)
    if unproven:
        print(f"note           {len(unproven)} of {len(records)} windows did not reach the "
              f"{(args.gap or 0.0):.1%} gap; their dual bounds are still valid")

    forecasts = None
    if args.forecast:
        from data.forecasts import refreshMatrix
        from rl.baseline import chunkWindows

        # An episode opens one interval before its first interval ends.
        dayStarts = [
            prices.index[start] - nem_interval for start in
            chunkWindows(len(prices), args.horizon)
        ]
        forecasts = refreshMatrix(dayStarts, args.region, args.horizon,
                                  _refreshIntervals(args, args.horizon))

    if args.heuristic:
        from rl.env import BessEnv
        from rl.heuristic import ThresholdPolicy

        env = BessEnv(prices, model, baseline=baselines,
                      calendar=not args.no_calendar, horizon=args.horizon,
                      rewardScale=args.reward_scale, forecasts=forecasts)
        policy = ThresholdPolicy(args.quantiles[0], args.quantiles[1])
        results = {0: evaluate.evaluatePolicy(policy, env, masking=True, bounds=bounds)}
        args.seeds = [0]
        args.algo = f"threshold {args.quantiles[0]}/{args.quantiles[1]}"
        _reportEval(evaluate.acrossSeeds(results), results, args)
        return 0

    checkpoints = {
        seed: Path(args.out) / f"{args.algo}_seed{seed}" / "agent" for seed in args.seeds
    }
    missing = [str(path) for path in checkpoints.values() if not path.with_suffix(".zip").exists()]
    if missing:
        raise SystemExit("no checkpoint at: " + ", ".join(missing))

    # Each seed is scored under the transform it trained with, so say which that is.
    from rl.train import loadScaler
    trained = loadScaler(Path(next(iter(checkpoints.values()))).parent)
    print(f"price scale    cap {trained.cap:.0f}, knee {trained.knee:.0f} "
          f"(from the checkpoint)")

    results = evaluate.evaluateSeeds(
        checkpoints, prices, model, baselines=baselines, algoName=args.algo,
        calendar=not args.no_calendar, horizon=args.horizon,
        rewardScale=args.reward_scale, forecasts=forecasts, bounds=bounds,
    )
    summary = evaluate.acrossSeeds(results)
    _reportEval(summary, results, args)
    return 0


def _parseFolds(value):
    """'0-3,10' as a set of fold indices. Raises SystemExit on any malformed part."""
    chosen = set()
    for part in value.split(","):
        part = part.strip()
        try:
            if "-" in part.lstrip("-"):
                first, last = (int(n) for n in part.split("-", 1))
                chosen.update(range(first, last + 1))
            else:
                chosen.add(int(part))
        except ValueError:
            raise SystemExit(f"--folds wants N or N-M, comma separated; got {part!r}")
    if not chosen:
        raise SystemExit(f"--folds selected nothing: {value!r}")
    return frozenset(chosen)


def _runPublish(args):
    """Put an existing checkpoint under trainedPolicies/.

    Training publishes on its own. This is for the agents trained before the
    convention, and for an agent moved between machines by hand.
    """
    if len(args.seeds) != len(args.checkpoint):
        raise SystemExit(
            f"{len(args.checkpoint)} directories against {len(args.seeds)} seeds: "
            f"give one --seeds value per directory, in the same order"
        )
    for seed, checkpoint in zip(args.seeds, args.checkpoint):
        published = policies.publish(
            checkpoint, args.region, args.algo, args.train_mode, seed,
            root=args.policies, command="publish",
        )
        print(f"published      {checkpoint} -> {published}")
    return 0


def _runRolling(args):
    # A bad span, a cold oracle cache and a changed manifest are all the operator's to
    # fix, so they exit with the message rather than a traceback.
    try:
        return _rolling(args)
    except ValueError as error:
        raise SystemExit(str(error))


def _rolling(args):
    from backtest import metrics, records, runner

    start, end = _window(args)
    sources = (("realised", "forecast") if args.price_source == "both"
               else (args.price_source,))
    cfg = runner.RollingConfig(
        region=args.region, start=start, end=end, capDate=args.cap_date,
        trainWeeks=args.train_weeks, testWeeks=args.test_weeks,
        stepWeeks=args.step_weeks, algos=tuple(args.algo), seeds=tuple(args.seeds),
        trainModes=tuple(args.trainModes), warmTimesteps=args.warm_timesteps,
        coldTimesteps=args.cold_timesteps, priceSources=sources,
        forecastEvery=_refreshIntervals(args, args.horizon),
        horizon=args.horizon, rewardScale=args.reward_scale,
        initialSoc=args.initial_soc, calendar=not args.no_calendar,
        discretePower=args.baseline_oracle == "discrete",
        gap=args.gap, timeLimit=args.time_limit, workers=args.workers,
        battery=(args.mw, args.mwh, args.replacement_cost),
        trajectorySeed=None if args.trajectory_seed < 0 else args.trajectory_seed,
        foldFilter=None if args.folds is None else _parseFolds(args.folds),
        allowSolve=args.allow_solve, outDir=Path(args.out),
        resultsDir=Path(args.results), policyRoot=Path(args.policies),
    )

    audit = runner.auditPlan(cfg)
    print(f"folds          {audit['selected']} of {audit['folds']}, "
          f"{audit['leftoverDays']} day(s) past the last one")
    print(f"first          train {audit['first'].trainStart:%Y-%m-%d} .. "
          f"{audit['first'].trainEnd:%Y-%m-%d}, test .. {audit['first'].testEnd:%Y-%m-%d}")
    print(f"last           test {audit['last'].testStart:%Y-%m-%d} .. "
          f"{audit['last'].testEnd:%Y-%m-%d}")
    print(f"oracle cache   {audit['days'] - audit['missing']} of {audit['days']} days "
          f"present, {audit['missing']} would be solved")
    if args.dry_run:
        return 0

    written = runner.runRolling(cfg, onProgress=lambda done, total, fold: print(
        f"  [{done:3d}/{total}] fold {fold.index:2d} test {fold.testStart:%Y-%m-%d}"))
    frame = records.readDays(cfg.resultsDir, cfg.region)
    print(f"wrote          {written} new unit(s), {len(frame)} rows in "
          f"{records.regionDir(cfg.resultsDir, cfg.region)}")
    print(f"published      {len(cfg.algos) * len(cfg.trainModes) * len(cfg.seeds)} "
          f"agent(s) to {Path(args.policies) / cfg.region}")
    _reportRolling(frame, metrics, cfg.gap)
    return 0


def _refreshIntervals(args, horizon):
    """--forecast-refresh minutes in dispatch intervals. Refuses one that does not divide
    the episode, since every refresh must start on the step grid."""
    every, rest = divmod(args.forecast_refresh, 5)
    if rest or every <= 0 or horizon % every:
        raise SystemExit(
            f"--forecast-refresh {args.forecast_refresh} minutes must be a whole number "
            f"of 5-minute intervals that divides the {horizon}-interval episode"
        )
    return every


def _runPrefetch(args):
    from data.forecasts import prefetchForecasts

    start, end = _window(args)
    # Always the full episode horizon: a shorter one is a slice of this cache.
    target = prefetchForecasts(
        start, end, args.region, config.episode_intervals,
        every=_refreshIntervals(args, config.episode_intervals))
    print(f"wrote {target}")
    return 0


def _reportRolling(frame, metrics, gap):
    # One block per (algo, trainMode, source): pooling them would average unlike runs.
    for (algo, trainMode, source), part in frame.groupby(
            ["algo", "trainMode", "priceSource"]):
        overall = metrics.summarise(part)
        days = part.drop_duplicates("testDay")
        # NaN is a solve with no gap at all, so it counts as loose too.
        loose = int((~(days.oracleGap <= gap + 1e-9)).sum())
        print(f"\n{algo} {trainMode} on {source} prices")
        print(f"days           {overall['days']}  ({loose} with a loose bound, gap "
              f"above {gap:.2%}: captured is a floor there)")
        print(f"agent profit   {overall['agentProfit']:12.4f}")
        print(f"oracle profit  {overall['oracleProfit']:12.4f}")
        print(f"captured       {overall['captured']:12.2%}  "
              f"(per-fold median {overall['foldMedian']:.2%}, "
              f"IQR {overall['foldIqr']:.2%})")
        print(f"win rate       {overall['winRate']:12.2%}")
        for high, got in metrics.byBucket(part).items():
            label = "high vol" if high else "low vol "
            print(f"  {label}     {got['days']:4d} days  agent {got['agentProfit']:10.4f}  "
                  f"oracle {got['oracleProfit']:10.4f}  captured {got['captured']:7.2%}")
    for (algo, trainMode), part in frame.groupby(["algo", "trainMode"]):
        if set(part.priceSource) >= {"realised", "forecast"}:
            delta = metrics.difference(part, "priceSource", "realised", "forecast")
            print(f"\n{algo} {trainMode}: realised minus forecast {delta.pooled:.4f} over "
                  f"{len(delta.perFold)} folds")


def _reportEval(summary, results, args):
    """The same report for a trained agent and for the rule-based control."""
    from rl import evaluate

    source = "AEMO forecast" if args.forecast else "realised prices"
    print(f"{args.algo} on {args.region}, {len(args.seeds)} seeds, "
          f"{summary['perSeed'][args.seeds[0]]['windows']} windows, "
          f"price slots from {source}")
    print(f"oracle profit  {summary['lpProfit']:12.4f}")
    print(f"agent profit   {summary['agentProfit']['mean']:12.4f}"
          f" +/- {summary['agentProfit']['sd']:.4f}")
    print(f"regret         {summary['regret']['mean']:12.4f}"
          f" +/- {summary['regret']['sd']:.4f}")
    print(f"captured       {summary['captured']['mean']:12.2%}"
          f" +/- {summary['captured']['sd']:.2%}")
    for seed, records in sorted(results.items()):
        one = evaluate.summarise(records)
        print(f"  seed {seed}       {one['agentProfit']:12.4f}"
              f"   regret {one['regret']:9.4f}   captured {one['captured']:7.2%}")

    if getattr(args, "result_file", None):
        heading, body = _resultSection(summary, results, args, source)
        _spliceSection(args.result_file, heading, body)
        print(f"result file    {args.result_file}  (section: {heading})")


def _resultSection(summary, results, args, source):
    """One markdown section for a results file: one region, one price source."""
    from rl import evaluate

    heading = "AEMO forecast" if args.forecast else "Realised prices"
    lines = [
        f"- region: {args.region}",
        f"- window: {args.start.isoformat()} .. {args.end.isoformat()}",
        f"- algo: {args.algo}",
        f"- seeds: {', '.join(str(s) for s in args.seeds)}",
        f"- battery: {args.mw} MW / {args.mwh} MWh, "
        f"replacement cost {args.replacement_cost:.0f}",
        f"- price slots from: {source}",
        f"- generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| metric | value |",
        "|---|---|",
        f"| oracle profit | {summary['lpProfit']:.4f} |",
        f"| agent profit | {summary['agentProfit']['mean']:.4f} "
        f"+/- {summary['agentProfit']['sd']:.4f} |",
        f"| regret | {summary['regret']['mean']:.4f} "
        f"+/- {summary['regret']['sd']:.4f} |",
        f"| captured | {summary['captured']['mean']:.2%} "
        f"+/- {summary['captured']['sd']:.2%} |",
        "",
        "| seed | agent profit | regret | captured |",
        "|---|---|---|---|",
    ]
    for seed, records in sorted(results.items()):
        one = evaluate.summarise(records)
        lines.append(f"| {seed} | {one['agentProfit']:.4f} | {one['regret']:.4f} "
                     f"| {one['captured']:.2%} |")
    lines.append("")
    return heading, "\n".join(lines) + "\n"


def _spliceSection(path, heading, body):
    """Insert or replace one `## heading` section in a markdown file; the rest of the
    file, including any other region's or price source's section, is left untouched."""
    target = Path(path)
    text = target.read_text() if target.exists() else ""
    marker = f"## {heading}\n"
    start = text.find(marker)
    if start == -1:
        if text and not text.endswith("\n"):
            text += "\n"
        text += marker + body
    else:
        end = text.find("\n## ", start + len(marker))
        end = len(text) if end == -1 else end + 1
        text = text[:start] + marker + body + text[end:]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cli", description=__doc__.splitlines()[0],
                                     formatter_class=_formatter)
    subcommands = parser.add_subparsers(dest="module", required=True)

    def addParser(name, helpText):
        return subcommands.add_parser(name, parents=[_sharedParser()],
                                      formatter_class=_formatter, help=helpText)

    def addSolverArgs(parser, gapDefault=None, limitDefault=None):
        parser.add_argument("--no-calendar", action="store_true",
                            help="drop calendar aging; roughly 5x faster")
        parser.add_argument("--gap", type=float, default=gapDefault, metavar="FRACTION",
                            help="relative MIP gap to stop at; HiGHS defaults to 0.0001, "
                                 "which costs 1555s on one discrete day against 82s at "
                                 f"0.005 (default: {gapDefault})")
        parser.add_argument("--time-limit", type=float, default=limitDefault,
                            metavar="SECONDS",
                            help="cap each solve; past it the incumbent is kept and the "
                                 "dual bound is still valid, but it is also loose, and "
                                 "the cache key does not record the limit, so a loose "
                                 f"record is permanent (default: {limitDefault})")

    def addRefreshArg(parser):
        minutes = config.forecast_refresh_intervals * 5
        parser.add_argument("--forecast-refresh", type=int, default=minutes,
                            metavar="MINUTES",
                            help="replace the forecast with AEMO's latest run this often "
                                 "within an episode, from its 04:00 start; 5 is every "
                                 f"interval, 1440 one run per day (default: {minutes})")

    def addRlArgs(parser, oracleDefault=None, forecastFlag=True, algoList=False):
        parser.add_argument("--algo", default=["maskableppo"] if algoList else "maskableppo",
                            nargs="+" if algoList else None,
                            choices=list(config.algorithms),
                            help="MaskablePPO is the only one that reads the action mask "
                                 "(default: maskableppo)")
        parser.add_argument("--seeds", type=int, nargs="+", default=list(config.seeds),
                            metavar="N",
                            help=f"one run per seed; results are reported as mean +/- SD "
                                 f"(default: {' '.join(str(s) for s in config.seeds)})")
        parser.add_argument("--horizon", type=int, default=config.episode_intervals,
                            metavar="N",
                            help=f"intervals per episode, and per oracle window; the two "
                                 f"must match (default: {config.episode_intervals})")
        parser.add_argument("--reward-scale", type=float, default=config.reward_scale,
                            metavar="X",
                            help=f"fixed divisor on the reward (default: {config.reward_scale})")
        parser.add_argument("--out", default="runs", metavar="DIR",
                            help="where checkpoints live (default: runs)")
        parser.add_argument("--policies", default=str(policies.defaultRoot),
                            metavar="DIR",
                            help=f"where the final trained agent of each run is "
                                 f"published, and where the dashboard reads it "
                                 f"(default: {policies.defaultRoot})")
        parser.add_argument("--verbose", action="store_true",
                            help="show an elapsed-time ticker per run")
        if oracleDefault is None:
            return  # train: the reward is raw profit, so no oracle is consulted
        parser.add_argument("--workers", type=int, default=1, metavar="N",
                            help="parallel oracle solves when the cache is cold "
                                 "(default: 1)")
        parser.add_argument("--baseline-oracle", choices=["continuous", "discrete"],
                            default=oracleDefault,
                            help="which oracle supplies the per-interval profit and the "
                                 "bound; a quoted bound must be discrete "
                                 f"(default: {oracleDefault})")
        if forecastFlag:
            parser.add_argument("--forecast", action="store_true",
                                help="fill the price slots from the AEMO forecast "
                                     "published at each episode's run time, instead of "
                                     "realised prices; settlement still uses realised "
                                     "prices")
            addRefreshArg(parser)

    lp = addParser("lp", "perfect-foresight MILP oracle")
    addSolverArgs(lp)
    lp.add_argument("--discrete-power", action="store_true",
                   help="restrict every interval to full rate or idle, which is the RL "
                        "agent's action set; this is the bound the agent is measured "
                        "against, and it is much slower")
    lp.set_defaults(run=_runLp)

    baselineCmd = addParser("baseline", "solve and cache one oracle schedule per window")
    addSolverArgs(baselineCmd, gapDefault=config.oracle_gap)
    baselineCmd.add_argument("--horizon", type=int, default=config.episode_intervals,
                             metavar="N",
                             help=f"intervals per window (default: {config.episode_intervals})")
    baselineCmd.add_argument("--workers", type=int, default=1, metavar="N",
                             help="solve windows in parallel; safe only because the SoC "
                                  "resets at every boundary (default: 1)")
    baselineCmd.add_argument("--continuous", action="store_true",
                             help="unrestricted power; far faster, but not a fair bound "
                                  "for a three-action agent")
    baselineCmd.add_argument("--verbose", action="store_true",
                             help="print each window as it finishes")
    baselineCmd.set_defaults(run=_runBaseline)

    train = addParser("train", "train an agent, one run per seed")
    train.add_argument("--no-calendar", action="store_true",
                       help="drop calendar aging from the environment's degradation cost")
    addRlArgs(train)
    train.add_argument("--timesteps", type=int, default=100000, metavar="N",
                       help="environment steps per seed (default: 100000)")
    train.add_argument("--cap-date", type=datetime.fromisoformat, default=None,
                       metavar="DATE",
                       help="pin the observation price scale to the AEMC market price "
                            "cap in force at DATE, instead of at the end of the training "
                            "window. Pass the end of the window you will score on to "
                            "keep full resolution over prices above the training year's "
                            "cap. The scale is saved beside the checkpoint either way")
    train.set_defaults(run=_runTrain)

    evalCmd = addParser("eval", "score trained agents against the oracle")
    addSolverArgs(evalCmd, gapDefault=config.oracle_gap)
    addRlArgs(evalCmd, "discrete" if config.bound_discrete_power else "continuous")
    evalCmd.add_argument("--heuristic", action="store_true",
                         help="score the rule-based threshold policy instead of a "
                              "checkpoint")
    evalCmd.add_argument("--quantiles", type=float, nargs=2, default=[0.15, 0.85],
                         metavar=("LOW", "HIGH"),
                         help="charge at or below LOW of the forward window, discharge at "
                              "or above HIGH (default: 0.15 0.85)")
    evalCmd.add_argument("--result-file", metavar="PATH",
                         help="write/update one markdown section at PATH, keyed by "
                              "realised vs forecast; other sections already in the file "
                              "are left untouched")
    evalCmd.set_defaults(run=_runEval)

    rolling = addParser("rolling", "walk-forward backtest: slide the training window, "
                                   "test the next week, repeat")
    addSolverArgs(rolling, gapDefault=config.backtest_gap,
                  limitDefault=config.backtest_time_limit)
    addRlArgs(rolling, "discrete", forecastFlag=False, algoList=True)
    rolling.add_argument("--train-weeks", type=int, required=True, metavar="N",
                         help="weeks of training data per fold; the window slides rather "
                              "than grows")
    rolling.add_argument("--test-weeks", type=int, default=1, metavar="N",
                         help="weeks scored per fold (default: 1)")
    rolling.add_argument("--step-weeks", type=int, default=1, metavar="N",
                         help="weeks between fold origins (default: 1)")
    rolling.add_argument("--cap-date", type=datetime.fromisoformat, required=True,
                         metavar="DATE",
                         help="pin the observation price scale to the market price cap in "
                              "force then; required, because a per-fold cap would make "
                              "folds either side of 1 July incomparable")
    rolling.add_argument("--train-modes", dest="trainModes", nargs="+",
                         default=["warm"], choices=["warm", "cold"],
                         help="warm continues the previous fold's checkpoint, cold starts "
                              "fresh each fold (default: warm)")
    rolling.add_argument("--warm-timesteps", type=int, default=30000, metavar="N",
                         help="top-up budget per warm fold (default: 30000)")
    rolling.add_argument("--cold-timesteps", type=int, default=300000, metavar="N",
                         help="budget for a fresh fit, also used for the first fold of a "
                              "warm chain (default: 300000)")
    rolling.add_argument("--price-source", choices=["realised", "forecast", "both"],
                         default="realised",
                         help="what fills the observation's price slots; settlement always "
                              "uses realised prices (default: realised)")
    addRefreshArg(rolling)
    rolling.add_argument("--results", default="results/rolling", metavar="DIR",
                         help="where the per-fold CSV and parquet go "
                              "(default: results/rolling)")
    rolling.add_argument("--trajectory-seed", type=int, default=0, metavar="N",
                         help="persist per-interval trajectories for this seed; -1 for "
                              "none (default: 0)")
    rolling.add_argument("--folds", metavar="SPEC",
                         help="restrict to a subset, as N or N-M, comma separated")
    rolling.add_argument("--allow-solve", action="store_true",
                         help="permit solving oracle windows that are not cached; without "
                              "it a cold cache stops the run instead of silently starting "
                              "hours of MILP")
    rolling.add_argument("--dry-run", action="store_true",
                         help="print the fold table and the cache audit, run nothing")
    # The cached oracle was solved without calendar aging, so matching it is the
    # default here and opting back in costs a full re-solve.
    rolling.add_argument("--calendar", dest="no_calendar", action="store_false",
                         help="keep calendar aging; every oracle window then misses the "
                              "cache and must be re-solved")
    rolling.set_defaults(run=_runRolling, out="runs/rolling", no_calendar=True)

    # Not addParser: this command takes directories, not a window, and the shared
    # parser's optional start/end positionals would swallow the first one.
    publish = subcommands.add_parser(
        "publish", formatter_class=_formatter,
        help="put an existing checkpoint under trainedPolicies/")
    publish.add_argument("checkpoint", nargs="+", metavar="DIR",
                         help="one training output directory per seed, each holding "
                              "agent.zip and scaler.json, in the order of --seeds")
    publish.add_argument("--region", default=config.default_region, metavar="ID",
                         help=f"NEM region the agent was trained for "
                              f"(default: {config.default_region})")
    publish.add_argument("--algo", default="maskableppo",
                         choices=list(config.algorithms),
                         help="which algorithm trained it (default: maskableppo)")
    publish.add_argument("--train-mode", default="warm", choices=["warm", "cold"],
                         help="warm continued an earlier checkpoint, cold started fresh "
                              "(default: warm)")
    publish.add_argument("--seeds", type=int, nargs="+", default=[0], metavar="N",
                         help="the seed each DIR was trained on (default: 0)")
    publish.add_argument("--policies", default=str(policies.defaultRoot), metavar="DIR",
                         help=f"where to publish (default: {policies.defaultRoot})")
    publish.set_defaults(run=_runPublish)

    prefetch = addParser("prefetch", "cache the AEMO forecast paths for a region")
    addRefreshArg(prefetch)
    prefetch.set_defaults(run=_runPrefetch)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

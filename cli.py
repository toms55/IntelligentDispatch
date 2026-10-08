"""Run an optimisation or RL module over a slice of NEM prices, until the dashboard exists.

    python -m cli lp 2024-01-10 2024-01-11 --no-calendar --csv day.csv
    python -m cli baseline --split train --workers 8
    python -m cli train --split train --algo maskableppo --seed 0
    python -m cli eval  --split val   --algo maskableppo

`lp` is the oracle. `baseline` solves and caches one oracle schedule per episode window,
which the reward subtracts. `train` and `eval` are the agent.

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
    from rl import baseline
    from rl.features import PriceScaler, scalerFor
    from rl.train import trainOne
    from utils.market import marketPriceCap

    prices = _prices(args)
    model = _battery(args)
    scaler = (PriceScaler(marketPriceCap(args.cap_date)) if args.cap_date
              else scalerFor(prices.index))
    print(f"price scale     cap {scaler.cap:.0f}, knee {scaler.knee:.0f}")
    baselines = None
    if not args.no_baseline:
        # The baseline only has to be a constant per window/interval, so it need not
        # share the environment's battery model -- dropping Q^CAL here is much cheaper.
        baselines = baseline.profitPerInterval(baseline.buildBaseline(
            prices, region=args.region, horizon=args.horizon,
            calendar=not (args.no_calendar or args.baseline_no_calendar),
            model=model, initialSoc=args.initial_soc,
            discretePower=args.baseline_oracle == "discrete",
            gap=args.gap, timeLimit=args.time_limit, workers=args.workers,
        ))
    for seed in args.seeds:
        with _stage(f"train {args.algo} seed {seed}", quiet=not args.verbose):
            _, path = trainOne(
                args.algo, prices, model, baselines=baselines, seed=seed,
                timesteps=args.timesteps, calendar=not args.no_calendar,
                horizon=args.horizon, rewardScale=args.reward_scale, outDir=args.out,
                scaler=scaler,
            )
        print(f"wrote          {path}.zip")
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
        from data.forecasts import forecastMatrix
        from rl.baseline import chunkWindows

        # Run time is one interval before the episode's first interval.
        runTimes = [
            prices.index[start] - nem_interval for start in
            chunkWindows(len(prices), args.horizon)
        ]
        forecasts = forecastMatrix(runTimes, args.region, args.horizon)

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


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cli", description=__doc__.splitlines()[0],
                                     formatter_class=_formatter)
    subcommands = parser.add_subparsers(dest="module", required=True)

    def addParser(name, helpText):
        return subcommands.add_parser(name, parents=[_sharedParser()],
                                      formatter_class=_formatter, help=helpText)

    def addSolverArgs(parser, gapDefault=None):
        parser.add_argument("--no-calendar", action="store_true",
                            help="drop calendar aging; roughly 5x faster")
        parser.add_argument("--gap", type=float, default=gapDefault, metavar="FRACTION",
                            help="relative MIP gap to stop at; HiGHS defaults to 0.0001, "
                                 "which costs 1555s on one discrete day against 82s at "
                                 f"0.005 (default: {gapDefault})")
        parser.add_argument("--time-limit", type=float, default=None, metavar="SECONDS",
                            help="cap each solve; past it the incumbent is kept and the "
                                 "dual bound is still valid")

    def addRlArgs(parser, oracleDefault):
        parser.add_argument("--algo", default="maskableppo",
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
        parser.add_argument("--workers", type=int, default=1, metavar="N",
                            help="parallel oracle solves when the cache is cold "
                                 "(default: 1)")
        parser.add_argument("--verbose", action="store_true",
                            help="show an elapsed-time ticker per run")
        parser.add_argument("--baseline-oracle", choices=["continuous", "discrete"],
                            default=oracleDefault,
                            help="which oracle supplies the per-interval profit; the "
                                 "reward baseline can use continuous, a quoted bound must "
                                 f"be discrete (default: {oracleDefault})")
        parser.add_argument("--forecast", action="store_true",
                            help="fill the price slots from the AEMO forecast published "
                                 "at each episode's run time, instead of realised prices; "
                                 "settlement still uses realised prices")

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
    addSolverArgs(train, gapDefault=config.oracle_gap)
    addRlArgs(train, "continuous" if not config.baseline_discrete_power else "discrete")
    train.add_argument("--timesteps", type=int, default=100000, metavar="N",
                       help="environment steps per seed (default: 100000)")
    train.add_argument("--cap-date", type=datetime.fromisoformat, default=None,
                       metavar="DATE",
                       help="pin the observation price scale to the AEMC market price "
                            "cap in force at DATE, instead of at the end of the training "
                            "window. Pass the end of the window you will score on to "
                            "keep full resolution over prices above the training year's "
                            "cap. The scale is saved beside the checkpoint either way")
    train.add_argument("--no-baseline", action="store_true",
                       help="reward is raw profit, with no oracle baseline subtracted")
    train.add_argument("--baseline-no-calendar", action="store_true",
                       help="drop calendar aging from the baseline solve only, leaving it "
                            "on in the environment")
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
    evalCmd.set_defaults(run=_runEval)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

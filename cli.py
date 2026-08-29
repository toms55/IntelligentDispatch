"""Run an optimisation module over a slice of NEM prices, until the dashboard exists.

    python -m cli lp 2024-01-10 2024-01-11 --no-calendar --csv day.csv

`rl` joins as a second subcommand at TODO.txt step 2; the shared arguments below
are already what it will need.
"""
import argparse
import contextlib
import sys
import threading
import time
from datetime import datetime

import pandas as pd

from data.prices import price_series
from degradation.schade import SchadeDegradation
from optimisation.lp import defaultSolver, solve


DEFAULT_PATH = ""  # --csv / --plot given without a PATH; filled in by _outputPath


def _formatter(prog):
    """Wider than argparse's default, so no option name wraps off its description."""
    return argparse.HelpFormatter(prog, max_help_position=28)


@contextlib.contextmanager
def _stage(label, quiet):
    """Announce a step on stderr and tick the elapsed seconds while it runs.

    Progress goes to stderr so that stdout stays a clean report. A MILP gives no
    intermediate signal, so an elapsed counter is the honest thing to show: it
    says the process is alive without pretending to know how far along it is.
    """
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
        # \r plus pad, to wipe the longest line the ticker could have left.
        prefix = "\r" if live else ""
        print(f"{prefix}  {label}... {elapsed:.1f}s{' ' * 8}", file=sys.stderr, flush=True)


def _sharedParser():
    """Arguments common to every module: the window, the market, the battery."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("start", type=datetime.fromisoformat,
                        help="start of the window, ISO date or datetime "
                             "(e.g. 2024-01-10 or 2024-01-10T06:00)")
    parser.add_argument("end", type=datetime.fromisoformat,
                        help="end of the window, exclusive, same format as start")
    parser.add_argument("--region", default="SA1", metavar="ID",
                        help="NEM region: SA1, NSW1, VIC1, QLD1, TAS1 (default: SA1)")
    # Battery defaults are the Schade 2024 Table 2 case.
    parser.add_argument("--mw", type=float, default=0.060,
                        help="charge/discharge power limit, in MW (default: 0.060)")
    parser.add_argument("--mwh", type=float, default=0.100,
                        help="usable energy capacity, in MWh (default: 0.100)")
    parser.add_argument("--replacement-cost", type=float, default=150_000.0, metavar="COST",
                        help="pack replacement, in $/MWh of capacity (default: 150000)")
    parser.add_argument("--initial-soc", type=float, default=0.5, metavar="SOC",
                        help="state of charge before the first interval, 0-1 (default: 0.5)")
    parser.add_argument("--csv", nargs="?", const=DEFAULT_PATH, metavar="PATH",
                        help="write the per-interval trajectory as CSV; "
                             "PATH defaults to <module>_<region>_<start>.csv in the cwd")
    parser.add_argument("--plot", nargs="?", const=DEFAULT_PATH, metavar="PATH",
                        help="save the price/SoC/dispatch figure; "
                             "PATH defaults to <module>_<region>_<start>.png in the cwd")
    parser.add_argument("--show", action="store_true",
                        help="display the figure in a window instead of saving it")
    return parser


def _battery(args):
    return SchadeDegradation(args.mw, args.mwh, args.replacement_cost)


def _outputPath(args, suffix):
    """<module>_<region>_<start>.<suffix>, e.g. lp_SA1_2024-01-10.csv."""
    return f"{args.module}_{args.region}_{args.start.date()}.{suffix}"


def _frame(prices, schedule):
    """One row per interval, timestamps from the price index."""
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
    """Price above, SoC and net dispatch below. Imported late: matplotlib is slow.

    destination is a path to save to, or None to open a window.
    """
    import matplotlib
    if destination is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (top, bottom) = plt.subplots(2, sharex=True, figsize=(12, 7))
    time = frame["settlementdate"]

    top.plot(time, frame["price"], linewidth=0.9)
    top.set_ylabel("$/MWh")
    top.axhline(0, color="grey", linewidth=0.5)

    # Signed so one bar series reads as charge below the line, discharge above.
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

    # Not truthiness: an empty string is the "given without a PATH" sentinel.
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
    prices = price_series(args.start, args.end, args.region)
    schedule = solve(
        prices.to_list(),
        _battery(args),
        initialSoc=args.initial_soc,
        calendar=not args.no_calendar,
    )
    _report(prices, schedule, args)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cli", description=__doc__.splitlines()[0],
                                     formatter_class=_formatter)
    subcommands = parser.add_subparsers(dest="module", required=True)

    lp = subcommands.add_parser("lp", parents=[_sharedParser()],
                                formatter_class=_formatter,
                                help="perfect-foresight MILP oracle")
    lp.add_argument("--no-calendar", action="store_true",
                    help="drop calendar aging; roughly 5x faster")
    lp.set_defaults(run=_runLp)

    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())

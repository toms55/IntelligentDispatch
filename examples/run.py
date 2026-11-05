"""Score a policy over a window, on realised or forecast prices.

    uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast
    uv run python -m examples.run rl.heuristic:ThresholdPolicy NSW1 2025-03-03 2025-03-10 realised

The policy is `module:Class`. The class takes no arguments.
"""
import argparse
import importlib
import json
from datetime import datetime

import config
from backtest.score import score


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("policy", help="module:Class of the policy")
    parser.add_argument("region", help="SA1, NSW1, VIC1, QLD1 or TAS1")
    parser.add_argument("start", help="first day, ISO date; the window opens at 04:00")
    parser.add_argument("end", help="last day, ISO date; the window closes at 04:00")
    parser.add_argument("source", choices=("realised", "forecast"),
                        help="what fills the price slots of the observation")
    parser.add_argument("--cycle-budget", type=float, default=config.cycle_budget,
                        help="equivalent full cycles per episode")
    parser.add_argument("--no-cycle-budget", action="store_true",
                        help="no cycle limit")
    parser.add_argument("--terminal-price", type=float, default=0.0,
                        help="$/MWh the pack is worth at the close")
    parser.add_argument("--initial-soc", type=float, default=config.initial_soc,
                        help="state of charge at the start of each episode, 0 to 1")
    parser.add_argument("--end-empty", action="store_true",
                        help="the pack must be empty at the close")
    parser.add_argument("--forecast-refresh", type=int,
                        default=config.forecast_refresh_intervals * 5, metavar="MINUTES",
                        help="how often the forecast is replaced by the latest AEMO run; "
                             "use the same value for `cli prefetch`")
    args = parser.parse_args()

    module, _, name = args.policy.partition(":")
    policy = getattr(importlib.import_module(module), name)()
    at4 = lambda day: datetime.fromisoformat(day).replace(hour=4)
    result = score(
        policy, args.region, at4(args.start), at4(args.end), args.source,
        cycleBudget=None if args.no_cycle_budget else args.cycle_budget,
        terminalPrice=args.terminal_price, initialSoc=args.initial_soc,
        endEmpty=args.end_empty, forecastEvery=args.forecast_refresh // 5)
    print(json.dumps({"summary": result["summary"], "settings": result["settings"]},
                     indent=2))


if __name__ == "__main__":
    main()

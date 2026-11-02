"""Rank the seeds of the dashboard runs, per run and region, and print the pinned map.

    uv run python scripts/rankSeeds.py [RUN ...]

The rule: the seed with the highest total agentProfit over the run's test days, on
forecast prices, because forecast is what both screens show. A run that scored no
forecast prices is ranked on realised prices instead, and the table says so.

The runs default to `config.dashboard_runs`. Paste the printed map into
`config.best_seed_by_run`, and re-run this whenever a run joins that list or a run is
published again. Each train mode is ranked on its own, because a pooled run's shared
agent and its regional fine-tune are two agents and need not agree on a seed.

Picking the top of five seeds on the same days the dashboard then draws is selection on
the test set. It is a presentational choice, not a result. Quote the spread over seeds in
the thesis, never this one number.
"""
import sys

import pandas as pd

# The repo root, so `import config` works when this file is run by path rather than as a
# module. conftest.py does the same job for the tests.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import config  # noqa: E402 - after the path above

runs = sys.argv[1:] or list(config.dashboard_runs)
regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")
head = ("region", "mode", "source", "days", "best", "best total", "seed 0", "gain",
        "spread", "worst")

best = {}
for run in runs:
    print(f"\n{run}\n")
    print("%-7s %-9s %-9s %5s %5s %12s %12s %10s %10s %6s" % head)
    picks = {}
    for region in regions:
        frame = pd.read_csv(f"results/backtestResults/{run}_{region}.csv")
        sources = set(frame.priceSource)
        source = "forecast" if "forecast" in sources else sorted(sources)[0]
        rows = frame[frame.priceSource == source]
        for mode in sorted(set(rows.trainMode)):
            totals = (rows[rows.trainMode == mode]
                      .groupby("seed").agentProfit.agg(["sum", "size"]))
            assert totals["size"].nunique() == 1, \
                f"{run} {region} {mode}: seeds cover different days"
            top = int(totals["sum"].idxmax())
            picks.setdefault(mode, {})[region] = top
            print("%-7s %-9s %-9s %5d s%-4d %12.2f %12.2f %+10.2f %10.2f %6s" % (
                region, mode, source, int(totals["size"].iloc[0]), top,
                totals["sum"][top], totals["sum"][0],
                totals["sum"][top] - totals["sum"][0],
                totals["sum"].max() - totals["sum"].min(),
                f"s{int(totals['sum'].idxmin())}"))
    best[run] = picks

print("\n\nbest_seed_by_run = {")
for run, modes in best.items():
    print(f'    "{run}": {{')
    for mode, picked in sorted(modes.items()):
        print(f'        "{mode}": ' + repr(picked) + ",")
    print("    },")
print("}")

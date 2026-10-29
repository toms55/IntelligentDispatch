"""Rank the seeds of a published run, per region, and print the `config.best_seed` map.

    uv run python scripts/rankSeeds.py [RUN]

The rule: the seed with the highest total agentProfit over the run's test days, on
forecast prices, because forecast is what the hosted dashboard shows. The run defaults
to the one scripts/publishSpace.sh opens the Space on.

The winner is specific to the run. Re-run this and update `config.best_seed` whenever
the Space opens on another run, or the map describes an experiment the Space no longer
shows. The second table prints the winner on every published run, so the spread is
visible: only QLD1 holds its seed across the three profit runs.

Picking the top of five seeds on the same days the dashboard then draws is selection on
the test set. It is a presentational choice, not a result. Quote the spread over seeds
in the thesis, never this one number.
"""
import sys

import pandas as pd

run = sys.argv[1] if len(sys.argv) > 1 else "rolling2023_profit_fc5m"
regions = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")

print(f"{run}, forecast prices\n")
head = ("region", "days", "best", "best total", "seed 0", "gain", "spread", "worst")
print("%-7s %5s %5s %11s %11s %9s %9s %8s" % head)

best = {}
for region in regions:
    frame = pd.read_csv(f"results/backtestResults/{run}_{region}.csv")
    rows = frame[frame.priceSource == "forecast"]
    totals = rows.groupby("seed").agentProfit.agg(["sum", "size"])
    assert totals["size"].nunique() == 1, f"{region}: seeds cover different days"
    top = int(totals["sum"].idxmax())
    best[region] = top
    spread = totals["sum"].max() - totals["sum"].min()
    print("%-7s %5d s%-4d %11.2f %11.2f %+9.2f %9.2f %8s" % (
        region, int(totals["size"].iloc[0]), top, totals["sum"][top],
        totals["sum"][0], totals["sum"][top] - totals["sum"][0], spread,
        f"s{int(totals['sum'].idxmin())}"))

print("\nbest_seed = " + repr(best))

# Does the ranking hold on the other runs, or is it an artefact of this one?
print("\nbest total per region, every published run")
print("%-30s %s" % ("run", "  ".join(f"{r:>6}" for r in regions)))
for other in ("rolling2023_profit", "rolling2023_profit_fc4h", "rolling2023_profit_fc5m",
              "rolling2023_fc4h", "rolling2023_fctrain", "rolling2023_profit_cold_fc4h"):
    picks = []
    for region in regions:
        frame = pd.read_csv(f"results/backtestResults/{other}_{region}.csv")
        rows = frame[frame.priceSource == "forecast"]
        picks.append(f"s{int(rows.groupby('seed').agentProfit.sum().idxmax()):<5d}")
    print("%-30s %s" % (other, "  ".join(f"{p:>6}" for p in picks)))

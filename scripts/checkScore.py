"""Check that backtest.score reproduces a published run, to the cent.

    python scripts/checkScore.py results/backtestResults/<run>_<REGION> trainedPolicies/<REGION>/<agent>

A rolling run publishes its last fold's agent. That agent's test days are already in the
run's rows, scored by the rolling backtest. This scores the same agent over the same
days through `backtest.score` instead, under the settings the run's manifest states, and
compares day by day. The agent columns (profit, revenue, degradation, cycles) must agree
to 1e-9, since both paths drive the same env with the same weights. The oracle bound is
reported but not asserted: a run published before the exact DP carries a MILP dual bound,
which sits at or above the DP optimum this recomputes.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.score import score  # noqa: E402
from rl import algos  # noqa: E402
from rl.train import loadScaler  # noqa: E402

tolerance = 1e-9
columns = {"agentProfit": "profit", "agentRevenue": "revenue",
           "agentDegradation": "degradation", "agentCycles": "cycles"}


class Trained:
    """A published SB3 agent as a policy."""

    def __init__(self, agent, readsMask):
        self.agent, self.reads_mask = agent, readsMask

    def predict(self, observation, deterministic=True, action_masks=None):
        if self.reads_mask:
            return self.agent.predict(observation, deterministic=deterministic,
                                      action_masks=action_masks)
        return self.agent.predict(observation, deterministic=deterministic)


def main(runStem, policyDir):
    rows = pd.read_csv(f"{runStem}.csv", parse_dates=["testDay", "episodeStart"])
    manifest = json.loads(Path(f"{runStem}.json").read_text())["manifest"]
    policy = json.loads((Path(policyDir) / "policy.json").read_text())
    start, end = (datetime.fromisoformat(day) for day in policy["testWindow"])

    agent = algos.load(policy["algo"], str(Path(policyDir) / "agent"))
    trained = Trained(agent, algos.supportsMasking(policy["algo"]))
    scaler = loadScaler(policyDir)

    worst = 0.0
    for source in manifest["priceSources"]:
        # The manifest, not policy.json: policy.json holds the training machine's config.
        result = score(
            trained, policy["region"], start, end, source,
            horizon=manifest["horizon"], cycleBudget=manifest["cycleBudget"],
            terminalPrice=manifest.get("terminalPrice", 0.0),
            initialSoc=manifest["initialSoc"], endEmpty=manifest["endEmpty"],
            calendar=manifest["calendar"], battery=tuple(manifest["battery"]),
            forecastEvery=manifest["forecastEvery"], scaler=scaler)

        published = rows[(rows.fold == policy["fold"]) & (rows.seed == policy["seed"])
                         & (rows.priceSource == source)].set_index("testDay").sort_index()
        episode = pd.Timedelta(minutes=5 * manifest["horizon"])
        scored = pd.DataFrame([
            {"testDay": start + record["window"] * episode + offset * pd.Timedelta(days=1),
             **day}
            for record in result["records"] for offset, day in enumerate(record["perDay"])
        ]).set_index("testDay").sort_index()
        if not published.index.equals(scored.index):
            sys.exit(f"{source}: published days {list(published.index)} differ from "
                     f"scored days {list(scored.index)}")

        print(f"\n{source}: {len(scored)} days, fold {policy['fold']}, seed {policy['seed']}")
        for published_name, scored_name in columns.items():
            gap = float(np.abs(published[published_name] - scored[scored_name]).max())
            worst = max(worst, gap)
            print(f"  {published_name:18s} published {published[published_name].sum():12.6f}"
                  f"  scored {scored[scored_name].sum():12.6f}  max |diff| {gap:.3e}")
        bound = np.array([record["lpProfit"] for record in result["records"]])
        shown = published.oracleDualBound.to_numpy()
        print(f"  oracle bound       published {np.nansum(shown):12.6f}  scored "
              f"{bound.sum():12.6f}  max |diff| {np.nanmax(np.abs(shown - bound)):.3e} "
              f"(not asserted)")

    print(f"\nlargest agent-column difference {worst:.3e} (limit {tolerance:.0e})")
    if worst > tolerance:
        sys.exit("FAIL: backtest.score does not reproduce the published rows")
    print("PASS: backtest.score reproduces the published rows")


if __name__ == "__main__":
    main(*sys.argv[1:3])

"""Score a trained policy against the oracle, window by window.

Regret is the oracle's profit minus the agent's on the same window. Running the same
agent twice -- once with realised prices, once with the AEMO forecast -- isolates the
cost of forecast error from policy quality.
"""
import statistics
from pathlib import Path

import numpy as np

import config
from rl import algos
from rl.env import BessEnv
from rl.train import loadScaler


def rolloutWindow(agent, env, windowIndex, masking=False, bound=None):
    """One deterministic episode on one window."""
    obs, _ = env.reset(options={"window": windowIndex})
    trajectory = []
    info = {}
    while True:
        if masking:
            action, _ = agent.predict(obs, deterministic=True,
                                      action_masks=env.action_masks())
        else:
            action, _ = agent.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        trajectory.append(info)
        if terminated or truncated:
            break
    totals = info["episode_totals"]
    # bound is the oracle's dual bound; the summed per-interval baseline can contribute
    # zeros from a timed-out window.
    lpProfit = totals["baseline"] if bound is None else bound
    return {
        "window": windowIndex,
        "agentProfit": totals["profit"],
        "agentRevenue": totals["revenue"],
        "agentDegradation": totals["degradation"],
        "agentCycles": totals["cycles"],
        "agentTerminalValue": totals["terminalValue"],
        "lpProfit": lpProfit,
        "regret": lpProfit - totals["profit"],
        "perDay": perDay(trajectory),
        "trajectory": trajectory,
    }


def perDay(trajectory, day=None):
    """One entry per trading day inside the episode, in order.

    The episode is the scoring grain and the trading day is the reporting grain, so a
    weekly episode still writes seven result rows. Computed here rather than in the
    backtest because this is where the per-interval infos are.

    `cycles` is the EFC spent inside that day. The environment reports a running total,
    so a day's own spend is the difference across it, and the seven days sum to the
    episode. The terminal value lands on the last day, where the environment paid it.
    """
    day = config.observation_intervals if day is None else day
    out = []
    for first in range(0, len(trajectory), day):
        block = trajectory[first:first + day]
        before = trajectory[first - 1]["cycles"] if first else 0.0
        out.append({
            "revenue": sum(step["revenue"] for step in block),
            "degradation": sum(step["degradation"] for step in block),
            "profit": sum(step["profit"] + step["terminalValue"] for step in block),
            "cycles": block[-1]["cycles"] - before,
            "terminalValue": sum(step["terminalValue"] for step in block),
        })
    return out


def evaluatePolicy(agent, env, algoName=None, masking=None, bounds=None):
    if masking is None:
        masking = algos.supportsMasking(algoName) if algoName else False
    return [
        rolloutWindow(agent, env, index, masking=masking,
                      bound=None if bounds is None else bounds[index])
        for index in range(len(env.windows))
    ]


def evaluateSeeds(checkpoints, prices, model, baselines=None, algoName="maskableppo",
                  calendar=True, horizon=None, rewardScale=None, forecasts=None,
                  bounds=None, initialSoc=config.initial_soc,
                  cycleBudget=config.cycle_budget, terminalPrice=0.0):
    """One evaluation per seed, over the same windows. Returns {seed: records}.

    Each env uses the price transform its own checkpoint trained under, read from
    scaler.json. Never the transform the evaluation window would pin: the market price
    cap moves between years, and a different divisor changes what every observation
    means.
    """
    extra = {}
    if horizon is not None:
        extra["horizon"] = horizon
    if rewardScale is not None:
        extra["rewardScale"] = rewardScale
    results = {}
    for seed, path in checkpoints.items():
        env = BessEnv(prices, model, baseline=baselines, calendar=calendar,
                      forecasts=forecasts, scaler=loadScaler(Path(path).parent),
                      initialSoc=initialSoc, cycleBudget=cycleBudget,
                      terminalPrice=terminalPrice, **extra)
        agent = algos.load(algoName, str(path), env=None)
        results[seed] = evaluatePolicy(agent, env, algoName=algoName, bounds=bounds)
    return results


def summarise(records):
    agentProfit = sum(r["agentProfit"] for r in records)
    lpProfit = sum(r["lpProfit"] for r in records)
    return {
        "windows": len(records),
        "agentProfit": agentProfit,
        "lpProfit": lpProfit,
        "regret": lpProfit - agentProfit,
        "captured": agentProfit / lpProfit if lpProfit else float("nan"),
    }


def acrossSeeds(resultsBySeed):
    """Mean and standard deviation over seeds."""
    summaries = {seed: summarise(records) for seed, records in resultsBySeed.items()}
    out = {"perSeed": summaries}
    for field in ("agentProfit", "regret", "captured"):
        values = [summary[field] for summary in summaries.values()]
        out[field] = {
            "mean": statistics.fmean(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    out["lpProfit"] = next(iter(summaries.values()))["lpProfit"] if summaries else 0.0
    return out


def randomPolicyProfit(env, seed=0, episodes=5):
    """A random-action control, to catch a policy that has learned nothing."""
    rng = np.random.default_rng(seed)
    totals = []
    for index in range(min(episodes, len(env.windows))):
        env.reset(options={"window": index})
        info = {}
        while True:
            mask = env.action_masks()
            allowed = np.flatnonzero(mask)
            _, _, terminated, truncated, info = env.step(int(rng.choice(allowed)))
            if terminated or truncated:
                break
        totals.append(info["episode_totals"]["profit"])
    return totals

"""The current forecast screen: the AEMO path published now, and the dispatch a policy
would run against it.

The backtest screen reads finished folds off disk. This one reads nothing off disk but
the policy itself. It asks AEMO for the forward price path at this minute, then rolls
one episode of `rl.env.BessEnv` over that path, so the action set, the degradation terms
and the state of charge rule are the same ones the backtest scored.

Two things differ from a backtest day, and both are stated on the screen.

1. The episode starts at the run time, not at 04:00, so it is as long as the forecast
   reaches. PREDISPATCH stops at the end of the trading day.
2. The state of charge starts at `config.initial_soc`, because the live state of a real
   battery is not in this repository.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data import forecasts
from degradation.schade import SchadeDegradation
from rl import algos, policies
from rl.env import BessEnv
from rl.evaluate import rolloutWindow
from rl.train import loadScaler
from utils import nem_interval

defaultCheckpoints = policies.defaultRoot

columns = ("settlementdate", "price", "action", "soc", "revenue", "profit")


def findCheckpoints(region, root=defaultCheckpoints):
    """Every agent published for `region`, as {label: directory}.

    `rl/policies.py` owns the layout. Training publishes its final agent there, and
    this screen reads it, so a machine that only runs the dashboard needs that one
    directory copied across and nothing else.
    """
    return policies.available(region, root)


def forecastPath(region, runTime=None, horizon=config.episode_intervals):
    """The forward price path AEMO publishes now, cut where the path stops.

    PREDISPATCH reaches the end of the trading day and no further, so a run late in the
    day returns fewer than `horizon` intervals. Trailing NaN is cut rather than dropped,
    so a gap inside the path can never close up and shift later prices earlier.
    """
    runTime = forecasts.nemNow().floor("5min") if runTime is None else pd.Timestamp(runTime)
    path = forecasts.forecastSeries(runTime, region, horizon, allowPartial=True)
    good = path.notna().to_numpy()
    cut = int(good.argmin()) if not good.all() else len(good)
    return path.iloc[:cut]


def checkpointKey(label):
    """The checkpoint behind a dashboard model label.

    The backtest screen names a model `algo-trainMode-sN-priceSource`, because the same
    agent is scored twice there. One agent trained it, and live there is only the
    forecast, so the price source is dropped.
    """
    return "-".join(label.split("-")[:3])


def _policy(label, root, region):
    """The agent and its price transform for one dropdown label.

    Every model on this screen is a published agent, so a label with no checkpoint
    behind it raises. The board then draws the price path alone and writes the reason
    over it.
    """
    where = Path(root) / region
    fix = ("Training writes the final agent there. Copy that directory from the "
           "machine that trained it.")
    if label is None:
        raise FileNotFoundError(f"No agent is published under {where}, so the "
                                f"dispatch blocks are not drawn. {fix}")
    directory = findCheckpoints(region, root).get(checkpointKey(label))
    if directory is None:
        raise FileNotFoundError(f"{label} is not published under {where}, so the "
                                f"dispatch blocks are not drawn. {fix}")
    algo = label.split("-")[0]
    agent = algos.load(algo, str(directory / "agent"), env=None)
    return agent, loadScaler(directory), algos.supportsMasking(algo)


def suggest(path, label, region, root=defaultCheckpoints,
            battery=(config.battery_mw, config.battery_mwh,
                     config.battery_replacement_cost),
            initialSoc=config.initial_soc, calendar=False):
    """One row per forecast interval: the price, the action, the state of charge after
    it, and the running revenue and profit. An empty or one-interval path returns an
    empty frame."""
    if len(path) < 2:
        return pd.DataFrame(columns=list(columns))
    agent, scaler, masking = _policy(label, root, region)
    env = BessEnv(path, SchadeDegradation(*battery), horizon=len(path),
                  initialSoc=initialSoc, calendar=calendar, scaler=scaler)
    steps = rolloutWindow(agent, env, 0, masking=masking)["trajectory"]
    bought = np.array([step["bought"] for step in steps])
    sold = np.array([step["sold"] for step in steps])
    return pd.DataFrame({
        "settlementdate": path.index,
        "price": path.to_numpy(dtype=float),
        "action": np.select([bought > 0, sold > 0], ["charge", "discharge"], "idle"),
        "soc": [step["soc"] for step in steps],
        # Both are run totals, so the last row covers the whole horizon. Revenue is the
        # money the trades take. Profit is what is left after the degradation they cost.
        "revenue": np.cumsum([step["revenue"] for step in steps]),
        "profit": np.cumsum([step["profit"] for step in steps]),
    }, columns=list(columns))


def blocks(frame):
    """Each run of one action as (action, start, end), with the stamps the chart needs.

    AEMO stamps an interval by its end, so a block opens one interval before its first
    row. Idle runs are left out: the screen marks what to do, not what not to do.
    """
    out = []
    if frame.empty or "action" not in frame:
        return out
    stamps = pd.to_datetime(frame.settlementdate)
    actions = list(frame.action)
    start = 0
    for index in range(1, len(actions) + 1):
        if index < len(actions) and actions[index] == actions[start]:
            continue
        if actions[start] != "idle":
            out.append((actions[start], stamps.iloc[start] - nem_interval,
                        stamps.iloc[index - 1]))
        start = index
    return out


def priceFrame(path):
    """The forecast path alone, in the layout `suggest` returns. This is what the board
    draws when no policy can run: the prices are AEMO's and need no agent."""
    return pd.DataFrame({"settlementdate": path.index,
                         "price": path.to_numpy(dtype=float)})

"""The current forecast screen: the AEMO path published now, and the dispatch a policy
would run against it.

The backtest screen reads finished folds off disk. This one reads nothing off disk but
the policy itself. It asks AEMO for the forward price path at this minute, then rolls
one episode of `rl.env.BessEnv` over that path, so the action set, the degradation terms
and the state of charge rule are the same ones the backtest scored.

Every other setting of the episode comes from the experiment the agent was trained and
scored under, through `experiments.envSettings`: the cycle budget, whether the pack must
close empty, what closing stock is worth. Not from `config`, which has moved since the
agents were trained, and not from `policy.json`, which records the training machine's
`config` at publish time.

Two things differ from a backtest day, and both are stated on the screen.

1. The episode starts at the run time, not at 04:00, so it is as long as the forecast
   reaches. PREDISPATCH stops at the end of the trading day. The cycle budget is cut to
   the share of the episode that is left: see `budgetFor`.
2. The state of charge starts where the reader sets it, and at `defaultSoc` until they
   do, because the live state of a real battery is not in this repository. The screen
   states the number the board was drawn for. That default is not `config.initial_soc`:
   the episode reset the oracle and the agent share is zero, and this screen reports a
   real battery part way through a day.
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
from visualise.dashboard import experiments

defaultCheckpoints = policies.defaultRoot

# The start state of charge the forecast screen opens on, until the reader sets one.
defaultSoc = 0.5

columns = ("settlementdate", "price", "action", "soc", "revenue", "profit")


def budgetFor(settings, intervals):
    """The cycle budget a path of `intervals` is due, out of the experiment's own budget.

    The screen starts its episode at the run time, so the path is a part of one episode,
    and the whole budget would let a late-afternoon board spend what a whole day is
    allowed. The share of the episode that is left gets the same share of the budget.
    No input box: the screen already states a start state of charge it did not measure
    either.
    """
    budget = settings.get("cycleBudget")
    horizon = settings.get("horizon") or config.episode_intervals
    if budget is None:
        return None
    return float(budget) * min(1.0, intervals / horizon)


# The live screen reruns the agent on every NEM dispatch interval, at minutes ending in 0
# or 5. The clock ticks this often and refetches once a new interval has opened.
clockTickMs = 15_000


def refreshSlot(now=None):
    """The dispatch interval open at `now` (default the NEM clock): its :x0 or :x5 start.
    The forecast fetched for it is the one `forecastPath` reads by default."""
    now = forecasts.nemNow() if now is None else pd.Timestamp(now)
    return now.floor("5min")


def forecastPath(region, runTime=None, horizon=config.episode_intervals):
    """The forward price path AEMO publishes now, cut where the path stops.

    PREDISPATCH reaches the end of the trading day and no further, so a run late in the
    day returns fewer than `horizon` intervals. Trailing NaN is cut rather than dropped,
    so a gap inside the path can never close up and shift later prices earlier.
    """
    runTime = refreshSlot() if runTime is None else pd.Timestamp(runTime)
    path = forecasts.forecastSeries(runTime, region, horizon, allowPartial=True)
    good = path.notna().to_numpy()
    cut = int(good.argmin()) if not good.all() else len(good)
    return path.iloc[:cut]


def _policy(model, region, root=defaultCheckpoints):
    """The agent and its price transform for one `experiments.Runnable`.

    Every model on this screen is a published agent of a published experiment, so a
    screen with no model to run raises. The board then draws the price path alone and
    writes the reason over it.
    """
    if model is None:
        where = Path(root) / region
        raise FileNotFoundError(
            f"No published agent under {where} belongs to an experiment in "
            f"config.dashboard_runs, so the dispatch blocks are not drawn. Training "
            f"writes the final agent there: copy that directory from the machine that "
            f"trained it, and publish the backtest it came from.")
    algo = model.entry.algo
    agent = algos.load(algo, str(model.directory / "agent"), env=None)
    return agent, loadScaler(model.directory), algos.supportsMasking(algo)


def suggest(path, model, region, root=defaultCheckpoints,
            battery=(config.battery_mw, config.battery_mwh,
                     config.battery_replacement_cost),
            initialSoc=defaultSoc, calendar=False):
    """One row per forecast interval: the price, the action, the state of charge after
    it, and the running revenue and profit. An empty or one-interval path returns an
    empty frame.

    `model` is an `experiments.Runnable`: the agent, and the experiment whose settings
    the episode runs under.
    """
    if len(path) < 2:
        return pd.DataFrame(columns=list(columns))
    agent, scaler, masking = _policy(model, region, root)
    settings = experiments.envSettings(model.experiment)
    env = BessEnv(path, SchadeDegradation(*battery), horizon=len(path),
                  initialSoc=initialSoc, calendar=calendar, scaler=scaler,
                  cycleBudget=budgetFor(settings, len(path)),
                  endEmpty=settings["endEmpty"],
                  terminalPrice=settings["terminalPrice"])
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

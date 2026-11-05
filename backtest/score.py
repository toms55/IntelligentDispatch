"""Score any policy against the exact oracle, on realised prices or on AEMO's forecast.

A policy is any object with `predict(observation, deterministic=True, action_masks=None)`
returning `(action, None)`, a `reads_mask` attribute saying whether it takes the mask, and
optionally `reset()`. Trained SB3 agents already have the first; `rl/heuristic.py` is the
smallest example. Nothing here trains, so a policy may be a rule, a controller or a model
fitted elsewhere.

The oracle is solved for the settings the policy is scored under, so a different cycle
budget, terminal price or start state of charge gets its own bound. `buildBaseline` serves
the cache and solves a miss, which the discrete oracle does in seconds. This is the same
env, oracle and scoring path as the rolling backtest, only without the fold and training
machinery, so a policy and a run agree when they are given the same settings.
"""
from types import SimpleNamespace

import config
from backtest import folds
from backtest.runner import _forecastSource
from data.forecasts import refreshMatrix
from data.prices import price_series
from degradation.schade import SchadeDegradation
from rl.baseline import buildBaseline
from rl.env import BessEnv
from rl.evaluate import evaluatePolicy, summarise

priceSources = ("realised", "forecast")


def score(policy, region, start, end, priceSource="forecast", *,
          horizon=config.episode_intervals, cycleBudget=config.cycle_budget,
          terminalPrice=0.0, initialSoc=config.initial_soc, endEmpty=False,
          calendar=False,
          battery=(config.battery_mw, config.battery_mwh, config.battery_replacement_cost),
          forecastEvery=config.forecast_refresh_intervals, scaler=None):
    """Run `policy` over `(start, end]` and return its profit against the oracle's.

    `start` and `end` are 04:00 trading-day boundaries, and the window is a whole number
    of episodes. `priceSource` is what fills the observation's price slots; settlement is
    always the realised price. Under "forecast" the AEMO runs must be prefetched over the
    window first (`python -m cli prefetch`), or each run is pulled live and slowly.

    `scaler` is the price transform the observation uses. It defaults to the market cap in
    force at `end`; pass the one a trained agent's `scaler.json` holds to score that agent.
    The defaults match `RollingConfig`, so a policy scored here faces the problem the
    published runs did.

    Returns {"summary", "records", "settings"}. `summary` is profit, the oracle's bound,
    regret and the captured fraction; `records` is one entry per episode, with its
    trajectory; `settings` is everything the score depended on, to quote beside it.
    """
    if priceSource not in priceSources:
        raise ValueError(f"priceSource must be one of {priceSources}, got {priceSource!r}")
    folds.requireTradingDayStart(start, "start")
    folds.requireTradingDayStart(end, "end")
    length = folds.episodeLength(horizon)
    if (end - start) % length or end <= start:
        raise ValueError(f"{start} to {end} is not a whole number of {horizon}-interval episodes")

    prices = price_series(start, end, region)
    folds.requireFullDays(prices, start, end)
    model = SchadeDegradation(*battery)
    episodes = [start + step * length for step in range((end - start) // length)]

    # gap is part of the oracle's cache key, though the exact DP has none to apply, so it
    # takes the rolling backtest's value to hit the records that run solved.
    oracle = buildBaseline(
        prices, region=region, horizon=horizon, calendar=calendar, discretePower=True,
        model=model, initialSoc=initialSoc, gap=config.backtest_gap, cycles=cycleBudget,
        terminalPrice=terminalPrice, endEmpty=endEmpty)

    forecasts = None
    if priceSource == "forecast":
        # shortcut: _forecastSource reads five fields off the rolling config, so a
        # namespace stands in for it; lift it out of runner.py when a third caller needs it.
        source = _forecastSource(SimpleNamespace(
            priceSources=("forecast",), trainSource="realised", region=region,
            horizon=horizon, forecastEvery=forecastEvery), episodes)
        # allowPartial=False: a missing run reads as a flat $0 stretch the policy idles
        # through, and the loss would land in the score looking like a decision.
        forecasts = refreshMatrix(episodes, region, horizon, forecastEvery,
                                  allowPartial=False, source=source)

    env = BessEnv(prices, model, baseline=[record["profitPerInterval"] for record in oracle],
                  calendar=calendar, horizon=horizon, forecasts=forecasts, scaler=scaler,
                  initialSoc=initialSoc, cycleBudget=cycleBudget,
                  terminalPrice=terminalPrice, endEmpty=endEmpty)
    records = evaluatePolicy(policy, env, bounds=[record["dualBound"] for record in oracle])
    return {
        "summary": summarise(records),
        "records": records,
        "settings": {
            "region": region, "start": start.isoformat(), "end": end.isoformat(),
            "priceSource": priceSource, "horizon": horizon, "cycleBudget": cycleBudget,
            "terminalPrice": terminalPrice, "initialSoc": initialSoc, "endEmpty": endEmpty,
            "calendar": calendar, "battery": list(battery),
            "forecastEvery": forecastEvery if priceSource == "forecast" else None,
        },
    }

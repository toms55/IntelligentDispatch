"""Every number the backtest quotes. This is the only module that divides.

`captured` is a ratio of sums, never a mean of ratios:

    captured = sum(agentProfit) / sum(oracleDualBound)

The oracle's profit is concentrated in a handful of volatile days, so a per-day ratio on
a quiet day is noise, and averaging those ratios lets the quiet days outvote the days the
money is in. Seeds are averaged into one row per day first, otherwise the dollar totals
come out multiplied by the seed count.
"""
from collections import namedtuple

import pandas as pd

from backtest import records

dayKeys = ["fold", "algo", "trainMode", "priceSource", "testDay"]
Difference = namedtuple("Difference", "perFold pooled")


def perDay(frame):
    """One row per day, with the seeds averaged into it."""
    if frame.empty:
        return frame
    return (frame.groupby(dayKeys, as_index=False)
                 .agg(agentProfit=("agentProfit", "mean"),
                      oracleProfit=("oracleDualBound", "mean"),
                      highVol=("highVol", "first")))


def _ratio(agent, oracle):
    return agent / oracle if oracle else float("nan")


def perFold(frame):
    """Agent and oracle dollars per fold, and the fold's captured share."""
    days = perDay(frame)
    if days.empty:
        return pd.DataFrame(columns=["agentProfit", "oracleProfit", "captured"])
    folds = days.groupby("fold")[["agentProfit", "oracleProfit"]].sum()
    folds["captured"] = [_ratio(a, o) for a, o in
                         zip(folds.agentProfit, folds.oracleProfit)]
    return folds


def summarise(frame):
    """Pooled dollars and share, the win rate, and the spread across folds."""
    days = perDay(frame)
    if days.empty:
        return {"days": 0, "agentProfit": 0.0, "oracleProfit": 0.0,
                "captured": float("nan"), "winRate": float("nan"),
                "foldMedian": float("nan"), "foldIqr": float("nan")}
    agent = days.agentProfit.sum()
    # min_count so a scope holding no bound at all sums to NaN rather than to zero. One
    # day of a weekly episode is such a scope: the bound sits on the episode's first day,
    # and "the oracle made $0.00" would be a wrong statement rather than a missing one.
    oracle = days.oracleProfit.sum(min_count=1)
    # A fold the oracle made nothing on has no share to contribute to the spread.
    captured = perFold(frame).captured.dropna()
    quartiles = captured.quantile([0.25, 0.75]) if len(captured) else None
    return {
        "days": len(days),
        "agentProfit": agent,
        "oracleProfit": oracle,
        "captured": _ratio(agent, oracle),
        "winRate": float((days.agentProfit > 0).mean()),
        # Median and IQR, not mean and standard deviation: most folds are quiet and a
        # few carry the year, so the distribution is skewed.
        "foldMedian": captured.median() if len(captured) else float("nan"),
        "foldIqr": (quartiles[0.75] - quartiles[0.25]) if quartiles is not None
                   else float("nan"),
    }


def dayFacts(frame):
    """One day's revenue, degradation cost and spike intensity, seeds averaged."""
    if frame.empty:
        return {"revenue": float("nan"), "degradation": float("nan"),
                "spikeIntensity": float("nan"), "highVol": False}
    return {"revenue": frame.agentRevenue.mean(),
            "degradation": frame.agentDegradation.mean(),
            "spikeIntensity": frame.spikeIntensity.mean(),
            "highVol": bool(frame.highVol.iloc[0])}


def perDayBound(frame):
    """Whether the oracle bound is a per-day figure.

    It is not once an episode holds several trading days: the state of charge carries
    across them, so a day-long oracle started empty is no bound on a day the agent
    entered holding energy, and the backtest writes the episode's bound on its first day
    with NaN on the rest.
    """
    return bool(frame.empty or "episodeStart" not in frame
                or (frame.episodeStart == frame.testDay).all())


def byBucket(frame):
    """The summary split on the volatility label. Both sides are always present.

    The captured share is dropped where the bound is not per day. A bucket holds a set of
    days, and the episode whose bound those days share may open in the other bucket, so
    the bucket's agent dollars have no denominator of their own. The dollars still mean
    what they say.
    """
    perDay = perDayBound(frame)
    out = {}
    for high in (True, False):
        summary = summarise(frame[frame.highVol == high] if not frame.empty else frame)
        if not perDay:
            summary.update({key: float("nan") for key in
                            ("oracleProfit", "captured", "foldMedian", "foldIqr")})
        out[high] = summary
    return out


def cycleUse(frame):
    """Mean equivalent full cycles per episode, and the budget they were spent under.

    A mean per episode rather than a total, so the figure reads against the budget at
    every scope: one fold, one month, or the whole run. Each (seed, price source,
    episode) group is one observation, so no seed's spend is counted twice and none is
    averaged away.
    """
    missing = {"cycles": float("nan"), "budget": float("nan")}
    if frame.empty or "agentCycles" not in frame or frame.agentCycles.isna().all():
        return missing
    grain = "episodeStart" if "episodeStart" in frame else "testDay"
    perEpisode = frame.groupby(
        ["algo", "trainMode", "priceSource", "seed", grain]).agentCycles.sum()
    budget = (frame.cycleBudget.max() if "cycleBudget" in frame else float("nan"))
    return {"cycles": float(perEpisode.mean()), "budget": budget}


def difference(frame, column, left, right):
    """`left` minus `right` in dollars, over the folds where both sides ran.

    Serves warm against cold, and realised prices against the AEMO forecast.
    """
    sides = {value: perFold(frame[frame[column] == value]).agentProfit
             for value in (left, right)}
    shared = sides[left].index.intersection(sides[right].index)
    delta = (sides[left][shared] - sides[right][shared]).rename("difference")
    return Difference(delta, delta.sum() if len(delta) else float("nan"))

"""Tests for backtest/outofbounds.py. Run with: uv run pytest backtest/ -q"""
from datetime import datetime

import pandas as pd
import pytest

from backtest import outofbounds

DAY = datetime(2024, 1, 1, 4)
REGION = "NSW1"


def dayRows(counts):
    """One published day row per entry of `counts`, keyed by (algo, seed)."""
    return pd.DataFrame([
        {"fold": 0, "algo": algo, "trainMode": "warm", "seed": seed,
         "priceSource": "realised", "testDay": pd.Timestamp(DAY), "intervals": 288,
         "agentProfit": 1.0, "outOfBounds": count}
        for (algo, seed), count in counts.items()
    ])


def actionRows(attempts):
    """One actions-export row per entry of `attempts`: (attemptedAction, action)."""
    return pd.DataFrame([
        {"intervalTime": pd.Timestamp(DAY) + pd.Timedelta(minutes=5 * step),
         "attemptedAction": attempted, "action": moved, "soc": 0.5,
         "realisedPrice": 50.0 + step, "forecastPrice": None, "forecastRunTime": None,
         "region": REGION, "fold": 0, "seed": 0, "algo": "dqn", "trainMode": "warm",
         "priceSource": "realised"}
        for step, (attempted, moved) in enumerate(attempts)
    ])


class TestCounts:
    # The totals come from the published day rows, so every seed is covered rather than
    # the one seed the fold trajectories keep.
    def test_it_totals_per_model(self):
        frame = dayRows({("dqn", 0): 5, ("dqn", 1): 2, ("maskableppo", 0): 0})
        got = outofbounds.counts(frame)
        assert got.set_index(["algo", "seed"]).outOfBounds.to_dict() == {
            ("dqn", 0): 5, ("dqn", 1): 2, ("maskableppo", 0): 0}

    # The rate is what compares two runs of different length.
    def test_it_reports_a_rate_per_interval(self):
        got = outofbounds.counts(dayRows({("dqn", 0): 72}))
        assert got.rate.iloc[0] == pytest.approx(0.25)

    # A masked algorithm cannot choose an out-of-bounds action at all, so zero is the
    # expected answer and must not look like a missing one.
    def test_a_masked_run_reports_zero(self):
        got = outofbounds.counts(dayRows({("maskableppo", 0): 0}))
        assert got.outOfBounds.iloc[0] == 0 and got.rate.iloc[0] == 0.0

    # A run from before the column cannot be summarised, and saying so beats printing a
    # zero that reads as "the agent never did it".
    def test_a_run_without_the_column_is_refused(self):
        frame = dayRows({("dqn", 0): 1}).drop(columns=["outOfBounds"])
        with pytest.raises(ValueError, match="outOfBounds"):
            outofbounds.counts(frame)


class TestIntervals:
    # An attempt that moved nothing is an attempt at a wall. An attempt that moved
    # something, even a part step, is a legal move.
    def test_it_finds_the_attempts_that_moved_nothing(self):
        frame = actionRows([("charge", "idle"), ("charge", "charge"),
                            ("idle", "idle"), ("discharge", "idle")])
        got = outofbounds.intervals(frame)
        assert list(got.attemptedAction) == ["charge", "discharge"]
        assert list(got.realisedPrice) == [50.0, 53.0]

    def test_a_clean_run_finds_nothing(self):
        frame = actionRows([("idle", "idle"), ("charge", "charge")])
        assert outofbounds.intervals(frame).empty

    def test_an_export_without_the_column_is_refused(self):
        frame = actionRows([("charge", "idle")]).drop(columns=["attemptedAction"])
        with pytest.raises(ValueError, match="attemptedAction"):
            outofbounds.intervals(frame)

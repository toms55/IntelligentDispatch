"""Tests for backtest/outofbounds.py. Run with: uv run pytest backtest/ -q"""
from datetime import datetime

import pandas as pd
import pytest

import config
from backtest import outofbounds

DAY = datetime(2024, 1, 1, 4)
REGION = "NSW1"


def dayRows(counts):
    """One published day row per entry of `counts`, keyed by (algo, seed)."""
    return pd.DataFrame([
        {"fold": 0, "algo": algo, "trainMode": "warm", "seed": seed,
         "priceSource": "realised", "testDay": pd.Timestamp(DAY),
         "agentProfit": 1.0, "outOfBounds": count}
        for (algo, seed), count in counts.items()
    ])


def actionRows(attempts):
    """One actions-export row per entry: (attemptedAction, action, outOfBounds)."""
    return pd.DataFrame([
        {"intervalTime": pd.Timestamp(DAY) + pd.Timedelta(minutes=5 * step),
         "attemptedAction": attempted, "action": moved, "outOfBounds": flagged,
         "soc": 0.5,
         "realisedPrice": 50.0 + step, "forecastPrice": None, "forecastRunTime": None,
         "region": REGION, "fold": 0, "seed": 0, "algo": "dqn", "trainMode": "warm",
         "priceSource": "realised"}
        for step, (attempted, moved, flagged) in enumerate(attempts)
    ])


class TestCounts:
    # The totals come from the published day rows, so every seed is covered rather than
    # the one seed the fold trajectories keep.
    def test_it_totals_per_model(self):
        frame = dayRows({("dqn", 0): 5, ("dqn", 1): 2, ("maskableppo", 0): 0})
        got = outofbounds.counts(frame)
        assert got.set_index(["algo", "seed"]).outOfBounds.to_dict() == {
            ("dqn", 0): 5, ("dqn", 1): 2, ("maskableppo", 0): 0}

    # The rate is what compares two runs of different length: attempts over the
    # intervals scored, which is one episode per test day.
    def test_it_reports_a_rate_per_interval(self):
        got = outofbounds.counts(dayRows({("dqn", 0): 72}))
        assert got.intervals.iloc[0] == config.episode_intervals
        assert got.rate.iloc[0] == pytest.approx(72 / config.episode_intervals)

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
    # The env's flag selects the rows, not the energy moved: `rl/env.py:action_masks` is
    # the only definition of out of bounds, and this reader must not invent a second one.
    def test_it_finds_the_flagged_intervals(self):
        frame = actionRows([("charge", "idle", True), ("charge", "charge", False),
                            ("idle", "idle", False), ("discharge", "idle", True)])
        got = outofbounds.intervals(frame)
        assert list(got.attemptedAction) == ["charge", "discharge"]
        assert list(got.realisedPrice) == [50.0, 53.0]

    # The case a count from the energy alone would lose. At the SoC boundary the mask is
    # a tolerance on the headroom, so a flagged step can still move an atom of energy and
    # read as a move.
    def test_a_flagged_interval_that_moved_an_atom_is_still_found(self):
        frame = actionRows([("charge", "charge", True)])
        assert len(outofbounds.intervals(frame)) == 1

    def test_a_clean_run_finds_nothing(self):
        frame = actionRows([("idle", "idle", False), ("charge", "charge", False)])
        assert outofbounds.intervals(frame).empty

    def test_an_export_without_the_column_is_refused(self):
        frame = actionRows([("charge", "idle", True)]).drop(columns=["outOfBounds"])
        with pytest.raises(ValueError, match="outOfBounds"):
            outofbounds.intervals(frame)

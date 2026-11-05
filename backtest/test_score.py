"""Tests for backtest/score.py. Run with: uv run pytest backtest/test_score.py -q

Prices are synthetic and dated 2031, past every oracle record on disk, so the DP solves
each window here and nothing reads a published bound. The oracle is real; only the price
feed and the forecast cache are replaced.
"""
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from backtest import runner, score as scoring
from rl.env import IDLE
from rl.heuristic import ThresholdPolicy
from utils import nem_interval

START = datetime(2031, 1, 1, 4)
END = datetime(2031, 1, 3, 4)  # two episodes


class Idle:
    reads_mask = False

    def __init__(self):
        self.resets = 0

    def predict(self, observation, deterministic=True, action_masks=None):
        return IDLE, None

    def reset(self):
        self.resets += 1


@pytest.fixture(scope="module")
def oracleDir(tmp_path_factory):
    """One scratch oracle cache for the module: a solve is keyed on its settings, so the
    tests that share settings share the solve."""
    return tmp_path_factory.mktemp("oracle")


@pytest.fixture(autouse=True)
def synthetic(monkeypatch, oracleDir):
    index = pd.date_range(START + nem_interval, END, freq=nem_interval)
    prices = pd.Series(100 + 80 * np.sin(np.arange(len(index)) * 2 * np.pi / 288),
                       index=index)
    monkeypatch.setattr(scoring, "price_series", lambda start, end, region: prices)
    # Written here, and read back from here too: readRoots is fixed at import, so the
    # scratch directory is added to it for the test.
    monkeypatch.setattr("rl.baseline.cacheRoot", oracleDir)
    monkeypatch.setattr("rl.baseline.readRoots", (oracleDir,))


def flatForecast(monkeypatch):
    """Every refresh run of both days, all at a flat $1."""
    runs = [pd.Timestamp(day) + offset * nem_interval
            for day in (START, START + pd.Timedelta(days=1)) for offset in range(0, 288, 48)]
    frame = pd.DataFrame(1.0, index=runs, columns=[str(n) for n in range(1, 289)])
    monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)


class TestScore:
    # The oracle bounds the policy, and the settings come back to be quoted.
    def test_realised_profit_sits_under_the_bound(self):
        result = scoring.score(ThresholdPolicy(), "NSW1", START, END, "realised")
        assert result["summary"]["windows"] == 2
        assert result["summary"]["agentProfit"] <= result["summary"]["lpProfit"] + 1e-9
        assert result["settings"]["priceSource"] == "realised"
        assert result["settings"]["cycleBudget"] == scoring.defaultCycleBudget == 1.0
        assert all(record["agentCycles"] <= 1.0 + 1e-9 for record in result["records"])

    # A policy that never moves earns nothing, so its regret is the whole bound, and the
    # optional reset() runs once per episode.
    def test_idle_regret_is_the_whole_bound(self):
        idle = Idle()
        result = scoring.score(idle, "NSW1", START, END, "realised")
        assert result["summary"]["agentProfit"] == 0.0
        assert result["summary"]["regret"] == result["summary"]["lpProfit"] > 0
        assert idle.resets == 2

    # Under "forecast" the observation holds the forecast and the settlement stays
    # realised: a flat $1 forecast makes the threshold rule idle through a volatile day.
    def test_forecast_fills_the_observation_not_the_settlement(self, monkeypatch):
        flatForecast(monkeypatch)
        realised = scoring.score(ThresholdPolicy(), "NSW1", START, END, "realised")
        forecast = scoring.score(ThresholdPolicy(), "NSW1", START, END, "forecast")
        assert realised["summary"]["agentProfit"] != 0.0
        assert forecast["summary"]["agentProfit"] == 0.0
        assert forecast["summary"]["lpProfit"] == realised["summary"]["lpProfit"]

    # A tighter cycle budget is a different problem and gets its own, lower, bound.
    def test_a_smaller_budget_lowers_the_bound(self):
        loose = scoring.score(Idle(), "NSW1", START, END, "realised", cycleBudget=None)
        tight = scoring.score(Idle(), "NSW1", START, END, "realised", cycleBudget=0.5)
        assert tight["summary"]["lpProfit"] < loose["summary"]["lpProfit"]

    # A policy that does not say whether it reads the mask is refused, not guessed.
    def test_a_policy_without_reads_mask_is_refused(self):
        class Bare:
            def predict(self, observation, deterministic=True):
                return IDLE, None
        with pytest.raises(ValueError, match="reads_mask"):
            scoring.score(Bare(), "NSW1", START, END, "realised")

    @pytest.mark.parametrize("kwargs, message", [
        ({"start": datetime(2031, 1, 1, 5)}, "04:00"),
        ({"end": datetime(2031, 1, 2, 16)}, "04:00"),
        ({"priceSource": "oracle"}, "priceSource"),
    ])
    def test_a_bad_window_or_source_is_refused(self, kwargs, message):
        args = {"start": START, "end": END, "priceSource": "realised"} | kwargs
        with pytest.raises(ValueError, match=message):
            scoring.score(Idle(), "NSW1", args["start"], args["end"], args["priceSource"])

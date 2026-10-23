"""Tests for backtest/actions.py. Run with: uv run pytest backtest/ -q"""
from datetime import datetime

import pandas as pd

from backtest import actions
from utils import nem_interval

DAY = datetime(2024, 1, 1, 4)


def trajectory(steps=288):
    return pd.DataFrame({
        "fold": 0, "algo": "maskableppo", "trainMode": "warm", "seed": 0,
        "priceSource": "forecast", "testDay": pd.Timestamp(DAY),
        "interval": range(steps),
        "settlementdate": [DAY + (step + 1) * nem_interval for step in range(steps)],
        "price": 50.0, "observedPrice": 48.0, "soc": 0.5, "bought": 0.0, "sold": 0.0,
        "profit": 0.0,
    })


class TestActionRows:
    # Each decision names the run it read: the 04:00 run until 08:00, and so on.
    def test_the_run_time_follows_the_refresh(self):
        got = actions.actionRows(trajectory(), "NSW1", every=48)
        assert got.forecastRunTime.iloc[0] == pd.Timestamp("2024-01-01 04:00")
        assert got.forecastRunTime.iloc[47] == pd.Timestamp("2024-01-01 04:00")
        assert got.forecastRunTime.iloc[48] == pd.Timestamp("2024-01-01 08:00")
        assert got.forecastRunTime.iloc[287] == pd.Timestamp("2024-01-02 00:00")

    # A recorded run is what the agent read, so it wins over the manifest's setting.
    def test_a_recorded_run_wins(self):
        frame = trajectory()
        frame["forecastRunTime"] = pd.Timestamp("2024-01-01 03:55")
        got = actions.actionRows(frame, "NSW1", every=48)
        assert (got.forecastRunTime == pd.Timestamp("2024-01-01 03:55")).all()

    # A result set from before the refresh read one 04:00 run all day.
    def test_an_older_run_reads_one_run_all_day(self):
        got = actions.actionRows(trajectory(), "NSW1")
        assert (got.forecastRunTime == pd.Timestamp("2024-01-01 04:00")).all()


class TestExportName:
    # Every run's CSV lands in one folder, so the name alone must say which run it is.
    def test_a_run_named_for_its_region_keeps_its_name(self):
        assert actions.exportName("results/rolling2023_fc4h_SA1", "SA1") \
            == "rolling2023_fc4h_SA1.csv"

    def test_a_run_without_a_region_gets_one(self):
        assert actions.exportName("results/rolling2023_profit", "NSW1") \
            == "rolling2023_profit_NSW1.csv"

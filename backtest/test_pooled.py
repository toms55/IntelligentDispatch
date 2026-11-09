"""Tests for backtest/pooled.py. Run with: uv run pytest backtest/test_pooled.py -q"""
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import config
from backtest import actions, pooled, runner
from degradation.schade import SchadeDegradation
from rl.env import BessEnv

DAY = config.observation_intervals


def window(start, days, level):
    """`days` whole trading days from `start`, every price `level`."""
    index = pd.date_range(start, periods=days * DAY, freq="5min")
    return pd.Series(float(level), index=index)


class TestPooledPrices:
    def test_every_episode_sits_inside_one_region(self):
        # Two regions over the same dates, told apart by their price level. A join that
        # broke the episode grid would show up as a window holding both.
        windows = {"A": window("2024-01-01 04:05", 3, 10.0),
                   "B": window("2024-01-01 04:05", 3, 20.0)}
        prices = pooled.pooledPrices(windows, DAY)
        env = BessEnv(prices, SchadeDegradation(config.battery_mw, config.battery_mwh,
                                                config.battery_replacement_cost),
                      horizon=DAY, calendar=False)
        levels = [set(env.prices[start:start + DAY]) for start in env.windows]
        assert levels == [{10.0}] * 3 + [{20.0}] * 3

    def test_a_window_off_the_episode_grid_is_refused(self):
        windows = {"A": window("2024-01-01 04:05", 2, 10.0).iloc[:-1],
                   "B": window("2024-01-01 04:05", 2, 20.0)}
        with pytest.raises(ValueError, match="straddle"):
            pooled.pooledPrices(windows, DAY)


class TestPaths:
    def cfg(self, seed):
        base = runner.RollingConfig(region="NSW1", start=datetime(2023, 1, 1, 4),
                                    end=datetime(2024, 1, 7, 4),
                                    capDate=datetime(2026, 10, 1, 4), trainWeeks=52)
        return pooled.PooledConfig(base=base, run="pooledtest", seed=seed,
                                   algo="maskableppo")

    def test_seeds_write_apart_and_share_one_actions_csv(self):
        zero = pooled.regionConfig(self.cfg(0), "SA1")
        one = pooled.regionConfig(self.cfg(1), "SA1")
        assert zero.resultsDir == Path("results/pooledtest_SA1/seed0")
        assert one.resultsDir == Path("results/pooledtest_SA1/seed1")
        assert zero.actionsFile == one.actionsFile == actions.liveFile(
            Path("results/pooledtest_SA1"), "SA1")

    def test_the_shared_agent_has_its_own_checkpoints(self):
        cfg = self.cfg(0)
        shared = runner.checkpointDir(pooled.pooledConfig(cfg), 3, "pooled",
                                      "maskableppo", 0)
        regional = runner.checkpointDir(pooled.regionConfig(cfg, "SA1"), 3, "finetune",
                                        "maskableppo", 0)
        assert shared == Path("runs/pooledtest_POOLED/POOLED/fold03/pooled/maskableppo_seed0")
        assert regional == Path("runs/pooledtest_SA1/SA1/fold03/finetune/maskableppo_seed0")

    def test_the_manifest_records_the_pooling(self):
        cfg = self.cfg(0)
        manifest = pooled.manifestOf(cfg, pooled.regionConfig(cfg, "SA1"))
        assert manifest["pooled"] == {"regions": list(pooled.regions),
                                      "pooledWarm": 150_000, "pooledCold": 1_500_000,
                                      "fineTune": 30_000}
        assert manifest["trainModes"] == ("pooled", "finetune")

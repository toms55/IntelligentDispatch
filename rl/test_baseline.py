"""Tests for rl/baseline.py. Run with: uv run pytest rl/ -q

`solve` is monkeypatched; the oracle is too slow to run here.
"""
import json

import pandas as pd
import pytest

import config
from degradation.schade import SchadeDegradation
from optimisation.lp import Schedule
from rl import baseline
from rl.env import BessEnv


@pytest.fixture
def paper():
    return SchadeDegradation(0.060, 0.100, 150_000.0)


@pytest.fixture
def series():
    values = [10.0 + n for n in range(36)]
    return pd.Series(
        values, index=pd.date_range("2024-01-10", periods=len(values), freq="5min"),
        name="RRP",
    )


@pytest.fixture
def fakeSolve(monkeypatch):
    """Stand in for the MILP, recording how often it ran."""
    calls = []

    def solve(prices, model, **kwargs):
        calls.append({"prices": list(prices), **kwargs})
        n = len(prices)
        return Schedule(
            kwargs.get("initialSoc", 0.5), [0.5] * n, [0.0] * n, [0.0] * n,
            3.0, 1.0, 2.0,
            revenuePerInterval=[3.0 / n] * n,
            cyclePerInterval=[0.5 / n] * n,
            calendarPerInterval=[0.5 / n] * n,
            degradationPerInterval=[1.0 / n] * n,
            profitPerInterval=[2.0 / n] * n,
            status="Optimal", mipGap=0.001, dualBound=2.01,
        )

    monkeypatch.setattr(baseline, "solve", solve)
    return calls


class TestChunkWindows:
    def test_matches_the_environment(self, paper, series):
        env = BessEnv(series, paper, horizon=12, calendar=False)
        assert baseline.chunkWindows(len(series), 12) == env.windows

    def test_drops_a_short_tail(self):
        assert baseline.chunkWindows(30, 12) == [0, 12]

    def test_windows_do_not_overlap(self):
        starts = baseline.chunkWindows(300, 24)
        assert all(b - a == 24 for a, b in zip(starts, starts[1:]))

    def test_a_series_shorter_than_one_window_gives_nothing(self):
        assert baseline.chunkWindows(5, 12) == []


class TestCacheKey:
    def _key(self, paper, **overrides):
        settings = dict(region="SA1", start=pd.Timestamp("2024-01-10"), horizon=288,
                        calendar=True, discretePower=True, model=paper,
                        initialSoc=0.5, gap=0.005)
        settings.update(overrides)
        return baseline.cacheKey(**settings)

    def test_is_stable_for_the_same_inputs(self, paper):
        assert self._key(paper) == self._key(paper)

    @pytest.mark.parametrize("field,value", [
        ("region", "NSW1"),
        ("horizon", 144),
        ("calendar", False),
        ("discretePower", False),
        ("initialSoc", 0.4),
        ("gap", 0.02),
    ])
    def test_changes_when_an_input_that_matters_changes(self, paper, field, value):
        assert self._key(paper) != self._key(paper, **{field: value})

    def test_changes_with_the_battery(self, paper):
        other = SchadeDegradation(0.120, 0.200, 150_000.0)
        assert self._key(paper) != self._key(paper, model=other)

    def test_names_the_window_readably(self, paper):
        assert self._key(paper).startswith("SA1_2024-01-10_0000_")


class TestSolveWindow:
    def test_solves_and_writes_a_cache_file(self, paper, series, fakeSolve, tmp_path):
        record = baseline.solveWindow(
            series.to_numpy(), list(series.index), "SA1", horizon=len(series),
            model=paper, cacheDir=tmp_path,
        )
        assert len(fakeSolve) == 1
        written = list(tmp_path.glob("*.json"))
        assert len(written) == 1
        assert json.loads(written[0].read_text()) == record

    def test_a_second_call_reads_the_cache(self, paper, series, fakeSolve, tmp_path):
        args = (series.to_numpy(), list(series.index), "SA1")
        kwargs = dict(horizon=len(series), model=paper, cacheDir=tmp_path)
        first = baseline.solveWindow(*args, **kwargs)
        second = baseline.solveWindow(*args, **kwargs)
        assert first == second
        assert len(fakeSolve) == 1  # not re-solved

    def test_keeps_the_dual_bound(self, paper, series, fakeSolve, tmp_path):
        record = baseline.solveWindow(
            series.to_numpy(), list(series.index), "SA1", horizon=len(series),
            model=paper, cacheDir=tmp_path,
        )
        assert record["dualBound"] == 2.01
        assert record["dualBound"] >= record["profit"]

    def test_passes_the_configured_gap_and_time_limit(self, paper, series, fakeSolve,
                                                     tmp_path):
        baseline.solveWindow(
            series.to_numpy(), list(series.index), "SA1", horizon=len(series),
            model=paper, cacheDir=tmp_path,
        )
        assert fakeSolve[0]["gap"] == config.oracle_gap
        assert fakeSolve[0]["timeLimit"] == config.oracle_time_limit

    def test_asks_for_the_discrete_oracle_by_default(self, paper, series, fakeSolve,
                                                    tmp_path):
        baseline.solveWindow(
            series.to_numpy(), list(series.index), "SA1", horizon=len(series),
            model=paper, cacheDir=tmp_path,
        )
        assert fakeSolve[0]["discretePower"] is True


class TestBuildBaseline:
    def test_solves_every_window(self, paper, series, fakeSolve, tmp_path):
        records = baseline.buildBaseline(series, horizon=12, model=paper,
                                        cacheDir=tmp_path)
        assert len(records) == 3
        assert len(fakeSolve) == 3

    def test_each_window_sees_only_its_own_prices(self, paper, series, fakeSolve,
                                                 tmp_path):
        baseline.buildBaseline(series, horizon=12, model=paper, cacheDir=tmp_path)
        for index, call in enumerate(fakeSolve):
            assert call["prices"] == list(series.iloc[index * 12:(index + 1) * 12])

    def test_every_window_starts_from_the_same_soc(self, paper, series, fakeSolve,
                                                  tmp_path):
        baseline.buildBaseline(series, horizon=12, model=paper, cacheDir=tmp_path)
        assert {call["initialSoc"] for call in fakeSolve} == {config.initial_soc}

    def test_profit_per_interval_is_shaped_for_the_environment(self, paper, series,
                                                              fakeSolve, tmp_path):
        records = baseline.buildBaseline(series, horizon=12, model=paper,
                                        cacheDir=tmp_path)
        rewards = baseline.profitPerInterval(records)
        assert len(rewards) == 3
        assert all(len(window) == 12 for window in rewards)

    def test_the_baseline_drops_straight_into_the_environment(self, paper, series,
                                                             fakeSolve, tmp_path):
        records = baseline.buildBaseline(series, horizon=12, model=paper,
                                        cacheDir=tmp_path)
        env = BessEnv(series, paper, baseline=baseline.profitPerInterval(records),
                      horizon=12, calendar=False)
        env.reset(seed=0)
        _, _, _, _, info = env.step(1)
        assert info["lpProfit"] == pytest.approx(2.0 / 12)

    def test_reports_progress(self, paper, series, fakeSolve, tmp_path):
        seen = []
        baseline.buildBaseline(series, horizon=12, model=paper, cacheDir=tmp_path,
                              onDone=lambda done, total, record: seen.append((done, total)))
        assert seen == [(1, 3), (2, 3), (3, 3)]

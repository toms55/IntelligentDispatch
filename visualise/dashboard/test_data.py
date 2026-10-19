"""Tests for visualise/dashboard/data.py. Run with: uv run pytest visualise/ -q

Everything here is a pure function of a frame, so no Dash import and no server.
"""
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import records
from visualise.dashboard import data

REGION = "NSW1"
DAY = datetime(2025, 1, 1, 4)


def dayRows(count=14, folds=2):
    rows = []
    for index in range(count):
        day = DAY + timedelta(days=index)
        rows.append({
            "fold": index // (count // folds), "algo": "maskableppo", "arm": "warm",
            "seed": 0, "priceSource": "realised", "testDay": day,
            "agentProfit": 10.0 - index, "agentRevenue": 12.0, "agentDegradation": 2.0,
            "oracleDualBound": 40.0, "spikeIntensity": 5.0 * (index % 2),
            "highVol": bool(index % 2), "timesteps": 30000, "warmFrom": "",
            "writtenAt": "2026-01-01T00:00:00",
        })
    return rows


@pytest.fixture
def run(tmp_path):
    """One finished two-fold run on disk: fold CSVs plus a manifest."""
    rows = dayRows()
    for fold in sorted({row["fold"] for row in rows}):
        records.writeFold(tmp_path, REGION, fold,
                          [row for row in rows if row["fold"] == fold])
    records.writeManifest(tmp_path, REGION, {
        "region": REGION, "algos": ["maskableppo"], "arms": ["warm"],
        "trainWeeks": 52, "warmTimesteps": 30000, "coldTimesteps": 300000,
        "seeds": [0], "capDate": "2026-01-01T04:00:00", "horizon": 288,
        "priceSources": ["realised"], "battery": [100.0, 200.0, 300000.0],
        "calendar": False, "discretePower": False, "gap": 0.02, "initialSoc": 0.5,
        "rewardScale": 1000.0, "testWeeks": 1, "stepWeeks": 1,
        "start": "2024-01-01T04:00:00", "end": "2026-01-01T04:00:00",
    })
    return tmp_path


class TestLoadRun:
    # A region with no fold CSV names the directory it looked in.
    def test_a_missing_run_names_the_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            data.loadRun("SA1", tmp_path)

    # A finished run comes back with its days and its manifest together.
    def test_a_finished_run_loads_both_parts(self, run):
        loaded = data.loadRun(REGION, run)
        assert len(loaded.days) == 14
        assert loaded.manifest["trainWeeks"] == 52

    # A manifest written beside no rows is still an error, not an empty dashboard.
    def test_a_manifest_without_rows_is_refused(self, tmp_path):
        records.writeManifest(tmp_path, REGION, {"region": REGION})
        with pytest.raises(FileNotFoundError):
            data.loadRun(REGION, tmp_path)


class TestSelectors:
    # The controls list what the data holds, never what config.py declares.
    def test_it_lists_only_what_is_present(self, run):
        got = data.selectors(data.loadRun(REGION, run).days)
        assert got == {"algo": ["maskableppo"], "arm": ["warm"],
                       "priceSource": ["realised"], "seed": [0]}

    # A second algorithm appears without any dashboard change.
    def test_a_new_algorithm_appears_on_its_own(self, run):
        frame = data.loadRun(REGION, run).days
        extra = frame.copy()
        extra["algo"] = "dqn"
        both = pd.concat([frame, extra], ignore_index=True)
        assert data.selectors(both)["algo"] == ["dqn", "maskableppo"]


class TestScope:
    # Day scope keeps that day and nothing else.
    def test_a_day_keeps_one_day(self, run):
        frame = data.loadRun(REGION, run).days
        got = data.scope(frame, day=DAY)
        assert len(got) == 1 and got.testDay.iloc[0] == pd.Timestamp(DAY)

    # Week scope keeps the seven days starting at the given day.
    def test_a_week_keeps_seven_days(self, run):
        frame = data.loadRun(REGION, run).days
        got = data.scope(frame, week=DAY)
        assert len(got) == 7
        assert got.testDay.min() == pd.Timestamp(DAY)
        assert got.testDay.max() == pd.Timestamp(DAY + timedelta(days=6))

    # Job scope is the whole frame, unchanged.
    def test_no_scope_returns_everything(self, run):
        frame = data.loadRun(REGION, run).days
        assert len(data.scope(frame)) == len(frame)

    # A day outside the run gives an empty frame rather than raising.
    def test_an_unknown_day_is_empty(self, run):
        frame = data.loadRun(REGION, run).days
        assert data.scope(frame, day=datetime(2020, 1, 1, 4)).empty

    # Asking for a day and a week at once is a caller error.
    def test_a_day_and_a_week_together_are_refused(self, run):
        frame = data.loadRun(REGION, run).days
        with pytest.raises(ValueError, match="one of"):
            data.scope(frame, day=DAY, week=DAY)


class TestFoldOfDay:
    # A day maps to the fold whose test window holds it, for the trajectory lookup.
    def test_a_day_maps_to_its_fold(self, run):
        frame = data.loadRun(REGION, run).days
        assert data.foldOfDay(frame, DAY) == 0
        assert data.foldOfDay(frame, DAY + timedelta(days=7)) == 1

    # A day no fold tested has no trajectory to show.
    def test_an_unknown_day_has_no_fold(self, run):
        frame = data.loadRun(REGION, run).days
        assert data.foldOfDay(frame, datetime(2020, 1, 1, 4)) is None


class TestTrajectories:
    # A day with no trajectory file gives an empty frame with the right columns.
    def test_a_missing_parquet_is_empty(self, run):
        loaded = data.loadRun(REGION, run)
        got = data.trajectories(loaded, DAY)
        assert got.empty and list(got.columns) == list(records.trajectoryColumns)

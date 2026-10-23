"""Tests for backtest/records.py. Run with: uv run pytest backtest/ -q

The output is one CSV and one parquet per fold, each rewritten whole and moved into
place with os.replace. That is why there is no torn-line handling anywhere: a fold file
is either the previous complete version or the new complete version, never half of one.
Rewriting a forty-row CSV after every unit costs nothing and buys unit-granular resume.
"""
from datetime import datetime

import pandas as pd
import pytest

from backtest import records

REGION = "NSW1"
DAY = datetime(2025, 1, 1, 4)


def row(fold=0, trainMode="warm", seed=0, priceSource="realised", day=DAY, profit=1.5):
    return {
        "fold": fold, "algo": "maskableppo", "trainMode": trainMode, "seed": seed,
        "priceSource": priceSource, "testDay": day,
        "agentProfit": profit, "agentRevenue": 2.0, "agentDegradation": 0.5,
        "oracleDualBound": 4.0, "oracleGap": 0.01, "spikeIntensity": 3.25, "highVol": True,
        "timesteps": 30000, "warmFrom": "runs/rolling/NSW1/fold00/warm",
        "writtenAt": "2026-10-19T12:00:00",
    }


def week(fold=0, **kwargs):
    return [row(fold=fold, day=DAY + n * pd.Timedelta(days=1), **kwargs)
            for n in range(7)]


class TestFoldCsv:
    # Rows survive a write and read with the types the dashboard and metrics expect.
    def test_rows_round_trip_with_their_dtypes(self, tmp_path):
        records.writeFold(tmp_path, REGION, 0, week())
        frame = records.readDays(tmp_path, REGION)
        assert len(frame) == 7
        assert frame.testDay.dtype.kind == "M"
        assert frame.highVol.dtype == bool
        assert frame.fold.dtype.kind == "i"
        assert frame.agentProfit.iloc[0] == 1.5

    # A row missing a declared column would write a silently empty cell.
    def test_a_row_missing_a_column_is_refused(self, tmp_path):
        short = row()
        del short["oracleDualBound"]
        with pytest.raises(ValueError, match="oracleDualBound"):
            records.writeFold(tmp_path, REGION, 0, [short])

    # An unknown column means the schema moved somewhere and not here.
    def test_a_row_with_an_unknown_column_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="captured"):
            records.writeFold(tmp_path, REGION, 0, [{**row(), "captured": 0.4}])

    # A fold is rewritten whole after every unit, so re-writing it must replace rather
    # than double its rows.
    def test_rewriting_a_fold_replaces_it(self, tmp_path):
        records.writeFold(tmp_path, REGION, 0, week())
        records.writeFold(tmp_path, REGION, 0, week() + week(trainMode="cold"))
        frame = records.readDays(tmp_path, REGION)
        assert len(frame) == 14
        assert set(frame.trainMode) == {"warm", "cold"}

    # The write is atomic, so an interrupted run cannot leave a partial file behind.
    def test_no_temporary_file_survives_a_write(self, tmp_path):
        records.writeFold(tmp_path, REGION, 0, week())
        assert list((tmp_path / REGION).glob("*.tmp")) == []


class TestReadDays:
    # Folds are concatenated in fold order, whatever order they were written in.
    def test_folds_are_concatenated_in_order(self, tmp_path):
        records.writeFold(tmp_path, REGION, 2, week(fold=2))
        records.writeFold(tmp_path, REGION, 0, week(fold=0))
        assert list(records.readDays(tmp_path, REGION).fold.unique()) == [0, 2]

    # A results directory with no folds yet is an empty frame, not an error, so a first
    # run and a resumed run take the same path.
    def test_an_empty_directory_gives_an_empty_frame(self, tmp_path):
        frame = records.readDays(tmp_path, REGION)
        assert frame.empty
        assert list(frame.columns) == list(records.dayColumns)

    # Two rows for the same unit and day would double-count that day in every total.
    def test_a_duplicated_day_is_refused(self, tmp_path):
        records.writeFold(tmp_path, REGION, 0, week())
        records.writeFold(tmp_path, REGION, 1, week(fold=0))  # fold 0 rows in fold 1's file
        with pytest.raises(ValueError, match="duplicate"):
            records.readDays(tmp_path, REGION)


class TestCompletedUnits:
    # A unit is one (fold, algo, trainMode, seed, priceSource), not one day, so a finished
    # week collapses to a single resume key.
    def test_one_key_per_unit_not_per_day(self, tmp_path):
        records.writeFold(tmp_path, REGION, 0, week() + week(trainMode="cold"))
        got = records.completedUnits(records.readDays(tmp_path, REGION))
        assert got == {(0, "maskableppo", "warm", 0, "realised"),
                       (0, "maskableppo", "cold", 0, "realised")}

    # Nothing completed yet is an empty set, which is what a first run resumes from.
    def test_an_empty_frame_has_no_units(self, tmp_path):
        assert records.completedUnits(records.readDays(tmp_path, REGION)) == set()


class TestManifest:
    # The manifest round trips, so a resumed run can compare against it.
    def test_round_trips(self, tmp_path):
        spec = {"trainWeeks": 52, "gap": 0.02}
        records.writeManifest(tmp_path, REGION, spec)
        records.requireSameManifest(tmp_path, REGION, spec)

    # Two configurations appended into one result set is the silent killer, so a changed
    # manifest is refused and the message names the field that moved.
    def test_a_changed_configuration_is_refused(self, tmp_path):
        records.writeManifest(tmp_path, REGION, {"trainWeeks": 52, "gap": 0.02})
        with pytest.raises(ValueError, match="trainWeeks"):
            records.requireSameManifest(tmp_path, REGION, {"trainWeeks": 26, "gap": 0.02})

    # A first run has no manifest to compare against and simply writes one.
    def test_a_missing_manifest_is_written(self, tmp_path):
        records.requireSameManifest(tmp_path, REGION, {"trainWeeks": 52})
        records.requireSameManifest(tmp_path, REGION, {"trainWeeks": 52})


class TestTrajectories:
    # One row per interval per series, so the day view can draw agent against oracle.
    def test_one_row_per_interval_per_series(self, tmp_path):
        rows = [
            {"fold": 0, "algo": "maskableppo", "trainMode": trainMode, "seed": 0,
             "priceSource": "realised", "testDay": DAY, "interval": n,
             "settlementdate": DAY + pd.Timedelta(minutes=5 * (n + 1)),
             "price": 50.0, "observedPrice": 48.0,
             "soc": 0.5, "bought": 0.0, "sold": 0.0, "profit": 0.0,
             "forecastRunTime": None}
            for trainMode in ("warm", "oracle") for n in range(288)
        ]
        path = records.writeTrajectories(tmp_path, REGION, 0, rows)
        frame = pd.read_parquet(path)
        assert len(frame) == 576
        assert set(frame.trainMode) == {"warm", "oracle"}
        assert frame.interval.max() == 287
        # What the agent was paid and what it saw are stored separately, so the day
        # view can draw the forecast against the realised price.
        assert frame.price.iloc[0] == 50.0 and frame.observedPrice.iloc[0] == 48.0

    # A trajectory written before forecastRunTime was recorded reads back with None, so
    # a resumed fold rewrites it under the current schema.
    def test_an_older_trajectory_reads_back_with_no_run(self, tmp_path):
        path = records.regionDir(tmp_path, REGION) / "fold00.parquet"
        path.parent.mkdir(parents=True)
        pd.DataFrame([{"fold": 0, "interval": 0}]).to_parquet(path)
        got = records.readTrajectories(tmp_path, REGION, 0)
        assert got[0]["forecastRunTime"] is None

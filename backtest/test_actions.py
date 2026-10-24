"""Tests for backtest/actions.py. Run with: uv run pytest backtest/ -q"""
from datetime import datetime

import pandas as pd
import pytest

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


class TestPriceSource:
    def both(self):
        realised = trajectory().assign(priceSource="realised", observedPrice=50.0)
        return pd.concat([trajectory(), realised], ignore_index=True)

    # The label is the runner's, copied: both sources come through, each as written.
    def test_both_sources_are_kept_under_their_own_label(self):
        got = actions.actionRows(self.both(), "NSW1", every=48)
        assert (got.priceSource.value_counts() == 288).all()
        assert set(got.priceSource) == {"forecast", "realised"}

    # A realised row read no forecast, so it names no run and no forecast price.
    def test_a_realised_row_has_no_forecast(self):
        got = actions.actionRows(self.both(), "NSW1", every=48)
        realised = got[got.priceSource == "realised"]
        assert realised.forecastRunTime.isna().all() and realised.forecastPrice.isna().all()
        assert (realised.realisedPrice == 50.0).all()
        forecast = got[got.priceSource == "forecast"]
        assert forecast.forecastRunTime.notna().all() and (forecast.forecastPrice == 48.0).all()

    # A realised agent saw what it was paid. One that saw anything else is mislabelled.
    def test_a_realised_row_that_saw_another_price_is_refused(self):
        frame = trajectory().assign(priceSource="realised")
        with pytest.raises(ValueError, match="not realised"):
            actions.actionRows(frame, "NSW1")

    def test_an_unknown_source_is_refused(self):
        with pytest.raises(ValueError, match="unknown priceSource"):
            actions.actionRows(trajectory().assign(priceSource="oracle"), "NSW1")

    # The oracle is not an agent decision, so it never reaches the export.
    def test_the_oracle_is_left_out(self):
        frame = pd.concat([trajectory(), trajectory().assign(algo="oracle")])
        assert set(actions.actionRows(frame, "NSW1").algo) == {"maskableppo"}


class TestExportName:
    # Every run's CSV lands in one folder, so the name alone must say which run it is.
    def test_a_run_named_for_its_region_keeps_its_name(self):
        assert actions.exportName("results/rolling2023_fc4h_SA1", "SA1") \
            == "rolling2023_fc4h_SA1.csv"

    def test_a_run_without_a_region_gets_one(self):
        assert actions.exportName("results/rolling2023_profit", "NSW1") \
            == "rolling2023_profit_NSW1.csv"


def rows(fold=0, seed=0, steps=3):
    frame = trajectory(steps).assign(fold=fold, seed=seed)
    frame["forecastRunTime"] = pd.Timestamp(DAY)
    return actions.actionRows(frame, "NSW1")


def appendFolds(path, seed, count):
    """One seed process: appends `count` folds of 288 rows, as the runner does."""
    for fold in range(count):
        actions.appendRows(path, rows(fold, seed, 288))


class TestLiveFile:
    # Every seed of a region writes the run's one CSV, beside where exportRun puts it.
    def test_a_seed_directory_names_the_run_by_its_parent(self, tmp_path):
        run = tmp_path / "results" / "rolling2024"
        assert actions.liveFile(run / "seed3", "NSW1") \
            == tmp_path / "results" / "actions" / "rolling2024_NSW1.csv"
        assert actions.liveFile(run, "NSW1") == actions.liveFile(run / "seed0", "NSW1")


class TestAppendRows:
    # The header is written once, by whichever fold arrives first.
    def test_one_header_however_many_folds(self, tmp_path):
        path = tmp_path / "actions" / "run_NSW1.csv"
        actions.appendRows(path, rows(0))
        actions.appendRows(path, rows(1))
        got = pd.read_csv(path)
        assert list(got.columns) == list(actions.columns)
        assert list(got.fold) == [0, 0, 0, 1, 1, 1]

    # Seeds are separate processes on one file. Each fold's block stays whole: every
    # row parses, none is lost, and a fold's rows sit together.
    def test_concurrent_seeds_do_not_interleave(self, tmp_path):
        import multiprocessing
        path = tmp_path / "run_NSW1.csv"
        context = multiprocessing.get_context("fork")
        workers = [context.Process(target=appendFolds, args=(path, seed, 20))
                   for seed in range(4)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
            assert worker.exitcode == 0
        got = pd.read_csv(path)
        assert len(got) == 4 * 20 * 288
        blocks = (got[["seed", "fold"]] != got[["seed", "fold"]].shift()).any(axis=1).sum()
        assert blocks == 4 * 20


class TestWrittenUnits:
    def test_a_missing_file_holds_nothing(self, tmp_path):
        assert actions.writtenUnits(tmp_path / "none.csv") == set()

    def test_it_lists_each_fold_seed_and_model(self, tmp_path):
        path = tmp_path / "run_NSW1.csv"
        actions.appendRows(path, rows(0, 0))
        actions.appendRows(path, rows(0, 1))
        assert actions.writtenUnits(path) == {(0, 0, "maskableppo", "warm", "forecast"),
                                             (0, 1, "maskableppo", "warm", "forecast")}


class TestRefuseExisting:
    # A run name is used once; a redo takes a new name.
    def test_a_fresh_name_starts(self, tmp_path):
        actions.refuseExisting(tmp_path / "results" / "new" / "seed0", "NSW1",
                               tmp_path / "backtestResults")

    def test_an_existing_actions_csv_refuses(self, tmp_path):
        run = tmp_path / "results" / "old"
        actions.appendRows(actions.liveFile(run, "NSW1"), rows())
        with pytest.raises(ValueError, match="already exists"):
            actions.refuseExisting(run / "seed0", "NSW1", tmp_path / "backtestResults")

    def test_a_published_region_refuses(self, tmp_path):
        (tmp_path / "backtestResults").mkdir()
        (tmp_path / "backtestResults" / "old_NSW1.csv").write_text("")
        with pytest.raises(ValueError, match="already exists"):
            actions.refuseExisting(tmp_path / "results" / "old_NSW1", "NSW1",
                                   tmp_path / "backtestResults")

    # Another region may still join a run name.
    def test_another_region_of_the_same_run_starts(self, tmp_path):
        (tmp_path / "backtestResults").mkdir()
        (tmp_path / "backtestResults" / "old_NSW1.csv").write_text("")
        actions.refuseExisting(tmp_path / "results" / "old", "SA1",
                               tmp_path / "backtestResults")

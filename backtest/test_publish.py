"""Tests for backtest/publish.py. Run with: uv run pytest backtest/ -q"""
import json

import pandas as pd
import pytest

from backtest import publish, records
from visualise.dashboard import data
from visualise.dashboard.test_data import REGION, dayRows, publishRun, writeActions


def finishedRun(results, name, region=REGION, sources=("realised",), every=48):
    """One region's merged rows, where the folds job leaves them."""
    rows = [dict(row, priceSource=source) for source in sources for row in dayRows()]
    for fold in sorted({row["fold"] for row in rows}):
        records.writeFold(results / name / "all", region, fold,
                          [row for row in rows if row["fold"] == fold])
    manifest = {"region": region}
    if every is not None:
        manifest["forecastEvery"] = every
    records.writeManifest(results / name / "all", region, manifest)
    return results / name


@pytest.fixture
def results(tmp_path):
    (tmp_path / "results").mkdir()
    return tmp_path / "results"


@pytest.fixture
def root(results):
    """Where the dashboard reads, beside results/actions/ as in the repository."""
    return results / "backtestResults"


class TestPublish:
    # Each region of an experiment is one CSV, named as its actions export is.
    def test_one_csv_per_region(self, results, root):
        for region in ("NSW1", "SA1"):
            publish.publish(finishedRun(results, f"rolling_{region}", region), root)
        assert sorted(path.name for path in root.iterdir()) == ["rolling_NSW1.csv",
                                                                 "rolling_SA1.csv"]
        assert data.availableRegions(root / "rolling") == ["NSW1", "SA1"]
        assert data.findResults(None, root=root) == root / "rolling"

    # Every fold's rows land in the one CSV, unchanged, with the forecast pull beside them.
    def test_the_copy_loads(self, results, root):
        run = finishedRun(results, "rolling")
        publish.publish(run, root)
        loaded = data.loadRun(REGION, root / "rolling")
        assert loaded.forecastEvery == 48
        pd.testing.assert_frame_equal(loaded.days,
                                      records.readDays(run / "all", REGION))

    # A forecast run is drawn with its actions export, found in results/actions where
    # git already tracks it, not copied.
    def test_a_forecast_run_reads_the_tracked_export(self, results, root):
        writeActions(results, "rolling")
        publish.publish(finishedRun(results, "rolling", sources=("forecast",)), root)
        assert data.actionsFile(root / "rolling", REGION).parent == results / "actions"
        assert [path.name for path in root.iterdir()] == [f"rolling_{REGION}.csv"]

    # A run the dashboard cannot draw is refused, and leaves nothing behind.
    def test_a_run_the_dashboard_cannot_load_is_refused(self, results, root):
        with pytest.raises(FileNotFoundError, match="actions export"):
            publish.publish(finishedRun(results, "rolling", sources=("forecast",)), root)
        assert not list(root.iterdir())

    def test_an_unstated_refresh_is_refused(self, results, root):
        writeActions(results, "rolling")
        with pytest.raises(ValueError, match="does not state forecastEvery"):
            publish.publish(finishedRun(results, "rolling", sources=("forecast",),
                                        every=None), root)

    # A run whose regions sit directly in it, as `cli rolling` writes one, publishes the
    # same way, and a merged `_all` run keeps its name without the suffix.
    def test_a_run_without_merged_seeds(self, results, root):
        records.writeFold(results / "rolling_all", REGION, 0, dayRows(2, folds=1))
        publish.publish(results / "rolling_all", root)
        assert data.availableRegions(root / "rolling") == [REGION]

    # A republish replaces the region whole, so a fold dropped upstream goes too.
    def test_a_republish_replaces_the_region(self, results, root):
        run = finishedRun(results, "rolling")
        publish.publish(run, root)
        records.regionDir(run / "all", REGION).joinpath("fold01.csv").unlink()
        publish.publish(run, root)
        assert set(data.loadRun(REGION, root / "rolling").days.fold) == {0}

    # A run from before the rename publishes under today's names, its values untouched.
    def test_an_old_run_is_published_under_current_names(self, results, root):
        run = finishedRun(results, "old")
        for path in records.regionDir(run / "all", REGION).glob("fold*.csv"):
            header, _, body = path.read_text().partition("\n")
            path.write_text(header.replace("trainMode", "arm") + "\n" + body)
        publish.publish(run, root)
        loaded = data.loadRun(REGION, root / "old").days
        assert "arm" not in loaded and set(loaded.trainMode) == {"warm"}
        assert len(loaded) == 14


def corpus(tmp_path, days, region=REGION, bound=38.0, gap=0.0, name="dp"):
    """A solved oracle corpus, as oracleResults/dp holds one: one JSON per region and
    day, named by its cache key."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    for index, day in enumerate(days):
        record = {"start": f"{day:%Y-%m-%d}T04:05:00", "profit": bound,
                  "dualBound": bound, "mipGap": gap, "status": "Optimal"}
        path = folder / f"{region}_{day:%Y-%m-%d}_0405_{index:016x}.json"
        path.write_text(json.dumps(record))
    return folder


class TestRestampOracle:
    """The oracle columns of a published run, rewritten from a solved corpus. Only the
    bound moves: the agent rows are what the run already measured."""

    def published(self, results, rows=None):
        return data.publishedFile(publishRun(results.parent, rows or dayRows()), REGION)

    def test_every_row_takes_the_corpus_bound(self, results, tmp_path):
        path = self.published(results)
        rows = records.readDayFile(path)
        before, after = publish.restampOracle(path, corpus(tmp_path, rows.testDay))
        stamped = records.readDayFile(path)
        assert list(stamped.oracleDualBound.unique()) == [38.0]
        assert list(stamped.oracleGap.unique()) == [0.0]
        # One bound per day, and both sums count a day once, as the dashboard does.
        assert (before, after) == (40.0 * 14, 38.0 * 14)

    def test_the_agent_rows_are_untouched(self, results, tmp_path):
        path = self.published(results)
        before = records.readDayFile(path)
        publish.restampOracle(path, corpus(tmp_path, before.testDay))
        after = records.readDayFile(path)
        keep = [column for column in after.columns
                if column not in ("oracleDualBound", "oracleGap")]
        pd.testing.assert_frame_equal(after[keep], before[keep])

    def test_a_missing_day_refuses_the_file(self, results, tmp_path):
        path = self.published(results)
        rows = records.readDayFile(path)
        with pytest.raises(FileNotFoundError, match="no oracle record"):
            publish.restampOracle(path, corpus(tmp_path, rows.testDay[:-1]))
        pd.testing.assert_frame_equal(records.readDayFile(path), rows)

    def test_two_records_for_one_day_refuse_the_file(self, results, tmp_path):
        path = self.published(results)
        rows = records.readDayFile(path)
        folder = corpus(tmp_path, rows.testDay)
        first = f"{REGION}_{rows.testDay[0]:%Y-%m-%d}_0405_ffffffffffffffff.json"
        (folder / first).write_text(json.dumps({"dualBound": 1.0, "mipGap": 0.0}))
        with pytest.raises(ValueError, match="different settings"):
            publish.restampOracle(path, folder)

    # A bound below an agent is not a bound. It means the corpus solved another problem,
    # so the file keeps the bound it was scored against.
    def test_a_bound_below_an_agent_refuses_the_file(self, results, tmp_path):
        path = self.published(results)
        rows = records.readDayFile(path)
        with pytest.raises(ValueError, match="beat the bound"):
            publish.restampOracle(path, corpus(tmp_path, rows.testDay, bound=5.0))
        pd.testing.assert_frame_equal(records.readDayFile(path), rows)

    # A file the dashboard cannot draw is not left on disk, as publishing leaves none.
    def test_a_file_the_dashboard_cannot_load_is_put_back(self, results, tmp_path):
        rows = [dict(row, priceSource="forecast") for row in dayRows()]
        path = self.published(results, rows)
        before = records.readDayFile(path)
        with pytest.raises(ValueError, match="does not state forecastEvery"):
            publish.restampOracle(path, corpus(tmp_path, before.testDay))
        pd.testing.assert_frame_equal(records.readDayFile(path), before)

    def test_the_restamped_file_still_loads(self, results, tmp_path):
        path = self.published(results)
        rows = records.readDayFile(path)
        publish.restampOracle(path, corpus(tmp_path, rows.testDay))
        run = path.parent / path.stem.removesuffix(f"_{REGION}")
        assert len(data.loadRun(REGION, run).days) == len(rows)

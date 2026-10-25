"""Tests for backtest/publish.py. Run with: uv run pytest backtest/ -q"""
import pandas as pd
import pytest

from backtest import publish, records
from visualise.dashboard import data
from visualise.dashboard.test_data import REGION, dayRows, writeActions


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

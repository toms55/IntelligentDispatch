"""Tests for backtest/publish.py. Run with: uv run pytest backtest/ -q"""
import json

import pytest

from backtest import publish, records
from visualise.dashboard import data
from visualise.dashboard.test_data import REGION, dayRows, writeActions


def finishedRun(results, name, region=REGION, sources=("realised",)):
    """One region's merged rows, where the folds job leaves them."""
    rows = [dict(row, priceSource=source) for source in sources for row in dayRows()]
    for fold in sorted({row["fold"] for row in rows}):
        records.writeFold(results / name / "all", region, fold,
                          [row for row in rows if row["fold"] == fold])
    manifest = {"region": region, "forecastEvery": 48}
    records.writeManifest(results / name / "all", region, manifest)
    return results / name


@pytest.fixture
def results(tmp_path, monkeypatch):
    # The export lookup falls back to results/actions, relative to the working directory.
    monkeypatch.setattr(data, "resultsRoot", tmp_path / "results")
    (tmp_path / "results").mkdir()
    return tmp_path / "results"


class TestPublish:
    # The five regional jobs of one experiment land in one directory the dashboard reads.
    def test_regions_of_one_experiment_share_a_run(self, results, tmp_path):
        root = tmp_path / "backtestResults"
        for region in ("NSW1", "SA1"):
            publish.publish(finishedRun(results, f"rolling_{region}", region), root)
        assert data.availableRegions(root / "rolling") == ["NSW1", "SA1"]
        assert data.findResults(None, root=root) == root / "rolling"

    # The copy holds the fold rows and the manifest, and loads as the dashboard loads it.
    def test_the_copy_loads(self, results, tmp_path):
        root = tmp_path / "backtestResults"
        publish.publish(finishedRun(results, "rolling"), root)
        run = data.loadRun(REGION, root / "rolling")
        assert len(run.days) == 14 and run.manifest["region"] == REGION

    # A forecast run is drawn with its actions export, found in results/actions where
    # git already tracks it, not copied.
    def test_a_forecast_run_reads_the_tracked_export(self, results, tmp_path):
        root = tmp_path / "backtestResults"
        writeActions(results, "rolling")
        publish.publish(finishedRun(results, "rolling", sources=("forecast",)), root)
        assert data.actionsFile(root / "rolling", REGION).parent == results / "actions"
        assert not (root / "actions").exists()

    # A run the dashboard cannot draw is refused, and leaves nothing behind.
    def test_a_run_the_dashboard_cannot_load_is_refused(self, results, tmp_path):
        root = tmp_path / "backtestResults"
        with pytest.raises(FileNotFoundError, match="actions export"):
            publish.publish(finishedRun(results, "rolling", sources=("forecast",)), root)
        assert not (root / "rolling" / REGION).exists()

    # A run whose regions sit directly in it, as `cli rolling` writes one, publishes the
    # same way, and a merged `_all` run keeps its name without the suffix.
    def test_a_run_without_merged_seeds(self, results, tmp_path):
        root = tmp_path / "backtestResults"
        records.writeFold(results / "rolling_all", REGION, 0, dayRows(2, folds=1))
        publish.publish(results / "rolling_all", root)
        assert data.availableRegions(root / "rolling") == [REGION]

    # A republish replaces the region whole, so a fold dropped upstream goes too.
    def test_a_republish_replaces_the_region(self, results, tmp_path):
        root = tmp_path / "backtestResults"
        publish.publish(finishedRun(results, "rolling"), root)
        stale = records.regionDir(root / "rolling", REGION) / "fold99.csv"
        stale.write_text("")
        publish.publish(results / "rolling", root)
        assert not stale.exists()

    # A run from before the rename publishes under today's names, its values untouched.
    def test_an_old_run_is_published_under_current_names(self, results, tmp_path):
        run = finishedRun(results, "old")
        regionDir = records.regionDir(run / "all", REGION)
        for path in regionDir.glob("fold*.csv"):
            header, _, body = path.read_text().partition("\n")
            path.write_text(header.replace("trainMode", "arm") + "\n" + body)
        manifest = records.manifestPath(run / "all", REGION)
        manifest.write_text(manifest.read_text().replace('"region"', '"arms": ["warm"], "region"'))
        root = tmp_path / "backtestResults"
        publish.publish(run, root)
        target = records.regionDir(root / "old", REGION)
        for path in regionDir.glob("fold*.csv"):
            got = (target / path.name).read_text().partition("\n")
            assert "trainMode" in got[0].split(",")
            assert got[2] == path.read_text().partition("\n")[2]
        manifest = json.loads(records.manifestPath(root / "old", REGION).read_text())
        assert manifest["trainModes"] == ["warm"] and "arms" not in manifest

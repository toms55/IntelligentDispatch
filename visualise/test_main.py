"""Tests for visualise/__main__.py. Run with: uv run pytest visualise/ -q

These drive `main` the way the shell does, with argv, so the argument wiring is under
test and not just the plotting functions below it.
"""
import pytest

from backtest import records
from visualise.__main__ import main
from visualise.test_rolling import _row, day1, day2

realisedFile = """## Realised prices
- region: NSW1
- window: 2025-01-01T04:00:00 .. 2026-01-01T04:00:00
- algo: maskableppo

| metric | value |
|---|---|
| oracle profit | 100.0000 |
| agent profit | 40.0000 +/- 10.0000 |
| regret | 60.0000 +/- 10.0000 |
| captured | 40.00% +/- 10.00% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | 30.0000 | 70.0000 | 30.00% |
| 1 | 50.0000 | 50.0000 | 50.00% |

## AEMO forecast
- region: NSW1
- window: 2025-01-01T04:00:00 .. 2026-01-01T04:00:00
- algo: maskableppo

| metric | value |
|---|---|
| oracle profit | 100.0000 |
| agent profit | 10.0000 +/- 5.0000 |
| regret | 90.0000 +/- 5.0000 |
| captured | 10.00% +/- 5.00% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | 5.0000 | 95.0000 | 5.00% |
| 1 | 15.0000 | 85.0000 | 15.00% |
"""


@pytest.fixture
def resultsFile(tmp_path):
    path = tmp_path / "NSW1.md"
    path.write_text(realisedFile)
    return path


@pytest.fixture
def rollingDir(tmp_path):
    rows = []
    for seed in (0, 1):
        for day in (day1, day2):
            rows.append(_row(seed, "realised", day, 4.0 + seed, 10.0))
            rows.append(_row(seed, "forecast", day, 1.0 + seed, 10.0))
    directory = tmp_path / "rolling"
    records.writeFold(directory, "NSW1", 0, rows)
    records.writeManifest(directory, "NSW1", {"trainWeeks": 52})
    return directory


class TestOutputPath:
    def test_makes_the_destination_directory(self, resultsFile, tmp_path):
        destination = tmp_path / "figures" / "deep" / "NSW1.png"
        assert main([str(resultsFile), "--compare-sources", "--out", str(destination)]) == 0
        assert destination.exists()

    def test_defaults_beside_a_results_file(self, resultsFile):
        assert main([str(resultsFile), "--compare-sources"]) == 0
        assert resultsFile.with_suffix(".png").exists()

    def test_defaults_inside_a_rolling_directory(self, rollingDir):
        assert main([str(rollingDir), "--region", "NSW1", "--compare-sources"]) == 0
        assert (rollingDir / "NSW1.png").exists()


class TestRollingInput:
    def test_a_directory_needs_a_region(self, rollingDir):
        with pytest.raises(SystemExit, match="--region"):
            main([str(rollingDir), "--compare-sources"])

    def test_reads_the_folds(self, rollingDir, tmp_path):
        destination = tmp_path / "NSW1.png"
        assert main([str(rollingDir), "--region", "NSW1", "--compare-sources",
                     "--out", str(destination)]) == 0
        assert destination.stat().st_size > 0


class TestTrainWindow:
    def test_both_halves_are_needed(self, resultsFile):
        with pytest.raises(SystemExit, match="must be given together"):
            main([str(resultsFile), "--compare-sources", "--train-start", "2024-01-01"])

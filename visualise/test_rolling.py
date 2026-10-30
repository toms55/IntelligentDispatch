"""Tests for visualise/rolling.py. Run with: uv run pytest visualise/ -q

The fixture writes real fold CSVs through `backtest.records`, so the column set under
test is the one the backtest actually writes, not a copy of it.
"""
import json

import pytest

from backtest import records
from visualise.rolling import sectionsFromRolling, trainNote

day1 = "2024-01-01 04:00:00"
day2 = "2024-01-02 04:00:00"


def _row(seed, source, day, agentProfit, oracle, fold=0):
    return {
        "fold": fold, "algo": "maskableppo", "trainMode": "warm", "seed": seed,
        "priceSource": source, "testDay": day, "episodeStart": day,
        "cycleBudget": 7.0, "agentProfit": agentProfit,
        "agentRevenue": agentProfit, "agentDegradation": 0.0, "agentCycles": 1.0,
        "oracleCycles": 6.0,
        "oracleDualBound": oracle, "oracleGap": 0.0, "spikeIntensity": 0.0,
        "highVol": False, "timesteps": 30000, "warmFrom": "",
        "writtenAt": "2026-01-01T00:00:00",
    }


@pytest.fixture
def resultsDir(tmp_path):
    """Two days, two price sources, two seeds. Realised beats forecast."""
    rows = []
    for seed, realised, forecast in ((0, 4.0, 1.0), (1, 6.0, 3.0)):
        rows.append(_row(seed, "realised", day1, realised, 10.0))
        rows.append(_row(seed, "realised", day2, realised, 10.0))
        rows.append(_row(seed, "forecast", day1, forecast, 10.0))
        rows.append(_row(seed, "forecast", day2, forecast, 10.0))
    records.writeFold(tmp_path, "NSW1", 0, rows)
    return tmp_path


class TestSectionsFromRolling:
    def test_one_section_per_price_source(self, resultsDir):
        sections = sectionsFromRolling(resultsDir, "NSW1")
        assert [s.heading for s in sections] == ["AEMO forecast", "Realised prices"]

    def test_the_oracle_counts_each_day_once_not_once_per_seed(self, resultsDir):
        # Two days at 10.0 each, over two seeds. Summing every row would give 40.0.
        sections = sectionsFromRolling(resultsDir, "NSW1")
        assert all(s.oracleProfit == pytest.approx(20.0) for s in sections)

    def test_a_seed_pools_its_own_days(self, resultsDir):
        realised = next(s for s in sectionsFromRolling(resultsDir, "NSW1")
                        if "Realised" in s.heading)
        assert {s.seed: s.agentProfit for s in realised.seeds} == {0: 8.0, 1: 12.0}

    def test_captured_is_a_ratio_of_sums(self, resultsDir):
        realised = next(s for s in sectionsFromRolling(resultsDir, "NSW1")
                        if "Realised" in s.heading)
        assert realised.bestSeed().seed == 1
        assert realised.bestSeed().captured == pytest.approx(12.0 / 20.0)

    def test_regret_is_the_oracle_less_the_agent(self, resultsDir):
        realised = next(s for s in sectionsFromRolling(resultsDir, "NSW1")
                        if "Realised" in s.heading)
        assert realised.bestSeed().regret == pytest.approx(8.0)

    def test_the_window_spans_the_test_days(self, resultsDir):
        section = sectionsFromRolling(resultsDir, "NSW1")[0]
        assert section.windowStart == "2024-01-01T04:00:00"
        assert section.windowEnd == "2024-01-02T04:00:00"

    def test_several_algos_without_a_filter_raise(self, resultsDir, tmp_path):
        rows = [_row(0, "realised", day1, 1.0, 10.0, fold=1)]
        rows[0]["algo"] = "dqn"
        records.writeFold(tmp_path, "NSW1", 1, rows)
        with pytest.raises(ValueError, match="several algo"):
            sectionsFromRolling(resultsDir, "NSW1")
        sections = sectionsFromRolling(resultsDir, "NSW1", algo="maskableppo")
        assert sections[0].algo == "maskableppo"

    def test_an_empty_region_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            sectionsFromRolling(tmp_path, "NSW1")

    def test_a_filter_that_matches_nothing_raises(self, resultsDir):
        with pytest.raises(ValueError, match="no rows left"):
            sectionsFromRolling(resultsDir, "NSW1", algo="a2c")


class TestTrainNote:
    def test_reads_the_train_length_from_the_manifest(self, resultsDir):
        records.writeManifest(resultsDir, "NSW1", {"trainWeeks": 52})
        assert trainNote(resultsDir, "NSW1") == "rolling 52 weeks before each test week"

    def test_is_empty_without_a_manifest(self, resultsDir):
        assert trainNote(resultsDir, "NSW1") == ""

    def test_is_empty_when_the_manifest_does_not_say(self, resultsDir):
        path = records.manifestPath(resultsDir, "NSW1")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"region": "NSW1"}))
        assert trainNote(resultsDir, "NSW1") == ""

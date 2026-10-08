"""Tests for visualise/parse.py. Run with: uv run pytest visualise/ -q"""
import pytest

from visualise.parse import loadResultsFile, parseSections

REALISED = """## Realised prices
- region: NSW1
- window: 2025-01-01T04:00:00 .. 2026-01-01T04:00:00
- algo: maskableppo
- seeds: 0, 1
- battery: 0.06 MW / 0.1 MWh, replacement cost 150000
- price slots from: realised prices
- generated: 2026-10-09T01:57:23

| metric | value |
|---|---|
| oracle profit | 15439.7117 |
| agent profit | 1811.3827 +/- 2780.0732 |
| regret | 13628.3289 +/- 2780.0732 |
| captured | 11.73% +/- 18.01% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | -1616.6409 | 17056.3526 | -10.47% |
| 1 | 2799.1074 | 12640.6043 | 18.13% |
"""

TWO_SECTIONS = REALISED + """
## AEMO forecast
- region: NSW1
- window: 2025-01-01T04:00:00 .. 2026-01-01T04:00:00
- algo: maskableppo
- seeds: 0, 1
- battery: 0.06 MW / 0.1 MWh, replacement cost 150000
- price slots from: AEMO forecast
- generated: 2026-10-09T02:23:31

| metric | value |
|---|---|
| oracle profit | 15439.7117 |
| agent profit | -5692.8550 +/- 12235.4069 |
| regret | 21132.5667 +/- 12235.4069 |
| captured | -36.87% +/- 79.25% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | -27514.3834 | 42954.0951 | -178.21% |
| 1 | 0.0000 | 15439.7117 | 0.00% |
"""


class TestParseSections:
    def test_reads_the_heading_and_identifying_fields(self):
        got = parseSections(REALISED)
        assert len(got) == 1
        section = got[0]
        assert section.heading == "Realised prices"
        assert section.region == "NSW1"
        assert section.algo == "maskableppo"

    def test_reads_the_window(self):
        section = parseSections(REALISED)[0]
        assert section.windowStart == "2025-01-01T04:00:00"
        assert section.windowEnd == "2026-01-01T04:00:00"

    def test_best_seed_is_the_highest_agent_profit(self):
        section = parseSections(REALISED)[0]
        best = section.bestSeed()
        assert best.seed == 1
        assert best.agentProfit == pytest.approx(2799.1074)

    def test_reads_the_summary_metrics(self):
        section = parseSections(REALISED)[0]
        assert section.oracleProfit == pytest.approx(15439.7117)
        assert section.agentProfitMean == pytest.approx(1811.3827)
        assert section.agentProfitSd == pytest.approx(2780.0732)
        assert section.regretMean == pytest.approx(13628.3289)
        assert section.regretSd == pytest.approx(2780.0732)

    def test_captured_is_a_fraction_not_a_percent(self):
        section = parseSections(REALISED)[0]
        assert section.capturedMean == pytest.approx(0.1173)
        assert section.capturedSd == pytest.approx(0.1801)

    def test_reads_one_row_per_seed(self):
        section = parseSections(REALISED)[0]
        assert [s.seed for s in section.seeds] == [0, 1]
        assert section.seeds[1].agentProfit == pytest.approx(2799.1074)
        assert section.seeds[1].regret == pytest.approx(12640.6043)
        assert section.seeds[1].captured == pytest.approx(0.1813)

    def test_a_negative_captured_percent_round_trips(self):
        section = parseSections(REALISED)[0]
        assert section.seeds[0].captured == pytest.approx(-0.1047)

    def test_two_sections_stay_independent(self):
        got = parseSections(TWO_SECTIONS)
        assert [s.heading for s in got] == ["Realised prices", "AEMO forecast"]
        assert got[0].agentProfitMean != got[1].agentProfitMean
        assert len(got[1].seeds) == 2

    def test_zero_profit_is_not_mistaken_for_missing(self):
        got = parseSections(TWO_SECTIONS)[1]
        assert got.seeds[1].agentProfit == 0.0
        assert got.seeds[1].captured == 0.0

    def test_refuses_a_section_with_no_seed_rows(self):
        truncated = REALISED.split("| seed |")[0]
        with pytest.raises(ValueError, match="per-seed table"):
            parseSections(truncated)


class TestLoadResultsFile:
    def test_reads_from_disk(self, tmp_path):
        path = tmp_path / "NSW1.md"
        path.write_text(TWO_SECTIONS)
        got = loadResultsFile(path)
        assert len(got) == 2
        assert got[0].region == "NSW1"

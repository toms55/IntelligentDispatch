"""Tests for visualise/plots.py. Run with: uv run pytest visualise/ -q

Only the non-rendering logic is asserted on directly (picking the best seed,
formatting timestamps); the figure builders are smoke-tested for "runs and
produces a file", not pixel content -- matplotlib's own test suite covers that.
"""
import pytest

from visualise.parse import parseSections
from visualise.plots import (
    _bestAcrossSections, _shortTimestamp, buildBestSeedFigure, buildSourceComparisonFigure,
    saveBestSeedFigure, saveSourceComparisonFigure,
)

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

FORECAST = """## AEMO forecast
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
| agent profit | 823.3769 +/- 100.0 |
| regret | 14616.3348 +/- 100.0 |
| captured | 5.33% +/- 1.00% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | 823.3769 | 14616.3348 | 5.33% |
| 1 | 0.0000 | 15439.7117 | 0.00% |
"""


class TestShortTimestamp:
    def test_keeps_the_hour_and_minute(self):
        assert _shortTimestamp("2025-01-01T04:00:00") == "2025-01-01 04:00"

    def test_a_bare_date_passes_through(self):
        assert _shortTimestamp("2025-01-01") == "2025-01-01"


class TestBestAcrossSections:
    def test_picks_the_higher_profit_seed_regardless_of_section(self):
        sections = parseSections(REALISED + "\n" + FORECAST)
        section, seed = _bestAcrossSections(sections)
        assert section.heading == "Realised prices"
        assert seed.seed == 1
        assert seed.agentProfit == pytest.approx(2799.1074)

    def test_a_single_section_just_returns_its_own_best(self):
        sections = parseSections(REALISED)
        section, seed = _bestAcrossSections(sections)
        assert seed.seed == 1


class TestBuildBestSeedFigure:
    def test_runs_without_a_train_window(self):
        sections = parseSections(REALISED)
        figure = buildBestSeedFigure(sections, "NSW1", "maskableppo")
        assert "train" not in figure._suptitle.get_text()
        assert "test" in figure._suptitle.get_text()

    def test_includes_both_dates_when_a_train_window_is_given(self):
        sections = parseSections(REALISED)
        figure = buildBestSeedFigure(
            sections, "NSW1", "maskableppo",
            trainWindow=("2024-01-01T04:00:00", "2025-01-01T04:00:00"),
        )
        title = figure._suptitle.get_text()
        assert "2024-01-01" in title
        assert "2025-01-01" in title
        assert "2026-01-01" in title  # the test window's end, still present

    def test_source_restricts_which_section_is_used(self):
        sections = parseSections(REALISED + "\n" + FORECAST)
        figure = buildBestSeedFigure(sections, "NSW1", "maskableppo", source="forecast")
        assert "AEMO forecast" in figure.axes[0].get_title() or any(
            "AEMO forecast" in label.get_text()
            for label in figure.axes[0].get_xticklabels()
        )


class TestSaveBestSeedFigure:
    def test_writes_a_file(self, tmp_path):
        sections = parseSections(REALISED)
        destination = tmp_path / "best.png"
        saveBestSeedFigure(sections, "NSW1", "maskableppo", destination)
        assert destination.exists()
        assert destination.stat().st_size > 0


class TestBuildSourceComparisonFigure:
    def test_draws_one_panel_per_source(self):
        sections = parseSections(REALISED + "\n" + FORECAST)
        figure = buildSourceComparisonFigure(sections, "NSW1", "maskableppo")
        assert len(figure.axes) == 2

    def test_each_panel_keeps_its_own_captured_share(self):
        sections = parseSections(REALISED + "\n" + FORECAST)
        figure = buildSourceComparisonFigure(sections, "NSW1", "maskableppo")
        titles = [ax.get_title() for ax in figure.axes]
        assert titles[0] != titles[1]
        assert all("of the oracle's profit" in title for title in titles)

    def test_each_panel_names_its_own_price_source(self):
        sections = parseSections(REALISED + "\n" + FORECAST)
        figure = buildSourceComparisonFigure(sections, "NSW1", "maskableppo")
        labels = [
            " ".join(label.get_text() for label in ax.get_xticklabels())
            for ax in figure.axes
        ]
        assert "Realised prices" in labels[0]
        assert "AEMO forecast" in labels[1]

    def test_every_panel_carries_the_train_window(self):
        sections = parseSections(REALISED + "\n" + FORECAST)
        figure = buildSourceComparisonFigure(
            sections, "NSW1", "maskableppo",
            trainWindow=("2024-01-01T04:00:00", "2025-01-01T04:00:00"),
        )
        headers = [panel._suptitle.get_text() for panel in figure.subfigs]
        assert len(headers) == 2
        assert all("train: 2024-01-01 04:00" in header for header in headers)


class TestSaveSourceComparisonFigure:
    def test_writes_a_file(self, tmp_path):
        sections = parseSections(REALISED + "\n" + FORECAST)
        destination = tmp_path / "compare.png"
        saveSourceComparisonFigure(sections, "NSW1", "maskableppo", destination)
        assert destination.exists()
        assert destination.stat().st_size > 0

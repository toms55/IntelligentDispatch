"""Tests for visualise/dashboard/figures.py. Run with: uv run pytest visualise/ -q

Each function returns numbers or a Plotly figure, so none of this needs a server.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest import metrics
from visualise.dashboard import figures
from visualise.dashboard.test_data import DAY, dayRows, run, REGION  # noqa: F401
from visualise.dashboard import data


@pytest.fixture
def frame():
    return pd.DataFrame(dayRows()).astype({"testDay": "datetime64[ns]"})


def trajectoryRows(algo="maskableppo", arm="warm", sources=("realised", "forecast")):
    rows = []
    for source in sources:
        for step in range(288):
            rows.append({
                "fold": 0, "algo": algo, "arm": arm, "seed": 0,
                "priceSource": source, "testDay": DAY, "interval": step,
                "settlementdate": DAY + timedelta(minutes=5 * (step + 1)),
                "price": 50.0 + step, "observedPrice": 50.0 + step + 3.0,
                "soc": 0.5, "bought": 0.0, "sold": 0.0, "profit": 0.1,
            })
    return rows


@pytest.fixture
def trajectory():
    """One day for the realised and the forecast arm, as the runner writes it."""
    return pd.DataFrame(trajectoryRows())


@pytest.fixture
def withOracle(trajectory):
    """The same day plus the arm="oracle" series the runner writes beside the agent."""
    oracle = pd.DataFrame(trajectoryRows(algo="oracle", arm="oracle",
                                         sources=("realised",)))
    oracle["soc"] = np.linspace(0.5, 0.9, 288)
    return pd.concat([trajectory, oracle], ignore_index=True)


class TestSummaryTiles:
    # Every tile value comes from backtest.metrics, never from a second calculation.
    def test_the_values_come_from_metrics(self, frame):
        tiles = {tile["label"]: tile["raw"] for tile in figures.summaryTiles(frame)}
        wanted = metrics.summarise(frame)
        assert tiles["Captured"] == pytest.approx(wanted["captured"])
        assert tiles["Agent profit"] == pytest.approx(wanted["agentProfit"])
        assert tiles["Win rate"] == pytest.approx(wanted["winRate"])

    # The high and low volatility split is always shown, both sides present.
    def test_both_volatility_buckets_are_reported(self, frame):
        buckets = figures.volatilityTiles(frame)
        assert [row["bucket"] for row in buckets] == ["High volatility", "Other days"]
        assert buckets[0]["days"] + buckets[1]["days"] == len(metrics.perDay(frame))

    # An empty selection gives tiles that say so rather than raising.
    def test_an_empty_frame_renders_a_message(self, frame):
        tiles = figures.summaryTiles(frame.iloc[0:0])
        assert all(tile["value"] == "no data" for tile in tiles)


class TestModelPanel:
    # Every row is read from the manifest, so config.py cannot drift from the display.
    def test_it_reads_the_manifest(self):
        rows = dict(figures.modelPanel({"algos": ["dqn"], "trainWeeks": 26,
                                        "warmTimesteps": 1, "coldTimesteps": 2,
                                        "seeds": [0, 1], "arms": ["warm"],
                                        "horizon": 288, "capDate": "2026-01-01T04:00:00",
                                        "priceSources": ["realised"]}))
        assert rows["Algorithm"] == "dqn"
        assert rows["Training window"] == "26 weeks"

    # An algorithm the dashboard has never seen renders without any change to it.
    def test_an_unknown_algorithm_renders(self):
        rows = dict(figures.modelPanel({"algos": ["sac"]}))
        assert rows["Algorithm"] == "sac"

    # The MDP description is fixed by config.py and states the observation layout.
    def test_the_mdp_rows_state_the_observation_layout(self):
        rows = dict(figures.modelPanel({}))
        assert "60" in rows["Observation"]
        assert "3" in rows["Actions"]


class TestCalendar:
    # One filled cell per test day, no gaps and no extras.
    def test_one_cell_per_test_day(self, frame):
        grid = figures.calendarFigure(frame).data[0].z
        assert int(np.isfinite(np.asarray(grid, dtype=float)).sum()) == len(
            metrics.perDay(frame))

    # The high-volatility marker sits on exactly the flagged days.
    def test_the_marker_marks_only_high_volatility_days(self, frame):
        figure = figures.calendarFigure(frame)
        marked = [trace for trace in figure.data if trace.name == "High volatility"][0]
        assert len(marked.x) == int(metrics.perDay(frame).highVol.sum())

    # Each cell carries its day and fold so the click callback knows what was picked.
    def test_each_cell_carries_its_day_and_fold(self, frame):
        cells = figures.calendarFigure(frame).data[0]
        stamped = {str(entry[0]) for week in cells.customdata for entry in week}
        assert str(pd.Timestamp(DAY).date()) in stamped
        assert np.asarray(cells.customdata).shape[:2] == np.asarray(cells.z).shape

    # Profit polarity is a diverging scale centred on zero, so a loss cannot read as a win.
    def test_the_colour_scale_is_centred_on_zero(self, frame):
        cells = figures.calendarFigure(frame).data[0]
        assert cells.zmid == 0 and cells.zmin == -cells.zmax

    # One outlier day must not flatten every other cell to the midpoint.
    def test_the_scale_clamps_on_an_outlier(self, frame):
        spiked = frame.copy()
        spiked.loc[spiked.index[0], "agentProfit"] = 10_000.0
        figure = figures.calendarFigure(spiked, metric="agentProfit")
        assert figure.data[0].zmax < spiked.agentProfit.max()

    # Captured share is the default, because day profit is too skewed to colour raw.
    def test_captured_is_the_default_metric(self, frame):
        cells = figures.calendarFigure(frame).data[0]
        assert cells.zmax <= 1.0
        assert "Captured" in cells.colorbar.title.text

    # An empty selection renders an explicit message rather than an empty grid.
    def test_an_empty_frame_renders_a_message(self, frame):
        figure = figures.calendarFigure(frame.iloc[0:0])
        assert figure.layout.annotations and "no" in figure.layout.annotations[0].text


class TestDayView:
    # The price panel draws both the realised path and the forecast the agent saw.
    def test_it_draws_both_price_paths(self, trajectory):
        figure = figures.dayFigure(trajectory, DAY)
        names = [trace.name for trace in figure.data]
        assert "Realised price" in names and "Forecast seen" in names

    # The oracle series the runner wrote is drawn beside the agent.
    def test_the_oracle_soc_joins_the_agent(self, withOracle):
        figure = figures.dayFigure(withOracle, DAY)
        assert "Oracle SoC" in [trace.name for trace in figure.data]

    # Selecting one agent arm must not filter the oracle series out with it.
    def test_an_arm_filter_keeps_the_oracle(self, withOracle):
        figure = figures.dayFigure(withOracle, DAY, algo="maskableppo", arm="warm")
        names = [trace.name for trace in figure.data]
        assert "Oracle SoC" in names and "Agent SoC" in names

    # A run written before the oracle series existed still renders, agent alone.
    def test_no_oracle_still_renders(self, trajectory):
        figure = figures.dayFigure(trajectory, DAY)
        assert "Oracle SoC" not in [trace.name for trace in figure.data]

    # Price and state of charge never share one y axis.
    def test_price_and_soc_are_separate_panels(self, trajectory):
        figure = figures.dayFigure(trajectory, DAY)
        axes = {trace.yaxis for trace in figure.data}
        assert axes == {"y", "y2"}

    # A day with no trajectory rows says so rather than drawing an empty frame.
    def test_an_empty_day_renders_a_message(self, trajectory):
        figure = figures.dayFigure(trajectory.iloc[0:0], DAY)
        assert figure.layout.annotations and "no" in figure.layout.annotations[0].text

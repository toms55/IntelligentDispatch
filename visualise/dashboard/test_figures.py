"""Tests for visualise/dashboard/figures.py. Run with: uv run pytest visualise/ -q

Each function returns numbers or a Plotly figure, so none of this needs a server.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import config
from backtest import metrics
from visualise.dashboard import figures
from visualise.dashboard.test_data import DAY, dayRows, run, REGION  # noqa: F401
from visualise.dashboard import data


@pytest.fixture
def frame():
    return pd.DataFrame(dayRows()).astype({"testDay": "datetime64[ns]"})


def trajectoryRows(algo="maskableppo", trainMode="warm", sources=("realised", "forecast")):
    rows = []
    for source in sources:
        for step in range(288):
            rows.append({
                "fold": 0, "algo": algo, "trainMode": trainMode, "seed": 0,
                "priceSource": source, "testDay": DAY, "interval": step,
                "settlementdate": DAY + timedelta(minutes=5 * (step + 1)),
                "price": 50.0 + step, "observedPrice": 50.0 + step + 3.0,
                "soc": 0.5, "bought": 0.0, "sold": 0.0, "profit": 0.1,
            })
    return rows


@pytest.fixture
def trajectory():
    """One day for the realised and the forecast price source, as the runner writes it."""
    return pd.DataFrame(trajectoryRows())


@pytest.fixture
def withOracle(trajectory):
    """The same day plus the trainMode="oracle" series the runner writes beside the agent."""
    oracle = pd.DataFrame(trajectoryRows(algo="oracle", trainMode="oracle",
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

    # One day has no spread across folds, so it gets tiles about the day itself.
    def test_one_day_describes_the_day(self, frame):
        one = frame[frame.testDay == frame.testDay.iloc[0]].iloc[0:1]
        tiles = {tile["label"]: tile["raw"] for tile in figures.summaryTiles(one)}
        assert "Win rate" not in tiles and "Days" not in tiles
        assert tiles["Revenue"] == pytest.approx(one.agentRevenue.iloc[0])
        assert "Degradation" not in tiles
        assert tiles["Cap Payout"] == pytest.approx(one.spikeIntensity.iloc[0])

    # The high and low volatility split is always shown, both sides present.
    def test_an_empty_frame_renders_a_message(self, frame):
        tiles = figures.summaryTiles(frame.iloc[0:0])
        assert all(tile["value"] == "no data" for tile in tiles)


class TestForecastTiles:
    def planned(self):
        """Two rows of a finished suggestion, with both totals already run forward."""
        return pd.DataFrame({
            "settlementdate": pd.date_range("2025-01-01 10:05", periods=2, freq="5min"),
            "price": [10.0, 300.0], "action": ["charge", "discharge"],
            "soc": [0.55, 0.40], "revenue": [-0.05, 1.45], "profit": [-0.07, 1.20],
        })

    # The screen names both cards, so neither number has to be guessed at.
    def test_it_names_the_two_cards(self):
        labels = [tile["label"] for tile in figures.forecastTiles(self.planned())]
        assert labels == ["Expected revenue", "Current state of charge"]

    # The last row is the whole horizon, so the card reads it rather than summing again.
    def test_the_revenue_is_the_last_running_total(self):
        frame = self.planned()
        tiles = {tile["label"]: tile for tile in figures.forecastTiles(frame)}
        assert tiles["Expected revenue"]["raw"] == pytest.approx(1.45)
        assert tiles["Expected revenue"]["value"] == "$1.45"
        # Profit sits under it, because revenue alone does not pay for the cycling.
        assert "$1.20 after degradation" == tiles["Expected revenue"]["sub"]

    # The card reports where the battery starts, which the screen assumes.
    def test_the_state_of_charge_is_the_starting_one(self):
        tiles = {tile["label"]: tile for tile in figures.forecastTiles(self.planned())}
        assert tiles["Current state of charge"]["raw"] == pytest.approx(config.initial_soc)
        assert tiles["Current state of charge"]["value"] == "50.0%"
        assert "40.0%" in tiles["Current state of charge"]["sub"]

    # A caller may hold a real battery state, so the starting point is not hard coded.
    def test_the_starting_state_of_charge_is_given(self):
        tiles = figures.forecastTiles(self.planned(), initialSoc=0.25)
        assert tiles[1]["value"] == "25.0%"

    # No forecast means no revenue, read the way every other tile reads no data.
    def test_an_empty_frame_has_no_revenue(self):
        tiles = figures.forecastTiles(pd.DataFrame())
        assert tiles[0]["value"] == "no data"
        # The state of charge is still known: the screen assumes it, the agent does not
        # choose it.
        assert tiles[1]["value"] == "50.0%"

    # A model with no checkpoint still draws prices, and those carry no dispatch.
    def test_a_price_only_frame_has_no_revenue(self):
        frame = pd.DataFrame({"settlementdate": [pd.Timestamp("2025-01-01 10:05")],
                              "price": [10.0]})
        tiles = figures.forecastTiles(frame)
        assert tiles[0]["value"] == "no data" and tiles[0]["sub"] == "no dispatch planned"


def _traces(figure, mode):
    """The scatter traces drawn in one mode. The heatmap has no mode at all."""
    return [trace for trace in figure.data if getattr(trace, "mode", None) == mode]


class TestMonth:
    # The grid holds whole weeks, and every day of the month carries its number.
    def test_it_draws_one_cell_per_day_of_the_month(self, frame):
        figure = figures.monthFigure(frame, DAY)
        grid = np.array(figure.data[0].z)
        assert grid.shape[1] == 7 and grid.size % 7 == 0
        numbers = _traces(figure, "text")[0]
        assert list(numbers.text) == [str(n) for n in range(1, 32)]

    # A day the backtest never tested has no fill, but keeps its number.
    def test_an_untested_day_is_unfilled(self, frame):
        figure = figures.monthFigure(frame, DAY)
        grid = np.array(figure.data[0].z, dtype=float)
        # The fixture writes 14 days from the 1st, so the 20th was never tested.
        assert np.isfinite(grid).sum() == 14
        numbers = _traces(figure, "text")[0]
        assert "20" in list(numbers.text)

    # Fill is profit, and the scale is centred on zero so a loss cannot read as a win.
    def test_the_colour_scale_is_centred_on_zero(self, frame):
        figure = figures.monthFigure(frame, DAY)
        heatmap = figure.data[0]
        assert heatmap.zmid == 0 and heatmap.zmin == -heatmap.zmax
        grid = np.array(heatmap.z, dtype=float)
        assert np.nanmax(grid) == frame.agentProfit.max()

    # Green above zero and red below, as the top of the scale and the bottom.
    def test_green_is_the_upper_pole(self, frame):
        scale = figures.monthFigure(frame, DAY).data[0].colorscale
        assert scale[0][1] == figures.lossPole and scale[-1][1] == figures.profitPole

    # An outlier day cannot flatten every other day to the midpoint.
    def test_the_scale_clamps_on_an_outlier(self, frame):
        loud = frame.copy()
        loud.loc[0, "agentProfit"] = 10000.0
        assert (figures.monthFigure(loud, DAY).data[0].zmax
                < figures.monthFigure(loud, DAY).data[0].z[0][2])

    # The amber dot marks the high-volatility days and nothing else.
    def test_the_dot_marks_only_high_volatility_days(self, frame):
        figure = figures.monthFigure(frame, DAY)
        dots = _traces(figure, "markers")[0]
        assert dots.marker.color == figures.volDot
        assert len(dots.x) == int(frame.highVol.sum())

    # The dot sits under the date, so hue is never the only thing that separates them.
    def test_the_dot_sits_under_the_number(self, frame):
        figure = figures.monthFigure(frame, DAY)
        numbers = _traces(figure, "text")[0]
        dots = _traces(figure, "markers")[0]
        assert dots.y[0] > numbers.y[0]

    # A month with no high-volatility day draws no marker trace at all.
    def test_a_quiet_month_has_no_dot(self, frame):
        quiet = frame.copy()
        quiet["highVol"] = False
        assert not _traces(figures.monthFigure(quiet, DAY), "markers")

    # Every cell carries the date the click handler reads, and its fold.
    def test_each_cell_carries_its_day_and_fold(self, frame):
        extra = np.array(figures.monthFigure(frame, DAY).data[0].customdata,
                         dtype=object)
        # 1 January 2025 is a Wednesday, so it sits in the third column.
        assert extra[0][2][0] == "2025-01-01"
        assert extra[0][2][1] == 0

    # The selected day is outlined, so the stepper and the field show where they are.
    def test_the_selected_day_is_outlined(self, frame):
        figure = figures.monthFigure(frame, DAY, selected="2025-01-08")
        assert len(figure.layout.shapes) == 1
        assert not figures.monthFigure(frame, DAY).layout.shapes

    # A day outside the drawn month cannot be outlined.
    def test_a_selection_in_another_month_draws_nothing(self, frame):
        assert not figures.monthFigure(frame, DAY, selected="2025-02-08").layout.shapes

    # A month with no results still draws the month rather than an error.
    def test_an_empty_frame_still_draws_the_month(self, frame):
        figure = figures.monthFigure(frame.iloc[0:0], DAY)
        numbers = _traces(figure, "text")[0]
        assert len(numbers.text) == 31



def forecastRows(every, start=DAY):
    """One day of the actions export: one forecast price per interval, read from a run
    published every `every` intervals from 04:00."""
    clock = pd.date_range(start + timedelta(minutes=5), periods=288, freq="5min")
    runs = [start + timedelta(minutes=5 * (step // every * every)) for step in range(288)]
    return pd.DataFrame({"settlementdate": clock, "forecastPrice": np.linspace(20, 80, 288),
                         "forecastRunTime": runs})


class TestForecastTile:
    def _one(self, frame, source):
        one = frame[frame.testDay == frame.testDay.iloc[0]].iloc[0:1].copy()
        one["priceSource"] = source
        return one

    # The one-day tiles read Revenue before the oracle, and the forecast tile takes the
    # place degradation had.
    def test_the_one_day_order(self, frame):
        tiles = figures.summaryTiles(self._one(frame, "forecast"), forecastRows(48))
        assert [tile["label"] for tile in tiles] == [
            "Captured", "Agent profit", "Revenue", "Oracle profit", "Forecast",
            "Cap Payout"]

    # The wording comes from the run times in the export, so any refresh reads right.
    @pytest.mark.parametrize("every, heading, sub", [
        (48, "Refreshed every 4 hours", "Every 4 hours, a new AEMO forecast was pulled "
         "for the backtest, starting at 04:00."),
        (1, "Refreshed every 5 minutes", "Every 5 minutes, a new AEMO forecast was "
         "pulled for the backtest, starting at 04:00."),
        (12, "Refreshed every hour", "Every hour, a new AEMO forecast was pulled for "
         "the backtest, starting at 04:00."),
        (288, "Pulled once a day", "One AEMO forecast was pulled at 04:00 and held for "
         "the whole trading day."),
    ])
    def test_the_refresh_is_read_from_the_runs(self, frame, every, heading, sub):
        tile = figures.summaryTiles(self._one(frame, "forecast"), forecastRows(every))[4]
        assert (tile["value"], tile["sub"]) == (heading, sub)
        assert tile["accent"] == figures.forecastLine

    def test_a_realised_day_says_no_forecast(self, frame):
        tile = figures.summaryTiles(self._one(frame, "realised"))[4]
        assert tile["value"] == "No forecast" and tile["accent"] is None

    def test_a_missing_export_says_so(self, frame):
        tile = figures.summaryTiles(self._one(frame, "forecast"), None)[4]
        assert tile["value"] == "Refresh unknown"

    # Several days have no single source, so the tile is not offered.
    def test_several_days_have_no_forecast_tile(self, frame):
        assert "Forecast" not in [tile["label"] for tile in figures.summaryTiles(frame)]


class TestForecastOverlay:
    def _path(self, trajectory):
        return data.dispatchPath(trajectory, priceSource="forecast")

    def test_the_forecast_line_and_chip(self, trajectory):
        figure = figures.dayFigure(self._path(trajectory), [], DAY,
                                   forecast=forecastRows(48))
        names = [trace.name for trace in figure.data]
        assert "Forecast acted on" in names
        assert "refreshed every 4 hours" in figure.layout.title.text
        rules = [shape for shape in figure.layout.shapes if shape.type == "line"
                 and shape.line.color == figures.forecastLine]
        assert len(rules) == 6

    # 288 runs would bury the chart in rules; the hover and the tile carry the run.
    def test_a_five_minute_refresh_draws_no_rules(self, trajectory):
        figure = figures.dayFigure(self._path(trajectory), [], DAY,
                                   forecast=forecastRows(1))
        assert not [shape for shape in figure.layout.shapes
                    if shape.line.color == figures.forecastLine]

    def test_a_realised_day_is_unchanged(self, trajectory):
        figure = figures.dayFigure(data.dispatchPath(trajectory, priceSource="realised"),
                                   [], DAY)
        assert "Forecast acted on" not in [trace.name for trace in figure.data]


class TestDayView:
    # The day view is the forecast screen's figure over realised prices, so it carries
    # the price path and the key to the two block colours.
    def test_it_draws_the_realised_path_and_the_block_key(self, trajectory):
        path = data.dispatchPath(trajectory, priceSource="realised")
        figure = figures.dayFigure(path, [], DAY)
        names = [trace.name for trace in figure.data]
        assert names == ["Realised price", "Discharge: sell", "Charge: buy"]

    # One y axis, because the figure draws price alone.
    def test_everything_sits_on_one_axis(self, trajectory):
        path = data.dispatchPath(trajectory, priceSource="realised")
        figure = figures.dayFigure(path, [], DAY)
        assert {trace.yaxis for trace in figure.data} == {None}

    # A charge block hangs below the mean price and a discharge block stands above it,
    # which is what makes the pattern readable against the path.
    def test_the_blocks_straddle_the_mean(self, trajectory):
        path = data.dispatchPath(trajectory, priceSource="realised")
        clock = pd.to_datetime(path.settlementdate)
        rule = float(path.price.mean())
        figure = figures.dayFigure(
            path, [("charge", clock.iloc[0], clock.iloc[5]),
                   ("discharge", clock.iloc[-6], clock.iloc[-1])], DAY)
        # The mean rule is a shape too, so it trails the two blocks.
        charge, discharge = figure.layout.shapes[:2]
        assert charge.y0 < rule and charge.y1 == pytest.approx(rule)
        assert discharge.y0 == pytest.approx(rule) and discharge.y1 > rule

    # Both screens fill a block with the same colour, so neither teaches a second key.
    def test_the_block_colours_match_the_forecast_screen(self, trajectory):
        path = data.dispatchPath(trajectory, priceSource="realised")
        clock = pd.to_datetime(path.settlementdate)
        figure = figures.dayFigure(path, [("charge", clock.iloc[0], clock.iloc[5])], DAY)
        assert figure.layout.shapes[0].fillcolor == figures.chargeBlock

    # A day with no trajectory rows says so rather than drawing an empty frame.
    def test_an_empty_day_renders_a_message(self, trajectory):
        figure = figures.dayFigure(data.dispatchPath(trajectory.iloc[0:0]), [], DAY)
        assert figure.layout.annotations and "no" in figure.layout.annotations[0].text

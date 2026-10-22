"""Tests for the Dash wiring in visualise/dashboard/__main__.py.

No server: the layout is built once and the callbacks are called as plain functions.
"""
from pathlib import Path

import pytest

from visualise.dashboard import __main__ as app
from rl import policies
from visualise.dashboard import data, live
from visualise.dashboard.test_data import DAY, REGION, run  # noqa: F401
from visualise.dashboard.test_live import path, sawtooth  # noqa: F401

MODEL = "maskableppo-warm-s0-realised"
LIVE_MODEL = "maskableppo-warm-s0"


def publish(root, region=None, seed=0):
    """One published agent under `root`, as training would leave it."""
    source = Path(root) / "checkpoint"
    source.mkdir(parents=True, exist_ok=True)
    (source / policies.agentFile).write_bytes(b"")
    (source / policies.scalerFile).write_text("{}")
    policies.publish(source, region or REGION, "maskableppo", "warm", seed,
                     root=Path(root) / "trainedPolicies")
    return Path(root) / "trainedPolicies"


def _ids(node, found=None):
    found = set() if found is None else found
    if getattr(node, "id", None):
        found.add(node.id)
    children = getattr(node, "children", None)
    for child in (children if isinstance(children, list) else [children]):
        if child is not None and hasattr(child, "_prop_names"):
            _ids(child, found)
    return found


def _callback(built, output):
    return [spec["callback"] for key, spec in built.callback_map.items()
            if output in key][0].__wrapped__


class TestLayout:
    # Every component a callback names is present in the layout.
    def test_every_callback_target_exists(self, run):
        built = app.build(REGION, run)
        present = _ids(built.layout)
        for key, spec in built.callback_map.items():
            for item in spec["inputs"] + spec.get("state", []):
                assert item["id"] in present, f"{item['id']} missing for {key}"

    # The sketch has two dropdowns, not three. Train mode and seed are in the name.
    def test_there_is_one_model_control(self, run):
        present = _ids(app.build(REGION, run).layout)
        assert {"pick-model", "pick-region"} <= present
        assert not {"pick-algo", "pick-trainMode", "pick-priceSource"} & present

    # The region list comes from the directory, never from config.py.
    def test_the_regions_come_from_disk(self, run):
        assert data.availableRegions(run) == [REGION]

    # The board opens on the last day the backtest tested, not on today.
    def test_it_opens_on_the_last_tested_day(self, run):
        store = [node for node in [app.build(REGION, run).layout]
                 if True][0].children[0]
        assert store.data["day"] == "2025-01-14"
        assert store.data["view"] == "calendar"


class TestModelOptions:
    # One entry per trained model scored one way.
    def test_it_lists_the_models_on_disk(self, run):
        built = app.build(REGION, run)
        options, value = _callback(built, "pick-model")(REGION, None)
        assert [item["value"] for item in options] == [MODEL]
        assert value == MODEL

    # A model the reader already picked survives a redraw of the list.
    def test_a_valid_choice_is_kept(self, run):
        built = app.build(REGION, run)
        assert _callback(built, "pick-model")(REGION, MODEL)[1] == MODEL


class TestNextState:
    base = {"region": REGION, "day": "2025-01-08", "view": "calendar", "model": MODEL}

    # Clicking a calendar cell opens that day.
    def test_a_cell_click_opens_its_day(self):
        click = {"points": [{"customdata": ["2025-01-05", 0, 12.0, 30.0]}]}
        got = app.nextState("board", self.base, click=click)
        assert got["day"] == "2025-01-05" and got["view"] == "day"

    # The same graph draws the day view, so a click there must not move the day.
    def test_a_click_in_the_day_view_changes_nothing(self):
        state = dict(self.base, view="day")
        click = {"points": [{"customdata": ["2025-01-05", 0, 12.0, 30.0]}]}
        assert app.nextState("board", state, click=click) == state

    # What Dash really sends for a heatmap cell: a row and a column, and no date.
    # Dash reads `customdata` out of the trace by the point number, and a heatmap
    # point number is a [row, column] pair, so the date the cell carries is dropped.
    # January 2025 starts on a Wednesday, so row 1 column 0 is the 6th.
    def test_a_cell_click_without_customdata_opens_its_day(self):
        click = {"points": [{"curveNumber": 0, "x": 0, "y": 1, "z": 12.0,
                             "pointNumber": [1, 0]}]}
        got = app.nextState("board", self.base, click=click)
        assert got["day"] == "2025-01-06" and got["view"] == "day"

    # The row and the column are read against the month the board draws, which is the
    # month of the selected day, not the month the reader started on.
    def test_a_cell_click_reads_the_month_on_screen(self):
        click = {"points": [{"curveNumber": 0, "x": 0, "y": 1, "z": 12.0,
                             "pointNumber": [1, 0]}]}
        state = dict(self.base, day="2025-02-20")
        # February 2025 starts on a Saturday, so row 1 column 0 is the 3rd.
        assert app.nextState("board", state, click=click)["day"] == "2025-02-03"

    # A cell outside the month, which the trailing blanks of the grid are, opens
    # nothing.
    def test_a_blank_cell_keeps_the_day(self):
        click = {"points": [{"curveNumber": 0, "x": 6, "y": 5}]}
        got = app.nextState("board", self.base, click=click)
        assert got["day"] == self.base["day"] and got["view"] == "calendar"

    # A click that lands on the volatility dot carries no date to open.
    def test_a_click_without_a_date_keeps_the_day(self):
        got = app.nextState("board", self.base, click={"points": [{"text": "7"}]})
        assert got["day"] == self.base["day"]

    # The calendar button comes back from a day without moving the day.
    def test_the_calendar_button_returns_to_the_month(self):
        got = app.nextState("show-calendar", dict(self.base, view="day"))
        assert got["view"] == "calendar" and got["day"] == self.base["day"]

    # In the calendar the arrows step one month, which is what that view draws.
    @pytest.mark.parametrize("trigger,wanted", [("step-prev", "2024-12-08"),
                                                ("step-next", "2025-02-08")])
    def test_the_arrows_step_a_month_in_the_calendar(self, trigger, wanted):
        assert app.nextState(trigger, self.base)["day"] == wanted

    # In the dispatch view they step one day, which is what that view draws.
    @pytest.mark.parametrize("trigger,wanted", [("step-prev", "2025-01-07"),
                                                ("step-next", "2025-01-09")])
    def test_the_arrows_step_a_day_in_the_day_view(self, trigger, wanted):
        state = dict(self.base, view="day")
        assert app.nextState(trigger, state)["day"] == wanted

    # A day step crosses a month boundary, and leaves the view alone.
    def test_a_day_step_crosses_the_month(self):
        state = dict(self.base, day="2025-01-31", view="day")
        got = app.nextState("step-next", state)
        assert got["day"] == "2025-02-01" and got["view"] == "day"

    # A month step into a shorter month holds the last day of it, and does not run
    # over into the month after.
    def test_a_month_step_into_a_short_month_clamps(self):
        state = dict(self.base, day="2025-01-31")
        assert app.nextState("step-next", state)["day"] == "2025-02-28"

    # The stepper says what it moves, because the arrows carry no unit of their own.
    @pytest.mark.parametrize("view,wanted", [("calendar", "month"), ("day", "day")])
    def test_the_stepper_names_its_unit(self, view, wanted):
        label, previous, following = app.stepLabels(view)
        assert label == f"Step {wanted}"
        assert (previous, following) == (f"Previous {wanted}", f"Next {wanted}")

    # The field is read as DD/MM/YYYY, which is the Australian order.
    def test_the_field_reads_day_first(self):
        got = app.nextState("day-text", self.base, text="05/02/2025")
        assert got["day"] == "2025-02-05"

    # A half-typed date must not throw the reader back to another day.
    @pytest.mark.parametrize("text", ["", "05/", "not a date", "2025-02-05", None])
    def test_an_unfinished_date_is_ignored(self, text):
        assert app.nextState("day-text", self.base, text=text)["day"] == self.base["day"]

    # Changing region returns to the month, because the day may not exist there.
    def test_a_region_change_returns_to_the_month(self):
        got = app.nextState("pick-region", dict(self.base, view="day"), region="SA1")
        assert got["region"] == "SA1" and got["view"] == "calendar"

    # A model that the new region does not hold falls back to one that it does.
    def test_a_missing_model_falls_back(self):
        state = dict(self.base, model="gone")
        assert app.nextState("pick-region", state, models=(MODEL,))["model"] == MODEL


class TestRedraw:
    # Every declared output is produced, in both views.
    @pytest.mark.parametrize("view", ["calendar", "day"])
    def test_it_fills_every_output(self, run, view):
        built = app.build(REGION, run)
        got = _callback(built, "tiles")({"region": REGION, "day": "2025-01-01",
                                         "view": view, "model": MODEL})
        assert len(got) == 3 and all(item is not None for item in got)

    # The month view still names its own month.
    def test_the_month_view_names_its_month(self, run):
        built = app.build(REGION, run)
        tiles, _, label = _callback(built, "tiles")(
            {"region": REGION, "day": "2025-02-01", "view": "calendar",
             "model": MODEL})
        assert label == "February 2025"
        assert tiles[0].children[1].children == "no data"

    # The tiles follow the selected day in the month view, not the month.
    def test_the_tiles_follow_the_day_in_the_month_view(self, run):
        built = app.build(REGION, run)
        redraw = _callback(built, "tiles")
        values = []
        for day in ("2025-01-01", "2025-01-02"):
            tiles, _, _ = redraw({"region": REGION, "day": day, "view": "calendar",
                                  "model": MODEL})
            labels = [tile.children[0].children for tile in tiles]
            values.append([tile.children[1].children for tile in tiles])
        assert "Days" not in labels
        assert values[0] != values[1]

    # The label is wired to the state, so the screen names the unit in both views.
    @pytest.mark.parametrize("view,wanted", [("calendar", "Step month"),
                                             ("day", "Step day")])
    def test_the_stepper_label_follows_the_view(self, run, view, wanted):
        built = app.build(REGION, run)
        got = _callback(built, "step-label")({"region": REGION, "day": "2025-01-01",
                                              "view": view, "model": MODEL})
        assert got[0] == wanted

    # A day with no trajectory file still renders the rest of the screen.
    def test_a_day_without_a_trajectory_still_renders(self, run):
        built = app.build(REGION, run)
        _, board, _ = _callback(built, "tiles")(
            {"region": REGION, "day": "2025-01-03", "view": "day", "model": MODEL})
        assert board.layout.annotations


class TestForecastScreen:
    # The screen runs the agent itself, so it offers what trainedPolicies/ holds, not
    # what the backtest scored. The backtest label carries a price source; this does not.
    def test_it_offers_the_published_agents(self, run, tmp_path):
        built = app.build(REGION, run, publish(tmp_path))
        _, options, value = _callback(built, "live-state")("live", REGION, None, 0)
        assert [item["value"] for item in options] == [live.thresholdLabel, LIVE_MODEL]
        assert value == live.thresholdLabel

    # A machine holding no agent still offers the rule, so the screen always works.
    def test_a_machine_without_agents_offers_the_rule(self, run, tmp_path):
        built = app.build(REGION, run, tmp_path)
        _, options, _ = _callback(built, "live-state")("live", REGION, None, 0)
        assert [item["value"] for item in options] == [live.thresholdLabel]

    # A model the new region does not hold falls back before the board fetches it.
    def test_an_unknown_model_falls_back(self, run, tmp_path):
        built = app.build(REGION, run, tmp_path)
        state, _, value = _callback(built, "live-state")("live", REGION, "gone", 0)
        assert value == live.thresholdLabel and state["model"] == live.thresholdLabel

    # One screen is shown and the other is hidden, never both.
    def test_the_tabs_swap_the_screens(self, run):
        swap = _callback(app.build(REGION, run), "backtest-screen")
        assert swap("live") == ({"display": "none"}, {})
        assert swap("backtest") == ({}, {"display": "none"})

    # The box over the board names the interval the forecast starts on.
    def test_the_box_names_the_start_of_the_forecast(self, monkeypatch, tmp_path):
        monkeypatch.setattr(live, "forecastPath", lambda region: path([10.0, 300.0]))
        figure, box, _ = app.forecastBoard(REGION, live.thresholdLabel, tmp_path)
        assert figure.data[0].name == "AEMO forecast"
        assert box[1].children == "01/01/2025 10:05"

    # The prices are AEMO's, so a model with no agent still gets its price path.
    def test_an_unpublished_model_still_draws_the_prices(self, monkeypatch, tmp_path):
        monkeypatch.setattr(live, "forecastPath", lambda region: path([10.0, 300.0]))
        figure, box, _ = app.forecastBoard(REGION, MODEL, tmp_path)
        assert figure.data[0].name == "AEMO forecast"
        assert not [s for s in figure.layout.shapes if s.type == "rect"]
        assert "not published" in figure.layout.annotations[0].text
        assert box[1].children == "01/01/2025 10:05"

    # The cards come back with the board, so one network read fills the whole screen.
    def test_the_board_returns_the_cards(self, monkeypatch, sawtooth, tmp_path):
        monkeypatch.setattr(live, "forecastPath", lambda region: sawtooth)
        _, _, tiles = app.forecastBoard(REGION, live.thresholdLabel, tmp_path)
        assert [tile["label"] for tile in tiles] == ["Expected revenue",
                                                     "Current state of charge"]
        assert tiles[0]["raw"] > 0

    # The cards have somewhere to be drawn.
    def test_the_cards_are_in_the_layout(self, run):
        assert "forecast-tiles" in _ids(app.build(REGION, run).layout)

    # A NEMWeb failure is written on the board, not swallowed into an empty chart.
    def test_a_failed_fetch_says_why(self, monkeypatch, tmp_path):
        def refuse(region):
            raise ValueError("AEMO has not published this run")

        monkeypatch.setattr(live, "forecastPath", refuse)
        figure, box, tiles = app.forecastBoard(REGION, live.thresholdLabel, tmp_path)
        assert "not published" in figure.layout.annotations[0].text
        assert box[1].children == "no run"
        assert tiles[0]["value"] == "no data"


class TestParseDay:
    def test_it_round_trips(self):
        assert app.parseDay(app.showDay("2025-03-09")) == "2025-03-09"

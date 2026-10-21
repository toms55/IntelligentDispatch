"""Tests for the Dash wiring in visualise/dashboard/__main__.py.

No server: the layout is built once and the callbacks are called as plain functions.
"""
import pytest

from visualise.dashboard import __main__ as app
from visualise.dashboard import data
from visualise.dashboard.test_data import DAY, REGION, run  # noqa: F401

MODEL = "maskableppo-warm-s0-realised"


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

    # A click that lands on the volatility dot carries no date to open.
    def test_a_click_without_a_date_keeps_the_day(self):
        got = app.nextState("board", self.base, click={"points": [{"x": 2}]})
        assert got["day"] == self.base["day"]

    # The calendar button comes back from a day without moving the day.
    def test_the_calendar_button_returns_to_the_month(self):
        got = app.nextState("show-calendar", dict(self.base, view="day"))
        assert got["view"] == "calendar" and got["day"] == self.base["day"]

    # The arrows step one day at a time, and may cross a month boundary.
    @pytest.mark.parametrize("trigger,wanted", [("step-prev", "2025-01-07"),
                                                ("step-next", "2025-01-09")])
    def test_the_arrows_step_one_day(self, trigger, wanted):
        assert app.nextState(trigger, self.base)["day"] == wanted

    def test_a_step_crosses_the_month(self):
        state = dict(self.base, day="2025-01-31")
        assert app.nextState("step-next", state)["day"] == "2025-02-01"

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

    # The month view is scoped to its own month, not to the whole job.
    def test_the_month_view_counts_only_that_month(self, run):
        built = app.build(REGION, run)
        tiles, _, label = _callback(built, "tiles")(
            {"region": REGION, "day": "2025-02-01", "view": "calendar",
             "model": MODEL})
        assert label == "February 2025"
        assert tiles[-1].children[1].children == "no data"

    # A day with no trajectory file still renders the rest of the screen.
    def test_a_day_without_a_trajectory_still_renders(self, run):
        built = app.build(REGION, run)
        _, board, _ = _callback(built, "tiles")(
            {"region": REGION, "day": "2025-01-03", "view": "day", "model": MODEL})
        assert board.layout.annotations


class TestParseDay:
    def test_it_round_trips(self):
        assert app.parseDay(app.showDay("2025-03-09")) == "2025-03-09"

"""Tests for the Dash wiring in visualise/dashboard/__main__.py.

No server: the layout is built once and the callbacks are called as plain functions.
"""
import pytest

from visualise.dashboard import __main__ as app
from visualise.dashboard import data
from visualise.dashboard.test_data import DAY, REGION, run  # noqa: F401


def _ids(node, found=None):
    found = set() if found is None else found
    if getattr(node, "id", None):
        found.add(node.id)
    children = getattr(node, "children", None)
    for child in (children if isinstance(children, list) else [children]):
        if child is not None and hasattr(child, "_prop_names"):
            _ids(child, found)
    return found


class TestLayout:
    # Every component a callback names is present in the layout.
    def test_every_callback_target_exists(self, run):
        built = app.build(REGION, run)
        present = _ids(built.layout)
        for key, spec in built.callback_map.items():
            for item in spec["inputs"] + spec.get("state", []):
                assert item["id"] in present, f"{item['id']} missing for {key}"

    # Nothing is read from config.py that the manifest should supply.
    def test_the_controls_come_from_the_data(self, run):
        built = app.build(REGION, run)
        assert "pick-algo" in _ids(built.layout)
        assert data.selectors(data.loadRun(REGION, run).days)["algo"] == ["maskableppo"]


class TestNextScope:
    # Clicking a calendar cell narrows the scope to that day.
    def test_a_cell_click_selects_its_day(self):
        click = {"points": [{"customdata": ["2025-02-05", 5, 12.0, 300.0]}]}
        assert app.nextScope("calendar", click, app.wholeJob) == {
            "kind": "day", "day": "2025-02-05"}

    # The reset control always returns to the whole job.
    def test_reset_returns_to_the_job(self):
        click = {"points": [{"customdata": ["2025-02-05", 5, 12.0, 300.0]}]}
        assert app.nextScope("reset-scope", click, {"kind": "day",
                                                    "day": "2025-02-05"}) == app.wholeJob

    # The high-volatility dot carries no customdata, so it must not clear the scope.
    def test_a_click_on_the_marker_keeps_the_scope(self):
        current = {"kind": "day", "day": "2025-02-05"}
        assert app.nextScope("calendar", {"points": [{"x": "Wed"}]}, current) == current

    # A callback fired with nothing selected falls back to the whole job.
    def test_no_click_is_the_whole_job(self):
        assert app.nextScope("calendar", None, {"kind": "day", "day": "x"}) == app.wholeJob


class TestRedraw:
    # Every declared output is produced, for both the job and the day scope.
    @pytest.mark.parametrize("scope", [{"kind": "job", "day": None},
                                       {"kind": "day", "day": "2025-01-01"}])
    def test_it_fills_every_output(self, run, scope):
        built = app.build(REGION, run)
        redraw = [spec["callback"] for key, spec in built.callback_map.items()
                  if "tiles" in key][0]
        got = redraw.__wrapped__(scope, "maskableppo", "warm", "realised")
        assert len(got) == 6
        assert all(item is not None for item in got)

    # A day with no trajectory file still renders the rest of the screen.
    def test_a_day_without_a_trajectory_still_renders(self, run):
        built = app.build(REGION, run)
        redraw = [spec["callback"] for key, spec in built.callback_map.items()
                  if "tiles" in key][0]
        got = redraw.__wrapped__({"kind": "day", "day": "2025-01-03"},
                                 "maskableppo", "warm", "realised")
        assert got[5].layout.annotations

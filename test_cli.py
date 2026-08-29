"""Tests for cli.py. No network: price_series is monkeypatched."""
from datetime import datetime

import pandas as pd
import pytest

import cli
from optimisation.lp import Schedule


@pytest.fixture
def series():
    """Four intervals of a cheap-then-dear day."""
    index = pd.to_datetime(
        ["2024-01-10 00:05", "2024-01-10 00:10", "2024-01-10 00:15", "2024-01-10 00:20"]
    )
    return pd.Series([10.0, 20.0, 300.0, 40.0], index=index, name="RRP")


@pytest.fixture
def schedule():
    return Schedule(0.5, [0.6, 0.7, 0.5, 0.4], [0.01, 0.01, 0.0, 0.0],
                    [0.0, 0.0, 0.02, 0.01], 6.4, 1.4, 5.0)


@pytest.fixture(autouse=True)
def calls(monkeypatch, series, schedule):
    """Record what the CLI hands to price_series and solve; do neither for real."""
    seen = {}

    def fakePrices(start, end, region="SA1"):
        seen["prices"] = (start, end, region)
        return series

    def fakeSolve(prices, model, **kwargs):
        seen["solve"] = (prices, model, kwargs)
        return schedule

    monkeypatch.setattr(cli, "price_series", fakePrices)
    monkeypatch.setattr(cli, "solve", fakeSolve)
    return seen


class TestArgs:
    def test_dates_reach_price_series(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        start, end, region = calls["prices"]
        assert start == datetime(2024, 1, 10)
        assert end == datetime(2024, 1, 11)
        assert region == "SA1"

    def test_datetimes_are_accepted(self, calls):
        cli.main(["lp", "2024-01-10T06:00", "2024-01-10T18:30"])
        start, end, _ = calls["prices"]
        assert start == datetime(2024, 1, 10, 6, 0)
        assert end == datetime(2024, 1, 10, 18, 30)

    def test_region_is_selectable(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--region", "NSW1"])
        assert calls["prices"][2] == "NSW1"

    def test_bad_date_exits_nonzero(self):
        with pytest.raises(SystemExit) as exit:
            cli.main(["lp", "the-tenth", "2024-01-11"])
        assert exit.value.code != 0

    def test_default_battery_is_the_paper_case(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        model = calls["solve"][1]
        assert (model.chargePower, model.MWh, model.replacementCost) == (
            0.060, 0.100, 150_000.0)

    def test_battery_is_configurable(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11",
                  "--mw", "1.5", "--mwh", "3.0", "--replacement-cost", "200000"])
        model = calls["solve"][1]
        assert (model.chargePower, model.MWh, model.replacementCost) == (1.5, 3.0, 200_000.0)

    def test_prices_are_passed_as_a_plain_list(self, calls):
        """solve() indexes by position and has no use for a DatetimeIndex."""
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        assert calls["solve"][0] == [10.0, 20.0, 300.0, 40.0]

    def test_calendar_aging_is_on_by_default(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        assert calls["solve"][2]["calendar"] is True

    def test_no_calendar_turns_it_off(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--no-calendar"])
        assert calls["solve"][2]["calendar"] is False

    def test_initial_soc_is_configurable(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--initial-soc", "0.25"])
        assert calls["solve"][2]["initialSoc"] == 0.25


class TestOutput:
    def test_summary_goes_to_stdout(self, capsys):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        out = capsys.readouterr().out
        assert "revenue" in out
        assert "degradation" in out
        assert "profit" in out

    def test_csv_has_one_row_per_interval(self, tmp_path):
        path = tmp_path / "day.csv"
        cli.main(["lp", "2024-01-10", "2024-01-11", "--csv", str(path)])
        frame = pd.read_csv(path)
        assert list(frame.columns) == ["settlementdate", "price", "soc", "bought", "sold"]
        assert len(frame) == 4
        assert frame["settlementdate"][0] == "2024-01-10 00:05:00"
        assert frame["price"].to_list() == [10.0, 20.0, 300.0, 40.0]
        assert frame["soc"].to_list() == [0.6, 0.7, 0.5, 0.4]

    def test_no_csv_writes_nothing(self, tmp_path):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        assert list(tmp_path.iterdir()) == []

    def test_plot_saves_a_figure(self, tmp_path):
        path = tmp_path / "day.png"
        cli.main(["lp", "2024-01-10", "2024-01-11", "--plot", str(path)])
        assert path.exists() and path.stat().st_size > 0

    def test_bare_csv_uses_a_derived_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cli.main(["lp", "2024-01-10", "2024-01-11", "--csv"])
        assert (tmp_path / "lp_SA1_2024-01-10.csv").exists()

    def test_bare_plot_uses_a_derived_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cli.main(["lp", "2024-01-10", "2024-01-11", "--plot"])
        assert (tmp_path / "lp_SA1_2024-01-10.png").exists()

    def test_default_path_follows_region_and_start(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cli.main(["lp", "2024-03-05T06:00", "2024-03-06", "--region", "VIC1", "--csv"])
        assert (tmp_path / "lp_VIC1_2024-03-05.csv").exists()

    def test_show_displays_instead_of_saving(self, tmp_path, monkeypatch):
        """--show must not leave a file behind, and must not need a PATH."""
        monkeypatch.chdir(tmp_path)
        shown = []
        monkeypatch.setattr(cli, "_plot", lambda frame, path: shown.append(path))
        cli.main(["lp", "2024-01-10", "2024-01-11", "--show"])
        assert shown == [None]
        assert list(tmp_path.iterdir()) == []


class TestEndToEnd:
    """The monkeypatched tests above never touch the solver. This one does."""

    def test_a_real_solve_runs(self, monkeypatch, capsys):
        from optimisation.lp import solve as realSolve

        index = pd.date_range("2024-01-10", periods=12, freq="5min")
        prices = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
                            300.0, 300.0, 300.0, 300.0, 300.0, 300.0], index=index)
        monkeypatch.setattr(cli, "price_series", lambda *a, **k: prices)
        monkeypatch.setattr(cli, "solve", realSolve)

        assert cli.main(["lp", "2024-01-10", "2024-01-11", "--no-calendar"]) == 0
        assert "profit" in capsys.readouterr().out

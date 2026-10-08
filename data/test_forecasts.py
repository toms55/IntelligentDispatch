"""Tests for data/forecasts.py. Run with: uv run pytest data/ -q

No network: nemseer is replaced by a fake returning hand-built frames.
"""
import numpy as np
import pandas as pd
import pytest

from data import forecasts


RUN = pd.Timestamp("2024-01-10 04:00")


def predispatchFrame(runs, region="SA1", periods=48, base=100.0):
    """One row per (run, 30-minute interval end), RRP = base + run offset + period."""
    rows = []
    for offset, run in enumerate(runs):
        for period in range(1, periods + 1):
            rows.append({
                "PREDISPATCH_RUN_DATETIME": pd.Timestamp(run),
                "DATETIME": pd.Timestamp(run) + pd.Timedelta(minutes=30 * period),
                "REGIONID": region,
                "INTERVENTION": 0,
                "RRP": base + 1000.0 * offset + period,
            })
    rows.append({  # intervention duplicate, must be dropped
        "PREDISPATCH_RUN_DATETIME": pd.Timestamp(runs[0]),
        "DATETIME": pd.Timestamp(runs[0]) + pd.Timedelta(minutes=30),
        "REGIONID": region, "INTERVENTION": 1, "RRP": -999.0,
    })
    return pd.DataFrame(rows)


def p5minFrame(runs, region="SA1", intervals=11, base=5.0):
    rows = []
    for offset, run in enumerate(runs):
        for step in range(1, intervals + 1):
            rows.append({
                "RUN_DATETIME": pd.Timestamp(run),
                "INTERVAL_DATETIME": pd.Timestamp(run) + pd.Timedelta(minutes=5 * step),
                "REGIONID": region,
                "RRP": base + 1000.0 * offset + step,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def fakeNemseer(monkeypatch):
    """Stand in for nemseer, recording the windows it was asked for."""
    asked = []
    state = {
        "PREDISPATCH": predispatchFrame([RUN, RUN + pd.Timedelta(minutes=30)]),
        "P5MIN": p5minFrame([RUN, RUN + pd.Timedelta(minutes=5)]),
    }

    class Fake:
        @staticmethod
        def compile_data(runStart, runEnd, forecastStart, forecastEnd, forecastType,
                         tables, raw_cache=None, **kwargs):
            asked.append({"runStart": runStart, "runEnd": runEnd,
                          "forecastStart": forecastStart, "forecastEnd": forecastEnd,
                          "type": forecastType, "tables": tables})
            return {tables: state[forecastType]}

    monkeypatch.setattr(forecasts, "_nemseer", lambda: Fake)
    Fake.asked = asked
    Fake.state = state
    return Fake


class TestRunTimeSelection:
    def test_predispatch_keeps_only_the_requested_run(self, fakeNemseer):
        got = forecasts.predispatchForecast(RUN, "SA1")
        assert got.max() < 1000.0

    def test_p5min_keeps_only_the_requested_run(self, fakeNemseer):
        got = forecasts.p5minForecast(RUN, "SA1")
        assert len(got) == 11
        assert got.max() < 1000.0

    def test_predispatch_floors_to_the_published_half_hour(self, fakeNemseer):
        got = forecasts.predispatchForecast(RUN + pd.Timedelta(minutes=20), "SA1")
        assert got.index[0] == RUN + pd.Timedelta(minutes=30)
        assert fakeNemseer.asked[-1]["runStart"] == "2024/01/10 04:00"

    def test_drops_intervention_rows(self, fakeNemseer):
        got = forecasts.predispatchForecast(RUN, "SA1")
        assert got.min() > 0

    def test_selects_the_region(self, fakeNemseer):
        fakeNemseer.state["PREDISPATCH"] = pd.concat([
            predispatchFrame([RUN], region="SA1", base=100.0),
            predispatchFrame([RUN], region="NSW1", base=9000.0),
        ])
        got = forecasts.predispatchForecast(RUN, "SA1")
        assert got.max() < 9000.0


class TestStitching:
    def test_a_half_hour_value_covers_its_six_intervals(self):
        halfHourly = pd.Series({pd.Timestamp("2024-01-10 04:30"): 42.0})
        got = forecasts._expandToFiveMinutes(halfHourly)
        assert len(got) == 6
        assert got.index[0] == pd.Timestamp("2024-01-10 04:05")
        assert got.index[-1] == pd.Timestamp("2024-01-10 04:30")
        assert set(got.to_numpy()) == {42.0}

    def test_holds_flat_rather_than_interpolating(self):
        halfHourly = pd.Series({
            pd.Timestamp("2024-01-10 04:30"): 10.0,
            pd.Timestamp("2024-01-10 05:00"): 90.0,
        })
        got = forecasts._expandToFiveMinutes(halfHourly)
        assert sorted(set(got.to_numpy())) == [10.0, 90.0]

    def test_p5min_wins_on_the_overlap(self, fakeNemseer):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        fine = forecasts.p5minForecast(RUN, "SA1")
        for stamp, price in fine.items():
            assert got[stamp] == pytest.approx(price)

    def test_predispatch_covers_everything_after_the_p5min_run(self, fakeNemseer):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        coarse = forecasts.predispatchForecast(RUN, "SA1")
        beyond = RUN + pd.Timedelta(minutes=60)
        assert got[beyond] == pytest.approx(coarse[beyond])


class TestForecastSeries:
    def test_is_indexed_from_the_next_interval_for_the_whole_horizon(self, fakeNemseer):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288)
        assert len(got) == 288
        assert got.index[0] == RUN + pd.Timedelta(minutes=5)
        assert got.index[-1] == RUN + pd.Timedelta(hours=24)

    def test_is_complete_at_the_trading_day_boundary(self, fakeNemseer):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288)
        assert int(got.isna().sum()) == 0

    def test_refuses_a_short_path(self, fakeNemseer):
        fakeNemseer.state["PREDISPATCH"] = predispatchFrame([RUN], periods=8)
        with pytest.raises(ValueError, match="of 288 intervals"):
            forecasts.forecastSeries(RUN, "SA1", horizon=288)

    def test_allow_partial_pads_with_nan_instead(self, fakeNemseer):
        fakeNemseer.state["PREDISPATCH"] = predispatchFrame([RUN], periods=8)
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        assert len(got) == 288
        assert int(got.isna().sum()) > 0

    def test_allow_partial_survives_a_run_with_no_archive_at_all(self, fakeNemseer, monkeypatch):
        """nemseer raises ValueError itself when a whole run has no archive, distinct
        from our own short-path check above. allowPartial must pad through that too."""
        def missing(*args, **kwargs):
            raise ValueError("Table(s) not available from MMS Historical Data SQL Loader")
        monkeypatch.setattr(forecasts, "predispatchForecast", missing)
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        assert len(got) == 288
        assert int(got.isna().sum()) == 288 - 11  # only the P5MIN run still lands

    def test_without_allow_partial_a_missing_archive_still_raises(self, fakeNemseer, monkeypatch):
        def missing(*args, **kwargs):
            raise ValueError("Table(s) not available from MMS Historical Data SQL Loader")
        monkeypatch.setattr(forecasts, "predispatchForecast", missing)
        with pytest.raises(ValueError, match="MMS Historical Data SQL Loader"):
            forecasts.forecastSeries(RUN, "SA1", horizon=288)

    def test_asks_for_only_the_horizon_it_needs(self, fakeNemseer):
        forecasts.forecastSeries(RUN, "SA1", horizon=288)
        predispatch = [a for a in fakeNemseer.asked if a["type"] == "PREDISPATCH"][-1]
        assert predispatch["forecastEnd"] == "2024/01/11 04:00"


class TestForecastMatrix:
    def test_one_row_per_run_time(self, fakeNemseer):
        runs = [RUN, RUN + pd.Timedelta(hours=24)]
        fakeNemseer.state["PREDISPATCH"] = predispatchFrame(
            [RUN, RUN + pd.Timedelta(hours=24)]
        )
        fakeNemseer.state["P5MIN"] = p5minFrame([RUN, RUN + pd.Timedelta(hours=24)])
        got = forecasts.forecastMatrix(runs, "SA1", horizon=288)
        assert got.shape == (2, 288)
        assert not np.isnan(got[0]).all()

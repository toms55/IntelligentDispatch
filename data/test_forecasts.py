"""Tests for data/forecasts.py. Run with: uv run pytest data/ -q

No network: the archive readers are replaced by fakes returning normalised rows, and the
CSV reader is fed hand-built zips.
"""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from data import forecasts


RUN = pd.Timestamp("2024-01-10 04:00")


def predispatchFrame(runs, region="SA1", periods=48, base=100.0):
    """Normalised rows, one per (run, 30-minute interval end), RRP = base + offset + period."""
    rows = []
    for offset, run in enumerate(runs):
        for period in range(1, periods + 1):
            rows.append({
                "run": pd.Timestamp(run),
                "interval": pd.Timestamp(run) + pd.Timedelta(minutes=30 * period),
                "REGIONID": region,
                "RRP": base + 1000.0 * offset + period,
            })
    return pd.DataFrame(rows, columns=forecasts.rowColumns)


def p5minFrame(runs, region="SA1", intervals=11, base=5.0):
    rows = []
    for offset, run in enumerate(runs):
        for step in range(1, intervals + 1):
            rows.append({
                "run": pd.Timestamp(run),
                "interval": pd.Timestamp(run) + pd.Timedelta(minutes=5 * step),
                "REGIONID": region,
                "RRP": base + 1000.0 * offset + step,
            })
    return pd.DataFrame(rows, columns=forecasts.rowColumns)


def fileMonth(kind, run):
    """Where AEMO files a run: PREDISPATCH by trading day, P5MIN by run time."""
    moment = run - forecasts.fileLag if kind == "PREDISPATCH" else run
    return moment.year, moment.month


def installFakeArchive(monkeypatch, state):
    """Months serve the rows AEMO would file under them; `unpublished` months are None
    and their runs come from `reports` instead. Every read is recorded."""
    asked = []

    def monthRows(kind, year, month):
        asked.append(("month", kind, year, month))
        if (year, month) in state.get("unpublished", ()):
            return None
        frame = state[kind]
        inMonth = np.array([fileMonth(kind, run) == (year, month) for run in frame.run],
                           dtype=bool)
        return frame[inMonth]

    def reportRows(kind, runTime):
        asked.append(("report", kind, runTime))
        frame = state.get("reports", {}).get(kind)
        if frame is None or not (frame.run == runTime).any():
            return None
        return frame[frame.run == runTime]

    monkeypatch.setattr(forecasts, "_monthRows", monthRows)
    monkeypatch.setattr(forecasts, "_reportRows", reportRows)
    return asked


class FakeArchive:
    def __init__(self, state, asked):
        self.state, self.asked = state, asked


@pytest.fixture
def fakeArchive(monkeypatch):
    state = {
        "PREDISPATCH": predispatchFrame([RUN, RUN + pd.Timedelta(minutes=30)]),
        "P5MIN": p5minFrame([RUN, RUN + pd.Timedelta(minutes=5)]),
    }
    return FakeArchive(state, installFakeArchive(monkeypatch, state))


class TestRunTimeSelection:
    def test_predispatch_keeps_only_the_requested_run(self, fakeArchive):
        got = forecasts.predispatchForecast(RUN, "SA1")
        assert got.max() < 1000.0

    def test_p5min_keeps_only_the_requested_run(self, fakeArchive):
        got = forecasts.p5minForecast(RUN, "SA1")
        assert len(got) == 11
        assert got.max() < 1000.0

    def test_predispatch_floors_to_the_published_half_hour(self, fakeArchive):
        got = forecasts.predispatchForecast(RUN + pd.Timedelta(minutes=20), "SA1")
        assert got.index[0] == RUN + pd.Timedelta(minutes=30)
        assert got.max() < 1000.0

    def test_selects_the_region(self, fakeArchive):
        fakeArchive.state["PREDISPATCH"] = pd.concat([
            predispatchFrame([RUN], region="SA1", base=100.0),
            predispatchFrame([RUN], region="NSW1", base=9000.0),
        ])
        got = forecasts.predispatchForecast(RUN, "SA1")
        assert got.max() < 9000.0

    def test_predispatch_stops_at_the_requested_hours(self, fakeArchive):
        got = forecasts.predispatchForecast(RUN, "SA1", hours=2)
        assert got.index[-1] == RUN + pd.Timedelta(hours=2)


class TestWhereARunIsFiled:
    # The run labelled 04:00 on the 1st is the previous trading day's last, filed in the
    # previous month. Looking only in the run's own month gave 11 of 288 intervals.
    def test_the_first_of_the_month_reads_the_previous_months_file(self, monkeypatch):
        first = pd.Timestamp("2024-02-01 04:00")
        state = {"PREDISPATCH": predispatchFrame([first]), "P5MIN": p5minFrame([first])}
        installFakeArchive(monkeypatch, state)
        got = forecasts.forecastSeries(first, "SA1", horizon=288)
        assert int(got.isna().sum()) == 0

    def test_an_unpublished_month_falls_back_to_the_reports(self, monkeypatch):
        state = {"PREDISPATCH": predispatchFrame([]), "P5MIN": p5minFrame([]),
                 "unpublished": {(2024, 1)},
                 "reports": {"PREDISPATCH": predispatchFrame([RUN]),
                             "P5MIN": p5minFrame([RUN])}}
        asked = installFakeArchive(monkeypatch, state)
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288)
        assert int(got.isna().sum()) == 0
        assert ("report", "PREDISPATCH", RUN) in asked

    def test_a_run_the_month_holds_never_touches_the_reports(self, fakeArchive):
        forecasts.forecastSeries(RUN, "SA1", horizon=288)
        assert not [entry for entry in fakeArchive.asked if entry[0] == "report"]


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

    def test_p5min_wins_on_the_overlap(self, fakeArchive):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        fine = forecasts.p5minForecast(RUN, "SA1")
        for stamp, price in fine.items():
            assert got[stamp] == pytest.approx(price)

    def test_predispatch_covers_everything_after_the_p5min_run(self, fakeArchive):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        coarse = forecasts.predispatchForecast(RUN, "SA1")
        beyond = RUN + pd.Timedelta(minutes=60)
        assert got[beyond] == pytest.approx(coarse[beyond])


class TestForecastSeries:
    def test_is_indexed_from_the_next_interval_for_the_whole_horizon(self, fakeArchive):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288)
        assert len(got) == 288
        assert got.index[0] == RUN + pd.Timedelta(minutes=5)
        assert got.index[-1] == RUN + pd.Timedelta(hours=24)

    def test_is_complete_at_the_trading_day_boundary(self, fakeArchive):
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288)
        assert int(got.isna().sum()) == 0

    def test_refuses_a_short_path(self, fakeArchive):
        fakeArchive.state["PREDISPATCH"] = predispatchFrame([RUN], periods=8)
        with pytest.raises(ValueError, match="of 288 intervals"):
            forecasts.forecastSeries(RUN, "SA1", horizon=288)

    def test_allow_partial_pads_with_nan_instead(self, fakeArchive):
        fakeArchive.state["PREDISPATCH"] = predispatchFrame([RUN], periods=8)
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        assert len(got) == 288
        assert int(got.isna().sum()) > 0

    def test_allow_partial_survives_a_run_published_nowhere(self, fakeArchive):
        fakeArchive.state["PREDISPATCH"] = predispatchFrame([])
        got = forecasts.forecastSeries(RUN, "SA1", horizon=288, allowPartial=True)
        assert int(got.isna().sum()) == 288 - 11  # only the P5MIN run still lands

    def test_without_allow_partial_a_missing_run_raises(self, fakeArchive):
        fakeArchive.state["PREDISPATCH"] = predispatchFrame([])
        with pytest.raises(ValueError, match="of 288 intervals"):
            forecasts.forecastSeries(RUN, "SA1", horizon=288)


class TestLive:
    # Live is the same path at the current NEM time, floored to the dispatch interval.
    def test_reads_the_latest_runs(self, monkeypatch):
        now = pd.Timestamp("2024-01-10 21:42")
        monkeypatch.setattr(forecasts, "nemNow", lambda: now)
        state = {"PREDISPATCH": predispatchFrame([pd.Timestamp("2024-01-10 21:30")]),
                 "P5MIN": p5minFrame([pd.Timestamp("2024-01-10 21:40")])}
        installFakeArchive(monkeypatch, state)
        got = forecasts.liveForecast("SA1")
        assert got.index[0] == pd.Timestamp("2024-01-10 21:45")
        assert got.iloc[0] == pytest.approx(5.0 + 1)  # the P5MIN run's first step

    def test_nem_time_is_aest_without_daylight_saving(self):
        utc = pd.Timestamp.now("UTC").tz_localize(None)
        assert abs(forecasts.nemNow() - utc - pd.Timedelta(hours=10)) < pd.Timedelta(
            minutes=1)


class TestForecastMatrix:
    def test_one_row_per_run_time(self, fakeArchive):
        runs = [RUN, RUN + pd.Timedelta(hours=24)]
        fakeArchive.state["PREDISPATCH"] = predispatchFrame(runs)
        fakeArchive.state["P5MIN"] = p5minFrame(runs)
        got = forecasts.forecastMatrix(runs, "SA1", horizon=288)
        assert got.shape == (2, 288)
        assert not np.isnan(got[0]).all()


# --- the NEMWeb readers -------------------------------------------------------

def zipped(text, name="file.CSV"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, text)
    buffer.seek(0)
    return buffer


PREDISPATCH_CSV = """C,SETP.WORLD,DVD_PREDISPATCHPRICE,AEMO,PUBLIC,2024/01/02
I,PREDISPATCH,CASE_SOLUTION,1,PREDISPATCHSEQNO,RUNNO
D,PREDISPATCH,CASE_SOLUTION,1,2023123148,1
I,PREDISPATCH,REGION_PRICES,2,PREDISPATCHSEQNO,RUNNO,REGIONID,PERIODID,INTERVENTION,RRP,LASTCHANGED,DATETIME
D,PREDISPATCH,REGION_PRICES,2,2023123148,1,NSW1,01,0,57.5,"2024/01/01 03:32:06","2024/01/01 04:30:00"
D,PREDISPATCH,REGION_PRICES,2,2023123148,1,NSW1,01,1,-999,"2024/01/01 03:32:06","2024/01/01 04:30:00"
D,PREDISPATCH,REGION_PRICES,2,2024010101,1,SA1,01,0,60.25,"2024/01/01 04:01:00","2024/01/01 05:00:00"
C,"END OF REPORT",7
"""

P5MIN_CSV = """C,NEMP.WORLD,P5MIN,AEMO,PUBLIC,2026/10/19
I,P5MIN,REGIONSOLUTION,10,RUN_DATETIME,INTERVENTION,INTERVAL_DATETIME,REGIONID,RRP,ROP
D,P5MIN,REGIONSOLUTION,10,"2026/10/19 21:40:00",0,"2026/10/19 21:45:00",NSW1,129.65,129.65
I,P5MIN,INTERCONNECTORSOLN,4,RUN_DATETIME,INTERVENTION,INTERCONNECTORID,RRP
D,P5MIN,INTERCONNECTORSOLN,4,"2026/10/19 21:40:00",0,N-Q-MNSP1,1.0
C,"END OF REPORT",4
"""


class TestReadTable:
    def test_predispatch_run_is_labelled_from_the_sequence_number(self):
        got = forecasts._readTable("PREDISPATCH", zipped(PREDISPATCH_CSV),
                                   forecasts.sources["PREDISPATCH"]["table"])
        # Period 48 of trading day 2023-12-31 is the run labelled 2024-01-01 04:00.
        assert list(got.run) == [pd.Timestamp("2024-01-01 04:00"),
                                 pd.Timestamp("2024-01-01 04:30")]
        assert list(got.columns) == forecasts.rowColumns

    def test_drops_intervention_rows(self):
        got = forecasts._readTable("PREDISPATCH", zipped(PREDISPATCH_CSV),
                                   forecasts.sources["PREDISPATCH"]["table"])
        assert got.RRP.min() > 0

    def test_reads_only_its_table_from_a_multi_table_report(self):
        got = forecasts._readTable("P5MIN", zipped(P5MIN_CSV),
                                   forecasts.sources["P5MIN"]["table"])
        assert len(got) == 1
        assert got.iloc[0].RRP == pytest.approx(129.65)
        assert got.iloc[0].interval == pd.Timestamp("2026-10-19 21:45")


class TestMonthRows:
    @pytest.fixture
    def offline(self, monkeypatch, tmp_path):
        monkeypatch.setattr(forecasts, "rawCache", tmp_path)
        urls = []

        def download(url, target):
            urls.append(url)
            return False
        monkeypatch.setattr(forecasts, "_download", download)
        return urls

    # AEMO renamed the monthly files in August 2024; both names must still resolve.
    @pytest.mark.parametrize("year, month, name", [
        (2024, 7, "PREDISP_ALL_DATA/PUBLIC_DVD_PREDISPATCHPRICE_202407010000.zip"),
        (2024, 8, "PREDISP_ALL_DATA/PUBLIC_ARCHIVE%23PREDISPATCHPRICE%23ALL%23FILE01"
                  "%23202408010000.zip"),
    ])
    def test_the_file_name_follows_the_layout_of_its_month(self, offline, year, month,
                                                           name):
        assert forecasts._monthRows("PREDISPATCH", year, month) is None
        assert offline[-1].endswith(f"MMSDM_{year}_{month:02d}/"
                                    f"MMSDM_Historical_Data_SQLLoader/{name}")

    def test_an_empty_published_month_raises_rather_than_caching(self, monkeypatch,
                                                                  tmp_path):
        monkeypatch.setattr(forecasts, "rawCache", tmp_path)

        def download(url, target):
            target.write_bytes(zipped("C,header\n").getvalue())
            return True
        monkeypatch.setattr(forecasts, "_download", download)
        with pytest.raises(ValueError, match="holds no"):
            forecasts._monthRows("P5MIN", 2025, 1)
        assert not list(tmp_path.glob("*.parquet"))


class TestArchivedReport:
    # The 00:00 run is published at 23:55 the day before, so it sits in that day's bundle.
    def test_finds_a_run_in_the_bundle_published_the_day_before(self, monkeypatch,
                                                                tmp_path):
        monkeypatch.setattr(forecasts, "rawCache", tmp_path)
        inner = "PUBLIC_P5MIN_202610200000_20261019235534.zip"
        monkeypatch.setattr(forecasts, "_listing", lambda folder: [
            "PUBLIC_P5MIN_20261018.zip", "PUBLIC_P5MIN_20261019.zip"])
        fetched = []

        def download(url, target):
            fetched.append(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as bundle:
                bundle.writestr(inner, zipped(P5MIN_CSV).getvalue())
            target.write_bytes(buffer.getvalue())
            return True
        monkeypatch.setattr(forecasts, "_download", download)
        got = forecasts._archivedReport("P5MIN", pd.Timestamp("2026-10-20 00:00"),
                                        "PUBLIC_P5MIN_202610200000_")
        assert len(got) == 1
        assert [url.rsplit("/", 1)[1] for url in fetched] == ["PUBLIC_P5MIN_20261019.zip"]


# --- prefetch -----------------------------------------------------------------

JAN = pd.Timestamp("2024-01-01 04:00")
PREFETCH_RUNS = [JAN, JAN + pd.Timedelta(days=1), pd.Timestamp("2024-02-01 04:00")]


@pytest.fixture
def fakeMonths(monkeypatch):
    """Two months of 04:00 runs, plus one 08:00 refresh run. The other refresh runs are
    absent, as runs AEMO never published are."""
    runs = PREFETCH_RUNS + [JAN + pd.Timedelta(hours=4)]
    state = {"PREDISPATCH": predispatchFrame(runs), "P5MIN": p5minFrame(runs)}
    return FakeArchive(state, installFakeArchive(monkeypatch, state))


class TestPrefetch:
    # Whole months per table, never one read per run time. The 1st of each month also
    # reads the month before, where its PREDISPATCH run is filed.
    def test_reads_whole_months_not_runs(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        months = sorted({entry for entry in fakeMonths.asked if entry[0] == "month"})
        assert months == sorted(("month", kind, year, month)
                                for kind in ("P5MIN", "PREDISPATCH")
                                for year, month in ((2023, 12), (2024, 1), (2024, 2),
                                                    (2024, 3)))

    # The cache holds one row per run time and one column per horizon step. Refresh runs
    # AEMO never published are left out rather than stored empty.
    def test_the_cache_is_one_row_per_run(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        cached = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        assert list(cached.index) == sorted(PREFETCH_RUNS + [JAN + pd.Timedelta(hours=4)])
        assert len(cached.columns) == 288

    # The refresh runs are kept beside the 04:00 one.
    def test_refresh_runs_are_kept(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        cached = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        assert JAN + pd.Timedelta(hours=4) in cached.index

    # A daily refresh keeps only the 04:00 runs.
    def test_a_daily_refresh_keeps_only_0400_runs(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path, every=288)
        cached = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        assert set(cached.index.time) == {JAN.time()}

    # A region with no cache file returns None rather than raising.
    def test_a_missing_cache_is_none(self, tmp_path):
        assert forecasts.loadForecastCache("NSW1", cacheDir=tmp_path) is None

    # Reading a prefetched source makes no nemseer call at all.
    def test_the_cached_path_touches_no_nemseer(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        source = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        before = len(fakeMonths.asked)
        forecasts.forecastMatrix(PREFETCH_RUNS, "SA1", source=source)
        assert len(fakeMonths.asked) == before

    # A 5-minute refresh reads the half-hourly PREDISPATCH run at or before each run time,
    # from the months alone, and matches the live path.
    def test_a_five_minute_refresh_floors_predispatch(self, monkeypatch, tmp_path):
        state = {"PREDISPATCH": predispatchFrame(
                     pd.date_range(JAN, periods=48, freq="30min")),
                 "P5MIN": p5minFrame(pd.date_range(JAN, periods=288, freq="5min"))}
        asked = installFakeArchive(monkeypatch, state)
        forecasts.prefetchForecasts(JAN, JAN, "SA1", cacheDir=tmp_path, every=1)
        assert not [entry for entry in asked if entry[0] == "report"]
        source = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        runs = list(pd.date_range(JAN, periods=288, freq="5min"))
        cached = forecasts.forecastMatrix(runs, "SA1", source=source)
        assert not np.isnan(cached[:, 0]).any()
        np.testing.assert_allclose(cached, forecasts.forecastMatrix(runs, "SA1"),
                                   equal_nan=True)

    # The cached matrix equals what the live per-run path produces.
    def test_the_cached_matrix_matches_the_live_one(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        source = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        cached = forecasts.forecastMatrix(PREFETCH_RUNS, "SA1", source=source)
        live = forecasts.forecastMatrix(PREFETCH_RUNS, "SA1")
        np.testing.assert_allclose(cached, live, equal_nan=True)

    # A run time the cache never saw becomes a NaN row, not an exception.
    def test_an_unknown_run_time_pads_with_nan(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        source = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        got = forecasts.forecastMatrix([pd.Timestamp("2023-05-05 04:00")], "SA1",
                                       source=source)
        assert got.shape == (1, 288) and np.isnan(got).all()

    # A horizon shorter than the cache slices it rather than refetching.
    def test_a_short_horizon_slices_the_cache(self, fakeMonths, tmp_path):
        forecasts.prefetchForecasts(JAN, pd.Timestamp("2024-02-29 04:00"), "SA1",
                                    cacheDir=tmp_path)
        source = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        got = forecasts.forecastMatrix(PREFETCH_RUNS, "SA1", horizon=12, source=source)
        assert got.shape == (3, 12)
        np.testing.assert_allclose(
            got, forecasts.forecastMatrix(PREFETCH_RUNS, "SA1", source=source)[:, :12],
            equal_nan=True)


# --- refresh within an episode --------------------------------------------------

REFRESH_RUNS = [RUN + pd.Timedelta(hours=4 * n) for n in range(6)]


@pytest.fixture
def refreshArchive(monkeypatch):
    """All six runs of RUN's trading day, each with its own price level."""
    state = {"PREDISPATCH": predispatchFrame(REFRESH_RUNS),
             "P5MIN": p5minFrame(REFRESH_RUNS)}
    return FakeArchive(state, installFakeArchive(monkeypatch, state))


class TestRefresh:
    # Every four hours from the 04:00 start, the last one at midnight.
    def test_the_run_times(self):
        got = forecasts.refreshRunTimes(RUN)
        assert [run.strftime("%H:%M") for run in got] == [
            "04:00", "08:00", "12:00", "16:00", "20:00", "00:00"]
        assert got[-1] == pd.Timestamp("2024-01-11 00:00")

    # A refresh that does not divide the episode would start off the step grid.
    def test_a_refresh_must_divide_the_episode(self):
        with pytest.raises(ValueError, match="does not divide"):
            forecasts.refreshRunTimes(RUN, every=50)

    # Row r is run r on the episode grid: NaN before it was published, its own path
    # after, cut at the episode end.
    def test_each_run_sits_at_its_own_offset(self, refreshArchive):
        grid = forecasts.refreshMatrix([RUN], "SA1")
        assert grid.shape == (1, 6, 288)
        for refresh, run in enumerate(REFRESH_RUNS):
            offset = refresh * 48
            assert np.isnan(grid[0, refresh, :offset]).all()
            expected = forecasts.forecastSeries(run, "SA1", 288 - offset).to_numpy()
            np.testing.assert_allclose(grid[0, refresh, offset:], expected)

    # A daily refresh is the old single 04:00 path.
    def test_a_daily_refresh_is_the_0400_path(self, refreshArchive):
        grid = forecasts.refreshMatrix([RUN], "SA1", every=288)
        np.testing.assert_allclose(grid[:, 0], forecasts.forecastMatrix([RUN], "SA1"))

    # The prefetched cache gives the same grid as the live path.
    def test_the_cached_grid_matches_the_live_one(self, refreshArchive, tmp_path):
        forecasts.prefetchForecasts(RUN, RUN, "SA1", cacheDir=tmp_path)
        source = forecasts.loadForecastCache("SA1", cacheDir=tmp_path)
        np.testing.assert_allclose(forecasts.refreshMatrix([RUN], "SA1", source=source),
                                   forecasts.refreshMatrix([RUN], "SA1"), equal_nan=True)

    # The agent reads the latest run published by each step.
    def test_the_active_run(self):
        grid = np.arange(6)[:, None] * np.ones((6, 288))
        assert [forecasts.activeRun(grid, step)[0] for step in (0, 47, 48, 287)] == [
            0, 0, 1, 5]

    # A single path is its own active run, whatever the step.
    def test_a_single_path_is_always_active(self):
        path = np.arange(288.0)
        np.testing.assert_array_equal(forecasts.activeRun(path, 200), path)

    # Each step names the run activeRun handed it, from the stack's own shape.
    def test_the_decision_runs(self):
        grid = np.zeros((6, 288))
        got = forecasts.decisionRuns(RUN, grid)
        start = pd.Timestamp(RUN)
        assert [got[step] - start for step in (0, 47, 48, 287)] == [
            pd.Timedelta(0), pd.Timedelta(0), pd.Timedelta(hours=4),
            pd.Timedelta(hours=20)]

    # A single path is the one run that opens the day.
    def test_a_single_path_is_one_run(self):
        got = forecasts.decisionRuns(RUN, np.zeros(288))
        assert set(got) == {pd.Timestamp(RUN)}

    # The decision path takes step k from the run active at k.
    def test_the_decision_path(self, refreshArchive):
        grid = forecasts.refreshMatrix([RUN], "SA1")[0]
        got = forecasts.decisionPath(grid)
        assert got.shape == (288,)
        for step in (0, 47, 48, 150, 287):
            assert got[step] == grid[step // 48, step]


def fakeSource(runs, horizon=288):
    """A prefetch cache holding one trading day of steps for every run, values keyed on
    (run index, step) so a misplaced column is visible rather than plausible."""
    columns = forecasts._stepColumns(horizon)
    values = [[1000.0 * r + step for step in range(1, horizon + 1)]
              for r in range(len(runs))]
    return pd.DataFrame(values, index=pd.DatetimeIndex(runs), columns=columns)


class TestWeeklyRefresh:
    """A weekly episode still reads one trading day ahead, because PREDISPATCH stops at
    the end of its own trading day and AEMO publishes no RRP past it."""

    weekly = 7 * 288

    def runsOf(self, every):
        return [RUN + offset * forecasts.nem_interval
                for offset in range(0, self.weekly, every)]

    def test_has_one_row_per_refresh(self):
        runs = self.runsOf(48)
        grid = forecasts.refreshMatrix([RUN], "SA1", horizon=self.weekly, every=48,
                                       source=fakeSource(runs))
        assert grid.shape == (1, 42, self.weekly)

    def test_each_run_is_cut_at_the_end_of_its_own_trading_day(self):
        runs = self.runsOf(48)
        grid = forecasts.refreshMatrix([RUN], "SA1", horizon=self.weekly, every=48,
                                       source=fakeSource(runs))
        for refresh, offset in enumerate(range(0, self.weekly, 48)):
            dayEnd = (offset // 288 + 1) * 288
            assert np.isnan(grid[0, refresh, :offset]).all()
            assert np.isnan(grid[0, refresh, dayEnd:]).all()
            covered = grid[0, refresh, offset:dayEnd]
            expected = [1000.0 * refresh + step
                        for step in range(1, dayEnd - offset + 1)]
            np.testing.assert_allclose(covered, expected)

    def test_a_single_daily_run_covers_its_own_day_only(self):
        runs = self.runsOf(288)
        grid = forecasts.refreshMatrix([RUN], "SA1", horizon=self.weekly, every=288,
                                       source=fakeSource(runs))
        assert grid.shape == (1, 7, self.weekly)
        for day in range(7):
            assert not np.isnan(grid[0, day, day * 288:(day + 1) * 288]).any()
            assert np.isnan(grid[0, day, (day + 1) * 288:]).all()

    # 224 divides 2016 but not 288, so its runs would land off the trading-day grid and
    # data.verifyRefresh, which reconstructs a run time from the day start, would disagree.
    def test_a_refresh_that_does_not_divide_a_trading_day_is_refused(self):
        with pytest.raises(ValueError, match="288"):
            forecasts.refreshRunTimes(RUN, horizon=self.weekly, every=224)

    # 336 is seven refreshes of 48, so it passes the episode check, but it is 28 hours:
    # one and a sixth trading days, so the second day's runs would start mid-day.
    def test_an_episode_must_be_whole_trading_days(self):
        with pytest.raises(ValueError, match="whole number of 288-interval trading days"):
            forecasts.refreshRunTimes(RUN, horizon=336, every=48)


class TestMatrixSizeGuard:
    """The grid grows with the square of the horizon, so a weekly episode at a 5-minute
    refresh is gigabytes. It says so rather than waiting for the OOM killer."""

    def test_a_modest_matrix_is_allowed(self, monkeypatch):
        monkeypatch.setattr(forecasts, "forecastMatrix",
                            lambda runs, region, cover, **k: np.zeros((len(runs), cover)))
        got = forecasts.refreshMatrix([pd.Timestamp("2024-01-10 04:00")], "NSW1",
                                      horizon=288, every=48)
        assert got.shape == (1, 6, 288)

    def test_an_oversized_matrix_is_refused_with_the_figure(self):
        """Weekly episodes at a 5-minute refresh. 200 of them is 6 GiB, which is what a
        several-year training grid would ask for."""
        weeks = [pd.Timestamp("2024-01-01 04:00") + pd.Timedelta(weeks=n)
                 for n in range(200)]
        with pytest.raises(ValueError, match="GiB, over the"):
            forecasts.refreshMatrix(weeks, "NSW1", horizon=2016, every=1)

    def test_a_year_of_weekly_training_episodes_still_fits(self):
        """1.6 GiB, which is large but is a configuration the backtest may legitimately
        ask for, so the cap must not refuse it."""
        weeks = [pd.Timestamp("2024-01-01 04:00") + pd.Timedelta(weeks=n)
                 for n in range(52)]
        forecasts._requireSize(len(weeks), 2016, 1)

    def test_the_cap_is_what_decides(self, monkeypatch):
        monkeypatch.setattr(forecasts, "matrixSizeCap", 1)
        with pytest.raises(ValueError, match="matrixSizeCap"):
            forecasts.refreshMatrix([pd.Timestamp("2024-01-10 04:00")], "NSW1",
                                    horizon=288, every=48)

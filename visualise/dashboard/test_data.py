"""Tests for visualise/dashboard/data.py. Run with: uv run pytest visualise/ -q

Everything here is a pure function of a frame, so no Dash import and no server.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import actions
from visualise.dashboard import data

REGION = "NSW1"
DAY = datetime(2025, 1, 1, 4)


def dayRows(count=14, folds=2):
    rows = []
    for index in range(count):
        day = DAY + timedelta(days=index)
        rows.append({
            "fold": index // (count // folds), "algo": "maskableppo", "trainMode": "warm",
            "seed": 0, "priceSource": "realised", "testDay": day,
            "agentProfit": 10.0 - index, "agentRevenue": 12.0, "agentDegradation": 2.0,
            "oracleDualBound": 40.0, "oracleGap": 0.01, "spikeIntensity": 5.0 * (index % 2),
            "highVol": bool(index % 2), "timesteps": 30000, "warmFrom": "",
            "writtenAt": "2026-01-01T00:00:00",
        })
    return rows


def publishRun(root, rows, name="rolling", region=REGION, every=None):
    """One published region, as `backtest.publish` writes it: root/backtestResults/
    <name>_<region>.csv. Returns the run, root/backtestResults/<name>."""
    frame = pd.DataFrame(rows)
    if every is not None:
        frame["forecastEvery"] = every
    run = root / "backtestResults" / name
    run.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data.publishedFile(run, region), index=False)
    return run


def writeActions(root, name, seeds=(0,), every=48, sources=("forecast",), live=False):
    """An actions export of a run named `name`, as `backtest.publish` writes it:
    root/actions/<name>_<region>.parquet. `live` writes the CSV a running backtest
    appends to instead, root/actions/<name>_<region>.csv."""
    rows = []
    for source in sources:
        for seed in seeds:
            for step in range(288):
                forecast = source == "forecast"
                rows.append({
                    "forecastRunTime": (DAY + timedelta(minutes=5 * (step // every * every))
                                        if forecast else None),
                    "intervalTime": DAY + timedelta(minutes=5 * (step + 1)),
                    "action": "charge" if step < 3 else "idle", "soc": 0.5,
                    "forecastPrice": 10.0 * seed + step if forecast else None,
                    "realisedPrice": 50.0 + step, "region": REGION, "fold": 0,
                    "seed": seed, "algo": "maskableppo", "trainMode": "warm",
                    "priceSource": source})
    frame = pd.DataFrame(rows, columns=list(actions.columns))
    if live:
        actions.appendRows(root / "actions" / f"{name}_{REGION}.csv", frame)
    else:
        frame["forecastRunTime"] = pd.to_datetime(frame.forecastRunTime)
        actions.writeExport(frame, root / "actions" / f"{name}_{REGION}.parquet")


@pytest.fixture
def run(tmp_path):
    """One finished two-fold run, published."""
    return publishRun(tmp_path, dayRows(), every=288)


class TestLoadRun:
    # A region with no published CSV names the file it looked for.
    def test_a_missing_run_names_the_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="rolling_SA1.csv"):
            data.loadRun("SA1", tmp_path / "backtestResults" / "rolling")

    # A finished run comes back with its days and its forecast pull together, and the
    # pull is not left in the days as a column.
    def test_a_finished_run_loads_both_parts(self, run):
        loaded = data.loadRun(REGION, run)
        assert len(loaded.days) == 14 and loaded.forecastEvery == 288
        assert "forecastEvery" not in loaded.days


class TestSelectors:
    # The controls list what the data holds, never what config.py declares.
    def test_it_lists_only_what_is_present(self, run):
        got = data.selectors(data.loadRun(REGION, run).days)
        assert got == {"algo": ["maskableppo"], "trainMode": ["warm"],
                       "priceSource": ["realised"], "seed": [0]}

    # A second algorithm appears without any dashboard change.
    def test_a_new_algorithm_appears_on_its_own(self, run):
        frame = data.loadRun(REGION, run).days
        extra = frame.copy()
        extra["algo"] = "dqn"
        both = pd.concat([frame, extra], ignore_index=True)
        assert data.selectors(both)["algo"] == ["dqn", "maskableppo"]


class TestScope:
    # Day scope keeps that day and nothing else.
    def test_a_day_keeps_one_day(self, run):
        frame = data.loadRun(REGION, run).days
        got = data.scope(frame, day=DAY)
        assert len(got) == 1 and got.testDay.iloc[0] == pd.Timestamp(DAY)

    # Week scope keeps the seven days starting at the given day.
    def test_a_week_keeps_seven_days(self, run):
        frame = data.loadRun(REGION, run).days
        got = data.scope(frame, week=DAY)
        assert len(got) == 7
        assert got.testDay.min() == pd.Timestamp(DAY)
        assert got.testDay.max() == pd.Timestamp(DAY + timedelta(days=6))

    # Job scope is the whole frame, unchanged.
    def test_no_scope_returns_everything(self, run):
        frame = data.loadRun(REGION, run).days
        assert len(data.scope(frame)) == len(frame)

    # A day outside the run gives an empty frame rather than raising.
    def test_an_unknown_day_is_empty(self, run):
        frame = data.loadRun(REGION, run).days
        assert data.scope(frame, day=datetime(2020, 1, 1, 4)).empty

    # Asking for a day and a week at once is a caller error.
    def test_a_day_and_a_week_together_are_refused(self, run):
        frame = data.loadRun(REGION, run).days
        with pytest.raises(ValueError, match="one of"):
            data.scope(frame, day=DAY, week=DAY)


class TestAvailableRegions:
    # Each region of a run is its own CSV; a longer run name is another run.
    def test_it_lists_this_runs_regions_only(self, tmp_path):
        for region in ("NSW1", "SA1"):
            publishRun(tmp_path, dayRows(2, folds=1), region=region)
        publishRun(tmp_path, dayRows(2, folds=1), name="rolling_fc4h")
        run = tmp_path / "backtestResults" / "rolling"
        assert data.availableRegions(run) == ["NSW1", "SA1"]


class TestFindResults:
    def make(self, root, name="rolling2023_profit"):
        """One finished run under `root`, named as an experiment of its own."""
        return publishRun(root, dayRows(2, folds=1), name=name)

    # A named run is taken as it is, whatever else sits beside it.
    def test_a_named_run_wins(self, run):
        assert data.findResults(run) == run

    # The regions of one run are one candidate, not five.
    def test_the_one_run_is_found(self, tmp_path):
        wanted = self.make(tmp_path)
        publishRun(tmp_path, dayRows(2, folds=1), name=wanted.name, region="SA1")
        assert data.findResults(None, root=tmp_path / "backtestResults") == wanted

    # Two experiments cannot be chosen between, so the message names both.
    def test_several_runs_name_themselves(self, tmp_path):
        self.make(tmp_path)
        self.make(tmp_path, "rolling2023_fc4h")
        with pytest.raises(FileNotFoundError, match="several runs"):
            data.findResults(None, root=tmp_path / "backtestResults")

    # An empty tree says what to run, not which path is missing.
    def test_an_empty_tree_says_what_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="cli rolling"):
            data.findResults(None, root=tmp_path / "gone")


class TestVerifyRefresh:
    """A forecast-priced run must state its test-day pull and the export must match."""

    def make(self, tmp_path, every, exportEvery=None, stated=True):
        rows = [dict(row, priceSource="forecast") for row in dayRows(count=2, folds=1)]
        run = publishRun(tmp_path, rows, every=every if stated else None)
        # The realised rows read no forecast, so they never disagree with it.
        writeActions(tmp_path, "rolling", every=every if exportEvery is None else exportEvery,
                     sources=("forecast", "realised"))
        return run

    @pytest.mark.parametrize("every", [1, 48, 288])
    def test_a_matching_export_loads(self, tmp_path, every):
        assert data.loadRun(REGION, self.make(tmp_path, every)).forecastEvery == every

    # A run backtested every 4 hours cannot be shown as a once-a-day run.
    def test_a_mismatched_export_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="disagrees with forecastEvery=288"):
            data.loadRun(REGION, self.make(tmp_path, 288, exportEvery=48))

    def test_an_unstated_refresh_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="does not state forecastEvery"):
            data.loadRun(REGION, self.make(tmp_path, 288, stated=False))

    def test_a_missing_export_is_refused(self, tmp_path):
        run = self.make(tmp_path, 48)
        (tmp_path / "actions" / f"rolling_{REGION}.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="actions export"):
            data.loadRun(REGION, run)


class TestDecisions:
    # A day's decisions come from the actions export, every model and source.
    def test_one_day_of_every_source(self, run, tmp_path):
        writeActions(tmp_path, "rolling", sources=("forecast", "realised"))
        got = data.decisions(data.loadRun(REGION, run), DAY)
        assert len(got) == 2 * 288 and set(got.priceSource) == {"forecast", "realised"}

    def test_another_day_is_empty(self, run, tmp_path):
        writeActions(tmp_path, "rolling")
        assert data.decisions(data.loadRun(REGION, run), DAY + timedelta(days=1)).empty

    # A run without an export draws an empty chart rather than raising.
    def test_no_export_is_empty(self, run):
        got = data.decisions(data.loadRun(REGION, run), DAY)
        assert got.empty and list(got.columns) == list(data.actionColumns)


class TestDispatchPath:
    def path(self, tmp_path, **overrides):
        """One day of decisions: charge, then idle."""
        writeActions(tmp_path, "rolling", seeds=(0,), sources=("forecast", "realised"))
        rows = data._readActions(str(tmp_path / "actions" / f"rolling_{REGION}.parquet"), 0)
        return rows.assign(**overrides)

    # The action is the one the agent took, the same reading the forecast screen takes
    # off its own rollout.
    def test_the_action_is_carried(self, tmp_path):
        got = data.dispatchPath(self.path(tmp_path), priceSource="realised")
        assert list(got.action[:3]) == ["charge"] * 3
        assert set(got.action[3:]) == {"idle"}

    # The price is the realised price the agent was paid, never the one it observed.
    def test_the_price_is_the_realised_one(self, tmp_path):
        got = data.dispatchPath(self.path(tmp_path), priceSource="forecast")
        assert list(got.price) == [50.0 + step for step in range(288)]

    # The chart reads the intervals in order, whatever order the file holds them in.
    def test_the_rows_come_back_in_interval_order(self, tmp_path):
        shuffled = self.path(tmp_path).sort_values("realisedPrice", ascending=False)
        got = data.dispatchPath(shuffled, priceSource="realised")
        assert list(got.price) == [50.0 + step for step in range(288)]

    # Picking one model must leave the other models and sources out of the chart.
    def test_it_filters_to_one_model(self, tmp_path):
        rows = self.path(tmp_path)
        both = pd.concat([rows, rows.assign(algo="dqn")], ignore_index=True)
        assert len(data.dispatchPath(both, algo="maskableppo",
                                     priceSource="forecast")) == 288

    # A seed that is not in the export must come back empty rather than draw seed 0's
    # dispatch under its own name.
    def test_a_seed_without_a_trajectory_is_empty(self, tmp_path):
        rows = self.path(tmp_path)
        assert data.dispatchPath(rows, seed=3).empty
        assert len(data.dispatchPath(rows, seed=0, priceSource="realised")) == 288


class TestForecastPath:
    # The export follows --results: the run's name, plus the region.
    def test_the_export_follows_the_results_directory(self, tmp_path):
        writeActions(tmp_path, "rolling_fc4h")
        writeActions(tmp_path, "rolling_fc5m")
        got = data.actionsFile(tmp_path / "backtestResults" / "rolling_fc4h", REGION)
        assert got.name == f"rolling_fc4h_{REGION}.parquet"
        assert data.actionsFile(tmp_path / "backtestResults" / "other", REGION) is None

    def test_one_day_of_one_seed(self, tmp_path):
        writeActions(tmp_path, "rolling", seeds=(0, 1))
        path = data.forecastPath(tmp_path / "backtestResults" / "rolling", REGION, DAY,
                                 algo="maskableppo", trainMode="warm", seed=1)
        assert len(path) == 288 and path.forecastPrice.iloc[0] == 10.0
        assert path.forecastRunTime.nunique() == 6

    # The realised rows read no forecast, so they never reach the forecast line.
    def test_the_realised_rows_are_left_out(self, tmp_path):
        writeActions(tmp_path, "rolling", sources=("forecast", "realised"))
        path = data.forecastPath(tmp_path / "backtestResults" / "rolling", REGION, DAY)
        assert len(path) == 288 and path.forecastPrice.notna().all()

    # A seed the export does not hold has no forecast, never another seed's.
    def test_a_missing_seed_is_empty(self, tmp_path):
        writeActions(tmp_path, "rolling")
        assert data.forecastPath(tmp_path / "backtestResults" / "rolling", REGION, DAY,
                                 seed=3).empty

    def test_no_export_is_none(self, tmp_path):
        assert data.forecastPath(tmp_path / "backtestResults" / "rolling", REGION,
                                 DAY) is None

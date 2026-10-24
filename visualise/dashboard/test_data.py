"""Tests for visualise/dashboard/data.py. Run with: uv run pytest visualise/ -q

Everything here is a pure function of a frame, so no Dash import and no server.
"""
import json
from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import records
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


@pytest.fixture
def run(tmp_path):
    """One finished two-fold run on disk: fold CSVs plus a manifest."""
    rows = dayRows()
    for fold in sorted({row["fold"] for row in rows}):
        records.writeFold(tmp_path, REGION, fold,
                          [row for row in rows if row["fold"] == fold])
    records.writeManifest(tmp_path, REGION, {
        "region": REGION, "algos": ["maskableppo"], "trainModes": ["warm"],
        "trainWeeks": 52, "warmTimesteps": 30000, "coldTimesteps": 300000,
        "seeds": [0], "capDate": "2026-01-01T04:00:00", "horizon": 288,
        "priceSources": ["realised"], "battery": [100.0, 200.0, 300000.0],
        "calendar": False, "discretePower": False, "gap": 0.02, "initialSoc": 0.5,
        "rewardScale": 1000.0, "testWeeks": 1, "stepWeeks": 1,
        "start": "2024-01-01T04:00:00", "end": "2026-01-01T04:00:00",
    })
    return tmp_path


class TestLoadRun:
    # A region with no fold CSV names the directory it looked in.
    def test_a_missing_run_names_the_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            data.loadRun("SA1", tmp_path)

    # A finished run comes back with its days and its manifest together.
    def test_a_finished_run_loads_both_parts(self, run):
        loaded = data.loadRun(REGION, run)
        assert len(loaded.days) == 14
        assert loaded.manifest["trainWeeks"] == 52

    # A manifest written beside no rows is still an error, not an empty dashboard.
    def test_a_manifest_without_rows_is_refused(self, tmp_path):
        records.writeManifest(tmp_path, REGION, {"region": REGION})
        with pytest.raises(FileNotFoundError):
            data.loadRun(REGION, tmp_path)


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


class TestFoldOfDay:
    # A day maps to the fold whose test window holds it, for the trajectory lookup.
    def test_a_day_maps_to_its_fold(self, run):
        frame = data.loadRun(REGION, run).days
        assert data.foldOfDay(frame, DAY) == 0
        assert data.foldOfDay(frame, DAY + timedelta(days=7)) == 1

    # A day no fold tested has no trajectory to show.
    def test_an_unknown_day_has_no_fold(self, run):
        frame = data.loadRun(REGION, run).days
        assert data.foldOfDay(frame, datetime(2020, 1, 1, 4)) is None


class TestFindResults:
    def make(self, root, name="rolling2023_profit_all"):
        """One finished run under `root`, named as an experiment of its own."""
        records.writeFold(root / name, REGION, 0, dayRows(2, folds=1))
        return root / name

    # A named directory is taken as it is, whatever else sits beside it.
    def test_a_named_directory_wins(self, run):
        assert data.findResults(run) == run

    # `cli rolling --results` names each experiment, so the run on disk is the one to
    # read even though its name is not the default one. The exports beside it are not
    # a run and must not be offered as one.
    def test_the_one_run_is_found(self, tmp_path):
        wanted = self.make(tmp_path)
        (tmp_path / "actions").mkdir()
        assert data.findResults(None, root=tmp_path) == wanted

    # Two experiments cannot be chosen between, so the message names both.
    def test_several_runs_name_themselves(self, tmp_path):
        self.make(tmp_path)
        self.make(tmp_path, "rolling2023_fc4h")
        with pytest.raises(FileNotFoundError, match="several runs"):
            data.findResults(None, root=tmp_path)

    # An empty tree says what to run, not which path is missing.
    def test_an_empty_tree_says_what_to_run(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="cli rolling"):
            data.findResults(None, root=tmp_path / "gone")


class TestVerifyRefresh:
    """A forecast-priced run must state its test-day pull and the export must match."""

    def make(self, tmp_path, every, exportEvery=None, stated=True):
        run = tmp_path / "rolling_all"
        rows = [dict(row, priceSource="forecast") for row in dayRows(count=2, folds=1)]
        records.writeFold(run, REGION, 0, rows)
        manifest = {"region": REGION, "horizon": 288}
        if stated:
            manifest["forecastEvery"] = every
        records.writeManifest(run, REGION, manifest)
        exportEvery = every if exportEvery is None else exportEvery
        clock = pd.date_range(DAY + timedelta(minutes=5), periods=288, freq="5min")
        (tmp_path / "actions").mkdir()
        pd.DataFrame({
            "forecastRunTime": [DAY + timedelta(minutes=5 * (step // exportEvery
                                                             * exportEvery))
                                for step in range(288)],
            "intervalTime": clock, "forecastPrice": 50.0, "seed": 0,
            "algo": "maskableppo", "trainMode": "warm",
        }).to_csv(tmp_path / "actions" / f"rolling_{REGION}.csv", index=False)
        return run

    @pytest.mark.parametrize("every", [1, 48, 288])
    def test_a_matching_export_loads(self, tmp_path, every):
        assert data.loadRun(REGION, self.make(tmp_path, every)).manifest[
            "forecastEvery"] == every

    # A run backtested every 4 hours cannot be shown as a once-a-day run.
    def test_a_mismatched_export_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="disagrees with forecastEvery=288"):
            data.loadRun(REGION, self.make(tmp_path, 288, exportEvery=48))

    def test_an_unstated_refresh_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="does not state forecastEvery"):
            data.loadRun(REGION, self.make(tmp_path, 288, stated=False))

    def test_a_missing_export_is_refused(self, tmp_path):
        run = self.make(tmp_path, 48)
        (tmp_path / "actions" / f"rolling_{REGION}.csv").unlink()
        with pytest.raises(FileNotFoundError, match="actions export"):
            data.loadRun(REGION, run)


class TestTrajectories:
    # A day with no trajectory file gives an empty frame with the right columns.
    def test_a_missing_parquet_is_empty(self, run):
        loaded = data.loadRun(REGION, run)
        got = data.trajectories(loaded, DAY)
        assert got.empty and list(got.columns) == list(records.trajectoryColumns)


class TestDispatchPath:
    def path(self, **overrides):
        """One hour of intervals: charge, then idle, then discharge."""
        rows = []
        for step in range(12):
            bought = 1.0 if step < 3 else 0.0
            sold = 1.0 if step >= 9 else 0.0
            rows.append({
                "fold": 0, "algo": "maskableppo", "trainMode": "warm", "seed": 0,
                "priceSource": "realised", "testDay": DAY, "interval": step,
                "settlementdate": DAY + timedelta(minutes=5 * (step + 1)),
                "price": 50.0 + step, "observedPrice": 90.0, "soc": 0.5,
                "bought": bought, "sold": sold, "profit": 0.1, **overrides,
            })
        return pd.DataFrame(rows)

    # A buy is a charge interval and a sale is a discharge one, which is the same
    # reading the forecast screen takes off its own rollout.
    def test_bought_and_sold_become_actions(self):
        got = data.dispatchPath(self.path())
        assert list(got.action[:3]) == ["charge"] * 3
        assert list(got.action[3:9]) == ["idle"] * 6
        assert list(got.action[9:]) == ["discharge"] * 3

    # The price is the realised price the agent was paid, never the one it observed.
    def test_the_price_is_the_realised_one(self):
        got = data.dispatchPath(self.path())
        assert list(got.price) == [50.0 + step for step in range(12)]

    # The chart reads the intervals in order, whatever order the file holds them in.
    def test_the_rows_come_back_in_interval_order(self):
        shuffled = self.path().sort_values("price", ascending=False)
        got = data.dispatchPath(shuffled)
        assert list(got.price) == [50.0 + step for step in range(12)]

    # Picking one model must leave the other models rows out of the chart.
    def test_it_filters_to_one_model(self):
        both = pd.concat([self.path(), self.path(algo="dqn")], ignore_index=True)
        assert len(data.dispatchPath(both, algo="maskableppo")) == 12

    # A fold writes one seed's trajectory, so a seed that is not on disk must come
    # back empty rather than draw seed 0's dispatch under its own name.
    def test_a_seed_without_a_trajectory_is_empty(self):
        assert data.dispatchPath(self.path(), seed=3).empty
        assert len(data.dispatchPath(self.path(), seed=0)) == 12

    # A day with no rows gives an empty frame with the columns the chart expects.
    def test_an_empty_trajectory_is_empty(self):
        got = data.dispatchPath(self.path().iloc[0:0])
        assert got.empty and list(got.columns) == list(data.dispatchColumns)


def writeActions(root, name, seeds=(0,)):
    """An actions export beside a run named `name`, as `backtest.actions` writes it."""
    rows = []
    for seed in seeds:
        for step in range(288):
            rows.append({
                "forecastRunTime": DAY + timedelta(minutes=5 * (step // 48 * 48)),
                "intervalTime": DAY + timedelta(minutes=5 * (step + 1)),
                "action": "idle", "soc": 0.5, "forecastPrice": 10.0 * seed + step,
                "realisedPrice": 1.0, "region": REGION, "fold": 0, "seed": seed,
                "algo": "maskableppo", "trainMode": "warm"})
    (root / "actions").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "actions" / f"{name}_{REGION}.csv", index=False)


class TestForecastPath:
    # The export follows --results: the run directory without "_all", plus the region.
    def test_the_export_follows_the_results_directory(self, tmp_path):
        writeActions(tmp_path, "rolling_fc4h")
        writeActions(tmp_path, "rolling_fc5m")
        got = data.actionsFile(tmp_path / "rolling_fc4h_all", REGION)
        assert got.name == f"rolling_fc4h_{REGION}.csv"
        assert data.actionsFile(tmp_path / "other_all", REGION) is None

    def test_one_day_of_one_seed(self, tmp_path):
        writeActions(tmp_path, "rolling", seeds=(0, 1))
        path = data.forecastPath(tmp_path / "rolling_all", REGION, DAY,
                                 algo="maskableppo", trainMode="warm", seed=1)
        assert len(path) == 288 and path.forecastPrice.iloc[0] == 10.0
        assert path.forecastRunTime.nunique() == 6

    # A seed the export does not hold has no forecast, never another seed's.
    def test_a_missing_seed_is_empty(self, tmp_path):
        writeActions(tmp_path, "rolling")
        assert data.forecastPath(tmp_path / "rolling_all", REGION, DAY, seed=3).empty

    def test_no_export_is_none(self, tmp_path):
        assert data.forecastPath(tmp_path / "rolling_all", REGION, DAY) is None

"""Tests for backtest/runner.py. Run with: uv run pytest backtest/ -q

Four seams are replaced: price_series, buildBaseline, trainOne and evaluateSeeds. So
nothing here touches the network, the MILP or torch, and what is pinned is the wiring:
which window each fold trains on, how the warm chain finds its parent, and that a
resumed run does not redo work.

The checkpoint on disk is the training record. There is no separate ledger: a unit is
done when its rows are in the fold CSV, and training is done when agent.zip exists.
"""
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import config
from backtest import folds, records, runner
from rl import policies
from utils import nem_interval

START = datetime(2024, 1, 1, 4)
REGION = "NSW1"


def refreshFrame(days, every=48, horizon=288):
    """A prefetched cache holding every refresh run of `days`, fully covered."""
    runs = [pd.Timestamp(day) + offset * nem_interval
            for day in days for offset in range(0, horizon, every)]
    return pd.DataFrame([[1.0] * horizon] * len(runs), index=runs,
                        columns=[str(n) for n in range(1, horizon + 1)])


@pytest.fixture
def prices():
    """Ten weeks of 04:00-aligned prices, enough for a 4-week train and six folds."""
    end = START + timedelta(weeks=10)
    index = pd.date_range(START + nem_interval, end, freq=nem_interval)
    return pd.Series(50.0, index=index, name="RRP")


@pytest.fixture
def config(tmp_path, prices):
    return runner.RollingConfig(
        region=REGION, start=START, end=prices.index[-1], capDate=datetime(2026, 1, 1, 4),
        trainWeeks=4, seeds=(0, 1), trainModes=("warm",),
        warmTimesteps=10, coldTimesteps=100,
        # buildBaseline is faked, so nothing solves. Without this the suite reads the
        # real data_cache/lp_baseline, which is gitignored and absent on a clean clone.
        allowSolve=True,
        outDir=tmp_path / "runs", resultsDir=tmp_path / "results",
        policyRoot=tmp_path / "trainedPolicies",
    )


@pytest.fixture
def spy(monkeypatch, prices):
    """Record every call across the four seams, and fake just enough to continue."""
    calls = {"prices": [], "baseline": [], "train": [], "evaluate": []}

    def fakePrices(start, end, region):
        calls["prices"].append((start, end, region))
        return folds.sliceWindow(prices, start, end)

    def fakeBaseline(window, **kwargs):
        calls["baseline"].append(kwargs)
        horizon = kwargs.get("horizon") or config.episode_intervals
        episodes = len(window) // horizon
        return [{"start": window.index[n * horizon].isoformat(),
                 "profitPerInterval": [0.01] * horizon, "dualBound": 10.0, "profit": 9.0,
                 "mipGap": 0.01, "cyclesUsed": 6.5,
                 "soc": [0.5] * horizon, "bought": [0.0] * horizon,
                 "sold": [0.0] * horizon}
                for n in range(episodes)]

    def fakeTrain(algo, trainPrices, model, **kwargs):
        calls["train"].append({"algo": algo, "prices": trainPrices, **kwargs})
        directory = kwargs["outDir"] / f"{algo}_seed{kwargs['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "agent.zip").write_bytes(b"")
        # The real trainOne writes the transform beside the weights, and publishing
        # refuses a checkpoint that holds only one of the two.
        (directory / "scaler.json").write_text('{"cap": 23200.0}')
        return None, directory / "agent"

    def fakeEvaluate(checkpoints, testPrices, model, **kwargs):
        calls["evaluate"].append({"checkpoints": dict(checkpoints),
                                  "prices": testPrices, **kwargs})
        horizon = kwargs.get("horizon") or config.episode_intervals
        windows = len(testPrices) // horizon
        days = horizon // 288
        return {seed: [{"window": w, "agentProfit": 1.0 * days, "agentRevenue": 2.0 * days,
                        "agentDegradation": 1.0 * days, "agentCycles": 0.5 * days,
                        "agentTerminalValue": 0.0, "lpProfit": 10.0, "regret": 9.0,
                        "perDay": [{"revenue": 2.0, "degradation": 1.0, "profit": 1.0,
                                    "cycles": 0.5, "terminalValue": 0.0}] * days,
                        "trajectory": [{"price": 50.0, "soc": 0.5, "bought": 0.0,
                                        "sold": 0.0, "profit": 0.0}] * horizon}
                       for w in range(windows)]
                for seed in checkpoints}

    monkeypatch.setattr(runner, "price_series", fakePrices)
    monkeypatch.setattr(runner, "buildBaseline", fakeBaseline)
    monkeypatch.setattr(runner, "trainOne", fakeTrain)
    monkeypatch.setattr(runner, "evaluateSeeds", fakeEvaluate)
    return calls


class TestSpanWideWork:
    # Prices are fetched once for the whole span. A per-day call re-reads the monthly
    # feather every time.
    def test_prices_are_fetched_once(self, config, spy):
        runner.runRolling(config)
        assert len(spy["prices"]) == 1
        assert spy["prices"][0][:2] == (config.start, config.end)

    # The oracle is built once for the whole span, not per fold: consecutive folds
    # overlap by all but one week, so a per-fold build re-reads the same cache entries.
    def test_the_oracle_is_built_once_for_the_span(self, config, spy):
        runner.runRolling(config)
        assert len(spy["baseline"]) == 1

    # The oracle must be asked for with the settings the run reports under. Discrete
    # power is the one the agent's three actions can reach, so `captured` divides by it.
    def test_the_oracle_is_asked_for_with_the_reported_settings(self, config, spy):
        runner.runRolling(config)
        asked = spy["baseline"][0]
        assert (asked["calendar"], asked["discretePower"]) == (False, True)
        assert asked["gap"] == 0.02

    # The reward is raw profit, so training days need no oracle: the build starts at the
    # first test day, and no training call is handed a baseline.
    def test_the_oracle_covers_the_test_days_only(self, config, monkeypatch, spy):
        windows = []
        fake = runner.buildBaseline
        monkeypatch.setattr(runner, "buildBaseline",
                            lambda window, **kw: windows.append(window) or fake(window, **kw))
        runner.runRolling(config)
        first = folds.enumerateFolds(config.start, config.end, 4)[0]
        assert windows[0].index[0] == first.testStart + nem_interval
        assert all("baselines" not in call for call in spy["train"])

    # The agent starts where the oracle starts. A start SoC that reaches the oracle but
    # not the agent's env pairs one episode's profit with another episode's bound.
    def test_the_agent_starts_at_the_oracle_soc(self, config, spy):
        runner.runRolling(config._replace(initialSoc=0.5))
        assert spy["baseline"][0]["initialSoc"] == 0.5
        assert {call["initialSoc"] for call in spy["train"]} == {0.5}
        assert {call["initialSoc"] for call in spy["evaluate"]} == {0.5}


class TestFoldWindows:
    # Each fold trains on its own sliding window, starting one interval after its 04:00
    # boundary because price_series is half-open.
    def test_each_fold_trains_on_its_own_window(self, config, spy):
        runner.runRolling(config)
        first = [call for call in spy["train"] if call["seed"] == 0]
        for fold, call in zip(folds.enumerateFolds(config.start, config.end, 4), first):
            assert call["prices"].index[0] == fold.trainStart + nem_interval
            assert len(call["prices"]) == 4 * 7 * 288

    # Each fold is scored on exactly its test window.
    def test_each_fold_evaluates_on_its_test_window(self, config, spy):
        runner.runRolling(config)
        for fold, call in zip(folds.enumerateFolds(config.start, config.end, 4),
                              spy["evaluate"]):
            assert call["prices"].index[0] == fold.testStart + nem_interval
            assert len(call["prices"]) == 7 * 288

    # The per-fold bound is sliced out of the span-wide records by day offset, so a
    # day-shifted record set would give wrong regret on every row and raise nothing.
    def test_a_day_shifted_oracle_record_set_is_refused(self, config, spy, monkeypatch):
        good = runner.buildBaseline
        monkeypatch.setattr(runner, "buildBaseline",
                            lambda window, **kw: good(window, **kw)[1:])
        with pytest.raises(ValueError, match="oracle"):
            runner.runRolling(config)


class TestWarmChain:
    # The first fold of a chain has no parent and trains for the full cold budget.
    def test_the_first_fold_starts_fresh(self, config, spy):
        runner.runRolling(config)
        first = spy["train"][0]
        assert first["warmFrom"] is None
        assert first["timesteps"] == config.coldTimesteps

    # Every later fold continues from the previous fold's checkpoint for the same seed,
    # on the shorter top-up budget.
    def test_later_folds_continue_from_the_previous_fold(self, config, spy):
        runner.runRolling(config)
        bySeed = {}
        for call in spy["train"]:
            bySeed.setdefault(call["seed"], []).append(call)
        for seed, calls in bySeed.items():
            for previous, current in zip(calls, calls[1:]):
                assert current["warmFrom"] == previous["outDir"] / f"maskableppo_seed{seed}"
                assert current["timesteps"] == config.warmTimesteps

    # A non-contiguous fold selection chains each fold to the previous SELECTED fold,
    # not to the one numerically before it.
    def test_a_sparse_selection_chains_to_the_previous_selected_fold(self, config, spy):
        sparse = config._replace(foldFilter=frozenset({0, 3}), seeds=(0,))
        runner.runRolling(sparse)
        assert [call["warmFrom"] for call in spy["train"]][0] is None
        assert spy["train"][1]["warmFrom"] == spy["train"][0]["outDir"] / "maskableppo_seed0"

    # Seeds are separate chains: seed 1 never inherits from seed 0.
    def test_seeds_are_separate_chains(self, config, spy):
        runner.runRolling(config)
        for call in spy["train"]:
            if call["warmFrom"] is not None:
                assert call["warmFrom"].name == f"maskableppo_seed{call['seed']}"

    # One rule, not three: the parent is the previous SELECTED fold. A selection that
    # starts partway through therefore begins a new chain rather than silently
    # inheriting, and the empty warmFrom on its first row says so.
    def test_a_selection_starting_partway_begins_a_new_chain(self, config, spy):
        runner.runRolling(config._replace(foldFilter=frozenset({2, 3}), seeds=(0,)))
        assert spy["train"][0]["warmFrom"] is None
        assert spy["train"][0]["timesteps"] == config.coldTimesteps
        frame = records.readDays(config.resultsDir, REGION)
        assert (frame[frame.fold == 2].warmFrom == "").all()
        assert (frame[frame.fold == 3].warmFrom != "").all()

    # The cold train mode is independent by construction and never inherits anything.
    def test_the_cold_mode_never_warm_starts(self, config, spy):
        runner.runRolling(config._replace(trainModes=("cold",), seeds=(0,)))
        assert all(call["warmFrom"] is None for call in spy["train"])
        assert all(call["timesteps"] == config.coldTimesteps for call in spy["train"])


class TestOutput:
    # One row per fold, day, algorithm, train mode, seed and price source.
    def test_one_row_per_fold_day_seed(self, config, spy):
        runner.runRolling(config)
        frame = records.readDays(config.resultsDir, REGION)
        expected = len(folds.enumerateFolds(config.start, config.end, 4)) * 7 * 2
        assert len(frame) == expected
        assert set(frame.trainMode) == {"warm"} and set(frame.seed) == {0, 1}

    # The volatility label is joined onto every row, so the calendar and the high/low
    # split need nothing but the CSV.
    def test_rows_carry_the_volatility_label(self, config, spy):
        runner.runRolling(config)
        frame = records.readDays(config.resultsDir, REGION)
        assert frame.spikeIntensity.notna().all()
        assert frame.highVol.dtype == bool

    # Trajectories are written for the configured seed only, which keeps the parquet to
    # tens of megabytes rather than hundreds.
    def test_trajectories_cover_only_the_configured_seed(self, config, spy):
        runner.runRolling(config._replace(trajectorySeed=1))
        written = pd.read_parquet(config.resultsDir / REGION / "fold00.parquet")
        assert set(written.seed) == {1}
        assert set(written.trainMode) == {"warm", "oracle"}

    # A forecast-priced row names the run the agent read; a realised one names none.
    def test_trajectories_record_the_run_read(self, config):
        fold = SimpleNamespace(index=0, testDays=[pd.Timestamp(START)])
        steps = [{"price": 1.0, "soc": 0.5, "bought": 0.0, "sold": 0.0, "profit": 0.0}] * 288
        perWindow = [{"window": 0, "trajectory": steps}]
        stack = np.zeros((1, 6, 288))
        rows = runner._resultTrajectories(config, fold, "maskableppo", "warm", "forecast",
                                          perWindow, None, stack)
        assert rows[48]["forecastRunTime"] == pd.Timestamp(START) + pd.Timedelta(hours=4)
        realised = runner._resultTrajectories(config, fold, "maskableppo", "warm",
                                              "realised", perWindow, None, None)
        assert realised[0]["forecastRunTime"] is None

    # Two algorithms get disjoint checkpoint directories, so neither overwrites the
    # other's warm chain.
    def test_two_algorithms_do_not_collide(self, config, spy):
        runner.runRolling(config._replace(algos=("maskableppo", "dqn"), seeds=(0,)))
        written = {call["outDir"] / f"{call['algo']}_seed0" for call in spy["train"]}
        assert len({path.name for path in written}) == 2


class TestPublishesTheFinalAgent:
    # The run ends by putting its last fold under trainedPolicies/, which is the only
    # place the dashboard looks. The per-fold checkpoints stay working state.
    def test_the_last_fold_is_published(self, config, spy):
        runner.runRolling(config)
        found = policies.available(REGION, config.policyRoot)
        assert sorted(found) == ["maskableppo-warm-s0", "maskableppo-warm-s1"]

    # policy.json says which fold it came from, because the weights do not.
    def test_it_records_the_fold_it_came_from(self, config, spy):
        runner.runRolling(config)
        facts = policies.describe(
            policies.policyDir(REGION, "maskableppo", "warm", 0, config.policyRoot))
        last = runner._plan(config).selected[-1]
        assert facts["fold"] == last.index and facts["command"] == "rolling"

    # A resumed run that trains nothing still publishes, so a run finished in two
    # passes still leaves the dashboard an agent to read.
    def test_a_resumed_run_still_publishes(self, config, spy):
        runner.runRolling(config)
        assert runner.runRolling(config) == 0
        assert policies.available(REGION, config.policyRoot)


class TestResume:
    # A finished unit is not redone, and nothing is trained for it either, because the
    # checkpoint on disk is the training record.
    def test_a_resumed_run_skips_finished_units(self, config, spy):
        runner.runRolling(config)
        trained, evaluated = len(spy["train"]), len(spy["evaluate"])
        runner.runRolling(config)
        assert len(spy["train"]) == trained
        assert len(spy["evaluate"]) == evaluated

    # A resumed run still reads prices and the oracle, which is cheap, and must not
    # start solving.
    def test_a_resumed_run_rereads_without_resolving(self, config, spy):
        runner.runRolling(config)
        runner.runRolling(config)
        assert len(spy["prices"]) == 2 and len(spy["baseline"]) == 2

    # Appending a differently configured run into one result set is the silent killer.
    def test_a_changed_configuration_is_refused(self, config, spy):
        runner.runRolling(config)
        with pytest.raises(ValueError, match="trainWeeks"):
            runner.runRolling(config._replace(trainWeeks=5))


class TestDryRun:
    # The audit reports the fold table and how much of the oracle cache is missing,
    # before anything trains.
    def test_the_audit_counts_folds_and_cache_misses(self, config, spy):
        report = runner.auditPlan(config)
        assert report["folds"] == 6
        # One oracle record per episode: six weekly test windows, not forty-two days.
        assert report["episodes"] == 6
        assert 0 <= report["missing"] <= report["episodes"]
        assert spy["train"] == [] and spy["evaluate"] == []



class TestAuditReportsTheSelection:
    # A fold selection narrows the reported first and last fold, not just the count.
    def test_first_and_last_follow_the_selection(self, config):
        audit = runner.auditPlan(config._replace(foldFilter=frozenset({1})))
        assert audit["selected"] == 1
        assert audit["first"].index == 1 and audit["last"].index == 1

    # A filter naming folds the span does not have must not inflate the count.
    def test_an_out_of_range_fold_is_not_counted(self, config):
        audit = runner.auditPlan(config._replace(foldFilter=frozenset({0, 99})))
        assert audit["selected"] == 1

    # With no filter the audit still spans every fold.
    def test_no_filter_reports_every_fold(self, config):
        audit = runner.auditPlan(config)
        assert audit["selected"] == audit["folds"] == 6


class TestForecastSource:
    # Every test here states a 288-interval episode. The subject is one trading day's
    # refresh runs, and one episode holds one day only at that horizon.

    # A span with no forecast price source never touches the forecast cache.
    def test_a_realised_only_run_loads_nothing(self, config, monkeypatch):
        monkeypatch.setattr(runner, "loadForecastCache",
                            lambda region: pytest.fail("loaded without a forecast price source"))
        assert runner._forecastSource(config, [START]) is None

    # No prefetched cache falls back to the live per-run path.
    def test_no_cache_falls_back_to_the_live_path(self, config, monkeypatch):
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: None)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        assert runner._forecastSource(cfg, [START]) is None

    # A test day the prefetch never reached names the prefetch command.
    def test_a_missing_day_names_the_prefetch(self, config, monkeypatch):
        frame = refreshFrame([START])
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        with pytest.raises(ValueError, match="cli prefetch"):
            runner._forecastSource(cfg, [START, START + folds.oneDay])

    # A refresh run AEMO never published is held at the latest earlier run, shifted onto
    # its own run time, as a live agent would have seen it.
    def test_a_missing_refresh_run_holds_the_earlier_one(self, config, monkeypatch):
        frame = refreshFrame([START])
        earlier = pd.Timestamp(START) + pd.Timedelta(hours=12)
        frame.loc[earlier] = np.arange(1.0, 289.0)
        lost = pd.Timestamp(START) + pd.Timedelta(hours=16)
        frame = frame.drop(lost)
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        source = runner._forecastSource(cfg, [START])
        np.testing.assert_array_equal(source.loc[lost].iloc[:240], np.arange(49.0, 289.0))
        assert source.loc[lost].iloc[240:].isna().all()

    # The episode's opening run has nothing earlier to hold, so its absence is refused.
    def test_a_missing_opening_run_is_refused(self, config, monkeypatch):
        frame = refreshFrame([START]).drop(pd.Timestamp(START))
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        with pytest.raises(ValueError, match="first 2024-01-01 04:00"):
            runner._forecastSource(cfg, [START])

    # A day present but only partly covered is refused, not padded to zero in silence.
    def test_a_partial_day_is_refused(self, config, monkeypatch):
        frame = refreshFrame([START])
        frame.iloc[0, 11:] = None
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        with pytest.raises(ValueError, match="11 of 288"):
            runner._forecastSource(cfg, [START])

    # A later run need only reach the episode end: the 16:00 run covers 144 steps.
    def test_a_refresh_run_need_only_reach_the_episode_end(self, config, monkeypatch):
        frame = refreshFrame([START])
        frame.loc[pd.Timestamp(START) + pd.Timedelta(hours=12), "145":] = None
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        assert runner._forecastSource(cfg, [START]) is frame

    # ...and one that stops short of it, as when AEMO skipped the PREDISPATCH run, keeps
    # its own steps and takes the rest from the earlier run.
    def test_a_refresh_run_short_of_the_episode_end_holds_the_tail(self, config, monkeypatch):
        frame = refreshFrame([START])
        frame.loc[pd.Timestamp(START) + pd.Timedelta(hours=8)] = 2.0
        short = pd.Timestamp(START) + pd.Timedelta(hours=12)
        frame.loc[short, "144":] = None
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        row = runner._forecastSource(cfg, [START]).loc[short]
        assert (row.iloc[:143] == 1.0).all() and row.iloc[143] == 2.0

    # One run per day is still a valid design, and needs only the 04:00 run.
    def test_a_daily_refresh_needs_only_the_0400_run(self, config, monkeypatch):
        frame = refreshFrame([START]).iloc[:1]
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",),
                                forecastEvery=288)
        assert runner._forecastSource(cfg, [START]) is frame

    # A fully covered day passes the cache straight through.
    def test_a_complete_day_is_accepted(self, config, monkeypatch):
        frame = refreshFrame([START])
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(horizon=288, priceSources=("forecast",))
        assert runner._forecastSource(cfg, [START]) is frame


class TestManifest:
    # The test-day forecast pull is recorded, so the dashboard can check what it shows.
    def test_a_forecast_run_records_the_refresh(self, config):
        cfg = config._replace(priceSources=("realised", "forecast"))
        assert runner.manifestOf(cfg)["forecastEvery"] == 48

    # A realised-only set records it too: every result set states its pull frequency.
    def test_a_realised_only_run_records_it_too(self, config):
        assert runner.manifestOf(config)["forecastEvery"] == 48


class TestTrainingOnForecasts:
    def _run(self, config, spy, monkeypatch, trainSource):
        days = [START + n * folds.oneDay for n in range(10 * 7)]
        monkeypatch.setattr(runner, "loadForecastCache",
                            lambda region: refreshFrame(days))
        runner.runRolling(config._replace(trainSource=trainSource, foldFilter=frozenset({0})))
        return spy["train"]

    # Training on realised prices passes no forecast, as before.
    def test_realised_training_passes_no_forecast(self, config, spy, monkeypatch):
        assert all(call["forecasts"] is None
                   for call in self._run(config, spy, monkeypatch, "realised"))

    # One refresh stack per training episode, on the episode's own step grid.
    def test_forecast_training_reads_one_stack_per_episode(self, config, spy,
                                                           monkeypatch):
        calls = self._run(config, spy, monkeypatch, "forecast")
        assert calls
        for call in calls:
            # One stack per training EPISODE, each on the episode's own step grid: four
            # weekly episodes of 2016 intervals, a refresh every 48 of them.
            assert call["forecasts"].shape == (4, 2016 // 48, 2016)
            assert len(call["prices"]) == 4 * 7 * 288

    # A training day the prefetch never reached stops the run before any fit.
    def test_a_training_day_missing_from_the_cache_is_refused(self, config, spy,
                                                              monkeypatch):
        late = [START + n * folds.oneDay for n in range(7, 10 * 7)]
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: refreshFrame(late))
        with pytest.raises(ValueError, match="cli prefetch"):
            runner.runRolling(config._replace(trainSource="forecast",
                                              foldFilter=frozenset({0})))
        assert spy["train"] == []

    # Only a forecast-trained set records its source, so older result sets still resume.
    def test_the_manifest_records_a_forecast_training_source(self, config):
        assert "trainSource" not in runner.manifestOf(config)
        cfg = config._replace(trainSource="forecast")
        assert runner.manifestOf(cfg)["trainSource"] == "forecast"
        assert "forecastEvery" in runner.manifestOf(cfg)


class TestActionsFile:
    # Every seed's decisions, both price sources, reach the CSV as each fold finishes,
    # not just the trajectory seed's.
    @pytest.fixture
    def forecastConfig(self, config, monkeypatch, tmp_path):
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: None)
        monkeypatch.setattr(runner, "refreshMatrix", lambda days, region, horizon, every, **kw:
                            np.full((len(days), horizon // every, horizon), 40.0))
        return config._replace(priceSources=("realised", "forecast"),
                               foldFilter=frozenset({0, 1}),
                               actionsFile=tmp_path / "actions" / "run_NSW1.csv")

    def test_every_seed_of_every_fold_is_written(self, forecastConfig, spy):
        runner.runRolling(forecastConfig)
        got = pd.read_csv(forecastConfig.actionsFile)
        assert set(got.seed) == {0, 1}
        assert set(got.fold) == {0, 1}
        assert len(got) == 2 * 2 * 2 * 7 * 288
        assert (got.realisedPrice == 50.0).all()
        forecast = got.priceSource == "forecast"
        assert forecast.sum() == len(got) // 2 and (got.forecastPrice[forecast] == 40.0).all()
        assert got.forecastPrice[~forecast].isna().all()

    def test_a_resumed_run_writes_no_row_twice(self, forecastConfig, spy):
        runner.runRolling(forecastConfig._replace(foldFilter=frozenset({0})))
        runner.runRolling(forecastConfig)
        got = pd.read_csv(forecastConfig.actionsFile)
        assert len(got) == 2 * 2 * 2 * 7 * 288
        assert not got.duplicated(["fold", "seed", "priceSource", "intervalTime"]).any()

    # A fold killed after its CSV rows but before its fold CSV is scored again, and its
    # rows are not appended a second time.
    def test_a_fold_killed_before_its_records_is_not_appended_again(
            self, forecastConfig, spy, monkeypatch):
        cfg = forecastConfig._replace(foldFilter=frozenset({0}))
        real = records.writeFold
        monkeypatch.setattr(records, "writeFold", lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("killed")))
        with pytest.raises(RuntimeError):
            runner.runRolling(cfg)
        monkeypatch.setattr(records, "writeFold", real)
        runner.runRolling(cfg)
        got = pd.read_csv(cfg.actionsFile)
        assert len(got) == 2 * 2 * 7 * 288


class TestWeeklyEpisodes:
    """An episode may hold several trading days. The day stays the reporting grain, so a
    weekly fold still writes seven rows per seed, but the oracle bound is the episode's
    and sits on its first day alone."""

    @pytest.fixture
    def weekly(self, config):
        return config._replace(horizon=2016, cycleBudget=7.0, seeds=(0,))

    def rows(self, cfg):
        return records.readDays(cfg.resultsDir, REGION)

    def test_the_oracle_is_solved_once_per_episode(self, weekly, spy):
        runner.runRolling(weekly)
        assert spy["baseline"][0]["horizon"] == 2016
        assert spy["baseline"][0]["cycles"] == 7.0

    def test_a_weekly_fold_still_writes_seven_day_rows(self, weekly, spy):
        runner.runRolling(weekly)
        frame = self.rows(weekly)
        agent = frame[frame.algo != "oracle"]
        perFold = agent.groupby("fold").size()
        assert set(perFold) == {7}
        assert agent.testDay.nunique() == len(agent)

    def test_every_day_names_the_episode_it_sat_in(self, weekly, spy):
        runner.runRolling(weekly)
        frame = self.rows(weekly)
        for episode, group in frame.groupby("episodeStart"):
            assert len(group) == 7
            assert group.testDay.min() == episode
            assert group.testDay.max() == episode + pd.Timedelta(days=6)

    def test_the_bound_sits_on_the_first_day_of_the_episode_only(self, weekly, spy):
        """A day-long oracle started empty is no bound on a day the agent entered
        holding energy, so there is no per-day bound to write."""
        runner.runRolling(weekly)
        frame = self.rows(weekly)
        first = frame.testDay == frame.episodeStart
        assert frame.loc[first, "oracleDualBound"].notna().all()
        assert frame.loc[~first, "oracleDualBound"].isna().all()
        assert frame.loc[~first, "oracleGap"].isna().all()
        assert frame.loc[~first, "oracleCycles"].isna().all()

    def test_a_whole_episode_scope_counts_the_bound_once(self, weekly, spy):
        """Which is the point of the NaN: pandas skips it, so summing a week's rows
        gives the one bound rather than seven copies of it."""
        runner.runRolling(weekly)
        frame = self.rows(weekly)
        week = frame[(frame.fold == 0) & (frame.algo != "oracle")]
        assert week.oracleDualBound.sum() == pytest.approx(10.0)
        assert week.oracleDualBound.count() == 1

    def test_the_day_rows_sum_to_the_episode(self, weekly, spy):
        runner.runRolling(weekly)
        frame = self.rows(weekly)
        week = frame[(frame.fold == 0) & (frame.algo != "oracle")]
        # The fake agent reports 1.0 of profit and 0.5 cycles a day over seven days.
        assert week.agentProfit.sum() == pytest.approx(7.0, abs=1e-9)
        assert week.agentCycles.sum() == pytest.approx(3.5, abs=1e-9)
        assert set(week.cycleBudget) == {7.0}

    def test_the_trajectory_is_still_written_in_trading_days(self, weekly, spy):
        """The dashboard draws one day at a time, so the export keeps 288-interval days
        whatever the episode length."""
        runner.runRolling(weekly)
        written = pd.read_parquet(weekly.resultsDir / REGION / "fold00.parquet")
        agent = written[written.algo != "oracle"]
        assert agent.interval.max() == 287
        assert agent.testDay.nunique() == 7
        assert len(agent) == 2016

    def test_the_manifest_carries_the_budget(self, weekly, spy):
        runner.runRolling(weekly)
        manifest = records.manifestPath(weekly.resultsDir, REGION)
        assert json.loads(manifest.read_text())["cycleBudget"] == 7.0

    def test_a_budget_free_result_set_refuses_a_budgeted_run(self, config, spy):
        free = config._replace(cycleBudget=None)
        runner.runRolling(free)
        with pytest.raises(ValueError, match="cycleBudget"):
            runner.runRolling(config)

    def test_an_unbudgeted_run_still_resumes_a_result_set_from_before_the_budget(
            self, config, spy):
        """`cycleBudget: None` must read as the absent key, or every published run would
        refuse to be added to."""
        free = config._replace(cycleBudget=None)
        runner.runRolling(free)
        stored = json.loads(records.manifestPath(free.resultsDir, REGION).read_text())
        assert "cycleBudget" not in stored
        assert runner.runRolling(free) == 0

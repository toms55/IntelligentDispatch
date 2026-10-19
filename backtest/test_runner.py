"""Tests for backtest/runner.py. Run with: uv run pytest backtest/ -q

Four seams are replaced: price_series, buildBaseline, trainOne and evaluateSeeds. So
nothing here touches the network, the MILP or torch, and what is pinned is the wiring:
which window each fold trains on, how the warm chain finds its parent, and that a
resumed run does not redo work.

The checkpoint on disk is the training record. There is no separate ledger: a unit is
done when its rows are in the fold CSV, and training is done when agent.zip exists.
"""
from datetime import datetime, timedelta

import pandas as pd
import pytest

from backtest import folds, records, runner
from utils import nem_interval

START = datetime(2024, 1, 1, 4)
REGION = "NSW1"


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
        trainWeeks=4, seeds=(0, 1), arms=("warm",),
        warmTimesteps=10, coldTimesteps=100,
        # buildBaseline is faked, so nothing solves. Without this the suite reads the
        # real data_cache/lp_baseline, which is gitignored and absent on a clean clone.
        allowSolve=True,
        outDir=tmp_path / "runs", resultsDir=tmp_path / "results",
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
        days = len(window) // 288
        return [{"start": window.index[n * 288].isoformat(),
                 "profitPerInterval": [0.01] * 288, "dualBound": 10.0, "profit": 9.0,
                 "soc": [0.5] * 288, "bought": [0.0] * 288, "sold": [0.0] * 288}
                for n in range(days)]

    def fakeTrain(algo, trainPrices, model, **kwargs):
        calls["train"].append({"algo": algo, "prices": trainPrices, **kwargs})
        directory = kwargs["outDir"] / f"{algo}_seed{kwargs['seed']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "agent.zip").write_bytes(b"")
        return None, directory / "agent"

    def fakeEvaluate(checkpoints, testPrices, model, **kwargs):
        calls["evaluate"].append({"checkpoints": dict(checkpoints),
                                  "prices": testPrices, **kwargs})
        windows = len(testPrices) // 288
        return {seed: [{"window": w, "agentProfit": 1.0, "agentRevenue": 2.0,
                        "agentDegradation": 1.0, "lpProfit": 10.0, "regret": 9.0,
                        "trajectory": [{"price": 50.0, "soc": 0.5, "bought": 0.0,
                                        "sold": 0.0, "profit": 0.0}] * 288}
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

    # The cold arm is independent by construction and never inherits anything.
    def test_the_cold_arm_never_warm_starts(self, config, spy):
        runner.runRolling(config._replace(arms=("cold",), seeds=(0,)))
        assert all(call["warmFrom"] is None for call in spy["train"])
        assert all(call["timesteps"] == config.coldTimesteps for call in spy["train"])


class TestOutput:
    # One row per fold, day, algorithm, arm, seed and price source.
    def test_one_row_per_fold_day_seed(self, config, spy):
        runner.runRolling(config)
        frame = records.readDays(config.resultsDir, REGION)
        expected = len(folds.enumerateFolds(config.start, config.end, 4)) * 7 * 2
        assert len(frame) == expected
        assert set(frame.arm) == {"warm"} and set(frame.seed) == {0, 1}

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
        assert set(written.arm) == {"warm", "oracle"}

    # Two algorithms get disjoint checkpoint directories, so neither overwrites the
    # other's warm chain.
    def test_two_algorithms_do_not_collide(self, config, spy):
        runner.runRolling(config._replace(algos=("maskableppo", "dqn"), seeds=(0,)))
        written = {call["outDir"] / f"{call['algo']}_seed0" for call in spy["train"]}
        assert len({path.name for path in written}) == 2


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
        assert report["days"] == 70          # fold 0's train start to fold 5's test end
        assert 0 <= report["missing"] <= report["days"]
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
    # A span with no forecast arm never touches the forecast cache.
    def test_a_realised_only_run_loads_nothing(self, config, monkeypatch):
        monkeypatch.setattr(runner, "loadForecastCache",
                            lambda region: pytest.fail("loaded without a forecast arm"))
        assert runner._forecastSource(config, [START]) is None

    # No prefetched cache falls back to the live per-run path.
    def test_no_cache_falls_back_to_the_live_path(self, config, monkeypatch):
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: None)
        cfg = config._replace(priceSources=("forecast",))
        assert runner._forecastSource(cfg, [START]) is None

    # A test day the prefetch never reached names the prefetch command.
    def test_a_missing_day_names_the_prefetch(self, config, monkeypatch):
        frame = pd.DataFrame([[1.0] * 288], index=[pd.Timestamp(START)],
                             columns=[str(n) for n in range(1, 289)])
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(priceSources=("forecast",))
        with pytest.raises(ValueError, match="cli prefetch"):
            runner._forecastSource(cfg, [START, START + folds.oneDay])

    # A day present but only partly covered is refused, not padded to zero in silence.
    def test_a_partial_day_is_refused(self, config, monkeypatch):
        row = [1.0] * 11 + [None] * 277
        frame = pd.DataFrame([row], index=[pd.Timestamp(START)],
                             columns=[str(n) for n in range(1, 289)])
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(priceSources=("forecast",))
        with pytest.raises(ValueError, match="11 of 288"):
            runner._forecastSource(cfg, [START])

    # A fully covered day passes the cache straight through.
    def test_a_complete_day_is_accepted(self, config, monkeypatch):
        frame = pd.DataFrame([[1.0] * 288], index=[pd.Timestamp(START)],
                             columns=[str(n) for n in range(1, 289)])
        monkeypatch.setattr(runner, "loadForecastCache", lambda region: frame)
        cfg = config._replace(priceSources=("forecast",))
        assert runner._forecastSource(cfg, [START]) is frame

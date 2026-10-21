"""Tests for cli.py. No network: price_series is monkeypatched."""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

import cli
import config
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
            0.060, 0.100, 150000.0)

    def test_battery_is_configurable(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11",
                  "--mw", "1.5", "--mwh", "3.0", "--replacement-cost", "200000"])
        model = calls["solve"][1]
        assert (model.chargePower, model.MWh, model.replacementCost) == (1.5, 3.0, 200000.0)

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

    def test_a_split_name_labels_the_default_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cli.main(["lp", "--split", "probe=2024-01-10..2024-01-11", "--csv"])
        assert (tmp_path / "lp_SA1_probe.csv").exists()

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


def _fakeCheckpoint(root, seed, cap=17500.0, algo="maskableppo"):
    """A checkpoint directory as rl/train.py leaves it: the agent and its price scale."""
    directory = root / f"{algo}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent.zip").write_bytes(b"")
    (directory / "scaler.json").write_text(
        json.dumps({"cap": cap, "floor": -1000.0, "knee": config.price_knee})
    )
    return directory


class TestWindow:
    """A window is two dates, or one --split NAME=START..END on the command line."""

    def test_a_split_supplies_the_dates(self, calls):
        cli.main(["lp", "--split", "val=2024-11-01T04:00..2025-01-01T04:00"])
        start, end, _ = calls["prices"]
        assert start == datetime(2024, 11, 1, 4)
        assert end == datetime(2025, 1, 1, 4)

    def test_a_plain_date_needs_no_time(self, calls):
        cli.main(["lp", "--split", "probe=2024-01-10..2024-01-11"])
        start, end, _ = calls["prices"]
        assert (start, end) == (datetime(2024, 1, 10), datetime(2024, 1, 11))

    def test_explicit_dates_beat_a_split(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11",
                  "--split", "test=2025-01-01T04:00..2026-01-01T04:00"])
        start, end, _ = calls["prices"]
        assert start == datetime(2024, 1, 10)
        assert end == datetime(2024, 1, 11)

    def test_neither_dates_nor_split_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp"])

    def test_a_start_without_an_end_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp", "2024-01-10"])

    @pytest.mark.parametrize("value", [
        "val",                                  # no name/window separator
        "=2024-01-10..2024-01-11",              # no name
        "val=2024-01-10",                       # no date separator
        "val=2024-01-10:2024-01-11",            # ':' is not the separator
        "val=notadate..2024-01-11",             # not ISO
        "val=2024-01-10..notadate",
    ])
    def test_a_malformed_split_is_refused(self, value):
        with pytest.raises(SystemExit):
            cli.main(["lp", "--split", value])

    def test_more_than_one_split_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp", "--split", "a=2024-01-10..2024-01-11",
                      "--split", "b=2024-02-10..2024-02-11"])

    def test_a_backwards_window_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp", "2024-01-11", "2024-01-10"])


class TestDiscretePowerFlag:
    def test_is_off_by_default(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        assert calls["solve"][2]["discretePower"] is False

    def test_the_flag_turns_it_on(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--discrete-power"])
        assert calls["solve"][2]["discretePower"] is True

    def test_the_gap_and_time_limit_reach_the_solver(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--gap", "0.02",
                  "--time-limit", "30"])
        assert calls["solve"][2]["gap"] == 0.02
        assert calls["solve"][2]["timeLimit"] == 30.0

    def test_lp_leaves_the_gap_at_the_solver_default(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        assert calls["solve"][2]["gap"] is None

    def test_the_dual_bound_is_reported_for_a_discrete_solve(self, calls, capsys):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--discrete-power"])
        assert "dual bound" in capsys.readouterr().out

    def test_no_dual_bound_line_for_a_continuous_solve(self, calls, capsys):
        cli.main(["lp", "2024-01-10", "2024-01-11"])
        assert "dual bound" not in capsys.readouterr().out


class TestRlSubcommands:
    """Argument plumbing only. Training itself is covered in rl/test_algos.py."""

    @pytest.fixture
    def rlCalls(self, monkeypatch, series):
        from rl import baseline, evaluate, train
        seen = {}

        def fakeBuild(prices, **kwargs):
            seen["baseline"] = kwargs
            return [{"profitPerInterval": [0.1] * len(prices), "profit": 1.0,
                     "dualBound": 1.01, "mipGap": 0.001,
                     "start": "2024-01-10T00:00:00"}]

        def fakeTrain(algoName, prices, model, **kwargs):
            seen.setdefault("train", []).append((algoName, kwargs))
            return None, Path("runs/fake/agent")

        def fakeEvaluateSeeds(checkpoints, prices, model, **kwargs):
            seen["eval"] = (sorted(checkpoints), kwargs)
            return {seed: [{"agentProfit": 1.0, "lpProfit": 2.0}] for seed in checkpoints}

        monkeypatch.setattr(baseline, "buildBaseline", fakeBuild)
        monkeypatch.setattr(train, "trainOne", fakeTrain)
        monkeypatch.setattr(evaluate, "evaluateSeeds", fakeEvaluateSeeds)
        return seen

    def test_baseline_reports_a_summary(self, rlCalls, capsys):
        cli.main(["baseline", "2024-01-10", "2024-01-11", "--horizon", "4"])
        out = capsys.readouterr().out
        assert "oracle profit" in out and "dual bound" in out

    def test_baseline_asks_for_the_discrete_oracle(self, rlCalls):
        cli.main(["baseline", "2024-01-10", "2024-01-11", "--horizon", "4"])
        assert rlCalls["baseline"]["discretePower"] is True

    def test_baseline_continuous_flag_relaxes_the_restriction(self, rlCalls):
        cli.main(["baseline", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--continuous"])
        assert rlCalls["baseline"]["discretePower"] is False

    def test_baseline_defaults_to_the_configured_gap(self, rlCalls):
        cli.main(["baseline", "2024-01-10", "2024-01-11", "--horizon", "4"])
        assert rlCalls["baseline"]["gap"] == config.oracle_gap

    def test_baseline_passes_the_worker_count(self, rlCalls):
        cli.main(["baseline", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--workers", "6"])
        assert rlCalls["baseline"]["workers"] == 6

    def test_train_runs_once_per_seed(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--seeds", "0", "1", "2", "--timesteps", "10"])
        assert len(rlCalls["train"]) == 3
        assert [kwargs["seed"] for _, kwargs in rlCalls["train"]] == [0, 1, 2]

    def test_train_defaults_to_the_configured_seeds(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10"])
        assert len(rlCalls["train"]) == len(config.seeds)

    def test_train_defaults_to_maskable_ppo(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10"])
        assert rlCalls["train"][0][0] == "maskableppo"

    def test_train_algo_is_selectable(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10", "--algo", "dqn"])
        assert rlCalls["train"][0][0] == "dqn"

    def test_train_refuses_an_unknown_algo(self, rlCalls):
        with pytest.raises(SystemExit):
            cli.main(["train", "2024-01-10", "2024-01-11", "--algo", "ppo2"])

    def test_train_solves_no_oracle(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10"])
        assert "baselines" not in rlCalls["train"][0][1]
        assert "baseline" not in rlCalls

    def test_train_pins_the_price_scale_to_the_window_end(self, rlCalls, capsys):
        """2024-01-10 to 2024-01-11 sits in 2023-24, so the cap is 16600."""
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10", "--seeds", "0"])
        assert rlCalls["train"][0][1]["scaler"].cap == 16600.0
        assert "cap 16600" in capsys.readouterr().out

    def test_cap_date_overrides_the_window_end(self, rlCalls):
        """Pin to the end of the window you will score on, so test-year prices above the
        training year's cap still resolve."""
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10", "--seeds", "0",
                  "--cap-date", "2026-01-01T04:00"])
        assert rlCalls["train"][0][1]["scaler"].cap == 20300.0

    def test_every_seed_shares_one_price_scale(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10", "--seeds", "0", "1", "2"])
        scalers = [kwargs["scaler"] for _, kwargs in rlCalls["train"]]
        assert all(one == scalers[0] for one in scalers)

    def test_the_horizon_reaches_the_env(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10"])
        assert rlCalls["train"][0][1]["horizon"] == 4

    def test_eval_refuses_a_missing_checkpoint(self, rlCalls, tmp_path):
        with pytest.raises(SystemExit, match="no checkpoint"):
            cli.main(["eval", "2024-01-10", "2024-01-11", "--horizon", "4",
                      "--out", str(tmp_path)])

    def test_eval_reports_regret_and_spread(self, rlCalls, tmp_path, capsys):
        for seed in (0, 1):
            _fakeCheckpoint(tmp_path, seed)
        cli.main(["eval", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--out", str(tmp_path), "--seeds", "0", "1"])
        out = capsys.readouterr().out
        assert "regret" in out and "captured" in out and "oracle profit" in out
        assert "cap 17500" in out  # the cap the checkpoint trained under, not the window's

    def test_eval_refuses_a_checkpoint_with_no_price_scale(self, rlCalls, tmp_path):
        """A checkpoint from before the dated price cap. Guessing the transform would
        silently change what every observation means."""
        directory = tmp_path / "maskableppo_seed0"
        directory.mkdir(parents=True)
        (directory / "agent.zip").write_bytes(b"")
        with pytest.raises(FileNotFoundError, match="scaler.json"):
            cli.main(["eval", "2024-01-10", "2024-01-11", "--horizon", "4",
                      "--out", str(tmp_path), "--seeds", "0"])


class TestRolling:
    """The rolling subcommand's argument translation. runRolling itself is faked."""

    @pytest.fixture
    def spy(self, monkeypatch):
        from backtest import runner
        seen = {}
        monkeypatch.setattr(runner, "runRolling",
                            lambda cfg, onProgress=None: seen.update(cfg=cfg) or 0)
        monkeypatch.setattr(runner, "auditPlan", lambda cfg: {
            "folds": 1, "selected": 1, "days": 7, "missing": 0, "leftoverDays": 0,
            "first": runner.folds.Fold(0, *[datetime(2025, 1, 1, 4)] * 4),
            "last": runner.folds.Fold(0, *[datetime(2025, 1, 1, 4)] * 4)})
        return seen

    def rolling(self, *extra):
        return cli.main([
            "rolling", "--split", "r=2024-01-01T04:00..2026-01-01T04:00",
            "--region", "NSW1", "--train-weeks", "52",
            "--cap-date", "2026-01-01T04:00", *extra,
        ])

    # Both price sources in one run becomes the pair the runner expects.
    def test_price_source_both_becomes_a_pair(self, spy):
        self.rolling("--price-source", "both", "--dry-run")
        self.rolling("--price-source", "both")
        assert spy["cfg"].priceSources == ("realised", "forecast")

    # A fold range and a fold list both resolve to a set of indices.
    @pytest.mark.parametrize("spec,expected", [
        ("0-3", {0, 1, 2, 3}),
        ("0,13", {0, 13}),
        ("0-2,10", {0, 1, 2, 10}),
        ("7", {7}),
    ])
    def test_folds_spec_resolves_to_indices(self, spy, spec, expected):
        self.rolling("--folds", spec)
        assert spy["cfg"].foldFilter == frozenset(expected)

    # A malformed fold spec exits rather than silently selecting nothing.
    def test_a_malformed_folds_spec_exits(self, spy):
        with pytest.raises(SystemExit):
            self.rolling("--folds", "one-two")

    # Several algorithms in one run, which train and eval do not accept.
    def test_several_algorithms_become_a_tuple(self, spy):
        self.rolling("--algo", "maskableppo", "dqn")
        assert spy["cfg"].algos == ("maskableppo", "dqn")

    # Without a pinned cap, folds either side of 1 July read a different price scale.
    def test_the_cap_date_is_required(self, spy):
        with pytest.raises(SystemExit):
            cli.main(["rolling", "--split", "r=2024-01-01T04:00..2026-01-01T04:00",
                      "--train-weeks", "52"])

    # The training length has no sensible default, so it must be given.
    def test_the_train_length_is_required(self, spy):
        with pytest.raises(SystemExit):
            cli.main(["rolling", "--split", "r=2024-01-01T04:00..2026-01-01T04:00",
                      "--cap-date", "2026-01-01T04:00"])

    # One window per run still holds, so _window is untouched by this subcommand.
    def test_two_splits_still_exit(self, spy):
        with pytest.raises(SystemExit):
            cli.main(["rolling", "--split", "a=2024-01-01T04:00..2025-01-01T04:00",
                      "--split", "b=2025-01-01T04:00..2026-01-01T04:00",
                      "--train-weeks", "52", "--cap-date", "2026-01-01T04:00"])

    # The cache-matching oracle settings are the default here, and --calendar opts back in.
    def test_calendar_is_off_by_default_and_can_be_turned_on(self, spy):
        self.rolling()
        assert spy["cfg"].calendar is False
        self.rolling("--calendar")
        assert spy["cfg"].calendar is True

    # A negative trajectory seed means persist nothing.
    def test_a_negative_trajectory_seed_disables_trajectories(self, spy):
        self.rolling("--trajectory-seed", "-1")
        assert spy["cfg"].trajectorySeed is None

    # --dry-run reports and stops.
    def test_dry_run_does_not_run(self, spy):
        assert self.rolling("--dry-run") == 0
        assert "cfg" not in spy


class TestExistingSubcommandsSurvive:
    # train and eval keep their single --algo after rolling made it a list elsewhere.
    @pytest.mark.parametrize("module", ["train", "eval"])
    def test_a_single_algo_is_still_a_string(self, module, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli, f"_run{module.capitalize()}",
                            lambda args: captured.update(algo=args.algo) or 0)
        parser_args = ["--split", "t=2024-01-01T04:00..2024-01-02T04:00",
                       "--algo", "dqn"]
        cli.main([module, *parser_args])
        assert captured["algo"] == "dqn"

    # train and eval keep --forecast, which rolling replaces with --price-source.
    def test_eval_still_takes_forecast(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(cli, "_runEval",
                            lambda args: captured.update(f=args.forecast) or 0)
        cli.main(["eval", "--split", "t=2024-01-01T04:00..2024-01-02T04:00",
                  "--forecast"])
        assert captured["f"] is True

    # rolling has no --forecast to collide with --price-source.
    def test_rolling_rejects_forecast(self):
        with pytest.raises(SystemExit):
            cli.main(["rolling", "--split", "r=2024-01-01T04:00..2026-01-01T04:00",
                      "--train-weeks", "52", "--cap-date", "2026-01-01T04:00",
                      "--forecast"])


class TestReportRolling:
    """The rolling report keeps each price source apart and counts loose bounds."""

    def frame(self):
        days = pd.date_range("2024-01-01 04:00", periods=7, freq="D")
        return pd.DataFrame([
            {"fold": 0, "algo": "maskableppo", "trainMode": "warm", "seed": 0,
             "priceSource": source, "testDay": day, "agentProfit": agent,
             "oracleDualBound": 10.0, "oracleGap": 0.5 if step == 0 else 0.01,
             "highVol": False}
            for source, agent in (("realised", 8.0), ("forecast", 2.0))
            for step, day in enumerate(days)])

    def test_sources_are_reported_apart_with_their_difference(self, capsys):
        from backtest import metrics
        cli._reportRolling(self.frame(), metrics, 0.02)
        out = capsys.readouterr().out
        assert "80.00%" in out and "20.00%" in out and "50.00%" not in out
        assert "1 with a loose bound" in out
        assert "realised minus forecast 42.0000 over 1 folds" in out

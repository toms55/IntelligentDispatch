"""Tests for cli.py. No network: price_series is monkeypatched."""
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


class TestWindow:
    """A window is two dates, or a named split out of config.py."""

    def test_a_split_supplies_the_dates(self, calls):
        cli.main(["lp", "--split", "val"])
        start, end, _ = calls["prices"]
        assert (start, end) == config.splits["val"]

    def test_full_train_is_the_whole_training_year(self, calls):
        cli.main(["lp", "--split", "full_train"])
        start, end, _ = calls["prices"]
        assert (start, end) == config.full_train

    def test_explicit_dates_beat_a_split(self, calls):
        cli.main(["lp", "2024-01-10", "2024-01-11", "--split", "test"])
        start, end, _ = calls["prices"]
        assert start == datetime(2024, 1, 10)
        assert end == datetime(2024, 1, 11)

    def test_neither_dates_nor_split_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp"])

    def test_a_start_without_an_end_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp", "2024-01-10"])

    def test_an_unknown_split_is_refused(self):
        with pytest.raises(SystemExit):
            cli.main(["lp", "--split", "holdout"])


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

    def test_train_subtracts_the_oracle_baseline_by_default(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10"])
        assert rlCalls["train"][0][1]["baselines"] is not None

    def test_no_baseline_leaves_the_reward_as_raw_profit(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10", "--no-baseline"])
        assert rlCalls["train"][0][1]["baselines"] is None
        assert "baseline" not in rlCalls  # no oracle solve at all

    def test_the_horizon_reaches_both_the_env_and_the_oracle(self, rlCalls):
        cli.main(["train", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--timesteps", "10"])
        assert rlCalls["train"][0][1]["horizon"] == 4
        assert rlCalls["baseline"]["horizon"] == 4

    def test_eval_refuses_a_missing_checkpoint(self, rlCalls, tmp_path):
        with pytest.raises(SystemExit, match="no checkpoint"):
            cli.main(["eval", "2024-01-10", "2024-01-11", "--horizon", "4",
                      "--out", str(tmp_path)])

    def test_eval_reports_regret_and_spread(self, rlCalls, tmp_path, capsys):
        for seed in (0, 1):
            directory = tmp_path / f"maskableppo_seed{seed}"
            directory.mkdir(parents=True)
            (directory / "agent.zip").write_bytes(b"")
        cli.main(["eval", "2024-01-10", "2024-01-11", "--horizon", "4",
                  "--out", str(tmp_path), "--seeds", "0", "1"])
        out = capsys.readouterr().out
        assert "regret" in out and "captured" in out and "oracle profit" in out

"""Tests for visualise/dashboard/live.py. No network: every path is built here."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl import policies
from rl.features import scalerFor
from rl.heuristic import ThresholdPolicy
from utils import nem_interval
from visualise.dashboard import experiments, figures, live

REGION = "NSW1"


def model(algo="maskableppo", trainMode="warm", seed=0, run="rolling", **settings):
    """One `experiments.Runnable`, as the catalogue hands it to the screen.

    The screen takes the episode from the experiment's manifest, so a test that cares
    about the budget or the closing rule states it here and nowhere else.
    """
    manifest = {"horizon": 288, "cycleBudget": None, "endEmpty": False,
                "terminalPrice": 0.0, "initialSoc": 0.0, **settings}
    experiment = experiments.Experiment(run, REGION, "a test run", manifest,
                                        Path("results/backtestResults") / run)
    return experiments.Runnable(
        experiments.Entry(run, REGION, algo, trainMode, seed), experiment,
        Path("trainedPolicies") / REGION / f"{algo}-{trainMode}-s{seed}")


MODEL = model()


def path(prices, start="2025-01-01 10:05"):
    stamps = pd.date_range(start, periods=len(prices), freq=nem_interval)
    return pd.Series(prices, index=stamps, name="RRP")


@pytest.fixture
def sawtooth():
    """Twelve cheap intervals then twelve dear ones, which any policy should trade.

    Neither block is flat. The rule idles on a flat forward window, by design, because
    a window with no spread holds no arbitrage.
    """
    return path(list(np.linspace(10.0, 50.0, 12)) + list(np.linspace(300.0, 200.0, 12)))


@pytest.fixture
def standIn(monkeypatch):
    """A policy in place of a published agent.

    `suggest` loads the weights off disk, and a test machine holds none. What the
    screen owns is the rollout and the frame it builds, so the rule from
    `rl/heuristic.py` stands in for an agent. The screen itself offers published
    agents only.
    """
    scaler = scalerFor(pd.DatetimeIndex([pd.Timestamp("2025-01-01")]))
    monkeypatch.setattr(live, "_policy",
                        lambda *args, **kwargs: (ThresholdPolicy(), scaler, True))


class TestSuggest:
    # The frame follows the policy: it buys in the cheap half and sells in the dear one.
    def test_it_follows_the_policy(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        assert len(frame) == len(sawtooth)
        assert set(frame.action) <= {"charge", "discharge", "idle"}
        assert "charge" in set(frame.action[:12])
        assert "discharge" in set(frame.action[12:])

    # The state of charge stays inside the battery at every interval.
    def test_the_state_of_charge_stays_in_range(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        assert frame.soc.between(0.0, 1.0).all()

    # Degradation is a cost, so a plan never keeps more than its trades take.
    def test_profit_never_exceeds_revenue(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        assert (frame.profit <= frame.revenue + 1e-9).all()
        assert frame.revenue.iloc[-1] > 0

    # Revenue runs forward, so an interval that trades nothing leaves the total alone.
    def test_an_idle_interval_does_not_move_the_revenue(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        moved = frame.revenue.diff().fillna(0.0)
        assert (moved[frame.action == "idle"].abs() < 1e-12).all()

    # A forecast too short to be an episode draws nothing rather than raising.
    @pytest.mark.parametrize("length", [0, 1])
    def test_a_short_path_gives_an_empty_frame(self, length):
        frame = live.suggest(path([10.0] * length), MODEL, REGION)
        assert frame.empty and list(frame.columns) == list(live.columns)

    # A machine with no agent of a published experiment has no model to run, and says
    # so over the price path rather than drawing nothing.
    def test_no_model_names_the_directory(self, sawtooth, tmp_path):
        with pytest.raises(FileNotFoundError, match="No published agent"):
            live.suggest(sawtooth, None, REGION, root=tmp_path)


class TestCheckpoints:
    # The default root is the one the convention names, not a run directory.
    def test_the_default_root_is_trainedpolicies(self):
        assert live.defaultCheckpoints == policies.defaultRoot


class TestBlocks:
    def frame(self, actions):
        return pd.DataFrame({"settlementdate": path([0.0] * len(actions)).index,
                             "action": actions})

    # One block per unbroken run, and idle is not a block.
    def test_it_groups_runs_and_drops_idle(self):
        got = live.blocks(self.frame(["charge", "charge", "idle", "discharge"]))
        assert [block[0] for block in got] == ["charge", "discharge"]

    # A block opens one interval before its first row, because AEMO stamps the end.
    def test_a_block_covers_its_own_intervals(self):
        (_, start, end), = live.blocks(self.frame(["idle", "charge", "charge"]))
        assert end - start == 2 * nem_interval

    def test_an_empty_frame_has_no_blocks(self):
        assert live.blocks(pd.DataFrame(columns=list(live.columns))) == []


class TestForecastFigure:
    # The key names both blocks, so the colours never have to be guessed.
    def test_the_key_names_both_blocks(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        figure = figures.forecastFigure(frame, live.blocks(frame), region=REGION)
        names = [trace.name for trace in figure.data]
        assert names == ["AEMO forecast", "Discharge: sell", "Charge: buy"]

    # Every block spans the full plot height, so price never sets its size.
    def test_every_block_spans_the_plot_height(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        figure = figures.forecastFigure(frame, live.blocks(frame))
        rects = [shape for shape in figure.layout.shapes if shape.type == "rect"]
        assert rects
        for shape in rects:
            assert (shape.yref, shape.y0, shape.y1) == ("y domain", 0, 1)

    # AEMO publishes one price per half hour past the P5MIN part, so the line carries
    # one point per half hour there, not six identical ones.
    def test_the_predispatch_part_is_drawn_at_its_own_half_hour(self):
        # 10:05 is the first interval, so the P5MIN part ends at 11:00.
        prices = path([float(step // 6) for step in range(48)])
        frame = pd.DataFrame({"settlementdate": prices.index, "price": prices.values})
        figure = figures.forecastFigure(frame)
        stamps = pd.DatetimeIndex(figure.data[0].x)
        assert len(stamps) < len(prices)
        assert list(stamps[:11]) == list(prices.index[:11])
        assert {stamp.minute for stamp in stamps[11:]} == {0, 30}

    # A dispatch price holds flat across its interval, so the line steps, never ramps.
    def test_the_line_steps(self, sawtooth, standIn):
        frame = live.suggest(sawtooth, MODEL, REGION)
        assert figures.forecastFigure(frame).data[0].line.shape == "vh"

    def test_no_forecast_draws_a_message(self):
        figure = figures.forecastFigure(pd.DataFrame(), note="NEMWeb is down")
        assert figure.layout.annotations[0].text == "NEMWeb is down"


class TestRefreshSlot:
    # The live screen refetches on the dispatch boundaries, minutes ending in 0 or 5.
    @pytest.mark.parametrize("now, slot", [
        ("2026-10-23 14:07:59", "2026-10-23 14:05"),
        ("2026-10-23 14:10:00", "2026-10-23 14:10"),
        ("2026-10-23 14:14:59", "2026-10-23 14:10"),
    ])
    def test_the_slot_starts_on_a_five_minute_boundary(self, now, slot):
        assert live.refreshSlot(now) == pd.Timestamp(slot)


class TestCycleBudget:
    """The screen opens its episode at the run time, so the path is part of an episode
    and it gets that part of the experiment's budget. The whole budget would let a
    late-afternoon board spend what a whole day is allowed."""

    def test_a_part_episode_gets_that_part_of_the_budget(self):
        settings = {"cycleBudget": 1.0, "horizon": 288}
        assert live.budgetFor(settings, 144) == pytest.approx(0.5)
        assert live.budgetFor(settings, 288) == pytest.approx(1.0)

    # A path longer than the episode is still one episode's budget.
    def test_the_share_never_exceeds_the_budget(self):
        assert live.budgetFor({"cycleBudget": 1.0, "horizon": 288}, 600) == 1.0

    def test_no_budget_stays_no_budget(self):
        assert live.budgetFor({"cycleBudget": None, "horizon": 288}, 144) is None

    # The experiment's budget, not config's: the agents were trained elsewhere.
    def test_it_reads_the_experiment_not_config(self, monkeypatch):
        import config as settings
        monkeypatch.setattr(settings, "cycle_budget", 7.0)
        assert live.budgetFor({"cycleBudget": 1.0, "horizon": 288}, 288) == 1.0

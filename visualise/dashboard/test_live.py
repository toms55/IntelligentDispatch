"""Tests for visualise/dashboard/live.py. No network: every path is built here."""
import numpy as np
import pandas as pd
import pytest

from rl import policies
from utils import nem_interval
from visualise.dashboard import figures, live

REGION = "NSW1"


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


class TestSuggest:
    # The rule control needs no checkpoint, so the screen works on a bare machine.
    def test_the_rule_charges_low_and_discharges_high(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        assert len(frame) == len(sawtooth)
        assert set(frame.action) <= {"charge", "discharge", "idle"}
        assert "charge" in set(frame.action[:12])
        assert "discharge" in set(frame.action[12:])

    # The state of charge stays inside the battery at every interval.
    def test_the_state_of_charge_stays_in_range(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        assert frame.soc.between(0.0, 1.0).all()

    # Degradation is a cost, so a plan never keeps more than its trades take.
    def test_profit_never_exceeds_revenue(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        assert (frame.profit <= frame.revenue + 1e-9).all()
        assert frame.revenue.iloc[-1] > 0

    # Revenue runs forward, so an interval that trades nothing leaves the total alone.
    def test_an_idle_interval_does_not_move_the_revenue(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        moved = frame.revenue.diff().fillna(0.0)
        assert (moved[frame.action == "idle"].abs() < 1e-12).all()

    # A forecast too short to be an episode draws nothing rather than raising.
    @pytest.mark.parametrize("length", [0, 1])
    def test_a_short_path_gives_an_empty_frame(self, length):
        frame = live.suggest(path([10.0] * length), live.thresholdLabel, REGION)
        assert frame.empty and list(frame.columns) == list(live.columns)

    # A label with no checkpoint behind it names the directory it looked in.
    def test_a_missing_checkpoint_names_the_directory(self, sawtooth, tmp_path):
        with pytest.raises(FileNotFoundError, match=REGION):
            live.suggest(sawtooth, "maskableppo-warm-s0", REGION, root=tmp_path)


class TestCheckpoints:
    # The backtest label names the price source it was scored under; the agent is one.
    def test_the_price_source_is_dropped(self):
        assert live.checkpointKey("maskableppo-warm-s0-forecast") == "maskableppo-warm-s0"

    # The screen reads trainedPolicies/, which is what training publishes.
    def test_it_reads_what_training_published(self, tmp_path):
        source = tmp_path / "fold51"
        source.mkdir()
        (source / policies.agentFile).write_bytes(b"")
        (source / policies.scalerFile).write_text("{}")
        policies.publish(source, REGION, "maskableppo", "warm", 0, root=tmp_path / "out")
        found = live.findCheckpoints(REGION, tmp_path / "out")
        assert list(found) == ["maskableppo-warm-s0"]

    # A directory the copy never finished is not offered.
    def test_a_checkpoint_without_its_zip_is_skipped(self, tmp_path):
        (tmp_path / REGION / "a2c-cold-s1").mkdir(parents=True)
        assert live.findCheckpoints(REGION, tmp_path) == {}

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
    def test_the_key_names_both_blocks(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        figure = figures.forecastFigure(frame, live.blocks(frame), region=REGION)
        names = [trace.name for trace in figure.data]
        assert names == ["AEMO forecast", "Discharge: sell", "Charge: buy"]

    # Red stands above the mean price line and blue hangs below it.
    def test_discharge_sits_above_the_rule_and_charge_below(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        figure = figures.forecastFigure(frame, live.blocks(frame))
        rule = float(np.mean(frame.price))
        for shape in figure.layout.shapes:
            if shape.type != "rect":
                continue
            if shape.fillcolor == figures.dischargeBlock:
                assert shape.y0 == rule and shape.y1 > rule
            else:
                assert shape.y1 == rule and shape.y0 < rule

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
    def test_the_line_steps(self, sawtooth):
        frame = live.suggest(sawtooth, live.thresholdLabel, REGION)
        assert figures.forecastFigure(frame).data[0].line.shape == "vh"

    def test_no_forecast_draws_a_message(self):
        figure = figures.forecastFigure(pd.DataFrame(), note="NEMWeb is down")
        assert figure.layout.annotations[0].text == "NEMWeb is down"

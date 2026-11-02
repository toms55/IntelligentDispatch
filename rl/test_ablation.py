"""Tests for rl/ablation.py and the knobs it turns. Run with: uv run pytest rl/ -q

The first class is the one to read: the baseline ablation must be the environment every
published agent trained in, or none of the variants is a one-element change.
"""
import numpy as np
import pandas as pd
import pytest

import config
from degradation.schade import SchadeDegradation
from rl import features
from rl.ablation import Ablation
from rl.env import CHARGE, DISCHARGE, IDLE, BessEnv
from rl.test_env import PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST, priceSeries

DAY = config.observation_intervals


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


def dayPrices():
    """One trading day of distinct prices, so every slot says where it came from."""
    return [50.0 + (index % 97) * 3.0 for index in range(DAY)]


def dayEnv(paper, **kwargs):
    settings = dict(horizon=DAY, calendar=False, rewardScale=1.0)
    settings.update(kwargs)
    return BessEnv(priceSeries(dayPrices()), paper, **settings)


class TestBaselineIsUntouched:
    def test_default_names_no_change(self):
        assert Ablation().isDefault()
        assert Ablation().asManifest() == {}

    def test_a_variant_names_only_what_it_moves(self):
        assert Ablation(lookahead=12, history=8).asManifest() == {"lookahead": 12,
                                                                  "history": 8}

    def test_manifest_round_trips(self):
        variant = Ablation(timeEncoding="sincos", hideMask=True)
        assert Ablation.fromManifest(variant.asManifest()) == variant

    def test_default_ablation_replays_the_unablated_env(self, paper):
        plain, ablated = dayEnv(paper), dayEnv(paper, ablation=Ablation())
        first, _ = plain.reset(seed=0)
        again, _ = ablated.reset(seed=0)
        assert first.shape == (config.observation_size,)
        np.testing.assert_array_equal(first, again)
        for action in [CHARGE] * 20 + [IDLE] * 5 + [DISCHARGE] * 20:
            a = plain.step(action)
            b = ablated.step(action)
            np.testing.assert_array_equal(a[0], b[0])
            assert a[1] == b[1]

    def test_scaler_json_of_the_baseline_gains_no_keys(self):
        assert set(features.PriceScaler(17500.0).asDict()) == {"cap", "floor", "knee"}

    def test_invalid_variants_are_refused(self):
        with pytest.raises(ValueError):
            Ablation(lookahead=0).validate()
        with pytest.raises(ValueError):
            Ablation(timeEncoding="hourly").validate()
        with pytest.raises(ValueError):
            Ablation(priceTransform="log").validate()


class TestLookahead:
    def prices(self, paper, lookahead):
        env = dayEnv(paper, ablation=Ablation(lookahead=lookahead))
        observed, _ = env.reset(seed=0)
        assert observed.shape == (config.observation_size,)
        return observed[features.priceSlice]

    def test_spot_only(self, paper):
        slots = self.prices(paper, 1)
        assert slots[0] != 0.0
        assert not slots[1:].any()

    def test_one_hour_keeps_every_five_minute_slot_and_no_bucket(self, paper):
        slots = self.prices(paper, 12)
        assert slots[:12].all()
        assert not slots[12:].any()

    def test_four_hours_keeps_six_half_hour_buckets(self, paper):
        slots = self.prices(paper, 48)
        assert slots[12:].astype(bool).sum() == 6

    def test_full_view_is_the_baseline(self, paper):
        full = self.prices(paper, DAY)
        base = dayEnv(paper).reset(seed=0)[0][features.priceSlice]
        np.testing.assert_array_equal(full, base)


class TestTimeEncoding:
    def slots(self, paper, encoding, steps=0):
        env = dayEnv(paper, ablation=Ablation(timeEncoding=encoding))
        observed, _ = env.reset(seed=0)
        for _ in range(steps):
            observed, *_ = env.step(IDLE)
        return observed[features.clockIndex], observed[features.episodeIndex]

    def test_none_hides_the_hour(self, paper):
        assert self.slots(paper, "none", steps=40) == (0.0, 0.0)

    def test_sincos_at_the_day_start(self, paper):
        clock, episode = self.slots(paper, "sincos")
        assert clock == pytest.approx(0.5)
        assert episode == pytest.approx(1.0)

    def test_sincos_stays_in_bounds_and_moves(self, paper):
        first = self.slots(paper, "sincos", steps=0)
        later = self.slots(paper, "sincos", steps=72)
        assert first != later
        assert all(0.0 <= value <= 1.0 for value in first + later)

    def test_a_multi_day_episode_is_refused(self, paper):
        with pytest.raises(ValueError):
            BessEnv(priceSeries([100.0] * (2 * DAY)), paper, horizon=2 * DAY,
                    calendar=False, ablation=Ablation(timeEncoding="none"))


class TestHistory:
    def test_size_grows_by_the_slot_count(self, paper):
        env = dayEnv(paper, ablation=Ablation(history=8))
        observed, _ = env.reset(seed=0)
        assert observed.shape == (config.observation_size + 8,)
        assert env.observation_space.shape == observed.shape

    def test_nothing_before_the_episode(self, paper):
        env = dayEnv(paper, ablation=Ablation(history=8))
        observed, _ = env.reset(seed=0)
        assert not observed[features.historyStart:].any()

    def test_newest_first_and_only_what_has_settled(self, paper):
        env = dayEnv(paper, ablation=Ablation(history=4))
        env.reset(seed=0)
        for _ in range(9):
            observed, *_ = env.step(IDLE)
        prices, scaler = dayPrices(), env.scaler
        history = observed[features.historyStart:]
        assert history[0] == pytest.approx(scaler(np.mean(prices[3:9])), abs=1e-6)
        assert history[1] == pytest.approx(scaler(np.mean(prices[0:3])), abs=1e-6)
        assert not history[2:].any()

    def test_a_second_episode_does_not_see_the_first(self, paper):
        values = dayPrices() + [9000.0] * DAY
        env = BessEnv(priceSeries(values), paper, horizon=DAY, calendar=False,
                      ablation=Ablation(history=8))
        observed, _ = env.reset(options={"window": 1})
        assert not observed[features.historyStart:].any()


class TestMaskHiding:
    def test_all_actions_show_legal_but_the_count_is_unchanged(self, paper):
        shown = dayEnv(paper, ablation=Ablation(hideMask=True))
        kept = dayEnv(paper)
        shown.reset(seed=0)
        kept.reset(seed=0)
        assert shown.action_masks().all()
        assert not kept.action_masks().all()
        # An empty pack cannot discharge: the same step is out of bounds in both.
        assert shown.step(DISCHARGE)[4]["outOfBounds"]
        assert kept.step(DISCHARGE)[4]["outOfBounds"]


class TestShapingBonus:
    def test_reward_only(self, paper):
        plain, shaped = dayEnv(paper), dayEnv(paper, ablation=Ablation(shapingBonus=2.0))
        plain.reset(seed=0)
        shaped.reset(seed=0)
        cheap = int(np.argmin(dayPrices()))
        rewards = []
        for env in (plain, shaped):
            for _ in range(cheap):
                env.step(IDLE)
            _, reward, _, _, info = env.step(CHARGE)
            rewards.append((reward, info["profit"], info["revenue"]))
        assert rewards[1][0] == pytest.approx(rewards[0][0] + 2.0)
        assert rewards[1][1:] == rewards[0][1:]


class TestPriceTransform:
    cap = 17500.0

    def test_linear_is_price_over_cap(self):
        scaler = Ablation(priceTransform="linear").scaler(self.cap)
        assert scaler(1750.0) == pytest.approx(0.1)
        assert scaler(self.cap) == 1.0

    def test_clip_flattens_the_tails(self):
        scaler = Ablation(priceTransform="clip").scaler(self.cap)
        low, high = config.price_clip_bounds
        assert scaler(5000.0) == scaler(high) == scaler(high + 1.0)
        assert scaler(-500.0) == scaler(low)
        assert scaler(100.0) == features.PriceScaler(self.cap)(100.0)

    def test_scaler_round_trips_through_its_dict(self):
        scaler = Ablation(priceTransform="clip").scaler(self.cap)
        assert features.PriceScaler(**scaler.asDict()) == scaler

    def test_a_clip_scaler_differs_from_the_baseline(self):
        assert Ablation(priceTransform="clip").scaler(self.cap) != features.PriceScaler(
            self.cap)

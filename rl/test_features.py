"""Tests for rl/features.py. Run with: uv run pytest rl/ -q"""
import math

import numpy as np
import pandas as pd
import pytest

import config
from rl import features


@pytest.fixture
def ramp():
    return [float(n) for n in range(config.episode_intervals)]


@pytest.fixture
def timestamps():
    return list(pd.date_range("2024-03-05", periods=config.episode_intervals, freq="5min"))


class TestNormalisePrice:
    def test_zero_maps_to_zero(self):
        assert features.normalisePrice(0.0) == 0.0

    def test_is_increasing(self):
        values = [features.normalisePrice(p) for p in range(-1000, 1000, 25)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_is_odd_about_zero(self):
        for price in (5.0, 50.0, 300.0, 900.0):
            assert features.normalisePrice(-price) == pytest.approx(
                -features.normalisePrice(price)
            )

    def test_the_market_cap_lands_on_one(self):
        assert features.normalisePrice(config.price_cap) == pytest.approx(1.0)

    def test_compresses_spikes_relative_to_ordinary_prices(self):
        ordinary = features.normalisePrice(50.0)
        spike = features.normalisePrice(5_000.0)
        assert spike / ordinary < 10

    def test_clips_beyond_the_administered_limits(self):
        assert features.normalisePrice(50_000.0) == pytest.approx(
            features.normalisePrice(config.price_cap)
        )
        assert features.normalisePrice(-9_000.0) == pytest.approx(
            features.normalisePrice(config.price_floor)
        )


class TestPriceFeatures:
    def test_has_the_configured_length(self, ramp):
        got = features.priceFeatures(ramp, 0, config.episode_intervals)
        assert len(got) == config.price_feature_count == 58

    def test_the_first_hour_is_five_minute_resolution(self, ramp):
        got = features.priceFeatures(ramp, 0, config.episode_intervals)
        for step in range(config.price_fine_intervals):
            assert got[step] == pytest.approx(features.normalisePrice(ramp[step]))

    def test_beyond_the_first_hour_is_half_hour_means(self, ramp):
        got = features.priceFeatures(ramp, 0, config.episode_intervals)
        first = config.price_fine_intervals
        window = ramp[first:first + config.price_bucket_intervals]
        assert got[first] == pytest.approx(
            features.normalisePrice(sum(window) / len(window))
        )

    def test_the_layout_closes_with_no_ragged_bucket(self):
        covered = (
            config.price_fine_intervals
            + config.price_bucket_count * config.price_bucket_intervals
        )
        assert covered == config.episode_intervals

    def test_is_forward_looking(self, ramp):
        got = features.priceFeatures(ramp, 100, config.episode_intervals)
        assert got[0] == pytest.approx(features.normalisePrice(ramp[100]))
        assert got[1] == pytest.approx(features.normalisePrice(ramp[101]))

    def test_pads_past_the_horizon_instead_of_reading_on(self, ramp):
        longer = ramp + [9_999.0] * 50
        got = features.priceFeatures(longer, config.episode_intervals - 2,
                                     config.episode_intervals)
        assert got[0] == pytest.approx(features.normalisePrice(ramp[-2]))
        assert got[1] == pytest.approx(features.normalisePrice(ramp[-1]))
        assert all(value == 0.0 for value in got[2:])

    def test_pads_a_part_full_bucket_from_what_is_there(self, ramp):
        start = config.episode_intervals - config.price_fine_intervals - 2
        got = features.priceFeatures(ramp, start, config.episode_intervals)
        tail = ramp[start + config.price_fine_intervals:]
        assert got[config.price_fine_intervals] == pytest.approx(
            features.normalisePrice(sum(tail) / len(tail))
        )

    def test_a_shorter_series_pads_rather_than_raising(self):
        got = features.priceFeatures([10.0, 20.0], 0, config.episode_intervals)
        assert len(got) == config.price_feature_count
        assert got[2] == 0.0


class TestTemporalFeatures:
    def test_is_two_sine_cosine_pairs(self):
        got = features.temporalFeatures(pd.Timestamp("2024-06-15 13:25"))
        assert len(got) == 4
        assert all(-1.0 <= value <= 1.0 for value in got)

    def test_midnight_and_the_interval_before_it_are_adjacent(self):
        late = features.temporalFeatures(pd.Timestamp("2024-06-15 23:55"))[:2]
        midnight = features.temporalFeatures(pd.Timestamp("2024-06-16 00:00"))[:2]
        distance = math.dist(late, midnight)
        noon = features.temporalFeatures(pd.Timestamp("2024-06-16 12:00"))[:2]
        assert distance < math.dist(midnight, noon)

    def test_the_same_clock_time_on_different_days_matches(self):
        a = features.temporalFeatures(pd.Timestamp("2024-02-01 06:30"))[:2]
        b = features.temporalFeatures(pd.Timestamp("2024-09-17 06:30"))[:2]
        assert a == pytest.approx(b)

    def test_the_season_pair_separates_january_from_july(self):
        january = features.temporalFeatures(pd.Timestamp("2024-01-15 12:00"))[2:]
        july = features.temporalFeatures(pd.Timestamp("2024-07-15 12:00"))[2:]
        assert math.dist(january, july) > 1.5


class TestObservation:
    def test_assembles_the_slots_in_order(self, ramp, timestamps):
        got = features.observation(0.42, ramp, timestamps, 0, config.episode_intervals)
        assert got.shape == (config.observation_size,)
        assert got[0] == pytest.approx(0.42, abs=1e-6)
        priceEnd = 1 + config.price_feature_count
        assert got[1:priceEnd] == pytest.approx(
            np.asarray(features.priceFeatures(ramp, 0, config.episode_intervals),
                       dtype=np.float32), abs=1e-6
        )
        assert got[priceEnd:priceEnd + 4] == pytest.approx(
            np.asarray(features.temporalFeatures(timestamps[0]), dtype=np.float32),
            abs=1e-6
        )
        assert got[-1] == pytest.approx(1.0)

    def test_a_forecast_equal_to_the_realised_prices_gives_the_same_vector(
        self, ramp, timestamps
    ):
        realised = features.observation(0.5, ramp, timestamps, 24,
                                        config.episode_intervals)
        forecast = features.observation(0.5, list(ramp), timestamps, 24,
                                       config.episode_intervals)
        assert np.array_equal(realised, forecast)

    def test_is_float32(self, ramp, timestamps):
        got = features.observation(0.5, ramp, timestamps, 0, config.episode_intervals)
        assert got.dtype == np.float32


class TestObservationBounds:
    def test_are_finite(self):
        low, high = features.observationBounds()
        assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))

    def test_low_is_below_high_everywhere(self):
        low, high = features.observationBounds()
        assert np.all(low < high)

    def test_contain_every_price_the_market_allows(self, timestamps):
        low, high = features.observationBounds()
        extremes = [config.price_floor, config.price_cap] * config.episode_intervals
        got = features.observation(1.0, extremes, timestamps, 0, config.episode_intervals)
        assert np.all(got >= low - 1e-6) and np.all(got <= high + 1e-6)

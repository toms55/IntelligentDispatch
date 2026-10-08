"""Tests for rl/features.py. Run with: uv run pytest rl/ -q"""
import numpy as np
import pytest

import config
from rl import features


@pytest.fixture
def ramp():
    return [float(n) for n in range(config.episode_intervals)]


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
        for step in range(config.five_minute_interval_count):
            assert got[step] == pytest.approx(features.normalisePrice(ramp[step]))

    def test_beyond_the_first_hour_is_half_hour_means(self, ramp):
        got = features.priceFeatures(ramp, 0, config.episode_intervals)
        first = config.five_minute_interval_count
        window = ramp[first:first + config.price_bucket_intervals]
        assert got[first] == pytest.approx(
            features.normalisePrice(sum(window) / len(window))
        )

    def test_the_layout_closes_with_no_ragged_bucket(self):
        covered = (
            config.five_minute_interval_count
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
        start = config.episode_intervals - config.five_minute_interval_count - 2
        got = features.priceFeatures(ramp, start, config.episode_intervals)
        tail = ramp[start + config.five_minute_interval_count:]
        assert got[config.five_minute_interval_count] == pytest.approx(
            features.normalisePrice(sum(tail) / len(tail))
        )

    def test_a_shorter_series_pads_rather_than_raising(self):
        got = features.priceFeatures([10.0, 20.0], 0, config.episode_intervals)
        assert len(got) == config.price_feature_count
        assert got[2] == 0.0


class TestObservation:
    def test_assembles_the_slots_in_order(self, ramp):
        got = features.observation(0.42, ramp, 0, config.episode_intervals)
        assert got.shape == (config.observation_size,)
        assert got[0] == pytest.approx(0.42, abs=1e-6)
        priceEnd = 1 + config.price_feature_count
        assert got[1:priceEnd] == pytest.approx(
            np.asarray(features.priceFeatures(ramp, 0, config.episode_intervals),
                       dtype=np.float32), abs=1e-6
        )
        assert got[-1] == pytest.approx(1.0)

    def test_a_forecast_equal_to_the_realised_prices_gives_the_same_vector(self, ramp):
        realised = features.observation(0.5, ramp, 24, config.episode_intervals)
        forecast = features.observation(0.5, list(ramp), 24, config.episode_intervals)
        assert np.array_equal(realised, forecast)

    def test_is_float32(self, ramp):
        got = features.observation(0.5, ramp, 0, config.episode_intervals)
        assert got.dtype == np.float32


class TestObservationBounds:
    def test_are_finite(self):
        low, high = features.observationBounds()
        assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))

    def test_low_is_below_high_everywhere(self):
        low, high = features.observationBounds()
        assert np.all(low < high)

    def test_contain_every_price_the_market_allows(self):
        low, high = features.observationBounds()
        extremes = [config.price_floor, config.price_cap] * config.episode_intervals
        got = features.observation(1.0, extremes, 0, config.episode_intervals)
        assert np.all(got >= low - 1e-6) and np.all(got <= high + 1e-6)

"""Tests for rl/features.py. Run with: uv run pytest rl/ -q"""
from datetime import datetime

import numpy as np
import pytest

import config
from rl import features
from rl.features import PriceScaler
from utils.market import market_price_floor


@pytest.fixture
def ramp():
    return [float(n) for n in range(config.observation_intervals)]


class TestPriceScaler:
    def test_zero_maps_to_zero(self, scaler):
        assert scaler(0.0) == 0.0

    def test_the_cap_lands_on_one(self, scaler):
        assert scaler(scaler.cap) == pytest.approx(1.0)
        assert scaler.high == 1.0

    def test_is_increasing(self, scaler):
        values = [scaler(price) for price in range(-1000, 1000, 25)]
        assert all(later > earlier for earlier, later in zip(values, values[1:]))

    def test_is_odd_about_zero(self, scaler):
        """asinh is odd by construction, so this needs no tolerance at all."""
        for price in (5.0, 50.0, 300.0, 900.0):
            assert scaler(-price) == -scaler(price)

    def test_the_knee_is_where_the_curve_bends(self, scaler):
        """Below the knee the transform is roughly linear, above it roughly
        logarithmic. Doubling inside the knee nearly doubles the output; doubling well
        past it does not."""
        inside = scaler(2 * scaler.knee / 20) / scaler(scaler.knee / 20)
        outside = scaler(20 * scaler.knee) / scaler(10 * scaler.knee)
        assert inside > 1.9
        assert outside < 1.3

    def test_compresses_spikes_relative_to_ordinary_prices(self, scaler):
        ordinary = scaler(50.0)
        spike = scaler(5000.0)
        assert spike / ordinary < 25

    def test_clips_beyond_the_administered_limits(self, scaler):
        assert scaler(50000.0) == scaler(scaler.cap)
        assert scaler(-9000.0) == scaler(scaler.floor)

    def test_the_floor_sets_the_lower_bound(self, scaler):
        assert scaler.low == scaler(market_price_floor)
        assert -1.0 < scaler.low < 0.0

    def test_a_different_cap_is_a_different_transform(self):
        """Why rl/train.py saves the scaler. Two caps disagree on the same price, so an
        agent scored under the wrong cap reads a number it never trained on."""
        trained = PriceScaler(17500.0)
        scored = PriceScaler(23200.0)
        assert trained(1000.0) != scored(1000.0)
        assert trained == PriceScaler(17500.0)
        assert trained != scored

    def test_round_trips_through_a_dict(self, scaler):
        assert PriceScaler(**scaler.asDict()) == scaler

    @pytest.mark.parametrize("kwargs", [
        {"cap": 0.0},
        {"cap": -1.0},
        {"cap": 17500.0, "knee": 0.0},
        {"cap": 17500.0, "floor": 10.0},
    ])
    def test_refuses_a_nonsense_scale(self, kwargs):
        with pytest.raises(ValueError):
            PriceScaler(**{"cap": 17500.0, **kwargs})


class TestScalerFor:
    def test_pins_the_cap_in_force_at_the_last_timestamp(self):
        """The last, not the first: the caps only rise, so the final cap is the largest
        over the window and nothing earlier clips."""
        window = [datetime(2025, 6, 1, 4), datetime(2025, 8, 1, 4)]
        assert features.scalerFor(window).cap == 20300.0

    def test_a_window_inside_one_financial_year(self):
        window = [datetime(2024, 1, 1, 4), datetime(2024, 6, 1, 4)]
        assert features.scalerFor(window).cap == 16600.0

    def test_carries_the_configured_knee(self):
        window = [datetime(2025, 1, 1, 4)]
        assert features.scalerFor(window).knee == config.price_knee
        assert features.scalerFor(window, knee=50.0).knee == 50.0

    def test_an_empty_window_is_refused(self):
        with pytest.raises(ValueError, match="empty window"):
            features.scalerFor([])

    def test_accepts_a_pandas_index(self):
        pandas = pytest.importorskip("pandas")
        index = pandas.date_range("2025-08-01 04:00", periods=12, freq="5min")
        assert features.scalerFor(index).cap == 20300.0


class TestPriceFeatures:
    def test_has_the_configured_length(self, ramp, scaler):
        got = features.priceFeatures(ramp, 0, config.observation_intervals, scaler)
        assert len(got) == config.price_feature_count == 58

    def test_the_first_hour_is_five_minute_resolution(self, ramp, scaler):
        got = features.priceFeatures(ramp, 0, config.observation_intervals, scaler)
        for step in range(config.five_minute_interval_count):
            assert got[step] == pytest.approx(scaler(ramp[step]))

    def test_beyond_the_first_hour_is_half_hour_means(self, ramp, scaler):
        got = features.priceFeatures(ramp, 0, config.observation_intervals, scaler)
        first = config.five_minute_interval_count
        window = ramp[first:first + config.bucket_window_intervals]
        assert got[first] == pytest.approx(scaler(sum(window) / len(window)))

    def test_the_layout_closes_with_no_ragged_bucket(self):
        covered = (
            config.five_minute_interval_count
            + config.price_bucket_count * config.bucket_window_intervals
        )
        assert covered == config.observation_intervals

    def test_is_forward_looking(self, ramp, scaler):
        got = features.priceFeatures(ramp, 100, config.observation_intervals, scaler)
        assert got[0] == pytest.approx(scaler(ramp[100]))
        assert got[1] == pytest.approx(scaler(ramp[101]))

    def test_pads_past_the_horizon_instead_of_reading_on(self, ramp, scaler):
        longer = ramp + [9999.0] * 50
        got = features.priceFeatures(longer, config.observation_intervals - 2,
                                     config.observation_intervals, scaler)
        assert got[0] == pytest.approx(scaler(ramp[-2]))
        assert got[1] == pytest.approx(scaler(ramp[-1]))
        assert all(value == 0.0 for value in got[2:])

    def test_pads_a_part_full_bucket_from_what_is_there(self, ramp, scaler):
        start = config.observation_intervals - config.five_minute_interval_count - 2
        got = features.priceFeatures(ramp, start, config.observation_intervals, scaler)
        tail = ramp[start + config.five_minute_interval_count:]
        assert got[config.five_minute_interval_count] == pytest.approx(
            scaler(sum(tail) / len(tail))
        )

    def test_a_shorter_series_pads_rather_than_raising(self, scaler):
        got = features.priceFeatures([10.0, 20.0], 0, config.observation_intervals, scaler)
        assert len(got) == config.price_feature_count
        assert got[2] == 0.0


class TestObservation:
    def test_assembles_the_slots_in_order(self, ramp, scaler):
        got = features.observation(0.42, ramp, 0, config.observation_intervals,
                                   scaler, 0.8, 0.6, 0.4)
        assert got.shape == (config.observation_size,)
        assert got[features.socIndex] == pytest.approx(0.42, abs=1e-6)
        assert got[features.priceSlice] == pytest.approx(
            np.asarray(features.priceFeatures(ramp, 0, config.observation_intervals,
                                              scaler), dtype=np.float32), abs=1e-6
        )
        assert got[features.clockIndex] == pytest.approx(0.8)
        assert got[features.cycleIndex] == pytest.approx(0.6)
        assert got[features.episodeIndex] == pytest.approx(0.4)

    def test_a_forecast_equal_to_the_realised_prices_gives_the_same_vector(self, ramp, scaler):
        realised = features.observation(0.5, ramp, 24, config.observation_intervals,
                                        scaler, 0.9, 1.0, 0.9)
        forecast = features.observation(0.5, list(ramp), 24,
                                        config.observation_intervals, scaler,
                                        0.9, 1.0, 0.9)
        assert np.array_equal(realised, forecast)

    def test_is_float32(self, ramp, scaler):
        got = features.observation(0.5, ramp, 0, config.observation_intervals,
                                   scaler, 1.0, 1.0, 1.0)
        assert got.dtype == np.float32


class TestTradingDayRemaining:
    """The clock slot. It must reproduce the old episode-relative formula on an aligned
    288-interval day, or every published agent reads a slot that moved under it."""

    def test_reproduces_the_old_formula_over_an_aligned_day(self):
        pandas = pytest.importorskip("pandas")
        stamps = pandas.date_range("2024-01-10 04:05", periods=config.observation_intervals,
                                   freq="5min")
        for step, stamp in enumerate(stamps):
            old = (config.observation_intervals - step) / config.observation_intervals
            assert features.tradingDayRemaining(stamp) == pytest.approx(old, abs=1e-12)

    def test_the_trading_day_opens_at_one_and_closes_at_one_interval(self):
        pandas = pytest.importorskip("pandas")
        assert features.tradingDayRemaining(pandas.Timestamp("2024-01-10 04:05")) == 1.0
        assert features.tradingDayRemaining(
            pandas.Timestamp("2024-01-11 04:00")
        ) == pytest.approx(1 / config.observation_intervals)

    def test_reads_the_clock_not_the_episode(self):
        """The bug this replaces: the current forecast screen starts its episode at the
        run time, so an episode-relative fraction read 1.0 at every hour of the day."""
        pandas = pytest.importorskip("pandas")
        assert features.tradingDayRemaining(
            pandas.Timestamp("2024-01-10 14:05")
        ) == pytest.approx(168 / config.observation_intervals)


class TestObservationBounds:
    def test_are_finite(self, scaler):
        low, high = features.observationBounds(scaler)
        assert np.all(np.isfinite(low)) and np.all(np.isfinite(high))

    def test_low_is_below_high_everywhere(self, scaler):
        low, high = features.observationBounds(scaler)
        assert np.all(low < high)

    def test_contain_every_price_the_market_allows(self, scaler):
        low, high = features.observationBounds(scaler)
        extremes = [scaler.floor, scaler.cap] * config.observation_intervals
        got = features.observation(1.0, extremes, 0, config.observation_intervals,
                                   scaler, 1.0, 1.0, 1.0)
        assert np.all(got >= low - 1e-6) and np.all(got <= high + 1e-6)


class TestLayoutIndices:
    """The layout is one definition, and it does not move with the episode length."""

    def test_the_slots_are_where_features_says_they_are(self):
        assert features.socIndex == 0
        assert features.priceSlice == slice(1, 1 + config.price_feature_count)
        assert features.clockIndex == 1 + config.price_feature_count
        assert features.cycleIndex == features.clockIndex + 1
        assert features.episodeIndex == features.cycleIndex + 1
        assert config.observation_size == features.episodeIndex + 1

    def test_the_lookahead_is_a_trading_day_not_an_episode(self):
        covered = (
            config.five_minute_interval_count
            + config.price_bucket_count * config.bucket_window_intervals
        )
        assert covered == config.observation_intervals

    def test_the_new_slots_are_fractions(self, scaler):
        low, high = features.observationBounds(scaler)
        for index in (features.cycleIndex, features.episodeIndex):
            assert (low[index], high[index]) == (0.0, 1.0)

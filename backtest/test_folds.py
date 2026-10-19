"""Tests for backtest/folds.py. Run with: uv run pytest backtest/ -q

Every number the rolling backtest produces depends on these boundaries being right, so
this file is the one to read first. The two facts it pins:

1. A window is half-open `(start, end]` and interval-ending, matching `price_series`.
   A 04:00 to 04:00 day runs 04:05 through next-day 04:00, which is 288 values and is
   what the `0405` in every oracle cache filename refers to.
2. An off-by-one-interval slice still yields 288 values and still solves. It just misses
   every oracle cache key and de-aligns the baseline from the environment's windows.
   `requireFullDays` is the only thing that catches it.
"""
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from backtest import folds
from utils import nem_interval

# A Monday 04:00, comfortably after Five Minute Settlement commenced.
SMALL_START = datetime(2022, 1, 3, 4)
SMALL_END = SMALL_START + timedelta(days=70)

# The NSW1 span the oracle cache currently covers.
NSW1_START = datetime(2024, 1, 1, 4)
NSW1_END = datetime(2026, 1, 1, 4)


def priceSeries(start, end):
    """A complete 5-minute series over `(start, end]`, the shape price_series returns."""
    index = pd.date_range(start + nem_interval, end, freq=nem_interval)
    return pd.Series(range(len(index)), index=index, dtype=float, name="RRP")


class TestEnumerateFolds:
    # The first fold starts where the caller said, and trains for the length it asked for.
    def test_fold_zero_starts_at_the_given_start(self):
        first = folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=4)[0]
        assert first.index == 0
        assert first.trainStart == SMALL_START
        assert first.trainEnd == SMALL_START + timedelta(weeks=4)
        assert first.testStart == first.trainEnd
        assert first.testEnd == first.trainEnd + timedelta(weeks=1)

    # The fold count follows from the span, the window lengths and the step.
    @pytest.mark.parametrize("trainWeeks,testWeeks,stepWeeks,expected", [
        (4, 1, 1, 6),
        (4, 1, 2, 3),
        (4, 2, 1, 5),
        (9, 1, 1, 1),
        (12, 1, 1, 0),
    ])
    def test_the_fold_count_follows_the_formula(self, trainWeeks, testWeeks, stepWeeks,
                                                expected):
        got = folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=trainWeeks,
                                   testWeeks=testWeeks, stepWeeks=stepWeeks)
        assert len(got) == expected

    # The worked example in the plan: a 52-week window over two years of NSW1 data.
    def test_the_nsw1_span_gives_fifty_two_weekly_folds(self):
        got = folds.enumerateFolds(NSW1_START, NSW1_END, trainWeeks=52)
        assert len(got) == 52
        assert got[0].testStart == datetime(2024, 12, 30, 4)
        assert got[-1].testStart == datetime(2025, 12, 22, 4)
        assert got[-1].testEnd == datetime(2025, 12, 29, 4)

    # Training ends exactly where testing begins, so no interval is both trained and scored.
    def test_training_ends_where_testing_begins(self):
        for fold in folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=4):
            assert fold.trainEnd == fold.testStart

    # Consecutive folds move forward by the step, sliding the training window rather than
    # growing it.
    @pytest.mark.parametrize("stepWeeks", [1, 2, 3])
    def test_consecutive_folds_step_by_the_step(self, stepWeeks):
        got = folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=4,
                                   stepWeeks=stepWeeks)
        step = timedelta(weeks=stepWeeks)
        for earlier, later in zip(got, got[1:]):
            assert later.trainStart - earlier.trainStart == step
            assert later.testStart - earlier.testStart == step
            assert later.index == earlier.index + 1

    # A span with no room for a single fold is an empty result, not an error.
    def test_a_span_too_short_for_one_fold_is_empty(self):
        assert folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=12) == []

    # Five Minute Settlement commenced 2021-10-01, and before it the market settled on
    # 30-minute averages, so the whole MDP is wrong over an earlier window.
    def test_a_span_before_five_minute_settlement_is_refused(self):
        with pytest.raises(ValueError, match="Five Minute Settlement"):
            folds.enumerateFolds(datetime(2021, 9, 30, 4), SMALL_END, trainWeeks=4)

    # The commencement day itself is valid.
    def test_the_commencement_day_is_accepted(self):
        got = folds.enumerateFolds(datetime(2021, 10, 1, 4),
                                   datetime(2021, 10, 1, 4) + timedelta(days=70),
                                   trainWeeks=4)
        assert got[0].trainStart == datetime(2021, 10, 1, 4)

    # 04:00 is the NEM trading day boundary and the only run time where PREDISPATCH
    # reaches a full 288 intervals, so a boundary anywhere else is refused by name.
    @pytest.mark.parametrize("label,start,end", [
        ("start", datetime(2022, 1, 3, 0), SMALL_END),
        ("start", datetime(2022, 1, 3, 4, 5), SMALL_END),
        ("end", SMALL_START, datetime(2022, 3, 14, 12)),
    ])
    def test_a_boundary_that_is_not_0400_is_refused(self, label, start, end):
        with pytest.raises(ValueError, match=label):
            folds.enumerateFolds(start, end, trainWeeks=4)


class TestArgumentValidation:
    """Everything a caller can get wrong, caught at the boundary with a named argument."""

    # A span that ends before it starts is refused, matching cli._window's wording.
    def test_an_end_before_the_start_is_refused(self):
        with pytest.raises(ValueError, match="ends before it starts"):
            folds.enumerateFolds(SMALL_END, SMALL_START, trainWeeks=4)

    # A zero-length span is refused rather than returning an empty fold list, so the
    # caller learns the span is wrong instead of that it was merely too short.
    def test_a_zero_length_span_is_refused(self):
        with pytest.raises(ValueError, match="ends before it starts"):
            folds.enumerateFolds(SMALL_START, SMALL_START, trainWeeks=4)

    # A week count below one is refused by name. stepWeeks=0 would never terminate.
    @pytest.mark.parametrize("label", ["trainWeeks", "testWeeks", "stepWeeks"])
    @pytest.mark.parametrize("weeks", [0, -1])
    def test_a_week_count_below_one_is_refused(self, label, weeks):
        with pytest.raises(ValueError, match=label):
            folds.enumerateFolds(SMALL_START, SMALL_END, **{"trainWeeks": 4, label: weeks})

    # A fractional week would put a boundary off 04:00 and break the forecast horizon.
    @pytest.mark.parametrize("label", ["trainWeeks", "testWeeks", "stepWeeks"])
    def test_a_fractional_week_count_is_refused(self, label):
        with pytest.raises(ValueError, match=label):
            folds.enumerateFolds(SMALL_START, SMALL_END, **{"trainWeeks": 4, label: 1.5})

    # NEM time is naive AEST year-round, and a tz-aware datetime silently shifts the
    # trading-day boundary across daylight saving.
    @pytest.mark.parametrize("label,start,end", [
        ("start", SMALL_START.replace(tzinfo=timezone.utc), SMALL_END),
        ("end", SMALL_START, SMALL_END.replace(tzinfo=timezone.utc)),
    ])
    def test_a_timezone_aware_boundary_is_refused(self, label, start, end):
        with pytest.raises(ValueError, match="naive NEM time"):
            folds.enumerateFolds(start, end, trainWeeks=4)

    # A plain date carries no time of day, so it cannot sit on the 04:00 boundary.
    def test_a_date_rather_than_a_datetime_is_refused(self):
        with pytest.raises(ValueError, match="must be a datetime"):
            folds.enumerateFolds(date(2022, 1, 3), SMALL_END, trainWeeks=4)

    # A pandas Timestamp is a datetime subclass and is accepted unchanged.
    def test_a_naive_pandas_timestamp_is_accepted(self):
        got = folds.enumerateFolds(pd.Timestamp(SMALL_START), pd.Timestamp(SMALL_END),
                                   trainWeeks=4)
        assert len(got) == 6


class TestFoldOrdering:
    # No fold can ever test on data earlier than it trained on: the test window is
    # derived from the training window, never supplied separately.
    @pytest.mark.parametrize("trainWeeks,testWeeks,stepWeeks", [
        (4, 1, 1), (4, 2, 3), (52, 1, 1),
    ])
    def test_training_always_precedes_testing(self, trainWeeks, testWeeks, stepWeeks):
        got = folds.enumerateFolds(NSW1_START, NSW1_END, trainWeeks=trainWeeks,
                                   testWeeks=testWeeks, stepWeeks=stepWeeks)
        assert got
        for fold in got:
            assert fold.trainStart < fold.trainEnd == fold.testStart < fold.testEnd

    # Every fold's test window sits inside the span the caller asked for.
    def test_every_fold_fits_inside_the_span(self):
        for fold in folds.enumerateFolds(NSW1_START, NSW1_END, trainWeeks=52):
            assert NSW1_START <= fold.trainStart and fold.testEnd <= NSW1_END


class TestFold:
    # The test window decomposes into one 04:00 start per trading day.
    @pytest.mark.parametrize("testWeeks,expected", [(1, 7), (2, 14)])
    def test_test_days_are_one_0400_start_per_day(self, testWeeks, expected):
        fold = folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=4,
                                    testWeeks=testWeeks)[0]
        days = fold.testDays
        assert len(days) == expected
        assert days[0] == fold.testStart
        assert all(day.hour == 4 and day.minute == 0 for day in days)
        assert days[-1] + timedelta(days=1) == fold.testEnd


class TestLeftoverDays:
    # Days at the tail that no fold reaches are reported rather than silently dropped.
    def test_reports_the_days_no_fold_reaches(self):
        got = folds.enumerateFolds(SMALL_START, SMALL_END, trainWeeks=4, stepWeeks=2)
        assert folds.leftoverDays(got, SMALL_END) == 7
        assert folds.leftoverDays(
            folds.enumerateFolds(NSW1_START, NSW1_END, trainWeeks=52), NSW1_END) == 3


class TestSliceWindow:
    # One trading day is 288 interval-ending stamps, 04:05 through next-day 04:00.
    def test_one_day_is_288_rows_from_0405(self):
        prices = priceSeries(SMALL_START, SMALL_START + timedelta(days=3))
        day = folds.sliceWindow(prices, SMALL_START, SMALL_START + timedelta(days=1))
        assert len(day) == 288
        assert day.index[0] == SMALL_START + nem_interval
        assert day.index[-1] == SMALL_START + timedelta(days=1)

    # Consecutive day slices abut without overlapping, which is what keeps the oracle
    # cache keys and the environment's windows aligned.
    def test_consecutive_days_do_not_overlap(self):
        prices = priceSeries(SMALL_START, SMALL_START + timedelta(days=3))
        first = folds.sliceWindow(prices, SMALL_START, SMALL_START + timedelta(days=1))
        second = folds.sliceWindow(prices, SMALL_START + timedelta(days=1),
                                   SMALL_START + timedelta(days=2))
        assert first.index.intersection(second.index).empty
        assert second.index[0] == first.index[-1] + nem_interval


class TestRequireFullDays:
    # A complete window passes.
    def test_a_complete_window_passes(self):
        end = SMALL_START + timedelta(days=2)
        folds.requireFullDays(priceSeries(SMALL_START, end), SMALL_START, end)

    # One missing dispatch interval slides every later window off 04:05, so it must raise.
    def test_a_missing_interval_is_refused(self):
        end = SMALL_START + timedelta(days=2)
        prices = priceSeries(SMALL_START, end).drop(index=SMALL_START + timedelta(hours=9))
        with pytest.raises(ValueError):
            folds.requireFullDays(prices, SMALL_START, end)

    # An intervention duplicate that escaped price_series would double-count an interval.
    def test_a_duplicated_interval_is_refused(self):
        end = SMALL_START + timedelta(days=2)
        prices = priceSeries(SMALL_START, end)
        doubled = pd.concat([prices, prices.iloc[[10]]]).sort_index()
        with pytest.raises(ValueError):
            folds.requireFullDays(doubled, SMALL_START, end)

    # An off-by-one-interval slice still holds whole days, so only the index comparison
    # catches it.
    def test_a_misaligned_start_is_refused(self):
        end = SMALL_START + timedelta(days=2)
        prices = priceSeries(SMALL_START, end).shift(1, freq=nem_interval)
        with pytest.raises(ValueError):
            folds.requireFullDays(prices, SMALL_START, end)

    # The message names both endpoints and both lengths, so a failure is diagnosable.
    def test_the_message_names_the_window_and_both_lengths(self):
        end = SMALL_START + timedelta(days=2)
        prices = priceSeries(SMALL_START, end).iloc[:-1]
        with pytest.raises(ValueError) as raised:
            folds.requireFullDays(prices, SMALL_START, end)
        message = str(raised.value)
        assert "2022-01-03" in message and "576" in message and "575" in message


class TestDayOffset:
    # Whole trading days between two 04:00 stamps, which is how a fold indexes into the
    # span-wide oracle records.
    def test_counts_whole_trading_days(self):
        assert folds.dayOffset(SMALL_START, SMALL_START) == 0
        assert folds.dayOffset(SMALL_START, SMALL_START + timedelta(days=1)) == 1
        assert folds.dayOffset(NSW1_START, datetime(2024, 12, 30, 4)) == 364

    # A moment that is not a whole number of days after the span start would index the
    # wrong record, so it raises rather than truncating.
    def test_a_partial_day_is_refused(self):
        with pytest.raises(ValueError):
            folds.dayOffset(SMALL_START, SMALL_START + timedelta(hours=30))

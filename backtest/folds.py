"""Fold boundaries for the rolling-origin backtest.

Windows are half-open `(start, end]` and interval-ending, exactly as `data/prices.py`
returns them, so a 04:00 to 04:00 day runs 04:05 through next-day 04:00. That is 288
values and it is what the `0405` in every oracle cache filename refers to.

Every boundary is 04:00, the NEM trading day boundary. PREDISPATCH only reaches 288
intervals from a 04:00 run, so it is the one run time where exactly one episode fits.
"""
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import pandas as pd

import config
from utils import nem_interval

tradingDayStart = time(4, 0)
oneDay = timedelta(days=1)


@dataclass(frozen=True)
class Fold:
    """One training window and the test window that follows it, back to back."""

    index: int
    trainStart: datetime
    trainEnd: datetime
    testStart: datetime
    testEnd: datetime

    @property
    def testDays(self):
        """The 04:00 start of each trading day in the test window."""
        count = (self.testEnd - self.testStart) // oneDay
        return [self.testStart + step * oneDay for step in range(count)]


def requireTradingDayStart(moment, label):
    if moment.time() != tradingDayStart:
        raise ValueError(
            f"{label} must be at 04:00, the NEM trading day boundary; got {moment}"
        )


def requireNemDatetime(moment, label):
    """NEM time is naive AEST year-round, and the whole codebase assumes it."""
    if not isinstance(moment, datetime):
        raise ValueError(
            f"{label} must be a datetime, got {type(moment).__name__}: {moment!r}"
        )
    if moment.tzinfo is not None:
        raise ValueError(
            f"{label} must be naive NEM time (AEST, no daylight saving); got {moment}"
        )


def enumerateFolds(start, end, trainWeeks, testWeeks=1, stepWeeks=1,
                   earliest=config.five_minute_settlement_start):
    """Sliding training window, test window immediately after it, stepped forward.

    Folds are emitted while the test window still fits inside `end`. Returns an empty
    list when the span has no room for one.
    """
    for label, moment in (("start", start), ("end", end)):
        requireNemDatetime(moment, label)
        requireTradingDayStart(moment, label)
    if end <= start:
        raise ValueError(f"the span ends before it starts: {start} to {end}")
    for label, weeks in (("trainWeeks", trainWeeks), ("testWeeks", testWeeks),
                         ("stepWeeks", stepWeeks)):
        # A fractional week would put a boundary off 04:00; stepWeeks=0 would not
        # terminate.
        if not isinstance(weeks, int):
            raise ValueError(f"{label} must be a whole number of weeks, got {weeks!r}")
        if weeks < 1:
            raise ValueError(f"{label} must be at least 1, got {weeks}")
    if start < earliest:
        raise ValueError(
            f"start {start:%Y-%m-%d} precedes {earliest:%Y-%m-%d}, when Five Minute "
            f"Settlement commenced (NER 3.15.1A). Before that date the market settled "
            f"on 30-minute averages, so an episode that settles each 5-minute interval "
            f"at its own RRP models a market that did not exist."
        )

    train, test, step = (timedelta(weeks=weeks)
                         for weeks in (trainWeeks, testWeeks, stepWeeks))
    out = []
    while True:
        trainStart = start + len(out) * step
        boundary = trainStart + train
        testEnd = boundary + test
        if testEnd > end:
            return out
        out.append(Fold(len(out), trainStart, boundary, boundary, testEnd))


def leftoverDays(folds, end):
    """Trading days past the last fold that no fold reaches."""
    return 0 if not folds else (end - folds[-1].testEnd) // oneDay


def sliceWindow(prices, start, end):
    """The `(start, end]` slice, matching what `price_series(start, end)` would return."""
    return prices.loc[start + nem_interval:end]


def requireFullDays(series, start, end):
    """Refuse anything but the exact 5-minute index `(start, end]` implies.

    One comparison catches a gap, a duplicate, a wrong length and a misaligned start.
    The misalignment is the one that matters: an off-by-one-interval slice still holds
    whole days and still solves, but it misses every oracle cache key and de-aligns the
    baseline from the environment's windows.
    """
    wanted = pd.date_range(start + nem_interval, end, freq=nem_interval)
    if series.index.equals(wanted):
        return
    got = (f"{len(series)} ending {series.index[0]:%Y-%m-%d %H:%M} to "
           f"{series.index[-1]:%Y-%m-%d %H:%M}") if len(series) else "0 intervals"
    raise ValueError(
        f"{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M} wants {len(wanted)} intervals "
        f"ending {wanted[0]:%Y-%m-%d %H:%M} to {wanted[-1]:%Y-%m-%d %H:%M}; got {got}"
    )


def dayOffset(spanStart, moment):
    """Whole trading days from `spanStart` to `moment`, for indexing span-wide records."""
    delta = moment - spanStart
    if delta % oneDay:
        raise ValueError(
            f"{moment} is not a whole number of trading days after {spanStart}"
        )
    return delta // oneDay

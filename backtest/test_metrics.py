"""Tests for backtest/metrics.py. Run with: uv run pytest backtest/ -q

Every ratio in the project is formed here, and the one that matters is `captured`:

    captured = sum(agentProfit) / sum(oracleDualBound)

A ratio of sums, not a mean of ratios. The two disagree badly, because the oracle's
profit is concentrated in a handful of volatile days: a quiet day where the oracle made
$0.02 produces a per-day ratio that is pure noise, and averaging those ratios lets the
quiet days outvote the days the money is actually in.
"""
import pandas as pd
import pytest

from backtest import metrics, records

DAY = pd.Timestamp("2025-01-01 04:00")


def rows(**overrides):
    """One day-row, defaulting to a day the agent captured half the oracle's profit."""
    base = {
        "fold": 0, "algo": "maskableppo", "arm": "warm", "seed": 0,
        "priceSource": "realised", "testDay": DAY,
        "agentProfit": 5.0, "agentRevenue": 6.0, "agentDegradation": 1.0,
        "oracleDualBound": 10.0, "spikeIntensity": 3.0, "highVol": True,
        "timesteps": 30000, "warmFrom": "", "writtenAt": "2026-10-19T12:00:00",
    }
    return {**base, **overrides}


def frameOf(*specs):
    return pd.DataFrame([rows(**spec) for spec in specs],
                        columns=list(records.dayColumns))


class TestCaptured:
    # A ratio of sums is dollar-weighted, so the day with the money in it decides the
    # figure. A mean of per-day ratios would let the quiet day outvote it.
    def test_pooled_captured_is_dollar_weighted(self):
        frame = frameOf(
            {"testDay": DAY, "agentProfit": 90.0, "oracleDualBound": 100.0},
            {"testDay": DAY + pd.Timedelta(days=1),
             "agentProfit": 0.0, "oracleDualBound": 0.02, "highVol": False},
        )
        assert metrics.summarise(frame)["captured"] == pytest.approx(90.0 / 100.02)
        perDayRatio = (90.0 / 100.0 + 0.0 / 0.02) / 2
        assert metrics.summarise(frame)["captured"] != pytest.approx(perDayRatio)

    # A day the oracle barely profited on cannot move the pooled figure, because it
    # contributes almost nothing to either side of the ratio.
    def test_a_near_zero_oracle_day_does_not_distort_the_pool(self):
        busy = {"agentProfit": 90.0, "oracleDualBound": 100.0}
        quiet = {"testDay": DAY + pd.Timedelta(days=1), "agentProfit": -0.01,
                 "oracleDualBound": 0.001, "highVol": False}
        assert metrics.summarise(frameOf(busy))["captured"] == pytest.approx(0.9)
        assert metrics.summarise(frameOf(busy, quiet))["captured"] == pytest.approx(
            0.9, abs=1e-3)

    # Seeds are repeats of the same day, so summing their rows would multiply the
    # dollar totals by the seed count. They are averaged into one day first.
    def test_seeds_are_averaged_before_pooling(self):
        frame = frameOf({"seed": 0, "agentProfit": 4.0},
                        {"seed": 1, "agentProfit": 6.0})
        got = metrics.summarise(frame)
        assert got["agentProfit"] == pytest.approx(5.0)
        assert got["oracleProfit"] == pytest.approx(10.0)
        assert got["days"] == 1

    # An oracle that made nothing has no ratio to report, and must not raise.
    def test_a_zero_oracle_gives_a_nan_rather_than_raising(self):
        got = metrics.summarise(frameOf({"agentProfit": 0.0, "oracleDualBound": 0.0}))
        assert pd.isna(got["captured"])


class TestWinRate:
    # Win rate is the share of days in profit, so one big winner does not outvote three
    # small losses the way a dollar total would.
    def test_counts_days_not_dollars(self):
        frame = frameOf(
            {"testDay": DAY, "agentProfit": 100.0},
            {"testDay": DAY + pd.Timedelta(days=1), "agentProfit": -1.0},
            {"testDay": DAY + pd.Timedelta(days=2), "agentProfit": -1.0},
            {"testDay": DAY + pd.Timedelta(days=3), "agentProfit": -1.0},
        )
        assert metrics.summarise(frame)["winRate"] == pytest.approx(0.25)


class TestEmptyInput:
    # Nothing to summarise is an explicit empty result, so a dashboard with no data
    # selected renders a message rather than propagating NaN arithmetic.
    def test_an_empty_frame_summarises_to_zero_days(self):
        got = metrics.summarise(records.readDays("/nonexistent", "NSW1"))
        assert got["days"] == 0 and got["agentProfit"] == 0.0
        assert pd.isna(got["captured"])

    # An empty bucket is reported as empty rather than omitted, so the high and low
    # split always has both sides.
    def test_an_empty_bucket_is_still_reported(self):
        got = metrics.byBucket(frameOf({"highVol": True}))
        assert set(got) == {True, False}
        assert got[False]["days"] == 0


class TestBuckets:
    # The volatility split loses no days: every day lands in exactly one bucket.
    def test_the_buckets_partition_the_days(self):
        frame = frameOf({"testDay": DAY, "highVol": True},
                        {"testDay": DAY + pd.Timedelta(days=1), "highVol": False},
                        {"testDay": DAY + pd.Timedelta(days=2), "highVol": False})
        got = metrics.byBucket(frame)
        assert got[True]["days"] + got[False]["days"] == metrics.summarise(frame)["days"]
        assert (got[True]["days"], got[False]["days"]) == (1, 2)


class TestPerFold:
    # One row per fold, which is what the per-fold median and IQR are taken over and
    # what the calendar's week scope aggregates to.
    def test_one_row_per_fold(self):
        frame = frameOf({"fold": 0}, {"fold": 1, "testDay": DAY + pd.Timedelta(days=7)})
        assert list(metrics.perFold(frame).index) == [0, 1]

    # The median survives the skew that a mean does not: most folds are quiet, a few
    # carry the year.
    def test_reports_the_median_and_iqr_of_per_fold_captured(self):
        frame = frameOf(
            *[{"fold": n, "testDay": DAY + pd.Timedelta(days=7 * n),
               "agentProfit": profit, "oracleDualBound": 10.0}
              for n, profit in enumerate([0.0, 2.0, 4.0, 6.0, 8.0])]
        )
        got = metrics.summarise(frame)
        assert got["foldMedian"] == pytest.approx(0.4)
        assert got["foldIqr"] == pytest.approx(0.4)


class TestDifference:
    # Warm against cold, and realised against forecast, are the same comparison: a
    # matched difference over the folds where both sides ran.
    @pytest.mark.parametrize("column,left,right", [
        ("arm", "warm", "cold"),
        ("priceSource", "realised", "forecast"),
    ])
    def test_compares_matched_folds_only(self, column, left, right):
        frame = frameOf(
            {"fold": 0, column: left, "agentProfit": 8.0},
            {"fold": 0, column: right, "agentProfit": 5.0},
            # fold 1 ran only the left side, so it is excluded from the comparison
            {"fold": 1, "testDay": DAY + pd.Timedelta(days=7), column: left,
             "agentProfit": 99.0},
        )
        got = metrics.difference(frame, column, left, right)
        assert list(got.perFold.index) == [0]
        assert got.pooled == pytest.approx(3.0)

    # No fold ran both sides, so there is nothing to compare and the result says so.
    def test_no_matched_folds_is_an_empty_comparison(self):
        got = metrics.difference(frameOf({"arm": "warm"}), "arm", "warm", "cold")
        assert got.perFold.empty and pd.isna(got.pooled)

"""Tests for backtest/stamp.py. Run with: uv run pytest backtest/ -q"""
import json
from datetime import datetime

import pytest

from backtest import records, stamp
from backtest.test_records import row

REGION = "NSW1"


def make(tmp_path, writtenAt, **manifest):
    records.writeFold(tmp_path, REGION, 0, [dict(row(), writtenAt=writtenAt)])
    records.writeManifest(tmp_path, REGION, {"region": REGION, "horizon": 288, **manifest})
    return tmp_path


def stated(resultsDir):
    return json.loads(records.manifestPath(resultsDir, REGION).read_text())


class TestStamp:
    # Every row predates the refresh, so each test day read one 04:00 run.
    def test_an_older_run_is_stamped_once_a_day(self, tmp_path):
        run = make(tmp_path, "2026-10-21T14:01:33")
        assert stamp.stampRegion(run, REGION) == 288
        assert stated(run)["forecastEvery"] == 288

    # A row written once the refresh existed could have read any pull, so it is refused.
    def test_a_later_run_is_refused(self, tmp_path):
        run = make(tmp_path, "2026-10-22T09:00:00")
        with pytest.raises(ValueError, match="cannot be known"):
            stamp.stampRegion(run, REGION)
        assert "forecastEvery" not in stated(run)

    # A manifest that already states it is left alone.
    def test_a_stated_run_is_untouched(self, tmp_path):
        run = make(tmp_path, "2026-10-22T09:00:00", forecastEvery=1)
        assert stamp.stampRegion(run, REGION) == 1

    def test_the_cutoff_is_the_refresh_commit(self):
        assert stamp.refreshIntroduced == datetime(2026, 10, 21, 20, 25, 48)

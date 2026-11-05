"""Score the threshold rule over a window, on realised or forecast prices.

    uv run python -m examples.threshold NSW1 2025-03-03 2025-03-10 forecast

Copy this file and swap `ThresholdPolicy()` for your own policy.
"""
import json
import sys
from datetime import datetime

from backtest.score import score
from rl.heuristic import ThresholdPolicy


def main(region, start, end, priceSource):
    # A window is whole trading days, so both ends sit on the 04:00 boundary.
    at4 = lambda day: datetime.fromisoformat(day).replace(hour=4)
    result = score(ThresholdPolicy(), region, at4(start), at4(end), priceSource)
    print(json.dumps({"summary": result["summary"], "settings": result["settings"]}, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:5])

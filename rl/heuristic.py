"""A rule-based threshold policy, as the control the agent has to beat (TODO.txt step 2).

Charge near the bottom of the forward window, discharge near the top, otherwise idle.
Reads the same observation vector the agent gets, so it is scored by the same path and
works unchanged on forecast-filled observations.
"""
import numpy as np

from rl.env import CHARGE, DISCHARGE, IDLE
from rl.features import priceSlice


class ThresholdPolicy:
    """Quantile thresholds on the forward price vector.

    minSpread guards the degenerate case: a flat window has no arbitrage, so it must
    idle rather than charge. Tests the window's full range, not the quantile gap, which
    collapses when one outlier sits among otherwise equal prices.
    """

    def __init__(self, lowQuantile=0.25, highQuantile=0.75, minSpread=1e-6):
        self.lowQuantile = lowQuantile
        self.highQuantile = highQuantile
        self.minSpread = minSpread

    def predict(self, observation, deterministic=True, action_masks=None):
        prices = np.asarray(observation, dtype=float)[priceSlice]
        # Padding past the episode end is a normalised zero; drop it so it doesn't drag
        # the quantiles down as the episode runs out.
        live = prices[prices != 0.0]
        if live.size < 2:
            return IDLE, None

        if live.max() - live.min() <= self.minSpread:
            return IDLE, None

        now = prices[0]
        low = np.quantile(live, self.lowQuantile)
        high = np.quantile(live, self.highQuantile)
        if now <= low:
            action = CHARGE
        elif now >= high:
            action = DISCHARGE
        else:
            action = IDLE

        if action_masks is not None and not action_masks[action]:
            action = IDLE
        return action, None

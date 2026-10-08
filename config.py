import math
from datetime import datetime


# --- Episode -----------------------------------------------------------------
episode_intervals = 288
initial_soc = 0.5
gamma = 1.0


# --- Observation -------------------------------------------------------------
price_fine_intervals = 12  # 5-minute values, the first hour
price_bucket_intervals = 6  # intervals per 30-minute bucket beyond that

price_bucket_count = (episode_intervals - price_fine_intervals) // price_bucket_intervals
price_feature_count = price_fine_intervals + price_bucket_count
observation_size = 1 + price_feature_count + 4 + 1  # soc, prices, temporal, time left

# normalisePrice(p) = sign(p) * log1p(|p| / scale) / divisor. The divisor is chosen so
# the market price cap lands on exactly 1.0.
price_log_scale = 50.0
price_log_divisor = math.log1p(350.0)  # = log1p(price_cap / price_log_scale)

price_floor = -1000.0
price_cap = 17_500.0


# --- Reward ------------------------------------------------------------------
reward_scale = 20.0


# --- Splits ------------------------------------------------------------------
# Train on 2024, test on 2025. `val` is the tail of 2024, excluded from `train`. Every
# boundary is 04:00, the NEM trading day boundary: PREDISPATCH only reaches 288
# intervals from a 04:00 run, so this is also what makes the forecast horizon available.
splits = {
    "train": (datetime(2024, 1, 1, 4), datetime(2024, 11, 1, 4)),
    "val": (datetime(2024, 11, 1, 4), datetime(2025, 1, 1, 4)),
    "test": (datetime(2025, 1, 1, 4), datetime(2026, 1, 1, 4)),
}

full_train = (datetime(2024, 1, 1, 4), datetime(2025, 1, 1, 4))

seeds = (0, 1, 2, 3, 4)

# rl/algos.py asserts its registry matches this tuple.
algorithms = ("maskableppo", "dqn", "a2c")


# --- Oracle --------------------------------------------------------------------
# Quote Schedule.dualBound, never Schedule.profit: a gap-limited solve returns an
# incumbent below the true optimum.
oracle_gap = 0.005
oracle_time_limit = 600

# The reward baseline only has to be a constant per window and interval, so it can use
# the cheap continuous solve. A quoted bound must be discrete, since the agent has three
# actions.
baseline_discrete_power = False
bound_discrete_power = True


# --- Market and battery ------------------------------------------------------
default_region = "SA1"

# The Schade 2024 Table 2 case.
battery_mw = 0.060
battery_mwh = 0.100
battery_replacement_cost = 150_000.0

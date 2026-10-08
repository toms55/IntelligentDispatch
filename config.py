# --- Episode -----------------------------------------------------------------
episode_intervals = 288
initial_soc = 0.5
gamma = 1.0


# --- Observation -------------------------------------------------------------
five_minute_interval_count = 12  # 5-minute values, the first hour of the forecast
bucket_window_intervals = 6  # dispatch intervals per 30-minute bucket beyond that

price_bucket_count = (episode_intervals - five_minute_interval_count) // bucket_window_intervals
price_feature_count = five_minute_interval_count + price_bucket_count
observation_size = 1 + price_feature_count + 1  # soc, prices, time left

# The knee of the asinh price transform, in $/MWh. Below the knee the transform is
# roughly linear, above it roughly logarithmic. The market price cap and floor are not
# set here: they are AEMC determinations and come from utils/market.py, keyed on date.
#
# NOT TUNED. 300 is a placeholder and needs an experiment. A larger knee spends more of
# the output range on spikes and compresses the $20 to $150 band that most intervals sit
# in. At a knee of 300 and a cap of 23200, $50 maps to 0.03 and $5000 maps to 0.70.
# A smaller knee does the opposite. Sweep this against captured profit on the validation
# window before you quote any result.
price_knee = 300.0


# --- Reward ------------------------------------------------------------------
reward_scale = 20.0


# --- Experiment design -------------------------------------------------------
# Windows are not set here. Pass them on the command line as --split NAME=START..END,
# because the design changes between experiments. Align every boundary to 04:00, the NEM
# trading day boundary: PREDISPATCH only reaches 288 intervals from a 04:00 run, so this
# is what makes the forecast horizon available.
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
battery_replacement_cost = 150000.0

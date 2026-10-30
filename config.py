from datetime import datetime

# --- Episode -----------------------------------------------------------------
episode_intervals = 288
# The battery starts every episode empty, and nothing values what it holds at the close.
# A start above zero would hand the agent energy it never bought, and it would then sell
# that energy at whatever the last interval pays. Starting empty removes both: under
# perfect foresight the oracle only buys what it can sell well, and the agent is paid the
# same way, so a day's profit is the trading it actually did.
initial_soc = 0.0
gamma = 1.0


# --- Observation -------------------------------------------------------------
# How far ahead the agent reads, which is NOT the episode length. AEMO publishes no RRP
# past the end of the current trading day, so a forecast can never reach further than
# this, and an episode longer than one day must still read only one day ahead or the
# realised-price agent trains on information the forecast-priced agent cannot have.
observation_intervals = 288

five_minute_interval_count = 12  # 5-minute values, the first hour of the forecast
bucket_window_intervals = 6  # dispatch intervals per 30-minute bucket beyond that

price_bucket_count = (observation_intervals - five_minute_interval_count) // bucket_window_intervals
price_feature_count = five_minute_interval_count + price_bucket_count
# soc, prices, trading-day clock, cycle budget left, episode left
observation_size = 1 + price_feature_count + 3

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
# Five Minute Settlement commenced in the NEM on this trading day, under NER 3.15.1A.
# Before it the market settled on the 30-minute TRADINGPRICE, so an episode that settles
# each 5-minute interval at its own RRP models a market that did not exist. The rolling
# backtest refuses any span that starts earlier.
five_minute_settlement_start = datetime(2021, 10, 1, 4)

# A high-volatility trading day, by mean excess over the spike threshold:
#     spikeIntensity = sum(max(RRP - spike_threshold, 0)) / episode_intervals
# A day is high-volatility when its intensity reaches the cutoff (inclusive).
# Only the excess counts, so negative prices never make a day high-volatility.
# Calibrated on NSW1: a 1.20 cutoff selects 17% of 2025 days carrying 64.6% of the
# oracle's profit, and 22% of 2024 days carrying 75.9%. Separate from price_knee, which
# happens to share the value 300 and means something else entirely.
spike_threshold = 300.0
high_vol_intensity = 1.20

# Windows are not set here. Pass them on the command line as --split NAME=START..END,
# because the design changes between experiments. Align every boundary to 04:00, the NEM
# trading day boundary: PREDISPATCH only reaches 288 intervals from a 04:00 run, so this
# is what makes the forecast horizon available.
seeds = (0, 1, 2, 3, 4)

# The seed the hosted dashboard shows, per region. Every seed stays in trainedPolicies/
# and in the published results, because the thesis reports the spread over seeds. The
# Space shows one of them, so a reader sees one dispatch per region rather than five
# that differ only by seed. scripts/publishSpace.sh reads this to pick which rows and
# which agent to stage, and nothing else reads it: a local dashboard offers every seed.
#
# Each entry is the seed with the highest total agentProfit on forecast prices, over the
# test days of the run the Space opens on, which is rolling2023_profit_fc5m.
# `scripts/rankSeeds.py` derives it and prints this line. The winner is specific to the
# run: only QLD1 keeps its seed across the three profit runs, so re-derive this whenever
# the Space opens on another run. Measured gain over seed 0 runs from nothing in QLD1 to
# $1293 of $19685 in SA1, against a spread over seeds of up to $2929.
best_seed = {"NSW1": 1, "QLD1": 0, "SA1": 4, "TAS1": 2, "VIC1": 1}

# How often the agent's forecast is replaced by AEMO's latest run inside an episode, in
# dispatch intervals. 48 is every four hours: 04:00, 08:00, 12:00, 16:00, 20:00 and
# 00:00. Every PREDISPATCH run reaches the end of its trading day, which is the episode's
# end, so each refresh covers the rest of the episode exactly. episode_intervals gives
# back the single 04:00 run per day. Must divide episode_intervals.
forecast_refresh_intervals = 48

# rl/algos.py asserts its registry matches this tuple.
algorithms = ("maskableppo", "dqn", "a2c")


# --- Oracle --------------------------------------------------------------------
# Quote Schedule.dualBound, never Schedule.profit: a gap-limited solve returns an
# incumbent below the true optimum.
oracle_gap = 0.005
oracle_time_limit = 600

# The gap the existing data_cache/lp_baseline entries were solved at. The cache key
# includes the gap, so the rolling backtest must ask for this one or it misses every
# entry and silently starts re-solving.
backtest_gap = 0.02

# The reward is the agent's own profit, so the oracle is a scoring device only. A quoted
# bound must be discrete, since the agent has three actions.
bound_discrete_power = True

# The rolling backtest solves the discrete oracle over its test days only; training days
# need none.
#
# The limit is 1800s rather than the 300s the oracle runs elsewhere, because cacheKey
# does not include the time limit. A day that stops on the limit with a wide gap writes a
# loose dual bound, and that record is then served from cache for good: raising the limit
# later cannot reach it. Measured on NSW1 2024-01-10, discrete, no calendar, 0.02 gap: a
# 300s limit stopped with the incumbent at 14.6996 and the bound still at 17.7368, which
# is above the continuous bound of 17.1258 and so useless as a bound.
backtest_time_limit = 1800


# --- Market and battery ------------------------------------------------------
default_region = "SA1"

# The Schade 2024 Table 2 case.
battery_mw = 0.060
battery_mwh = 0.100
battery_replacement_cost = 150000.0

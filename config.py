from datetime import datetime

# --- Episode -----------------------------------------------------------------
# One trading day, 288 intervals. This is NOT the lookahead: see observation_intervals.
# A multi-day episode (a weekly cycle budget, say) is supported by the code but not run:
# the state of charge would carry across the days inside it and reset at its boundary
# only, which the weekly-episode experiment tried and dropped.
episode_intervals = 288
# The battery starts every episode empty, and nothing values what it holds at the close.
# A start above zero would hand the agent energy it never bought, and it would then sell
# that energy at whatever the last interval pays. Starting empty removes both: under
# perfect foresight the oracle only buys what it can sell well, and the agent is paid the
# same way, so a day's profit is the trading it actually did.
initial_soc = 0.0
gamma = 1.0

# The most equivalent full cycles an episode may spend, standing in for a cycle warranty.
# One EFC is the nominal usable capacity removed cell-side, so a partial discharge counts
# proportionally. `None` is no budget at all, which is every result published so far.
cycle_budget = None
# How tightly the oracle's cycle budget is solved. The budget is priced rather than
# capped, so the answer is a sandwich: the relative width between the feasible schedule
# and the bound. See `optimisation.dp.solve`.
cycle_dual_gap = 1e-4

# What energy still in the pack at the episode close is worth, in $/MWh, per region.
# Without it nothing values closing stock, so the last interval's best move is always to
# sell whatever is left: a few cents on a day-long episode, a Sunday-night dump on a week.
#
# Pinned from the median RRP of the 2023 calendar year, which precedes every test window
# the backtest scores, measured over all 105120 intervals of each region. A median rather
# than a mean, because the mean is pulled up by the spikes the agent is meant to trade
# into and should not be paid for holding. Set it too high and the battery hoards instead
# of trading, so it belongs below what the market typically pays.
#
# Never re-pin this from a test window. It is a modelling constant with the same
# discipline as price_knee.
terminal_energy_price = {"NSW1": 86.0, "QLD1": 75.8, "SA1": 59.0, "TAS1": 54.3,
                         "VIC1": 53.5}


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

# The band the "clip" price transform of the ablation study clamps the observation to:
# the 10th and 90th percentile of every region's 2023 RRP, pooled (525600 intervals), so it
# precedes every test window. Kortmann et al. (2025) clip to the same percentiles. NEM-wide
# rather than per region because the pooled agent shares one scaler. Never re-pin this
# from a test window; it has the same discipline as price_knee and terminal_energy_price.
price_clip_bounds = (-28.1, 150.0)


# --- Reward ------------------------------------------------------------------
reward_scale = 20.0

# What the agent loses for choosing to charge a full pack or to discharge an empty one,
# in reward units. MaskablePPO cannot choose either, because the mask removes the action,
# so this term is identically zero for it and only DQN and A2C can ever incur it. Both
# algorithms still train under the same reward function.
#
# It enters the reward and nothing else. Profit, revenue, degradation, regret and the
# captured fraction stay raw dollars, which is what keeps the discrete oracle a valid
# bound: the oracle's schedule is feasible, so the term is zero along it.
#
# The size, against what one interval is worth. The battery moves 0.060 MW for one
# twelfth of an hour, so one interval at rated throughput turns over 5 kWh, which at
# $300/MWh is $1.50, or a reward of 0.075 once reward_scale divides it. So 1.0 costs
# about thirteen such intervals, or roughly a day's oracle profit for this battery. That
# is deliberately far above the interval the action wastes.
#
# NOT TUNED. Sweep it on the validation window, never on a test window, and watch for the
# failure mode of too large a value: the agent stops filling the pack, trades a narrower
# SoC band and earns less. `python -m backtest.outofbounds` reads the attempt counts that
# measure it. RollingConfig records the value, so requireSameManifest refuses to put two
# penalties in one result set, and every run published before this carried no penalty.
invalid_action_penalty = 1.0


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

# The experiments the dashboard offers, in the order it lists them. One name per
# published run under results/backtestResults/, and the only list either screen reads:
# the backtest screen offers these runs and nothing else, and the current forecast screen
# offers the agents these runs trained. So every agent a reader can run has a scored
# experiment behind it.
#
# A run reaches this list when it tests something the thesis reports and its manifest
# states what that was. Three groups of published runs stay out:
#
# 1. The runs from before the manifest, which record no description and were scored from
#    a 0.5 state of charge. `initial_soc` is 0.0 now, so their oracle bound answers
#    another problem.
# 2. The weekly 7-cycle runs. An episode is one trading day here, and a weekly episode
#    carries the state of charge across the days inside it, so the two cannot share a
#    screen without the bound meaning two things.
# 3. The two masked-DQN runs. Their agents were trained by a `MaskableDQN` that is not
#    on this branch, and bringing it over would make the algorithm name `dqn` mean two
#    things: the pooled DQN run above and the daily one were both scored unmasked under
#    that name. See rl/algos.py on `dqnnomask`, which is the unmasked arm and is here.
dashboard_runs = (
    "rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast",
    "rolling2023_dqn_daily_1_cycle_end_empty_5min_forecast",
    "rolling2023_ppo_pooled_regions_daily_1_cycle_end_empty_5min_forecast",
    "rolling2023_dqn_pooled_regions_daily_1_cycle_end_empty_5min_forecast",
    "rolling2023_ppo_daily_no_cycle_limit_start_empty_5min_forecast_rerun",
)

# The seed each dashboard run shows, per run, train mode and region. Every seed stays in
# trainedPolicies/ and in the published results, because the thesis reports the spread
# over seeds. Each screen shows one of them, so a reader sees one dispatch per region
# rather than five that differ only by seed.
#
# The winner is specific to the run: no region keeps its seed across all four. A pooled
# run holds two train modes, and the shared agent and the regional fine-tune are two
# agents, so each is ranked on its own.
#
# Each entry is the seed with the highest total agentProfit on forecast prices, over the
# test days of that run in that region. `scripts/rankSeeds.py` derives the map and prints
# it. Pinned rather than ranked when the screen draws, so the thesis quotes this file and
# a reader of the dashboard sees the same seed every time.
#
# Picking the top of five seeds on the days the dashboard then draws is selection on the
# test window. It is presentational: the screen says which seed of how many it shows, and
# the thesis quotes the spread over seeds, never one of these numbers.
best_seed_by_run = {
    "rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast": {
        "warm": {"NSW1": 1, "QLD1": 4, "SA1": 0, "TAS1": 3, "VIC1": 4},
    },
    "rolling2023_dqn_daily_1_cycle_end_empty_5min_forecast": {
        "warm": {"NSW1": 2, "QLD1": 0, "SA1": 1, "TAS1": 2, "VIC1": 0},
    },
    "rolling2023_ppo_pooled_regions_daily_1_cycle_end_empty_5min_forecast": {
        "finetune": {"NSW1": 2, "QLD1": 4, "SA1": 0, "TAS1": 0, "VIC1": 4},
        "pooled": {"NSW1": 3, "QLD1": 2, "SA1": 2, "TAS1": 4, "VIC1": 2},
    },
    "rolling2023_dqn_pooled_regions_daily_1_cycle_end_empty_5min_forecast": {
        "finetune": {"NSW1": 4, "QLD1": 4, "SA1": 4, "TAS1": 1, "VIC1": 4},
        "pooled": {"NSW1": 4, "QLD1": 4, "SA1": 4, "TAS1": 1, "VIC1": 4},
    },
    "rolling2023_ppo_daily_no_cycle_limit_start_empty_5min_forecast_rerun": {
        "warm": {"NSW1": 2, "QLD1": 4, "SA1": 0, "TAS1": 1, "VIC1": 1},
    },
}

# How often the agent's forecast is replaced by AEMO's latest run inside an episode, in
# dispatch intervals. 48 is every four hours: 04:00, 08:00, 12:00, 16:00, 20:00 and
# 00:00. Every PREDISPATCH run reaches the end of its trading day, which is the episode's
# end, so each refresh covers the rest of the episode exactly. episode_intervals gives
# back the single 04:00 run per day. Must divide episode_intervals.
forecast_refresh_intervals = 48

# rl/algos.py asserts its registry matches this tuple. `dqnnomask` is a second name for
# the unmasked DQN, which a published run records as its algorithm; see rl/algos.py.
algorithms = ("maskableppo", "dqn", "dqnnomask", "a2c")

# DQN's replay buffer, in whole episodes. The only SB3 default that rl/algos.py overrides
# for DQN, and it is overridden because the buffer now travels between folds on disk
# (rl/train.py). SB3 defaults to 1000000 transitions, and the buffer holds the
# observation twice over, so at 60 float32 slots that default is about 480 MB in memory
# per seed and the same again on disk every fold. 200 episodes is 57600 transitions, about
# two warm folds of experience and about 28 MB.
replay_buffer_size = 200 * episode_intervals


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

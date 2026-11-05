# MDP ablation: results

Written 2026-11-05 from the 44 published ablation runs in `results/backtestResults/`
(22 variants x PPO and DQN, less the runs that were never submitted, plus the two reference
runs). Nothing here was tuned or selected on the test window.

## How the numbers were collected

1. **Inputs.** For every run and region, `results/backtestResults/<run>_<region>.csv`. A
   row is one seed, price source, train mode and test day. The runs are
   `rolling2023_ablate_<variant>[_dqn]` and the two published references
   `rolling2023_{ppo,dqn}_pooled_regions_daily_1_cycle_end_empty_5min_forecast`. The
   one-line description of each job is the `description` field of the run's JSON manifest
   beside the CSV.
2. **Scope.** `trainMode == finetune` only, which is the per-region agent. The pooled
   (shared) agent's rows are in the script's CSV output but not in the tables. Both
   `priceSource` values are reported: `realised` (the agent reads true prices) and
   `forecast` (it reads the AEMO forecast it would have had). All agents trained on
   realised prices.
3. **Metric.** `backtest.metrics.summarise`, unchanged. Seeds are averaged into one row per
   day first (`perDay`), then

       captured = sum(agentProfit) / sum(oracleDualBound)

   a ratio of sums, never a mean of ratios. The bound is the run's own `oracleDualBound`
   column. The `deg_*` and `lev_*` variants change the problem, so they are scored against
   an oracle re-solved for the same problem; I did not check that column.
4. **High-volatility scope.** The same function on the rows with `highVol == True`. That
   is 151 days in NSW1, 147 in QLD1, 163 in SA1, 73 in TAS1 and 65 in VIC1, out of 1001
   test days per region (2023-12-31 to 2026-09-26, not 2025 alone).
5. **Seed spread.** The same function per seed gives the min and max across the 5 seeds,
   stored in the script's CSV as `seedMin` and `seedMax`.
6. **Script.** A throwaway script, not in the repo (see the appendix). It writes one row
   per run x region x train mode x price source x scope to
   a CSV that was not kept (rerun the script to regenerate it). "Mean" columns are the plain mean of
   the five regional ratios, not a ratio of pooled sums.

Caveats on the data:

- `px_knee1000` has no PPO run. Only DQN was submitted, though the plan listed both.
- `lev_5` and `look_4h` (PPO) each have one seed that captured exactly 0% in every region
  and both price sources: the agent never traded. Their means are dragged down by it.
- TAS1 forecast results for DQN include seeds with negative capture, the reference DQN
  among them.
- The DQN reference is published under the algorithm name `dqnnomask`.
- The reference runs have no `outOfBounds` or `agentCycles` columns.

## Headline: mean over the 5 regions, captured % of oracle (PPO / DQN)

| Job | What it changed | Realised, all | Realised, high-vol | Forecast, all | Forecast, high-vol |
|---|---|---|---|---|---|
| **Reference** | Pooled agent, daily episode, 1 cycle budget, empty at the close, 3 actions | 71 / 48 | 74 / 73 | 38 / 26 | 39 / 42 |
| deg_none | No wear cost in the reward | 79 / 46 | 78 / 68 | 54 / 27 | 48 / 35 |
| deg_flat | Flat $/MWh wear at the Schade full-discharge rate | 72 / 45 | 74 / 68 | 38 / 19 | 44 / 34 |
| deg_full | Schade cycling plus the calendar term | 72 / 45 | 76 / 69 | 39 / 21 | 42 / 36 |
| look_12h | Sees 12 h of prices ahead instead of the full 288 intervals | 67 / 45 | 70 / 68 | 39 / 21 | 40 / 33 |
| look_4h | Sees 4 h ahead | 44 / 42 | 48 / 62 | 24 / 17 | 26 / 28 |
| look_1h | Sees 1 h ahead | 41 / 38 | 42 / 55 | 16 / 10 | 17 / 18 |
| look_spot | Sees the spot price only | 14 / 10 | 7 / 10 | 10 / 5 | 6 / 5 |
| time_sincos | Sine and cosine of the time of day | 61 / 40 | 65 / 62 | 35 / 18 | 35 / 30 |
| time_none | No time-of-day input | 61 / 46 | 65 / 69 | 36 / 22 | 36 / 34 |
| hist_1h | Adds 1 h of settled-price history | 69 / 44 | 72 / 66 | 39 / 20 | 41 / 31 |
| hist_4h | Adds 4 h of history | 68 / 46 | 71 / 67 | 38 / 22 | 40 / 34 |
| hist_24h | Adds 24 h of history | 60 / 40 | 61 / 61 | 35 / 22 | 35 / 35 |
| lev_5 | 5 power levels instead of 3 | 52 / 42 | 54 / 64 | 27 / 20 | 28 / 33 |
| lev_9 | 9 power levels | 61 / 36 | 62 / 54 | 34 / 19 | 34 / 31 |
| lev_17 | 17 power levels | 51 / 32 | 51 / 51 | 26 / 15 | 25 / 27 |
| shaping | Buy-low/sell-high bonus in the reward only | 73 / 55 | 79 / 72 | 40 / 28 | 46 / 38 |
| px_knee100 | Price squashing (asinh) knee at 100 | 65 / 44 | 67 / 62 | 38 / 23 | 39 / 34 |
| px_knee1000 | Knee at 1000 | not run / 47 | not run / 71 | not run / 21 | not run / 32 |
| px_linear | Prices scaled linearly to the cap | 61 / 49 | 68 / 75 | 37 / 22 | 41 / 35 |
| px_clip | Prices clipped to the 10th-90th percentile, then squashed | 30 / 22 | 19 / 16 | 22 / 14 | 15 / 10 |
| mask_hidden | MaskablePPO with the action mask switched off | 15 / - | 25 / - | 0 / - | 5 / - |
| pen_0 | DQN, no out-of-bounds penalty | - / 47 | - / 72 | - / 26 | - / 44 |
| pen_0.1 | DQN, a tenth of the penalty | - / 46 | - / 69 | - / 24 | - / 38 |
| pen_10 | DQN, ten times the penalty | - / 34 | - / 54 | - / 15 | - / 32 |

Readings (single-run observations, no significance test):

- Removing the wear cost is the only change that clearly helps PPO (71 to 79 realised, 38
  to 54 forecast).
- Lookahead matters: spot-only 14%, 1 h 41%, 12 h 67%, full day 71%.
- Clipping prices is the worst change after spot-only.
- More action levels hurt, and more levels hurt more (52, 61 and 51% against 71%).
- Time encodings and price history are neutral to negative.
- Hiding the mask costs most of PPO's capture (15%) and produces about 3 out-of-bounds
  moves per day.
- DQN matches PPO on high-volatility days in the reference (73 against 74) but trails far
  on all days (48 against 71). With no penalty it makes about 144 out-of-bounds moves per
  day on average.

## Per-region tables and job descriptions

The tables below give every job by region. The descriptions are copied from the manifests.

## What each job tested

- deg_flat: flat per-MWh wear at the Schade full-discharge rate
- deg_flat_dqn: flat per-MWh wear at the Schade full-discharge rate
- deg_full: Schade cycling plus calendar wear
- deg_full_dqn: Schade cycling plus calendar wear
- deg_none: no wear cost at all
- deg_none_dqn: no wear cost at all
- hist_1h: one hour of settled prices
- hist_1h_dqn: one hour of settled prices
- hist_24h: a day of settled prices (zero before the episode)
- hist_24h_dqn: a day of settled prices (zero before the episode)
- hist_4h: four hours of settled prices
- hist_4h_dqn: four hours of settled prices
- lev_17: seventeen power levels
- lev_17_dqn: seventeen power levels
- lev_5: five power levels
- lev_5_dqn: five power levels
- lev_9: nine power levels
- lev_9_dqn: nine power levels
- look_12h: twelve hours ahead
- look_12h_dqn: twelve hours ahead
- look_1h: one hour ahead
- look_1h_dqn: one hour ahead
- look_4h: four hours ahead
- look_4h_dqn: four hours ahead
- look_spot: spot price only
- look_spot_dqn: spot price only
- mask_hidden: MaskablePPO without the action mask
- pen_0.1_dqn: a tenth of the pinned penalty
- pen_0_dqn: no out-of-bounds penalty
- pen_10_dqn: ten times the pinned penalty
- px_clip: 10th to 90th percentile band then asinh
- px_clip_dqn: 10th to 90th percentile band then asinh
- px_knee100: asinh knee at 100
- px_knee1000_dqn: asinh knee at 1000
- px_knee100_dqn: asinh knee at 100
- px_linear: price over the cap
- px_linear_dqn: price over the cap
- shaping: buy-low/sell-high bonus; reward only
- shaping_dqn: buy-low/sell-high bonus; reward only
- time_none: no time of day
- time_none_dqn: no time of day
- time_sincos: sine and cosine of the time of day
- time_sincos_dqn: sine and cosine of the time of day


### realised prices, all days (finetune, captured % of oracle)

| run | NSW1 | QLD1 | SA1 | TAS1 | VIC1 | mean |
|---|---|---|---|---|---|---|
| REFERENCE_ppo | 76% | 76% | 72% | 61% | 71% | 71% |
| REFERENCE_dqn | 61% | 60% | 54% | 26% | 38% | 48% |
| deg_flat | 76% | 80% | 74% | 60% | 71% | 72% |
| deg_flat_dqn | 57% | 50% | 52% | 36% | 31% | 45% |
| deg_full | 74% | 79% | 70% | 64% | 71% | 72% |
| deg_full_dqn | 56% | 51% | 49% | 36% | 34% | 45% |
| deg_none | 82% | 85% | 77% | 74% | 80% | 79% |
| deg_none_dqn | 53% | 49% | 48% | 40% | 36% | 46% |
| hist_1h | 72% | 74% | 70% | 63% | 68% | 69% |
| hist_1h_dqn | 54% | 50% | 47% | 37% | 32% | 44% |
| hist_24h | 64% | 68% | 56% | 55% | 59% | 60% |
| hist_24h_dqn | 51% | 45% | 43% | 33% | 28% | 40% |
| hist_4h | 69% | 73% | 68% | 61% | 70% | 68% |
| hist_4h_dqn | 57% | 50% | 50% | 40% | 33% | 46% |
| lev_17 | 54% | 55% | 47% | 46% | 53% | 51% |
| lev_17_dqn | 42% | 33% | 35% | 26% | 23% | 32% |
| lev_5 | 55% | 53% | 51% | 50% | 53% | 52% |
| lev_5_dqn | 54% | 44% | 48% | 34% | 31% | 42% |
| lev_9 | 63% | 63% | 61% | 56% | 62% | 61% |
| lev_9_dqn | 47% | 37% | 41% | 29% | 26% | 36% |
| look_12h | 72% | 75% | 66% | 56% | 65% | 67% |
| look_12h_dqn | 53% | 48% | 49% | 39% | 34% | 45% |
| look_1h | 44% | 43% | 39% | 41% | 38% | 41% |
| look_1h_dqn | 49% | 41% | 40% | 35% | 27% | 38% |
| look_4h | 48% | 46% | 43% | 40% | 41% | 44% |
| look_4h_dqn | 53% | 45% | 46% | 35% | 31% | 42% |
| look_spot | 14% | 17% | 12% | 11% | 13% | 14% |
| look_spot_dqn | 13% | 10% | 12% | 7% | 8% | 10% |
| mask_hidden | 26% | 15% | 18% | 9% | 8% | 15% |
| pen_0.1_dqn | 55% | 51% | 52% | 36% | 36% | 46% |
| pen_0_dqn | 60% | 59% | 54% | 28% | 35% | 47% |
| pen_10_dqn | 46% | 35% | 40% | 24% | 23% | 34% |
| px_clip | 35% | 42% | 23% | 22% | 30% | 30% |
| px_clip_dqn | 26% | 29% | 17% | 17% | 23% | 22% |
| px_knee1000_dqn | 57% | 51% | 49% | 42% | 37% | 47% |
| px_knee100 | 73% | 74% | 62% | 51% | 65% | 65% |
| px_knee100_dqn | 54% | 50% | 49% | 32% | 34% | 44% |
| px_linear | 72% | 63% | 63% | 51% | 57% | 61% |
| px_linear_dqn | 61% | 52% | 54% | 39% | 38% | 49% |
| shaping | 76% | 83% | 76% | 58% | 71% | 73% |
| shaping_dqn | 65% | 63% | 58% | 42% | 47% | 55% |
| time_none | 66% | 69% | 61% | 49% | 61% | 61% |
| time_none_dqn | 57% | 49% | 49% | 42% | 34% | 46% |
| time_sincos | 65% | 69% | 63% | 51% | 58% | 61% |
| time_sincos_dqn | 51% | 43% | 45% | 34% | 26% | 40% |

### realised prices, high-volatility days (finetune, captured % of oracle)

| run | NSW1 | QLD1 | SA1 | TAS1 | VIC1 | mean |
|---|---|---|---|---|---|---|
| REFERENCE_ppo | 79% | 77% | 73% | 69% | 74% | 74% |
| REFERENCE_dqn | 78% | 80% | 73% | 67% | 68% | 73% |
| deg_flat | 77% | 81% | 75% | 71% | 67% | 74% |
| deg_flat_dqn | 73% | 72% | 70% | 64% | 59% | 68% |
| deg_full | 76% | 81% | 70% | 75% | 75% | 76% |
| deg_full_dqn | 73% | 74% | 67% | 67% | 65% | 69% |
| deg_none | 81% | 82% | 75% | 75% | 75% | 78% |
| deg_none_dqn | 71% | 72% | 66% | 70% | 62% | 68% |
| hist_1h | 74% | 73% | 71% | 72% | 68% | 72% |
| hist_1h_dqn | 70% | 72% | 64% | 64% | 58% | 66% |
| hist_24h | 65% | 68% | 55% | 63% | 57% | 61% |
| hist_24h_dqn | 67% | 66% | 60% | 58% | 53% | 61% |
| hist_4h | 70% | 73% | 69% | 69% | 75% | 71% |
| hist_4h_dqn | 73% | 71% | 67% | 68% | 55% | 67% |
| lev_17 | 54% | 50% | 45% | 55% | 52% | 51% |
| lev_17_dqn | 57% | 50% | 50% | 51% | 47% | 51% |
| lev_5 | 56% | 51% | 51% | 58% | 54% | 54% |
| lev_5_dqn | 71% | 64% | 66% | 61% | 58% | 64% |
| lev_9 | 62% | 58% | 62% | 66% | 62% | 62% |
| lev_9_dqn | 61% | 53% | 56% | 52% | 48% | 54% |
| look_12h | 75% | 77% | 68% | 64% | 66% | 70% |
| look_12h_dqn | 70% | 70% | 67% | 69% | 64% | 68% |
| look_1h | 46% | 42% | 38% | 48% | 35% | 42% |
| look_1h_dqn | 63% | 60% | 52% | 57% | 45% | 55% |
| look_4h | 53% | 50% | 45% | 50% | 41% | 48% |
| look_4h_dqn | 68% | 65% | 62% | 60% | 56% | 62% |
| look_spot | 10% | 9% | 6% | 7% | 1% | 7% |
| look_spot_dqn | 15% | 10% | 10% | 10% | 3% | 10% |
| mask_hidden | 35% | 23% | 25% | 20% | 20% | 25% |
| pen_0.1_dqn | 70% | 72% | 71% | 64% | 66% | 69% |
| pen_0_dqn | 76% | 77% | 72% | 72% | 65% | 72% |
| pen_10_dqn | 62% | 53% | 55% | 54% | 47% | 54% |
| px_clip | 30% | 33% | 14% | 12% | 6% | 19% |
| px_clip_dqn | 23% | 22% | 10% | 13% | 11% | 16% |
| px_knee1000_dqn | 74% | 73% | 67% | 71% | 68% | 71% |
| px_knee100 | 76% | 75% | 61% | 56% | 65% | 67% |
| px_knee100_dqn | 68% | 68% | 64% | 53% | 56% | 62% |
| px_linear | 78% | 64% | 68% | 66% | 65% | 68% |
| px_linear_dqn | 80% | 76% | 75% | 75% | 71% | 75% |
| shaping | 79% | 86% | 77% | 80% | 72% | 79% |
| shaping_dqn | 77% | 78% | 72% | 66% | 67% | 72% |
| time_none | 67% | 70% | 61% | 60% | 65% | 65% |
| time_none_dqn | 74% | 71% | 67% | 70% | 62% | 69% |
| time_sincos | 67% | 71% | 64% | 63% | 58% | 65% |
| time_sincos_dqn | 68% | 66% | 64% | 59% | 53% | 62% |

### forecast prices, all days (finetune, captured % of oracle)

| run | NSW1 | QLD1 | SA1 | TAS1 | VIC1 | mean |
|---|---|---|---|---|---|---|
| REFERENCE_ppo | 46% | 52% | 39% | 13% | 40% | 38% |
| REFERENCE_dqn | 38% | 43% | 30% | -4% | 22% | 26% |
| deg_flat | 46% | 54% | 42% | 6% | 44% | 38% |
| deg_flat_dqn | 30% | 31% | 23% | -5% | 14% | 19% |
| deg_full | 47% | 55% | 34% | 14% | 43% | 39% |
| deg_full_dqn | 30% | 30% | 26% | 2% | 18% | 21% |
| deg_none | 58% | 63% | 51% | 40% | 59% | 54% |
| deg_none_dqn | 34% | 36% | 27% | 16% | 23% | 27% |
| hist_1h | 47% | 51% | 35% | 20% | 44% | 39% |
| hist_1h_dqn | 29% | 32% | 23% | 3% | 15% | 20% |
| hist_24h | 43% | 50% | 30% | 14% | 37% | 35% |
| hist_24h_dqn | 31% | 32% | 24% | 6% | 19% | 22% |
| hist_4h | 43% | 50% | 37% | 19% | 41% | 38% |
| hist_4h_dqn | 30% | 33% | 25% | 5% | 19% | 22% |
| lev_17 | 32% | 35% | 25% | 8% | 31% | 26% |
| lev_17_dqn | 23% | 21% | 17% | 2% | 14% | 15% |
| lev_5 | 33% | 33% | 27% | 12% | 32% | 27% |
| lev_5_dqn | 28% | 30% | 24% | 3% | 15% | 20% |
| lev_9 | 41% | 41% | 34% | 14% | 38% | 34% |
| lev_9_dqn | 26% | 27% | 21% | 4% | 17% | 19% |
| look_12h | 47% | 51% | 39% | 15% | 40% | 39% |
| look_12h_dqn | 28% | 29% | 23% | 5% | 17% | 21% |
| look_1h | 18% | 25% | 17% | 5% | 15% | 16% |
| look_1h_dqn | 15% | 12% | 15% | 2% | 8% | 10% |
| look_4h | 31% | 37% | 22% | 6% | 24% | 24% |
| look_4h_dqn | 25% | 22% | 20% | 5% | 15% | 17% |
| look_spot | 10% | 17% | 8% | 3% | 9% | 10% |
| look_spot_dqn | 6% | 7% | 7% | -1% | 4% | 5% |
| mask_hidden | 2% | 0% | 5% | -4% | -2% | 0% |
| pen_0.1_dqn | 31% | 32% | 30% | 5% | 21% | 24% |
| pen_0_dqn | 37% | 41% | 33% | -7% | 28% | 26% |
| pen_10_dqn | 25% | 21% | 20% | -3% | 13% | 15% |
| px_clip | 25% | 38% | 18% | 9% | 22% | 22% |
| px_clip_dqn | 18% | 25% | 11% | 3% | 13% | 14% |
| px_knee1000_dqn | 29% | 30% | 22% | 5% | 19% | 21% |
| px_knee100 | 45% | 53% | 37% | 15% | 40% | 38% |
| px_knee100_dqn | 30% | 35% | 26% | 6% | 18% | 23% |
| px_linear | 48% | 47% | 39% | 11% | 39% | 37% |
| px_linear_dqn | 30% | 33% | 23% | 3% | 20% | 22% |
| shaping | 47% | 58% | 42% | 10% | 45% | 40% |
| shaping_dqn | 37% | 40% | 30% | 5% | 29% | 28% |
| time_none | 43% | 47% | 36% | 14% | 39% | 36% |
| time_none_dqn | 30% | 29% | 25% | 6% | 18% | 22% |
| time_sincos | 40% | 48% | 36% | 13% | 37% | 35% |
| time_sincos_dqn | 28% | 25% | 19% | 5% | 14% | 18% |

### forecast prices, high-volatility days (finetune, captured % of oracle)

| run | NSW1 | QLD1 | SA1 | TAS1 | VIC1 | mean |
|---|---|---|---|---|---|---|
| REFERENCE_ppo | 48% | 54% | 39% | 16% | 38% | 39% |
| REFERENCE_dqn | 48% | 56% | 41% | 27% | 40% | 42% |
| deg_flat | 47% | 57% | 44% | 26% | 46% | 44% |
| deg_flat_dqn | 40% | 46% | 34% | 18% | 32% | 34% |
| deg_full | 49% | 60% | 34% | 20% | 45% | 42% |
| deg_full_dqn | 40% | 45% | 37% | 21% | 38% | 36% |
| deg_none | 55% | 58% | 46% | 32% | 48% | 48% |
| deg_none_dqn | 41% | 47% | 34% | 24% | 31% | 35% |
| hist_1h | 49% | 52% | 33% | 27% | 45% | 41% |
| hist_1h_dqn | 37% | 46% | 31% | 14% | 27% | 31% |
| hist_24h | 43% | 51% | 29% | 17% | 33% | 35% |
| hist_24h_dqn | 41% | 47% | 33% | 17% | 36% | 35% |
| hist_4h | 44% | 50% | 38% | 27% | 40% | 40% |
| hist_4h_dqn | 38% | 46% | 34% | 19% | 34% | 34% |
| lev_17 | 32% | 32% | 23% | 14% | 24% | 25% |
| lev_17_dqn | 32% | 32% | 25% | 16% | 31% | 27% |
| lev_5 | 33% | 31% | 27% | 17% | 33% | 28% |
| lev_5_dqn | 37% | 44% | 34% | 17% | 31% | 33% |
| lev_9 | 41% | 37% | 34% | 20% | 36% | 34% |
| lev_9_dqn | 35% | 39% | 30% | 19% | 33% | 31% |
| look_12h | 50% | 54% | 41% | 18% | 40% | 40% |
| look_12h_dqn | 37% | 43% | 33% | 21% | 32% | 33% |
| look_1h | 19% | 27% | 18% | 7% | 13% | 17% |
| look_1h_dqn | 22% | 19% | 21% | 11% | 16% | 18% |
| look_4h | 34% | 43% | 22% | 8% | 22% | 26% |
| look_4h_dqn | 32% | 32% | 28% | 18% | 29% | 28% |
| look_spot | 8% | 11% | 4% | 3% | 1% | 6% |
| look_spot_dqn | 6% | 7% | 6% | 2% | 1% | 5% |
| mask_hidden | 5% | 3% | 10% | 4% | 4% | 5% |
| pen_0.1_dqn | 40% | 46% | 41% | 21% | 40% | 38% |
| pen_0_dqn | 46% | 53% | 45% | 22% | 53% | 44% |
| pen_10_dqn | 36% | 37% | 30% | 20% | 35% | 32% |
| px_clip | 22% | 32% | 11% | 7% | 4% | 15% |
| px_clip_dqn | 16% | 20% | 6% | 3% | 2% | 10% |
| px_knee1000_dqn | 38% | 43% | 29% | 15% | 34% | 32% |
| px_knee100 | 47% | 56% | 37% | 17% | 37% | 39% |
| px_knee100_dqn | 38% | 48% | 34% | 20% | 31% | 34% |
| px_linear | 51% | 49% | 43% | 19% | 43% | 41% |
| px_linear_dqn | 39% | 46% | 32% | 21% | 37% | 35% |
| shaping | 50% | 63% | 43% | 29% | 47% | 46% |
| shaping_dqn | 44% | 48% | 36% | 18% | 41% | 38% |
| time_none | 44% | 47% | 35% | 16% | 39% | 36% |
| time_none_dqn | 40% | 43% | 34% | 20% | 34% | 34% |
| time_sincos | 41% | 48% | 36% | 15% | 35% | 35% |
| time_sincos_dqn | 38% | 39% | 28% | 16% | 31% | 30% |

## Appendix: the script

```python
import glob, re, sys, pandas as pd
sys.path.insert(0, '.')
from backtest import metrics
R = ['NSW1','QLD1','SA1','TAS1','VIC1']
rows = []
ref = {'ppo':'rolling2023_ppo_pooled_regions_daily_1_cycle_end_empty_5min_forecast',
       'dqn':'rolling2023_dqn_pooled_regions_daily_1_cycle_end_empty_5min_forecast'}
runs = {}
for a,p in ref.items(): runs[f'REFERENCE_{a}'] = p
for f in sorted(glob.glob('results/backtestResults/rolling2023_ablate_*_NSW1.csv')):
    p = re.sub(r'_NSW1\.csv$','',f.split('/')[-1]); runs[p.replace('rolling2023_ablate_','')] = p
for name, p in runs.items():
    for r in R:
        d = pd.read_csv(f'results/backtestResults/{p}_{r}.csv')
        for (algo, tm, ps), g in d.groupby(['algo','trainMode','priceSource']):
            for scope, h in (('all', g), ('highVol', g[g.highVol])):
                s = metrics.summarise(h)
                per = [metrics.summarise(h[h.seed==k])['captured'] for k in sorted(h.seed.unique())]
                rows.append(dict(run=name, algo=algo, trainMode=tm, priceSource=ps, region=r, scope=scope,
                    days=s['days'], agent=s['agentProfit'], oracle=s['oracleProfit'], captured=s['captured'],
                    seedMin=min(per), seedMax=max(per), seeds=len(per),
                    oob=(h.outOfBounds.mean() if 'outOfBounds' in h else float('nan')), cycles=(h.agentCycles.mean() if 'agentCycles' in h else float('nan'))))
out = pd.DataFrame(rows); out.to_csv(sys.argv[1], index=False); print(out.shape)
```

Run as `uv run python summ.py ablation_summary.csv` from the repo root. It takes a few minutes.

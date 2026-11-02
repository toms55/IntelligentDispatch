# MDP ablation study: candidate options

This is the study Gap 2 of `SLR.tex` calls for: hold data, tuning budget and environment
constant, vary one MDP design element at a time. Only Sage et al. (2024) and Sage & Zhao
(2025) do this in the corpus, on hourly Alberta/German prices with an oracle that is not a
true bound. This repo has the missing ingredients: 5-minute post-2021 NEM prices, an exact
full-horizon discrete oracle (`optimisation/dp.py`), 5 seeds, and a strict chronological
split (train 2023-24, test 2025). Say so in the thesis.

Baseline for every row: PPO, daily episode, 1 EFC budget, end-empty terminal value,
start SoC 0, 5-minute forecast refresh, pinned hyperparameters in `rl/algos.py`.
Report the spread over 5 seeds, never the best seed.

Two kinds of ablation, because the oracle must keep bounding the agent:

- **Agent-side** (observation, reward, algorithm): the problem is unchanged, so the
  existing oracle corpus applies and results compare directly.
- **Problem-side** (budget, terminal value, start SoC, degradation, action set): the oracle
  must be re-solved under the same change, or the capture ratio means nothing
  (`TestMatchesTheBudgetedOracle` must still hold).

## Ten options, each tied to a claim in the review

| # | Element | Variants | Kind | Where | Review hook |
|---|---------|----------|------|-------|-------------|
| 1 | **Degradation in the reward** | full Schade / cycling only (no calendar) / flat $/MWh throughput / none | problem + reward | `BessEnv(calendar=...)`, `degradation/schade.py` | Neves (13%->4% of profit) and Kortmann (less cycling, *lower* profit) disagree on whether pricing degradation pays. 10 studies omit it, 10 use a flat rate. Headline result. |
| 2 | **Lookahead / forecast-augmented observation** | 1 h / 4 h / 12 h / full 288 intervals; also price slots zeroed (SoC + clock only) | agent | `config.observation_intervals`, `rl/features.py` | Sage: multi-horizon forecasts lift reward up to 60%. 8 of 29 use forecasts, none isolates them across markets. |
| 3 | **Time encoding** | trading-day clock slot (now) / sin-cos hour / none / plus weekday-season (`day0`, already run) | agent | `rl/features.py` | Sage & Zhao: cyclic time encodings shift reward up to 51%. 12 of 29 use temporal features. |
| 4 | **Price history / observation stacking** | none / 1 h / 24 h (`hist0`, already run, worse) | agent | `rl/features.py` | Sage & Zhao: stacking shifts reward. Part of this exists; the rerun makes the arm comparable to the rest of the grid. |
| 5 | **Action discretisation** | 3 (now) / 5 / 9 / 17 power levels | problem | `rl/env.py`, `optimisation/dp.py` | Sage & Zhao compared 3 vs 17. 10 of 29 are discrete, 13 continuous, and the choice is tied to the algorithm. Needs the DP to take the same level set; the state is still one number, so it is exact. |
| 6 | **Cycle budget** | none / 0.5 / 1 / 2 / 3 EFC | problem | `config.cycle_budget` | Karimi Madahi caps daily cycles in one variant; no reviewed study prices a warranty. Also tests whether the budget-left slot is used. |
| 7 | **Terminal value** | 0 / 0.5x / 1x median / 2x | problem | `config.terminal_energy_price` | Sun penalises terminal SoC; Neves uses a day-end SoC continuity penalty. Tests dumping vs hoarding. |
| 8 | **Reward shaping and constraint penalty** | out-of-bounds penalty 0 / 0.1 / 1 / 10; PPO masked vs unmasked; plus a buy-low/sell-high bonus (Li 2024, Zhang 2023) | agent | `config.invalid_action_penalty`, `rl/algos.py` | 11 of 29 add a violation penalty. Sage & Zhao found wrappers had little effect and sometimes hurt. What does the mask buy? (`backtest.outofbounds` already counts it.) |
| 9 | **Price tails in the observation** | asinh knee 100 / 300 / 1000 / linear / clipped at the 10th-90th percentile | agent | `config.price_knee`, scaler | Kortmann clip to the 10th-90th percentile and the review says that removes the spikes arbitrage lives on. `price_knee` is marked NOT TUNED, so this also justifies 300. Pick the knee on validation, never on 2025. |
| 10 | **Start state of charge** | 0 / 0.5 / random per episode | problem | `config.initial_soc` | Start state hands the agent unbought energy; this measures how much of the gap it explains. The 0.5 corpus exists (`oracleResults/dp`). |

Optional hyperparameter sensitivity, lower priority: reward scale (5 / 20 / 80) and gamma
(1.0 / 0.999 / 0.99; 0.995 already regressed DQN).

## Already covered by published runs (do not repeat)

Daily vs weekly episode, realised vs forecast price at test, forecast refresh (4 h vs
5 min), trained-on-forecast agent (`fctrain`), 24 h history and weekday/season
(`hist0`, `day0`), pooled vs per-region. Rows 3 and 4 reuse those arms.

## How to read the results

Report each variant as median capture of the oracle over 5 seeds with the interquartile
range, on both price sources. Rank elements by effect size, which is the review's own
claim to test: "design choices move reward more than algorithm choice". The algorithm arm
(PPO vs DQN) is already published and gives the comparison scale.

## Suggested order and cost

1. Agent-side first (2, 3, 4, 8, 9): no new oracle.
2. Then 6, 7, 10 via `dp.solve(cycles=..., terminalPrice=...)`, seconds per window. Solve
   into a scratch folder and pack into `oracleResults/dp_<variant>`.
3. 1 needs a calendar-off and a flat-cost oracle; 5 needs the DP generalised to N levels.

Each variant needs its own `--out` and a descriptive run name (e.g.
`rolling2023_ppo_ablate_lookahead_1h`), because `trainingManifestOf` pins what a
checkpoint was trained on. About 5 regions x 5 seeds x 52 folds per variant.

## Open decisions

- Regions: all five, or NSW1 + TAS1 only (best and worst performers)?
- PPO only, or PPO and DQN?
- Which of the ten to run (all ten is roughly 35 variant-runs).

## Decided scope (user, 2026-11-02)

Elements 1-5, 8, 9. All five regions. PPO and DQN. Every variant runs as the
pooled-regions agent (`backtest/pooled.py`, `scripts/pooled.pbs`): one shared agent per
seed over all regions, then a per-region fine-tune. Reference arm is the published
`rolling2023_{ppo,dqn}_pooled_regions_daily_1_cycle_end_empty_5min_forecast`.

Variant count (non-baseline): 1 degradation 3, 2 lookahead 4, 3 time encoding 3,
4 history 2, 5 action levels 3, 8 penalty/mask/shaping 5, 9 price tails 4 = 24.
24 variants x 2 algorithms x 5 seeds = 240 seed-jobs (each up to 48 h on medq), plus 48
merge jobs. Code still to write: flags for 1, 2, 3, 4, 5, 9 (none exist today), and new
oracle corpora for 1 and 5.

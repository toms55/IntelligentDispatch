# MDP ablation study: candidate options

Baseline for every row: PPO, daily episode, 1 EFC budget, end-empty terminal value,
start SoC 0, 5-minute forecast refresh, pinned hyperparameters in `rl/algos.py`.
One factor moves per run. Report the spread over 5 seeds, not the best seed.

Two kinds of ablation, because the oracle must keep bounding the agent:

- **Agent-side** (observation, reward, algorithm): the problem is unchanged, so the
  existing oracle corpus still applies and results are directly comparable.
- **Problem-side** (budget, terminal value, start SoC, degradation): the oracle must be
  re-solved under the same change, or the capture ratio means nothing
  (`TestMatchesTheBudgetedOracle` must still hold).

| # | Element | Variants | Kind | Where it lives | Question |
|---|---------|----------|------|----------------|----------|
| 1 | Cycle budget | none / 0.5 / 1 / 2 / 3 EFC | problem | `config.cycle_budget` | How much of the agent's profit does a warranty cost, and does the budget slot in the observation help it respect one? |
| 2 | Terminal value | 0 / 0.5x / 1x (median) / 2x | problem | `config.terminal_energy_price` | Does valuing closing stock stop end-of-day dumping, and when does it cause hoarding? |
| 3 | Start SoC | 0 / 0.5 / uniform random per episode | problem | `config.initial_soc` | How much of the gap is the start state rather than policy? (0.5 corpus exists in `oracleResults/dp`) |
| 4 | Degradation in the reward | full Schade / cycling only (no calendar) / none | problem + reward | `BessEnv(calendar=...)`, `degradation/schade.py` | Does the agent need to see wear to trade sensibly, or does the budget already carry it? |
| 5 | Lookahead length | 1 h (12 slots) / 4 h / 12 h / full 288 | agent | `config.observation_intervals`, `rl/features.py` | How far ahead does the agent actually use the price path? |
| 6 | Observation slots | drop each of: clock, cycles-left, episode-left; drop 30-min buckets | agent | `rl/features.py` | Which state features carry the policy? |
| 7 | Price transform | asinh knee 100 / 300 / 1000 / linear clip | agent | `config.price_knee`, scaler | `price_knee` is marked NOT TUNED, so this either justifies 300 or replaces it. Choose it on validation, never on 2025. |
| 8 | Reward scale | 5 / 20 / 80 | agent | `config.reward_scale` | Sensitivity of PPO and DQN to reward magnitude. |
| 9 | Out-of-bounds penalty and masking | penalty 0 / 0.1 / 1 / 10, plus PPO masked vs unmasked | agent | `config.invalid_action_penalty`, `rl/algos.py` | What does the mask buy, and how much of DQN's deficit is out-of-bounds moves? (`backtest.outofbounds` already counts them.) |
| 10 | Discount | gamma 1.0 / 0.999 / 0.99 | agent | `config.gamma`, `rl/algos.py` | Does a shorter horizon of credit help on a 288-step episode? (gamma 0.995 already regressed for DQN, so run it for PPO and 0.99 / 0.999.) |

## Already covered by published runs (do not repeat)

Daily vs weekly episode (`rolling-efc7`), realised vs forecast price source at test,
forecast refresh interval (4 h vs 5 min), trained-on-forecast agent (`fctrain`), 24 h price
history and weekday/season features (`hist0`, `day0`, both worse), pooled vs per-region.

## Suggested order and cost

1. Agent-side first (5-10): no new oracle, so cheapest, and they reuse `oracleResults`.
2. Then 1-3 using `dp.solve(cycles=..., terminalPrice=...)`, which takes seconds per
   window. Solve into a scratch folder and pack into a new `oracleResults/dp_<variant>`.
3. 4 needs a calendar-off oracle (`--no-calendar` cache key already exists).

Each variant needs its own `--out` and descriptive run name (e.g.
`rolling2023_ppo_ablate_lookahead_1h`), because `trainingManifestOf` pins what a
checkpoint was trained on. Roughly 5 regions x 5 seeds x 52 folds per variant, so choose
a subset of regions (suggest NSW1 and TAS1, the best and worst performers) before
launching the full grid.

## Open decisions

- Regions: all five, or NSW1 + TAS1 only?
- PPO only, or PPO and DQN?
- Which of 1-10 to run (all ten is ~35 variant-runs).

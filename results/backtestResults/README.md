# Published backtests

Written by `python -m backtest.publish`; do not edit by hand. Each
`<run>_<region>.csv` has a `<run>_<region>.json` beside it holding the run's full
manifest (episode, budget, end-empty, refresh, timesteps, pooling) and where its
raw results live.

| run | regions | what it tested |
|---|---|---|
| `rolling2023_daily_1_cycle_end_empty` | NSW1 QLD1 SA1 TAS1 VIC1 | MaskablePPO, daily episode, budget of 1 equivalent full cycle per day, battery forced empty by the close (--end-empty), terminal price 0, 5-minute forecast refresh, 30k warm / 300k cold steps. The deployment setup that later runs compare against. |
| `rolling2023_dqn_daily_1_cycle_end_empty` | NSW1 QLD1 SA1 TAS1 VIC1 | DQN (unmasked) on the daily 1-cycle end-empty setup of rolling2023_daily_1_cycle_end_empty. |
| `rolling2023_dqn_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_dqn_weekly_7_cycle_budget` | NSW1 QLD1 SA1 TAS1 VIC1 | DQN (unmasked) on the weekly 7-cycle setup of rolling2023_weekly_7_cycle_budget. |
| `rolling2023_fc4h` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_fctrain` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_masked_dqn_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | DQN masked at action selection only (gamma 1.0) on the start-at-0-SoC setup. The mask is the only change against rolling2023_dqn_start_at_0_SoC_five_min_refresh. |
| `rolling2023_pooled_daily_1_cycle_end_empty` | NSW1 QLD1 SA1 TAS1 VIC1 | One MaskablePPO agent trained on all five regions end to end (150k warm / 1.5M cold steps per fold), then a fresh 30k-step fine-tune per region each fold. trainMode pooled is the shared agent, finetune the regional copy. Daily 1-cycle end-empty setup: compare with rolling2023_daily_1_cycle_end_empty. |
| `rolling2023_pooled_masked_dqn_daily_1_cycle_end_empty` | NSW1 QLD1 SA1 TAS1 VIC1 | The pooled setup of rolling2023_pooled_daily_1_cycle_end_empty with DQN masked at action selection only. |
| `rolling2023_price_history_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_price_scale_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_cold_fc4h` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc4h` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc5m` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc5m_allseeds` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc5m_allseeds_soc0` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_start_at_0_SoC_five_min_refresh_rerun` | NSW1 QLD1 SA1 TAS1 VIC1 | MaskablePPO rerun of rolling2023_start_at_0_SoC_five_min_refresh (daily, starts empty, no cycle budget, terminal price 0, 5-minute refresh) on the current code, as the PPO side of the masked-DQN comparison. |
| `rolling2023_weekday_season_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_weekly_7_cycle_budget` | NSW1 QLD1 SA1 TAS1 VIC1 | MaskablePPO, weekly episode (2016 intervals) with a budget of 7 equivalent full cycles per week and the region's terminal price on closing stock; forecast refreshed every 4 h; 30k warm / 300k cold steps. Tests weekly against daily episodes: compare with rolling2023_daily_1_cycle_end_empty. |

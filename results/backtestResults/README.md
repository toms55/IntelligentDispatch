# Published backtests

Written by `python -m backtest.publish`; do not edit by hand. Each
`<run>_<region>.csv` has a `<run>_<region>.json` beside it holding the run's full
manifest (episode, budget, end-empty, refresh, timesteps, pooling) and where its
raw results live.

| run | regions | what it tested |
|---|---|---|
| `rolling2023_ablate_deg_flat` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation deg_flat: flat per-MWh wear at the Schade full-discharge rate. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_deg_flat_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation deg_flat: flat per-MWh wear at the Schade full-discharge rate. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_deg_full` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation deg_full: Schade cycling plus calendar wear. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_deg_full_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation deg_full: Schade cycling plus calendar wear. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_deg_none` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation deg_none: no wear cost at all. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_deg_none_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation deg_none: no wear cost at all. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_hist_1h` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation hist_1h: one hour of settled prices. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_hist_1h_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation hist_1h: one hour of settled prices. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_hist_24h` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation hist_24h: a day of settled prices (zero before the episode). Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_hist_24h_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation hist_24h: a day of settled prices (zero before the episode). Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_hist_4h` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation hist_4h: four hours of settled prices. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_hist_4h_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation hist_4h: four hours of settled prices. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_lev_17` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation lev_17: seventeen power levels. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_lev_17_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation lev_17: seventeen power levels. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_lev_5` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation lev_5: five power levels. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_lev_5_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation lev_5: five power levels. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_lev_9` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation lev_9: nine power levels. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_lev_9_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation lev_9: nine power levels. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_12h` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_12h: twelve hours ahead. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_12h_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_12h: twelve hours ahead. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_1h` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_1h: one hour ahead. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_1h_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_1h: one hour ahead. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_4h` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_4h: four hours ahead. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_4h_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_4h: four hours ahead. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_spot` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_spot: spot price only. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_look_spot_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation look_spot: spot price only. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_mask_hidden` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation mask_hidden: MaskablePPO without the action mask. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_pen_0.1_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation pen_0.1: a tenth of the pinned penalty. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_pen_0_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation pen_0: no out-of-bounds penalty. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_pen_10_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation pen_10: ten times the pinned penalty. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_clip` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_clip: 10th to 90th percentile band then asinh. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_clip_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_clip: 10th to 90th percentile band then asinh. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_knee1000_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_knee1000: asinh knee at 1000. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_knee100` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_knee100: asinh knee at 100. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_knee100_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_knee100: asinh knee at 100. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_linear` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_linear: price over the cap. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_px_linear_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation px_linear: price over the cap. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_shaping` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation shaping: buy-low/sell-high bonus; reward only. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_shaping_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation shaping: buy-low/sell-high bonus; reward only. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_time_none` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation time_none: no time of day. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_time_none_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation time_none: no time of day. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_time_sincos` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation time_sincos: sine and cosine of the time of day. Pooled-regions maskableppo; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_ablate_time_sincos_dqn` | NSW1 QLD1 SA1 TAS1 VIC1 | MDP ablation time_sincos: sine and cosine of the time of day. Pooled-regions dqn; daily; 1 EFC; end empty; otherwise the reference run. |
| `rolling2023_dqn_daily_1_cycle_end_empty_5min_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | DQN (unmasked) on the daily 1-cycle end-empty setup of rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast. |
| `rolling2023_dqn_pooled_regions_daily_1_cycle_end_empty_5min_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | The pooled setup of rolling2023_ppo_pooled_regions_daily_1_cycle_end_empty_5min_forecast with plain DQN and no action mask (algo dqnnomask), so the mask is the only change against rolling2023_masked_dqn_pooled_regions_daily_1_cycle_end_empty_5min_forecast. |
| `rolling2023_dqn_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_dqn_weekly_7_cycles_terminal_value_4h_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | DQN (unmasked) on the weekly 7-cycle setup of rolling2023_ppo_weekly_7_cycles_terminal_value_4h_forecast. |
| `rolling2023_fc4h` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_fctrain` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_masked_dqn_daily_no_cycle_limit_start_empty_5min_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | DQN masked at action selection only (gamma 1.0) on the start-at-0-SoC setup. The mask is the only change against rolling2023_dqn_start_at_0_SoC_five_min_refresh. |
| `rolling2023_masked_dqn_pooled_regions_daily_1_cycle_end_empty_5min_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | The pooled setup of rolling2023_ppo_pooled_regions_daily_1_cycle_end_empty_5min_forecast with DQN masked at action selection only. |
| `rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | MaskablePPO, daily episode, budget of 1 equivalent full cycle per day, battery forced empty by the close (--end-empty), terminal price 0, 5-minute forecast refresh, 30k warm / 300k cold steps. The deployment setup that later runs compare against. |
| `rolling2023_ppo_daily_no_cycle_limit_start_empty_5min_forecast_rerun` | NSW1 QLD1 SA1 TAS1 VIC1 | MaskablePPO rerun of rolling2023_start_at_0_SoC_five_min_refresh (daily, starts empty, no cycle budget, terminal price 0, 5-minute refresh) on the current code, as the PPO side of the masked-DQN comparison. |
| `rolling2023_ppo_pooled_regions_daily_1_cycle_end_empty_5min_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | One MaskablePPO agent trained on all five regions end to end (150k warm / 1.5M cold steps per fold), then a fresh 30k-step fine-tune per region each fold. trainMode pooled is the shared agent, finetune the regional copy. Daily 1-cycle end-empty setup: compare with rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast. |
| `rolling2023_ppo_weekly_7_cycles_terminal_value_4h_forecast` | NSW1 QLD1 SA1 TAS1 VIC1 | MaskablePPO, weekly episode (2016 intervals) with a budget of 7 equivalent full cycles per week and the region's terminal price on closing stock; forecast refreshed every 4 h; 30k warm / 300k cold steps. Tests weekly against daily episodes: compare with rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast. |
| `rolling2023_price_history_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_price_scale_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_cold_fc4h` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc4h` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc5m` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc5m_allseeds` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_profit_fc5m_allseeds_soc0` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |
| `rolling2023_weekday_season_start_at_0_SoC_five_min_refresh` | NSW1 QLD1 SA1 TAS1 VIC1 | no description recorded |

## Realised prices
- region: NSW1
- window: 2025-01-01T04:00:00 .. 2026-01-01T04:00:00
- algo: maskableppo
- seeds: 0, 1, 2, 3, 4
- battery: 0.06 MW / 0.1 MWh, replacement cost 150000
- price slots from: realised prices
- generated: 2026-10-09T01:57:23

| metric | value |
|---|---|
| oracle profit | 15439.7117 |
| agent profit | 1811.3827 +/- 2780.0732 |
| regret | 13628.3289 +/- 2780.0732 |
| captured | 11.73% +/- 18.01% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | -1616.6409 | 17056.3526 | -10.47% |
| 1 | 2799.1074 | 12640.6043 | 18.13% |
| 2 | 5908.3681 | 9531.3435 | 38.27% |
| 3 | 859.1703 | 14580.5414 | 5.56% |
| 4 | 1106.9088 | 14332.8029 | 7.17% |

## AEMO forecast
- region: NSW1
- window: 2025-01-01T04:00:00 .. 2026-01-01T04:00:00
- algo: maskableppo
- seeds: 0, 1, 2, 3, 4
- battery: 0.06 MW / 0.1 MWh, replacement cost 150000
- price slots from: AEMO forecast
- generated: 2026-10-09T02:23:31

| metric | value |
|---|---|
| oracle profit | 15439.7117 |
| agent profit | -5692.8550 +/- 12235.4069 |
| regret | 21132.5667 +/- 12235.4069 |
| captured | -36.87% +/- 79.25% |

| seed | agent profit | regret | captured |
|---|---|---|---|
| 0 | -27514.3834 | 42954.0951 | -178.21% |
| 1 | -1773.2686 | 17212.9803 | -11.49% |
| 2 | 0.0000 | 15439.7117 | 0.00% |
| 3 | 823.3769 | 14616.3348 | 5.33% |
| 4 | 0.0000 | 15439.7117 | 0.00% |


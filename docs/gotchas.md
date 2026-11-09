# Gotchas

Each item below cost measured time to find. Read this before changing the
oracle, the environment, the observation layout or the publish path.

- Energy is tracked on two sides of the meter in `lp.py`: `stored` is cell-side,
  `bought`/`sold` are meter-side. Charging loses `nu_C` in, discharging loses `nu_D`
  out. Mixing them up silently changes the profit figure.
- The oracle is only a valid bound on the agent if both use the same episode horizon,
  the same SoC reset rule, and the same degradation terms. One window is one episode,
  and a window that reads past its own last interval breaks the bound, because the
  agent cannot read there either.
- `--no-calendar` drops the Q^CAL term and is much faster (19.2s vs 44.2s on one
  continuous day), but the degradation it reports then counts cycling only — fine for
  iterating, wrong for any number that gets quoted. The term is worth about 1.4% of
  profit.
- **Always pass `threads=1` when solving windows in parallel.** HiGHS otherwise takes
  every core in every worker, and the oversubscription costs an order of magnitude: four
  unrestricted workers ran at about 8 minutes a window against 44 seconds for one solve
  alone. `buildBaseline` sets it automatically when `workers > 1`. This applies to the
  continuous windows only. A discrete window goes to `optimisation/dp.py`, which is
  single-threaded numpy and takes milliseconds, so `--workers` buys it nothing.
- **Quote `Schedule.dualBound`, never `Schedule.profit`, as the bound.** A gap-limited or
  time-limited solve returns an incumbent that can sit *below* the true optimum, and a
  bound that is too low understates the agent's shortfall. Measured on SA1 2024-01-10
  discrete: 1555s to prove 21.8041, or 82s to return 21.7269 with a dual bound of
  21.8353. Note HiGHS minimises internally, so pulp negates a maximisation and the raw
  `mip_dual_bound` comes back with the opposite sign — `_solverInfo` flips it.
  A discrete window no longer has this problem: `dp.py` is exact and sets `dualBound`
  equal to `profit` and `mipGap` to 0. Quoting `dualBound` stays the rule, because it
  is right in both modes.
- **An episode starts the battery empty, and that is a choice, not an accident.**
  `config.initial_soc` is 0.0. A start above zero hands the agent energy it never bought,
  and because nothing values what the battery holds at the close, it then sells that
  energy at whatever the last interval pays. Starting empty removes both: the oracle only
  buys what it can sell well, and the agent is paid the same way, so a day's profit is
  the trading it did. `cacheKey` records the initial SoC, so every `data_cache/lp_baseline`
  entry solved at 0.5 no longer matches and the oracle re-solves. The forecast screen is
  separate and still opens at 50% (`live.defaultSoc`), because it reports a real battery
  part way through a day.
- **`oracleResults/dp` is solved at a 0.5 start, which `config.initial_soc` is not any
  more.** The corpus bounds the published `rolling2023` runs, which were scored at 0.5,
  and `--restamp-oracle` wrote those bounds into `results/backtestResults/`. A run at
  today's 0.0 start misses every record in it and solves its own, which the DP does in
  milliseconds. The 0.0-start corpus is `oracleResults/dp_soc0`, and `findRecord` reads
  it too. To add solves to a set, solve into a scratch folder and pack it in:
  `cli baseline --split ... --no-calendar --gap 0.02 --cache-dir /tmp/solves`, then
  `python -m rl.oracleArchive pack /tmp/solves oracleResults/dp_soc0`.
- **`cacheKey` records the cycle budget and the terminal price, so every oracle record
  from before them is re-solved.** A budgeted window is a different problem, and serving
  an unbudgeted record for it would hand the agent a bound it cannot reach. The existing
  `data_cache/lp_baseline` and `oracleResults/` records still match an unbudgeted
  request (both default `None`/`0.0`), so nothing already solved is clobbered.
  `baseline.solveWindow` also refuses to cache a record whose `cyclesUsed` exceeds the
  budget, because a cached record is served for good.
- **`cacheKey` does not record the solver time limit.** A window that stops on the limit
  writes a loose dual bound, and `solveWindow` then serves that record from cache for
  good. Raising `--time-limit` later cannot reach it, so delete the files to redo them.
  Measured on NSW1 2024-01-10, discrete, no calendar, 0.02 gap: a 300s limit stopped with
  the incumbent at 14.6996 and the bound still at 17.7368, above the continuous bound of
  17.1258 and so useless as a bound. This is why the rolling backtest uses 1800s.
  This now bites the continuous windows only. `cacheKey` records which solver wrote a
  record, so the discrete entries from before `dp.py` are not served, and the DP cannot
  write a loose bound in the first place.
- **The reward is raw profit; no oracle baseline is subtracted.** The oracle is only the
  bound the agent is scored against, so `train` solves nothing, and the rolling backtest
  needs its discrete oracle cache over the test days only (from the first fold's test
  start). The rolling manifest carries `reward: profit`, so a result set from the old
  baselined reward refuses to resume under it. The existing `data_cache/lp_baseline`
  entries are continuous, so none of them match the rolling oracle.
- **A rolling checkpoint is found by fold number, so `--out` carries its own manifest.**
  `runs/.../<region>/manifest.json` pins what fold N was trained on and how
  (`runner.trainingManifestOf`). A run that moves the span start, the train or step
  weeks, the timesteps or the train source needs a new `--out`, or it would score
  agents trained on other weeks, possibly after their test days. A rescore that only
  changes how it scores reuses the agents. A warm fold's parent is the fold before it
  in the whole table, so `--folds` resumes a chain and refuses to skip part of one.
- **Rolling test windows never overlap.** `enumerateFolds` refuses `stepWeeks` below
  `testWeeks`, because a shared trading day would count twice in every pooled figure.
- **Episode windows align to 04:00, the NEM trading day boundary, and that is
  load-bearing.** PREDISPATCH always stops at the end of a trading day, so its horizon
  depends on the time of day: a 04:00 run reaches 288 intervals, 08:00 reaches 240, and
  12:00 only 192. 04:00 is the one run time where exactly one trading day of forecast
  fits. That is why the observation reads 288 intervals whatever the episode length.
  A week-long episode therefore opens at 04:00 and reads one trading day ahead.
  `refreshMatrix` cuts each run at the end of its own trading day, not at the episode
  end. Without that cut the realised-price agent would train on prices the
  forecast-priced agent can never have.
- `gamma` is 1.0: an episode is one NEM week of profit, and every interval counts the
  same. An undiscounted 2016-step episode carries about seven times the return variance
  of a daily one, and a fold holds one episode rather than seven, so the timestep count
  per fold is a flag with a manifest entry and not a fixed number.
- **The cycle rent reaches no reported number.** `dp.solve(cycles=...)` prices the
  budget and bisects the rent, but every figure in the returned `Schedule` comes from
  the true cost arrays along the chosen path. Quote `Schedule.profit` as the achievable
  figure, because it is a schedule inside the budget, and `Schedule.dualBound` as the
  ceiling, because it is the least of (value after rent + rent * budget) over every rent
  tried. That is the same contract the MILP already has. The two never meet exactly: the
  battery discharges in whole intervals, so realised usage steps and no rent lands on
  the budget. Measured on NSW1, the week of 2024-08-05, 2016 intervals, no calendar
  term: no budget gives $4574995 at 17.885 EFC in 24.6s, and a budget of 7 gives
  $4470982 against a bound of $4472087, a 2.5e-4 gap, at 6.961 EFC in 14.1s. The shadow
  price of a cycle of warranty is $142.07 per MWh. A leak of the rent into a reported
  cost breaks the 1e-9 replay invariant, which is the right way to catch it.
- **There is no per-day oracle bound once the state of charge carries across the days
  inside an episode.** A day-long oracle started empty is no bound on a day the agent
  entered holding energy. So `oracleDualBound`, `oracleGap` and `oracleCycles` sit on
  the episode's first day and are NaN on the other six. Pandas skips NaN in `mean` and
  `sum`, so a whole-episode scope counts the bound exactly once, and `summarise` passes
  `min_count=1` so a scope holding no bound at all sums to NaN rather than to zero.
  Two costs follow. A single-day scope shows no capture ratio, and `metrics.byBucket`
  blanks `captured`, `oracleProfit`, `foldMedian` and `foldIqr`, so it reports agent
  dollars with no share. A volatility bucket holds a set of days, and the episode whose
  bound those days share may open in the other bucket. `metrics.perDayBound` is the one
  test of this, and it reads `episodeStart == testDay`. Both costs come from the weekly
  episode, not from the budget.
- **One equivalent full cycle is the nominal usable capacity removed CELL-side.**
  `EFC = sum(sold / dischargeEfficiency) / model.MWh`. A 100 per cent to 0 per cent
  discharge is exactly 1.0 and a partial counts proportionally. Measuring meter-side
  would understate the spend by `nu_D`, a 5 per cent error, and a budget that is 5 per
  cent loose is not the budget anybody asked for. `_StateSpace.cellSide` in `dp.py` and
  `BessEnv.spentMWh` both hold the cell-side figure.
- **A masked-DQN agent loads here and must not be run.** Three published families
  (`dqn-warm-daily_dump_soc0_maskdqn`, `dqn-pooled-pooled_daily_empty_efc1` and
  `dqn-finetune-pooled_daily_empty_efc1`) were trained by a `dqn` that read the action
  mask when it acted. `rl/algos.py` here has no such algorithm. A masked DQN is a plain
  DQN with another `predict`, so the weights load and nothing refuses them, and the
  forecast screen would run them unmasked under their own name. That is a wrong dispatch
  run, not a missing one. `policies.maskedDqnPolicies` names the three, `loadable` drops
  them, and `publishSpace.sh` stage 3 filters through `loadable`, so they never reach the
  Space. The list is closed: `publish` now records `readsMask` from
  `algos.supportsMasking`, and `loadable` drops an agent whose record disagrees with this
  build. Their result rows stay published, because the rows are what the masked run
  actually earned.
- **The observation went from 60 slots to 62, so an agent published under the old layout
  cannot be loaded into the new environment.** `policies.loadable(region, root)` filters
  `available()` on the `observationSize` that `publish` now records, and `live.py` reads
  `loadable` so the forecast dropdown never offers a model the screen is certain to fail
  on. An agent published before that fact was recorded states no size, so `loadable`
  reads it as `policies.unrecordedObservationSize` (60), which is the layout every one of
  them trained under. A warm start raises on its own: `algos.load` names both spaces when
  the layout has moved.
- **`refreshMatrix` has a size cap.** The grid is
  `episodes * (horizon / every) * horizon` float64 values, so it grows with the square of
  the horizon at a fixed refresh. A 52-week training grid of weekly episodes is 1.6 GiB
  at a 5-minute refresh against 35 MB at a 30-minute one. `forecasts.matrixSizeCap`
  (4 GiB) refuses anything larger and names the figure, rather than waiting for the OOM
  killer to name it. Add a new input that needs more and raise the cap deliberately.
- **A PREDISPATCH run is filed under its trading day.** The run labelled 04:00 on the
  1st is the previous trading day's last and sits in the previous month's MMSDM file;
  `fileLag` in `forecasts.py` is why `runRows` reads that month too.
- **Dash drops a heatmap point's `customdata` on click.** It reads `customdata` out of
  the trace by the point number, and a heatmap point number is a `[row, column]` pair
  rather than one integer. The calendar therefore finds the clicked day from the row
  and the column that do arrive, against the month on screen. See `clickedDay` in
  `visualise/dashboard/__main__.py`. A per-cell date in `customdata` reaches the hover
  template but never the callback.
- **An old actions export holds seed 0 only.** Exports made by `python -m
  backtest.actions` come from the fold trajectories, which hold `trajectorySeed` (0)
  only, so the day view of another seed is empty by design. It must stay empty:
  drawing seed 0's dispatch under another seed's name is a wrong chart, not a missing
  one. A backtest that writes its CSV as it runs logs every seed.
- **The actions CSV is written by the backtest, every seed and price source, as each
  fold finishes.** `cli rolling` appends one fold's decisions to
  `results/actions/<run>_<region>.csv` under a lock, because every seed's process shares
  the file. `priceSource` is copied from the runner's row, never inferred, and a
  realised row whose observed price is not its paid price is refused. A run name is
  used once: a fresh start refuses a name whose actions CSV or
  `results/backtestResults/<run>_<region>.csv` exists, and `--resume` (`RESUME=1` in the
  folds job) continues one without appending a fold twice. `python -m backtest.actions`
  is only for runs from before this. The CSV is working state: `backtest.publish`
  writes it out as `results/actions/<run>_<region>.parquet`, which is what lives in
  the HF dataset `toms55/IntelligentDispatchExperiments` and the dashboard reads.
- `nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first
  call for a date range is slow and network-bound while later calls are local. The
  directory is gitignored and safe to delete.

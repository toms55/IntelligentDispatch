# Architecture

Flat package layout at the repo root — no `src/`. Tests sit beside the code they
cover.

- **`data/`** — market data access.
  - `nemosis.py` wraps `nemosis.dynamic_data_compiler` for three AEMO dispatch
    tables: `fetch_dispatch_data` (DISPATCHREGIONSUM), `fetch_price_data`
    (DISPATCHPRICE), `fetch_interconnector_data` (DISPATCHINTERCONNECTORRES). Each
    takes `datetime` start/end and a `selected_columns` list.
  - `prices.py` is the layer above it: `price_series()` returns a clean 5-minute RRP
    series for one region, dropping AEMO's intervention duplicates. This is what the
    oracle consumes — prefer it over calling `nemosis.py` directly.
  - `forecasts.py` is the inference-time price source, read straight from NEMWeb.
    `forecastSeries(runTime, region, horizon)` stitches P5MIN REGIONSOLUTION (5-minute,
    11 intervals, 55 minutes ahead) under PREDISPATCH PRICE (30-minute, to the end of a
    trading day), preferring P5MIN on the overlap. PDPASA and STPASA are not used and
    cannot be: they are adequacy runs and carry no RRP column at all. `runRows` finds a
    run in its MMSDM month, else Reports/Archive, else Reports/Current, so history and
    `liveForecast()` share one path.

- **`degradation/`** — `schade.py` implements Schade & Egging-Bratseth (2024),
  doi:10.1002/est2.588. The battery is 16 stacked energy segments; emptying a high
  segment costs more than a low one, and those per-segment costs are constants, so
  the same model serves as both an exact MILP objective row and a causal per-step RL
  reward. `degradation.py` is a stub holding a second paper's citation.

- **`optimisation/`** — `lp.py` is the perfect-foresight oracle. Despite the name it
  is a MILP: the staircase fill (a segment may hold energy only if the one below it
  is full) needs a binary per segment, and calendar aging adds SOS2 binaries.
  `solve()` returns a `Schedule` with soc/bought/sold/revenue/degradationCost/profit,
  plus per-interval splits of each, the solver `status`, the `mipGap` and the
  `dualBound`.
  `discretePower=True` restricts every interval to the rated throughput or nothing,
  which is the agent's own action set, and is the only mode that bounds the agent
  fairly. Four extra binaries per interval do it; `_discretePower` explains why
  `atFull`/`atEmpty` cannot be relaxed to continuous.
  `dp.py` solves that discrete mode exactly, and is what `rl/baseline.py` now calls
  for it. With three actions the state is one number, so the problem is a shortest
  path over about 5700 reachable energy levels, and value iteration is exact. There
  is no gap, no time limit and no dual bound above the optimum. Measured on one week
  of NSW1: 2.5s against 363s, and the MILP's incumbent was short by up to 1.2%.
  `lp.py` stays for `discretePower=False`, where power is continuous and no finite
  state set exists. `dp.valueFunction` also returns the optimum at every state and
  interval, not only along one path, at the same cost.
  `dp.solve(cycles=...)` prices the cycle budget rather than capping it. Every MWh
  removed cell-side pays a rent, and `solve` bisects that rent until the schedule lands
  on the budget. A hard cap would need a second state dimension, and that is not
  possible. The pack energy saturates: under 8000 states for the paper battery, and 9121
  for the project battery over a week, the same count at 1008 intervals and at 2016.
  Cumulative discharge does not saturate. About a third of the oracle's
  discharges are partial, and they cluster at 0.9025 of a full step, which is
  `nu_C * nu_D`. Any integer grid therefore over-counts the EFC by 6 to 7.5 per cent,
  and a grid fine enough to hold the error under 0.1 per cent costs about 1.5e10
  states. The rent gives two honest numbers instead. `cyclePrice` is the rent that
  bought the incumbent, and `cyclesUsed` is what the schedule spent.
  `terminalPrice=...` makes the last value layer the worth of the closing stock, out of
  `degradation.schade.SchadeDegradation.terminalValue`, and value iteration carries it
  backwards for free. `lp.py` states the same budget as one exact constraint row and the
  same terminal term in the objective. That path is for cross-checking short horizons
  only. Do not wire it into the backtest, because the cycle row couples every interval.

- **`rl/`** — the agent.
  - `features.py` owns the observation layout and is the single definition of it.
    62 slots: SoC, 12 five-minute prices, 46 half-hour bucket means, the fraction of the
    trading day left, the fraction of the cycle budget left, and the fraction of the
    episode left. Index by `socIndex`, `priceSlice`, `clockIndex`, `cycleIndex` and
    `episodeIndex`, never by a literal or by -1. Training and inference both call
    `observation()` and neither is told which it is. The clock slot comes from
    `tradingDayRemaining`, which reads the clock: an episode-relative fraction matches
    it only while an episode is a whole 04:00 trading day, and the current forecast
    screen starts its episode at the run time, so it read 1.0 at every hour of the day.
    The clock repeats every 24 hours, which is why it cannot stand in for the episode
    slot on a week-long episode.
    `config.observation_intervals` (288) is how far ahead the agent reads and
    `config.episode_intervals` (288, the same number by default) is how long an
    episode runs. They are two different knobs now, kept apart so a later multi-day
    episode (a weekly cycle budget, say) can move the second without moving the first.
    The price features derive from the first one, because AEMO publishes no RRP past
    the end of the current trading day.
  - `env.py` — `BessEnv`, three actions, masking plus clipping at the SoC boundary.
    Takes a price `Series` rather than a date range, which is what keeps tests off the
    network. `forecasts=` swaps what the agent sees without touching what it is paid.
    `cycleBudget=` caps the equivalent full cycles one episode may spend and
    `terminalPrice=` values the closing stock. `_dischargeRoom()` holds both discharge
    clips, the SoC floor and the remaining budget, so the mask and the move cannot
    disagree. The clip in `_applyAction` is what enforces the budget, so it binds under
    DQN and A2C as well, which read no mask. Both default to the unbudgeted problem
    (`None`, `0.0`): no published agent trains with a cycle budget yet.
  - `baseline.py` — per-window oracle solutions, read from one of four places and
    written to the first. `findRecord` is the lookup: `data_cache/lp_baseline`, then the
    tracked sets `oracleResults/dp`, `oracleResults/dp_soc0` and `oracleResults/milp`.
    `oracleResults/dp` is the discrete oracle solved exactly by `optimisation/dp.py` for
    every trading day of all five regions, 2023-01-01 to 2026-09-26; `milp` is the MILP
    corpus from before it. A cache key states every parameter of its solve, the solver
    included, so a key matches a record of the same problem only, whichever set holds
    it. `--cache-dir` names one directory and then reads that one alone.
  - `oracleArchive.py` — the format of a tracked set: one Parquet per region
    (`oracleResults/<set>/<REGION>.parquet`), never one JSON per day, because 21,206
    small files made every checkout crawl on the shared filesystem. Each row keeps the
    cache file's name and its JSON text, so `findRecord` serves a record straight out of
    the Parquet and writes nothing. `python -m rl.oracleArchive pack <json folder>
    oracleResults/<set>` merges solves into a set, a new record replacing an old one of
    the same key, and `unpack` writes byte-identical cache JSONs back out.
  - `algos.py` — the `--algo` registry: MaskablePPO, DQN, A2C. `algos.load` reads a
    fourth name, `dqnnomask`, which `build` refuses. Agents of the pooled-regions
    experiment family carry it, because the branch that trained them had a second DQN
    that read the mask. Here DQN reads none, so the two names are one plain DQN and
    only `dqn` can start a run.
  - `train.py`, `evaluate.py` — one run per seed, and regret against the oracle.
  - `policies.py` — `trainedPolicies/`: the layout of a finished agent, and the only
    code that writes or lists one. See
    [A trained agent lives in `trainedPolicies/`](conventions.md#a-trained-agent-lives-in-trainedpolicies).

- **`backtest/`** — the rolling-origin backtest: `runner.py` runs the folds,
  `records.py` writes them, `actions.py` exports the dispatch, and `publish.py` writes a
  finished run into `results/backtestResults/`. `outofbounds.py` reads how often an
  unmasked agent chose a move it could not make, and which intervals it chose it in.
  See [A dashboard backtest lives in `results/backtestResults/`](conventions.md#a-dashboard-backtest-lives-in-resultsbacktestresults).
  A test day and a test episode are the same thing while an episode is one day, which
  is every published run so far.
  `folds.py` holds both regardless: `Fold.testDays` and `dayOffset` name the days, and
  `Fold.testEpisodes(horizon)` and `episodeOffset(spanStart, moment, horizon)` name the
  episodes. `episodeLength(horizon)` is one episode as a timedelta, and it refuses a
  horizon that is not a whole number of trading days. The oracle is solved one record
  per episode, and `_foldRecords` indexes it with `episodeOffset`. `evaluate.perDay`
  then splits one episode's trajectory into its trading days, so a multi-day episode
  would still write one result row per day, though no run runs one.
  `pooled.py` is the pooled-regions variant: each fold trains one shared agent on every
  region's training window end to end, warm from the last fold's shared agent, then
  fine-tunes a fresh copy per region. Both are scored, as trainMode `pooled` and
  `finetune`, and both are published. `scripts/hpc/pooled.pbs` runs one seed per job, then
  merges and publishes.

- **`config.py`** — experiment design pinned in one place: episode horizon, observation
  horizon, SoC reset, the cycle budget, seeds, splits, the observation layout constants,
  the two oracle roles, and the dashboard catalogue (`dashboard_runs`,
  `best_seed_by_run`). Module constants are lower_snake, matching
  `utils/constants.py`.

- **`utils/constants.py`** — `nem_interval` (5 min) and `nem_interval_hours` (1/12),
  used to scale per-hour rates.

- **`cli/`** — front-end over the modules above. `app.py` holds the subcommands and
  the parser, and `__main__.py` is the `python -m cli` entry point. Four subcommands
  share one window, region and battery argument set: `lp` inspects one oracle solve,
  `baseline` solves and caches one per window, `train` fits one agent per seed, `eval`
  scores them. A window is two dates or a named `--split`.

- **`scripts/overnight.sh`** — the five stages of the full experiment in order, each
  resumable because the oracle cache is keyed on its own parameters.
- **`scripts/rolling.sh`** — the two stages of the rolling-origin backtest: the discrete
  oracle cache over the whole span, then the 52 folds.
- **`scripts/pushMain.sh`** — one push for both remotes: main to GitHub, then the same
  code to the Space through `scripts/publishSpace.sh`.
- **`scripts/dashboardRuns.sh`** — `config.dashboard_runs` as one results path per line.
  It parses the block with sed rather than starting an interpreter, because the pre-push
  hook reads it on every push.
- **`scripts/hpc/`** — the PBS batch jobs, one file per submission: `rolling.pbs`,
  `rolling2023Oracle.pbs`, `rolling2023Folds.pbs`, `pooled.pbs`, `prefetch.pbs`,
  `nsw1Overnight.pbs` and `checkScore.pbs`. Each one runs under `qsub` from the repo
  root, which is what `cd "${PBS_O_WORKDIR:-$PWD}"` relies on, so every path inside
  them stays relative to the root and not to this directory.

- **`scripts/hooks/prePush`** — the same job on a plain `git push` that updates main.
  `scripts/installHooks.sh` installs it, and `scripts/spaceFingerprint.sh` tells it
  whether anything the Space serves has changed.

- **`reporting/`** — static reporting: `parse.py` reads a `results/<REGION>.md` file,
  `rolling.py` turns a rolling result set into the same sections, and `plots.py` draws
  the matplotlib figures. `python -m reporting <file>` is the entry point. This is the
  only package besides `cli/` that imports matplotlib, and `deploy/requirements.txt`
  carries no matplotlib, so nothing under `visualise/` may import it.

- **`visualise/dashboard/`** — the Dash app, and all that `visualise/` now holds. Two
  screens behind tabs.
  `experiments.py` is the catalogue both screens select from, `data.py` reads the
  rolling results, `figures.py` draws, `live.py` runs a published agent over the AEMO
  path published right now. `--policies` points at `trainedPolicies/`. With no
  `--results` both screens offer every experiment `config.dashboard_runs` names, at the
  seed `config.best_seed_by_run` pins, and each entry carries the run its numbers come
  from. `--results results/backtestResults/<run>`, named once per run, narrows them to
  the runs it names, and the screens open on the first. That is what the hosted
  dashboard passes, one name per staged run. `data.findResults` is the only reader of
  it, and it holds every run named to one root.
  A day view reads the run's actions export, `results/actions/<run>_<region>.parquet`
  (see `data.actionsFile`): its rows are the dispatch the chart draws, for either
  price source, and its `forecastRunTime` column is the only source of the forecast
  line, so pointing `--results` at another run changes both along with every other
  number. The "refreshed every …" tile states the published `forecastEvery` column,
  copied from the manifest: the test-day forecast pull frequency (training has
  none). `data.verifyRefresh` refuses a forecast-priced run that omits it or
  whose export disagrees with it, so the tile never describes a pull the backtest did
  not make. Every manifest records it, realised-only runs included. Each forecast
  trajectory row records the run the agent actually read (`forecastRunTime`, from
  `forecasts.decisionRuns`), and the export copies it, so the check compares the
  manifest against what happened. Never edit a finished manifest by hand:
  `python -m backtest.stamp` is the only code that does, and it stamps an old run once
  a day only when every row predates the intra-day refresh.
  A Cycles tile states the mean equivalent full cycles per episode against the budget,
  from `metrics.cycleUse`. A mean per episode, not a total, so the figure reads against
  the budget at every scope. `figures._cycleTile` appends it last and omits it entirely
  for a run that records no cycles, so the tiles a published run already shows keep
  their places. `data.loadRun` reads a run that carries none of the new columns, so
  every run published so far still draws.
  The current forecast screen refetches and reruns the agent on every NEM dispatch
  interval, at minutes ending in 0 or 5 (`live.refreshSlot`), as a live BESS would
  submit a new action each interval. It plans one trading day, so it passes
  `live.dailyBudget()`, which is a seventh of the week's budget. The whole week's would
  let a Monday spend the lot.
  Either screen hands its numbers over as a CSV. The backtest button downloads the
  published rows of the region on screen, over the days the two date boxes name. The
  boxes open on the whole run (`runSpan`), and a date outside it downloads nothing and
  turns red, as a bad start state of charge does (`downloadRange`). A test day is an
  04:00 trading day start, so the range matches on the calendar date. The forecast
  button hands over the intervals already drawn, held in a store by `redrawForecast`,
  so the file and the chart cannot disagree.

- **`deploy/`** — the hosted dashboard: the Dockerfile, the runtime requirements, the
  WSGI entry point and the Space card. `scripts/publishSpace.sh` stages and pushes it.
  See [One definition of the hosted dashboard](conventions.md#one-definition-of-the-hosted-dashboard).

- **`conftest.py`** — exists only to put the repo root on `sys.path` so
  `degradation/schade.py` can resolve `from utils import ...` during collection.


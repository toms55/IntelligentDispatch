# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Deep reinforcement learning for battery energy storage (BESS) dispatch against the
Australian National Electricity Market (NEM). Realised prices come from AEMO via
`nemosis`; price forecasts come via `nemseer`.

Built so far: the data layer, a Schade degradation model, a perfect-foresight MILP
oracle in two modes, a Gymnasium environment, and three agents behind one `--algo`
flag. The agent trains on realised 2024 prices and is scored on 2025, twice: once
with realised prices in its observation and once with the AEMO forecast it would
actually have had. See `TODO.txt` for what remains.

The central invariant: the environment and `solve(discretePower=True)` are the same
problem stated twice. Same three actions, same partial-top-up rule at the SoC
boundary, same episode length, same starting SoC, same degradation terms. Break any
one and the oracle stops bounding the agent, and every number downstream means
nothing. `rl/test_env.py::TestMatchesTheOracle` replays an oracle schedule through
the environment and holds the two to 1e-9.

## Environment

`flake.nix` pins the toolchain to a specific nixpkgs revision, so the interpreter and
system libraries are identical on every machine and stay that way over time. It
provides `python311` and `uv`; Python dependencies live in `pyproject.toml` and are
pinned in `uv.lock`. Never use `pip` or `conda` directly.

The interpreter is 3.11 rather than 3.13 because `nemseer`, the only maintained way to
reach AEMO's forecast tables, declares `requires-python >=3.8,<3.12` and pins
`packaging<22`. That pin rules out pytest 9, so pytest sits at 8.4.2. Nothing in the
numerical or RL stack moves: numpy, pandas, torch, gymnasium, stable-baselines3 and
nemosis all keep their versions, and the whole suite passes unchanged. nemseer is no
longer imported (it cannot read AEMO's archive from August 2024 on), so the pin could
now be lifted.

```bash
nix develop        # pinned interpreter + uv
uv sync            # dependencies into .venv
```

To add a package, put it in `pyproject.toml` and run `uv lock`. `flake.lock` pins
nixpkgs (tracking the `nixos-26.05` release branch).

`uv sync` alone also works and installs the same pinned dependencies — it just fetches
its own CPython, so the interpreter and system libraries are no longer pinned. The
`LD_LIBRARY_PATH`/`UV_PYTHON` exports in the flake's `shellHook` point uv at the Nix
interpreter and satisfy PyPI wheels that expect an FHS layout.

## Running code

Packages live at the repo root and are imported as such (`from data.prices import ...`),
so run from the repo root as modules, not by path, and under `uv run`:

```bash
uv run python -m cli lp 2024-01-10 2024-01-11 --no-calendar
uv run python -m cli lp --help
uv run pytest
```

## Conventions (mandatory)

These are rules, not defaults. Break one and the training machine and the dashboard
machine stop agreeing.

### Two algorithms, one problem

The thesis compares DQN against MaskablePPO, so a number is only worth quoting if both
agents faced the same problem and the same budget. `rl/env.py` is the one environment,
`rl/train.py:trainOne` the one training path and `backtest/runner.py` the one fold
driver. Nothing branches on the algorithm except the registry in `rl/algos.py` and the
one line in `rl/evaluate.py` that decides whether `predict` reads the action mask.

Rules:

1. Every hyperparameter is pinned in `rl/algos.py`, with a comment, including values
   that equal the SB3 default. The thesis quotes that one file. Nothing is tuned on a
   test window.
2. Budgets are equal in environment steps, not in gradient updates. `_dqn` states both
   counts, because the two differ by about 60% at the same step budget.
3. MaskablePPO reads the mask and DQN does not. That is a property of the algorithm, not
   a change to the problem, and the out-of-bounds rate measures what it costs.
4. The out-of-bounds penalty (`config.invalid_action_penalty`) enters the reward and
   nothing else. Profit, revenue, degradation, regret and the captured fraction stay raw
   dollars, so the discrete oracle keeps bounding the agent: the oracle's schedule is
   feasible, so the term is zero along it, and `rl/test_env.py` proves the bound does not
   move. Both algorithms train under the same reward function, and a masked one can never
   incur the term.
5. Out of bounds means `not action_masks()[action]`, and `rl/env.py:action_masks` is the
   only definition. Never write a second test for it, and never penalise a partial
   top-up: `solve(discretePower=True)` allows a part step where physics forces one, so
   penalising it would make the oracle solve a different problem.
6. `outOfBounds` is counted in four places and named the same in all of them: per step in
   the env's `info`, per training episode in `monitor.csv`, per test day as a published
   day-row column for every seed, and per interval in the actions export, where
   `attemptedAction` names what the agent chose and `action` what moved. An attempt that
   moved nothing is an attempt at a wall. `python -m backtest.outofbounds <run> <region>`
   reads both ends.
7. The penalty, the initial SoC and the forecast refresh are flags, and `RollingConfig`
   records all three, so `requireSameManifest` refuses to put two settings in one result
   set. Two runs are comparable when their manifests differ in `algos` alone.
8. A DQN chain carries its replay buffer between folds (`rl/train.py:saveReplay`). SB3's
   `save()` excludes it, so without this a warm fold would keep its weights and relearn
   from empty experience, and "warm" would mean two different things. Only the live end
   of each chain stays on disk, and the buffer never reaches `trainedPolicies/`.

### A trained agent lives in `trainedPolicies/`

Training runs on one machine and the dashboard runs on another, so the agents have to
travel. There is exactly one layout for a finished agent:

```
trainedPolicies/<region>/<algo>-<trainMode>-s<seed>/
    agent.zip      the weights
    scaler.json    the price transform the weights were trained under
    policy.json    which command wrote it, from which window, and when
```

Rules:

1. Never build that path by hand. `rl/policies.py` is the only definition of it. Call
   `policies.publish(...)` to write and `policies.available(region)` to read.
2. Every command that trains an agent publishes its final agent there before it exits.
   `cli train` publishes each seed. `cli rolling` publishes the last fold of the run.
3. Always copy `scaler.json` with the weights. The transform is pinned to a market
   price cap. Weights read under a different cap mean something else.
4. The directory name is the dashboard label. Do not add a second naming scheme.
5. Working state stays out. The rolling backtest keeps one checkpoint per fold under
   `--out` (`runs/rolling`), because it warm starts the next fold from the last one.
   Those are not published, and nothing reads them across machines.
6. `trainedPolicies/` is gitignored. Move it with `rsync` or `scp`, not with git.

To move a trained agent to the dashboard machine, copy one region directory:

```bash
rsync -a trainedPolicies/NSW1/ dashboard-host:~/IntelligentDispatch/trainedPolicies/NSW1/
```

### A dashboard backtest lives in `results/backtestResults/`

`results/` is working output and git ignores it, apart from the published runs and
their actions exports. Those are tracked, so every clone can draw them. Each is one CSV
per run per region, named the same way:

```
results/backtestResults/<run>_<region>.csv   one row per seed, price source and test
                                             day, every fold, plus forecastEvery
results/actions/<run>_<region>.csv           one row per agent decision, every seed and
                                             price source; the day view draws it
```

Rules:

1. `backtest/publish.py` is the only code that writes `results/backtestResults/`. Never
   copy a run in by hand.
2. The folds job publishes each region once it has merged the seeds. `<run>` is the
   results directory's name without the region, so the five regional jobs of one
   experiment share one name.
3. A region is published only if `data.loadRun` can read it, so a run the dashboard
   cannot draw is refused at publish time.
4. `forecastEvery` is the one manifest value the dashboard reads, so it travels as a
   column. The manifest and the fold trajectories stay with the raw results.
5. Publishing commits nothing. Commit `results/backtestResults/` and `results/actions/`
   after a backtest.

### The two dashboard screens read different things

The backtest screen reads finished rows from `--results`. It replays numbers already
written, so it needs no weights and its model list comes from the results.

The current forecast screen runs the agent now, so it needs the weights. Its model
list comes from `trainedPolicies/` and from nowhere else. Never fill it from the
results: that offers a model the screen cannot load.

### One definition of the hosted dashboard

The dashboard is hosted on a Hugging Face Space, which builds `deploy/Dockerfile` and
serves `deploy/app.py`. `scripts/publishSpace.sh` is the only code that stages the
upload, and the upload is a subset: the packages the two screens import, one run from
`results/backtestResults/` with its actions exports, one published agent per region,
and `deploy/`.

The Space runs on CPU Basic hardware, which has no hourly cost but needs a PRO plan to
create. Only static Spaces are free, and a Dash application is not static.

Rules:

1. `deploy/requirements.txt` holds the runtime dependencies of the two screens only.
   It is not a copy of `pyproject.toml`. pulp, highspy, nemosis, nemseer and matplotlib
   stay out, because no screen solves and no screen reads the AEMO archive.
2. Stage 4 of the script builds the app with those five names blocked. If a new import
   on the dashboard path reaches one of them, that check fails and the Space does not.
3. The agents travel through `policies.publish`, so the Space gets the same layout every
   other machine has. Never stage weights with `cp`.
4. The script pushes one commit over the last one. The weights are binary, so a history
   of them would grow the Space by their size on every update.
5. main and the Space hold the same code. The Space is a separate git repository, so a
   push to GitHub does not reach it. The `pre-push` hook in `scripts/hooks/prePush`
   publishes the Space whenever a push updates main. Run `scripts/installHooks.sh` once
   per machine, because git does not track hooks. `scripts/pushMain.sh` does the same
   job on demand.
6. A Space publish is driven by a fingerprint, not by a git diff. `models/` is
   gitignored, so a retrained agent changes the Space without changing one tracked
   file, and a new backtest changes `results/backtestResults/` before anyone commits
   it. `scripts/spaceFingerprint.sh` covers both: the
   git tree hash of each copied code item, and the name, size and modification time of
   every result file and weight that goes up. Add a new input to the upload and add it
   there in the same commit, or the hook stops noticing that input.
7. The Hugging Face write token lives in a secret store, never in the repository. The
   script reads `$HF_TOKEN`, else the token file `hf auth login` writes, else it asks on
   the terminal. It hands the token to one `git push` through a credential helper, so
   the token reaches no file and no command line. Never put a token in a remote URL.
8. Stage 0 resolves the token and tests it against the Space LFS endpoint, before the
   script stages anything. The push itself blocks every git prompt. git otherwise asks
   through `SSH_ASKPASS`, and a machine with no working askpass program pushes with
   empty credentials. Hugging Face answers that anonymous push with "You have read
   access but not the required permissions", which names the wrong cause.

### One dispatch chart, drawn once

A charge or discharge run reads the same way on both screens. `figures._dispatchFigure`
is the only code that draws it, and `live.blocks` is the only code that finds the runs.
The backtest day view feeds it realised prices from `data.dispatchPath`. The forecast
view feeds it the AEMO path. Do not write a second dispatch chart, and do not give a
block a second colour.

## Architecture

Flat package layout at the repo root — no `src/`. Tests sit beside the code they
cover.

- **`data/`** — market data access.
  - `NEMOSIS.py` wraps `nemosis.dynamic_data_compiler` for three AEMO dispatch
    tables: `fetch_dispatch_data` (DISPATCHREGIONSUM), `fetch_price_data`
    (DISPATCHPRICE), `fetch_interconnector_data` (DISPATCHINTERCONNECTORRES). Each
    takes `datetime` start/end and a `selected_columns` list.
  - `prices.py` is the layer above it: `price_series()` returns a clean 5-minute RRP
    series for one region, dropping AEMO's intervention duplicates. This is what the
    oracle consumes — prefer it over calling `NEMOSIS.py` directly.
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

- **`rl/`** — the agent.
  - `features.py` owns the observation layout and is the single definition of it.
    60 slots: SoC, 12 five-minute prices, 46 half-hour bucket means, and the fraction
    of the trading day left. Training and inference both call this and neither is told
    which it is. The last slot comes from `tradingDayRemaining`, which reads the clock:
    an episode-relative fraction matches it only while an episode is a whole 04:00
    trading day, and the current forecast screen starts its episode at the run time, so
    it read 1.0 at every hour of the day.
  - `env.py` — `BessEnv`, three actions, masking plus clipping at the SoC boundary.
    Takes a price `Series` rather than a date range, which is what keeps tests off the
    network. `forecasts=` swaps what the agent sees without touching what it is paid.
  - `baseline.py` — per-window oracle solutions, cached on disk under `data_cache/`.
  - `algos.py` — the `--algo` registry: MaskablePPO, DQN, A2C.
  - `train.py`, `evaluate.py` — one run per seed, and regret against the oracle.
  - `policies.py` — `trainedPolicies/`: the layout of a finished agent, and the only
    code that writes or lists one. See the convention above.

- **`backtest/`** — the rolling-origin backtest: `runner.py` runs the folds,
  `records.py` writes them, `actions.py` exports the dispatch, and `publish.py` writes a
  finished run into `results/backtestResults/`. `outofbounds.py` reads how often an
  unmasked agent chose a move it could not make, and which intervals it chose it in.
  See the convention above.

- **`config.py`** — experiment design pinned in one place: episode horizon, SoC reset,
  seeds, splits, the observation layout constants, and the two oracle roles. Module
  constants are lower_snake, matching `utils/constants.py`.

- **`utils/constants.py`** — `nem_interval` (5 min) and `nem_interval_hours` (1/12),
  used to scale per-hour rates.

- **`cli.py`** — front-end over the modules above. Four subcommands sharing one window,
  region and battery argument set: `lp` inspects one oracle solve, `baseline` solves and
  caches one per window, `train` fits one agent per seed, `eval` scores them. A window is
  two dates or a named `--split`.

- **`scripts/overnight.sh`** — the five stages of the full experiment in order, each
  resumable because the oracle cache is keyed on its own parameters.
- **`scripts/rolling.sh`** — the two stages of the rolling-origin backtest: the discrete
  oracle cache over the whole span, then the 52 folds.
- **`scripts/pushMain.sh`** — one push for both remotes: main to GitHub, then the same
  code to the Space through `scripts/publishSpace.sh`.
- **`scripts/hooks/prePush`** — the same job on a plain `git push` that updates main.
  `scripts/installHooks.sh` installs it, and `scripts/spaceFingerprint.sh` tells it
  whether anything the Space serves has changed.

- **`visualise/dashboard/`** — the Dash app. Two screens behind tabs. `data.py` reads
  the rolling results, `figures.py` draws, `live.py` runs a published agent over the
  AEMO path published right now. `--results` points at the records, `--policies` at
  `trainedPolicies/`. `--results` names a run as `results/backtestResults/<run>`, the
  path of its CSVs without the region. With no `--results`, `data.findResults` reads
  the one run under `results/backtestResults/`. Two runs there raise and name both.
  A day view reads the run's actions export, `results/actions/<run>_<region>.csv`
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
  The current forecast screen refetches and reruns the agent on every NEM dispatch
  interval, at minutes ending in 0 or 5 (`live.refreshSlot`), as a live BESS would
  submit a new action each interval.

- **`deploy/`** — the hosted dashboard: the Dockerfile, the runtime requirements, the
  WSGI entry point and the Space card. `scripts/publishSpace.sh` stages and pushes it.
  See the convention above.

- **`conftest.py`** — exists only to put the repo root on `sys.path` so
  `degradation/schade.py` can resolve `from utils import ...` during collection.

## Gotchas

- Energy is tracked on two sides of the meter in `lp.py`: `stored` is cell-side,
  `bought`/`sold` are meter-side. Charging loses `nu_C` in, discharging loses `nu_D`
  out. Mixing them up silently changes the profit figure.
- The oracle is only a valid bound on the agent if both use the same episode horizon,
  the same SoC reset rule, and the same degradation terms. `TODO.txt` step 1 explains
  why lookahead past a chunk boundary breaks this.
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
- **Rolling test windows never overlap.** `enumerateFolds` refuses `stepWeeks` below
  `testWeeks`, because a shared trading day would count twice in every pooled figure.
- **Episode windows align to 04:00, the NEM trading day boundary, and that is
  load-bearing.** PREDISPATCH always stops at the end of a trading day, so its horizon
  depends on the time of day: a 04:00 run reaches 288 intervals, 08:00 reaches 240, and
  12:00 only 192. 04:00 is the one run time where exactly one episode fits.
- `gamma` is 1.0: an episode is one trading day of profit, and every interval counts
  the same.
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
  is only for runs from before this.
- `nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first
  call for a date range is slow and network-bound while later calls are local. The
  directory is gitignored and safe to delete.

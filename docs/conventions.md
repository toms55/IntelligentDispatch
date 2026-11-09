# Conventions (mandatory)

These are rules, not defaults. Break one and the training machine and the
dashboard machine stop agreeing. `CLAUDE.md` at the repo root is the index.

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
   `dqnnomask` is a second name for that same unmasked DQN, and one builder serves both.
   It exists because a published run records it as its algorithm, and the dashboard loads
   the weights by that name. Never give it settings of its own: two names for one agent
   is already one too many, and two agents under two names would end the comparison.
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
   day-row column for every seed, and per interval in the actions export.
   `python -m backtest.outofbounds <run> <region>` reads both ends.
   The env's flag travels to all four. Never re-derive it from the energy moved, even
   though `attemptedAction` and `action` sit side by side in the export and look like
   they imply it. The mask is a tolerance on the headroom, so an out-of-bounds step can
   still move about 1e-12 MWh, which makes `action` read as a move. Deriving the flag
   there made the interval list disagree with the day-row count.
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
trainedPolicies/<region>/<algo>-<trainMode>[-<tag>]-s<seed>/
    agent.zip      the weights
    scaler.json    the price transform the weights were trained under
    policy.json    which command wrote it, from which window and run, and when
```

Rules:

1. Never build that path by hand. `rl/policies.py` is the only definition of it. Call
   `policies.publish(...)` to write and `policies.available(region)` to read.
   `policies.loadable(region)` is the reader for a screen that must run the agent: it
   drops every agent whose recorded `observationSize` is not the one `config` states now,
   and every agent in `policies.maskedDqnPolicies`. See
   [the masked-DQN gotcha](gotchas.md).
2. Every command that trains an agent publishes its final agent there before it exits.
   `cli train` publishes each seed. `cli rolling` publishes the last fold of the run.
3. Always copy `scaler.json` with the weights. The transform is pinned to a market
   price cap. Weights read under a different cap mean something else.
4. The directory name is the dashboard label. Do not add a second naming scheme. A run
   whose agents mean something else (another episode start, say) passes
   `cli rolling --policy-tag` so it publishes beside the existing agents, not over them.
5. Working state stays out. The rolling backtest keeps one checkpoint per fold under
   `--out` (`runs/rolling`), because it warm starts the next fold from the last one.
   Those are not published, and nothing reads them across machines.
6. `trainedPolicies/` is not tracked by git. It lives in the Hugging Face dataset
   `toms55/IntelligentDispatchExperiments`, alongside `results/` and `oracleResults/`.
   `hf upload toms55/IntelligentDispatchExperiments trainedPolicies trainedPolicies
   --repo-type=dataset` moves a finished agent to the dashboard machine. `hf download`
   with the same repo pulls the current set down. Always pass
   `--include 'results/*' 'trainedPolicies/*' 'oracleResults/*'`. The dataset holds its
   own `README.md` and `.gitattributes`, and a download without `--include` writes both
   into the repo root, over this project's README.

### A dashboard backtest lives in `results/backtestResults/`

`results/` is working output and git ignores it. The published runs and their actions
exports live in the Hugging Face dataset `toms55/IntelligentDispatchExperiments`
instead, so every machine can draw them with `hf download`. Each is one file per run
per region, named the same way:

```
results/backtestResults/<run>_<region>.csv   one row per seed, price source and test
                                             day, every fold, plus forecastEvery
results/actions/<run>_<region>.parquet       one row per agent decision, every seed and
                                             price source; the day view draws it
results/backtestResults/<run>_<region>.json  the run's manifest and a one-line
                                             description of what it tested
results/backtestResults/README.md            the index: every published run, one line
```

The actions export is Parquet because every seed under both price sources is about
270 MB of CSV per region (30 to 40 MB as Parquet). The CSV beside it,
`results/actions/<run>_<region>.csv`, is the live file a backtest appends to as it
runs. It stays off the dataset, and the dashboard never reads it.

Rules:

1. `backtest/publish.py` is the only code that writes `results/backtestResults/` and
   turns the live actions CSV into the Parquet export. Never copy a run in by hand.
   It also restamps a published run's two oracle columns from a solved corpus:
   `python -m backtest.publish --restamp-oracle <csv>...` replaces
   `oracleDualBound` with the optimum in `oracleResults/dp` and `oracleGap` with zero.
   The agent rows stay as they are, so the run is rescored against a tighter bound and
   not rerun. A day the corpus does not hold, or a bound below a row's `agentProfit`,
   refuses the whole file. Both mean the corpus solved another problem.
2. The folds job publishes each region once it has merged the seeds. `<run>` is the
   results directory's name without the region, so the five regional jobs of one
   experiment share one name.
3. A region is published only if `data.loadRun` can read it, so a run the dashboard
   cannot draw is refused at publish time.
4. `forecastEvery` is the one manifest value the dashboard reads, so it travels as a
   column. The whole manifest travels too, in the JSON beside the CSV, with a
   `--description` saying what the run tested and against which run, because the raw
   results never leave the machine that ran them. Always pass `--description` on a
   first publish. A republish without one keeps the old description, and
   `--context-only` adds context to a run already published. The fold trajectories
   stay with the raw results.
5. Publishing writes nothing to the dataset. Run
   `hf upload toms55/IntelligentDispatchExperiments results results --repo-type=dataset`
   after a backtest to push `results/backtestResults/` and `results/actions/`.
6. A day row carries `episodeStart` and `cycleBudget` beside `testDay`, and
   `agentCycles` and `oracleCycles` beside the profit columns. `episodeStart` names the
   episode the day sat in. The oracle columns `oracleDualBound`, `oracleGap` and
   `oracleCycles` hold the episode's figures on its first day and are empty on the rest.
   See [the gotchas](gotchas.md) for why there is no per-day bound.
7. `publish.requireWithinBudget` refuses a region where any
   `(seed, priceSource, episodeStart)` group sums `agentCycles` above its `cycleBudget`.
   The environment cannot exceed the budget, so this never fires. It runs anyway,
   because it is the only check that reads a whole run on real prices, and it runs
   before anything is written.

### One catalogue of experiments, read by both screens

An experiment is one published run of one region: its day rows, the manifest it ran
under, and the one line saying what it tested. `config.dashboard_runs` is the list of
them, `config.best_seed_by_run` names the seed each one shows, and
`visualise/dashboard/experiments.py` is the only reader of either.

The backtest screen offers one entry per experiment, train mode and price source. The
forecast screen offers the agents those same experiments trained, at the same seed. So
every model a reader can run has its numbers on the other screen.

Rules:

1. A run reaches `config.dashboard_runs` when it tests something the thesis reports and
   its manifest states what that was. The comment there says which published runs are
   out and why. Everything else stays published, in the thesis, and off the screen.
2. One seed per model, pinned in `config.best_seed_by_run`, never ranked while the
   screen draws. `scripts/rankSeeds.py` derives the map and prints it. Picking the top
   of five seeds on the test days is selection on the test window, so the settings box
   says which seed of how many it shows and the thesis quotes the spread.
3. Selecting an experiment selects its oracle. Every published row carries the bound of
   its own run, so the screen reads the rows of the selected run and nothing else. No
   screen solves, and no screen re-derives a bound.
4. The settings box comes from `experiments.settings` and from nowhere else, on both
   screens. It reads the published manifest, so it describes the backtest that ran
   rather than the settings this checkout holds now.
5. The forecast screen builds its environment from `experiments.envSettings`: the
   episode, the cycle budget, the closing rule, the terminal price and the start state
   of charge, all from the manifest. Not from `config`, which moves between
   experiments, and not from `policy.json`, which records the training machine's
   `config` at publish time and is wrong for every agent published so far. The screen
   plans part of a trading day, so `live.budgetFor` gives it that part of the budget.
6. An agent is tied to its experiment by `rawRun` in `policy.json`, the results
   directory that trained it, against `rawResults` in the run's manifest.
   `experiments.policyRun` is the one reader. An agent with no experiment behind it, or
   one `policies.loadable` refuses, is held back from the forecast screen rather than
   offered.

### One definition of the hosted dashboard

The dashboard is hosted on a Hugging Face Space, which builds `deploy/Dockerfile` and
serves `deploy/app.py`. `scripts/publishSpace.sh` is the only code that stages the
upload, and the upload is a subset: the packages the two screens import, the runs of
`config.dashboard_runs` with their actions exports and their manifests, the agents those
runs trained at one seed each, and `deploy/`.

The Space runs on CPU Basic hardware, which has no hourly cost but needs a PRO plan to
create. Only static Spaces are free, and a Dash application is not static.

Rules:

1. `deploy/requirements.txt` holds the runtime dependencies of the two screens only.
   It is not a copy of `pyproject.toml`. pulp, highspy, nemosis, nemseer and matplotlib
   stay out, because no screen solves and no screen reads the AEMO archive.
2. Stage 4 of the script builds the app with those five names blocked. If a new import
   on the dashboard path reaches one of them, that check fails and the Space does not.
   The matplotlib half of that check is also a package boundary: the static figures live
   in `reporting/`, which the upload never stages, and `visualise/` holds the dashboard
   alone. Keep it that way. A matplotlib import under `visualise/` would reach the Space
   and fail stage 4 there instead of here.
3. The agents travel through `policies.publish`, so the Space gets the same layout every
   other machine has. Never stage weights with `cp`.
4. The script pushes one commit over the last one. The weights are binary, so a history
   of them would grow the Space by their size on every update.
5. main and the Space hold the same code. The Space is a separate git repository, so a
   push to GitHub does not reach it. The `pre-push` hook in `scripts/hooks/prePush`
   publishes the Space whenever a push updates main. Run `scripts/installHooks.sh` once
   per machine, because git does not track hooks. `scripts/pushMain.sh` does the same
   job on demand.
6. A Space publish is driven by a fingerprint, not by a git diff.
   `trainedPolicies/` is gitignored, so a retrained agent changes the Space without
   changing one tracked file, and a new backtest changes `results/backtestResults/` before anyone commits
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
9. The Space carries one seed per train mode and region, the one
   `config.best_seed_by_run` names, and every other seed stays on this machine. The
   thesis reports the spread over seeds, so `trainedPolicies/` holds all of them and
   the published results hold all of them. The Space shows one, because five dispatch
   runs that differ only by seed is not a thing a reader can read. Stage 2 filters the
   result rows and the actions export to that seed, and stage 3 stages the agents
   `experiments.runnable` offers for the staged runs, so the two screens of the Space
   name the same models.
10. The staged runs default to `config.dashboard_runs`, through
   `scripts/dashboardRuns.sh`, and `--results` replaces that list. The first is the
   one the Space opens. Stage 2 writes every staged run into the staged Dockerfile as
   one comma-separated `ENV DASHBOARD_RESULTS`, so both screens offer all of them and a
   reader compares the experiments in one dropdown. A Space variable of that name
   reorders or narrows the staged runs with no republish.
11. An actions export written by `python -m backtest.actions` holds seed 0 only. Stage 2
   refuses to stage a seed the export does not hold, because the calendar would draw and
   every day view would be empty. Either name a seed the export holds, or re-run the
   backtest, which logs every seed as each fold finishes.

### One dispatch chart, drawn once

A charge or discharge run reads the same way on both screens. `figures._dispatchFigure`
is the only code that draws it, and `live.blocks` is the only code that finds the runs.
The backtest day view feeds it realised prices from `data.dispatchPath`. The forecast
view feeds it the AEMO path. Do not write a second dispatch chart, and do not give a
block a second colour.


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

- **`rl/`** — the agent.
  - `features.py` owns the observation layout and is the single definition of it.
    60 slots: SoC, 12 five-minute prices, 46 half-hour bucket means, and intervals
    remaining. Training and inference both call this and neither is told which it is.
  - `env.py` — `BessEnv`, three actions, masking plus clipping at the SoC boundary.
    Takes a price `Series` rather than a date range, which is what keeps tests off the
    network. `forecasts=` swaps what the agent sees without touching what it is paid.
  - `baseline.py` — per-window oracle solutions, cached on disk under `data_cache/`.
  - `algos.py` — the `--algo` registry: MaskablePPO, DQN, A2C.
  - `train.py`, `evaluate.py` — one run per seed, and regret against the oracle.

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
  alone. `buildBaseline` sets it automatically when `workers > 1`.
- **Quote `Schedule.dualBound`, never `Schedule.profit`, as the bound.** A gap-limited or
  time-limited solve returns an incumbent that can sit *below* the true optimum, and a
  bound that is too low understates the agent's shortfall. Measured on SA1 2024-01-10
  discrete: 1555s to prove 21.8041, or 82s to return 21.7269 with a dual bound of
  21.8353. Note HiGHS minimises internally, so pulp negates a maximisation and the raw
  `mip_dual_bound` comes back with the opposite sign — `_solverInfo` flips it.
- **`cacheKey` does not record the solver time limit.** A window that stops on the limit
  writes a loose dual bound, and `solveWindow` then serves that record from cache for
  good. Raising `--time-limit` later cannot reach it, so delete the files to redo them.
  Measured on NSW1 2024-01-10, discrete, no calendar, 0.02 gap: a 300s limit stopped with
  the incumbent at 14.6996 and the bound still at 17.7368, above the continuous bound of
  17.1258 and so useless as a bound. This is why the rolling backtest uses 1800s.
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
- `nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first
  call for a date range is slow and network-bound while later calls are local. The
  directory is gitignored and safe to delete.

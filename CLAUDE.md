# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Deep reinforcement learning for battery energy storage (BESS) dispatch against the
Australian National Electricity Market (NEM). Market data comes from AEMO via the
`nemosis` library.

Built so far: the data layer, a Schade degradation model, and a perfect-foresight
MILP oracle that bounds what any agent could earn. The RL environment and agent do
not exist yet — see `TODO.txt` for the roadmap and the open design questions.

## Environment

`flake.nix` pins the toolchain to a specific nixpkgs revision, so the interpreter and
system libraries are identical on every machine and stay that way over time. It
provides `python313` and `uv`; Python dependencies live in `pyproject.toml` and are
pinned in `uv.lock`. Never use `pip` or `conda` directly.

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

- **`degradation/`** — `schade.py` implements Schade & Egging-Bratseth (2024),
  doi:10.1002/est2.588. The battery is 16 stacked energy segments; emptying a high
  segment costs more than a low one, and those per-segment costs are constants, so
  the same model serves as both an exact MILP objective row and a causal per-step RL
  reward. `degradation.py` is a stub holding a second paper's citation.

- **`optimisation/`** — `lp.py` is the perfect-foresight oracle. Despite the name it
  is a MILP: the staircase fill (a segment may hold energy only if the one below it
  is full) needs a binary per segment, and calendar aging adds SOS2 binaries.
  `solve()` returns a `Schedule` with soc/bought/sold/revenue/degradationCost/profit.

- **`utils/constants.py`** — `nem_interval` (5 min) and `nem_interval_hours` (1/12),
  used to scale per-hour rates.

- **`cli.py`** — inspection front-end over the modules above; `lp` is the only
  subcommand so far, `rl` is meant to slot in beside it against the same shared
  arguments.

- **`conftest.py`** — exists only to put the repo root on `sys.path` so
  `degradation/schade.py` can resolve `from utils import ...` during collection.

## Gotchas

- Energy is tracked on two sides of the meter in `lp.py`: `stored` is cell-side,
  `bought`/`sold` are meter-side. Charging loses `nu_C` in, discharging loses `nu_D`
  out. Mixing them up silently changes the profit figure.
- The oracle is only a valid bound on the agent if both use the same episode horizon,
  the same SoC reset rule, and the same degradation terms. `TODO.txt` step 1 explains
  why lookahead past a chunk boundary breaks this.
- `--no-calendar` drops the Q^CAL term and is much faster (18.8s vs 92.5s on one day),
  but the degradation it reports then counts cycling only — fine for iterating, wrong
  for any number that gets quoted.
- `defaultSolver()` uses HiGHS, which keeps its default 0.01% MIP gap deliberately —
  immaterial next to the degradation model's own error, and closing it costs most of
  the runtime.
- `nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first
  call for a date range is slow and network-bound while later calls are local. The
  directory is gitignored and safe to delete.

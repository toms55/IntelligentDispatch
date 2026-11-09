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
actually have had.

An episode is one NEM week, 2016 intervals, and it carries a cycle budget. The
battery may spend at most `config.cycle_budget` equivalent full cycles in that week,
which stands in for a cycle warranty. The `terminalPrice` setting values what the pack
still holds at the close, so the agent is not paid to dump its stock into the last
interval. That price is a pinned modelling constant, not a forecast.

The central invariant: the environment and `solve(discretePower=True)` are the same
problem stated twice. Same three actions, same partial-top-up rule at the SoC
boundary, same episode length, same starting SoC, same degradation terms, same cycle
budget, same terminal value. Break any one and the oracle stops bounding the agent,
and every number downstream means nothing.
`rl/test_env.py::TestMatchesTheOracle` replays an oracle schedule through the
environment and holds the two to 1e-9.
`rl/test_env.py::TestMatchesTheBudgetedOracle` does the same with a budget and a
terminal value in force, over both calendar modes, and holds soc, profit, cycles and
terminal value to 1e-9.

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
ruff check .
```

`ruff` comes from the Nix dev shell, not from `uvx`. The PyPI binary is dynamically
linked against an FHS layout and does not start on NixOS. Its settings are pinned in
`pyproject.toml`, and `.github/workflows/ci.yml` runs the same check with the suite on
every push to main.

## Documentation

The reference material lives in `docs/`. This file is the index. Read the first two
before you change anything, because they hold rules that are not derivable from the
code:

- [`docs/conventions.md`](docs/conventions.md) — the mandatory rules: one problem for
  two algorithms, the layout of a trained agent, how a backtest is published, the
  experiment catalogue, the hosted dashboard, and the one dispatch chart.
- [`docs/gotchas.md`](docs/gotchas.md) — the measured traps. Meter-side against
  cell-side energy, why to quote `dualBound` and never `profit`, the empty start state
  of charge, the two oracle corpora, the masked-DQN agents that load but must not run,
  and the 62-slot observation layout.
- [`docs/architecture.md`](docs/architecture.md) — the package tour, module by module.
- [`docs/methodology.md`](docs/methodology.md) — the experiment design.
- [`docs/ablation.md`](docs/ablation.md) — results of the 44 published ablation runs.

`README.md` is the front page for a human arriving at the repository. `AGENTS.md` is a
symlink to this file.

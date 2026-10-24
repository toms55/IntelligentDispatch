# DRL Battery Cycle Optimisation

Deep reinforcement learning (DRL) for optimsing Battery Energy Storage Systems (BESSs) connected to the Australian National Electricity Market (NEM). Market data comes from AEMO via
[`nemosis`](https://github.com/UNSW-CEEM/NEMOSIS).

## Setup

Dependencies live in `pyproject.toml` and are pinned in `uv.lock`, so both paths below
install the same versions. `flake.nix` additionally pins the interpreter and system
libraries to a fixed nixpkgs revision, making the whole environment reproducible.

With Nix (fully pinned):

```bash
nix develop
uv sync
```

Without Nix: [install uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
```

`uv` fetches its own CPython, so no system Python is needed. Requires Python >= 3.12.

To add a dependency, put it in `pyproject.toml` and run `uv lock`.

## Running

Run the LP oracle over a price window:

```bash
uv run python -m cli lp 2024-01-10 2024-01-11 --no-calendar
uv run python -m cli lp --help
```

Tests:

```bash
uv run pytest
```

## Trained agents

A finished agent lives under `trainedPolicies/`. There is one layout:

```
trainedPolicies/<region>/<algo>-<trainMode>-s<seed>/
    agent.zip      the weights
    scaler.json    the price transform the weights were trained under
    policy.json    which command wrote it, from which window, and when
```

The directory name is the label the dashboard shows. `trainedPolicies/` is gitignored,
so an agent never travels by git.

`cli train` and `cli rolling` publish their own agents at the end of a run. Use
`cli publish` for an agent trained before that rule, or for one that arrived from
another machine.

To add an agent:

1. Find the checkpoint directory. It holds `agent.zip` and `scaler.json`.
2. Publish it. Name the region and the seed the checkpoint was trained on.

```bash
uv run python -m cli publish models/rolling2023_profit_fc4h/NSW1/seed4 \
    --region NSW1 --seeds 4
```

To add several seeds in one call, give one directory per seed, in the order of
`--seeds`:

```bash
uv run python -m cli publish \
    models/rolling2023_profit_fc4h/NSW1/seed0 \
    models/rolling2023_profit_fc4h/NSW1/seed4 \
    --region NSW1 --seeds 0 4
```

`--algo` and `--train-mode` set the rest of the label. They default to `maskableppo`
and `warm`.

Never build the path by hand. `rl/policies.py` is the only definition of it, and
`cli publish` is the only way to write one. The command copies `scaler.json` beside the
weights, because the transform is pinned to a market price cap. Weights read under a
different cap mean something else.

To remove an agent, delete its directory.

To move an agent to the dashboard machine, copy one region directory:

```bash
rsync -a trainedPolicies/NSW1/ dashboard-host:~/IntelligentDispatch/trainedPolicies/NSW1/
```

The current forecast screen lists the agents in `trainedPolicies/` and nothing else. An
agent that is not published there cannot run live.

## The Dashboard

If you would like to run it locally, the command is:

```bash
uv run python -m visualise.dashboard --results results/[your backtest results]
```

I host it with some models I've trained here: https://huggingface.co/spaces/toms55/IntelligentDispatch

## Data

`nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first run
for a date range is slow and network-bound; later runs are local. The directory is
gitignored and safe to delete.

## Licence

MIT. See [LICENSE](LICENSE).

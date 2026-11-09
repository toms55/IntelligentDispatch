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

`uv` fetches its own CPython, so no system Python is needed. The project uses Python 3.11.

To add a dependency, put it in `pyproject.toml` and run `uv lock`.

## Running

Run the LP oracle over a price window:

```bash
uv run python -m cli lp 2024-01-10 2024-01-11 --no-calendar
uv run python -m cli lp --help
```

The first run for any date range downloads AEMO archive files, so it is network-bound
and slower than later runs over the same range. See [Data](#data).

Tests and lint:

```bash
uv run pytest
ruff check .
```

The suite needs no network. `ruff` comes from the Nix dev shell, because the PyPI
binary does not start on NixOS. GitHub Actions runs both on every push to main.

## Score your own policy

Already have a trained policy? Wrap it in a small class and backtest it against the
exact oracle on this framework, using the same battery model and rules as the
published backtests, so the score means the same thing.

The oracle is the most profit that any policy can earn on the same prices. The score
shows the share of that profit your policy earns.

### Wrap your policy

Save this in a file named `my_policy.py` in the repository root. Point `load` at your
own checkpoint, and use the algorithm name you trained under (`maskableppo`, `dqn` or
`a2c`):

```python
from rl.algos import load


class MyPolicy:
    reads_mask = True  # False if your policy does not use the action mask

    def __init__(self):
        self.agent = load("maskableppo", "path/to/agent.zip")

    def predict(self, observation, deterministic=True, action_masks=None):
        return self.agent.predict(observation, deterministic=deterministic,
                                   action_masks=action_masks)
```

This works directly for a stable-baselines3 agent (MaskablePPO, DQN, A2C), since it
already has a matching `predict` method. For anything else, call your own policy
inside `predict` instead and return `(action, None)`, where `action` is `0` (charge),
`1` (idle) or `2` (discharge) — see `rl.env.CHARGE`/`IDLE`/`DISCHARGE`. The
observation is 62 values; `rl/features.py` names the position of each one.

### Run an experiment

1. Make sure you finished the Setup steps above.
2. Save your policy in `my_policy.py`.
3. If you want forecast prices, download the AEMO forecasts for your window first.

   ```bash
   uv run python -m cli prefetch 2025-03-03T04:00 2025-03-10T04:00 --region NSW1
   ```

4. Run the policy over the window.

   ```bash
   uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast
   ```

5. Run the same command with `realised` to fill the price slots with the real prices.

The first run also downloads the price data from AEMO. Without step 3, the run reads each forecast from AEMO. This is slow and can fail.

### Read the result

The command prints a summary and the settings.

- `agentProfit` is the profit of your policy in dollars.
- `lpProfit` is the profit of the oracle on the same window.
- `regret` is `lpProfit` minus `agentProfit`.
- `captured` is `agentProfit` divided by `lpProfit`.

The settings list every value that the score depends on. Quote them with the score. Two scores compare only if the settings match.

### Change the constraints

By default the battery spends at most 1 equivalent full cycle each day and starts each day empty. The oracle solves again for each new setting, so the bound always matches your policy. A new setting takes a few seconds on the first run.

```bash
# Make the battery end each day empty. The limit of 1 cycle each day still applies.
uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast \
    --end-empty

# Allow 2 cycles each day.
uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast \
    --cycle-budget 2

# Remove the cycle limit.
uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast \
    --no-cycle-budget

# Start each day at half charge. Value the energy left at the close at 40 $/MWh.
uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast \
    --initial-soc 0.5 --terminal-price 40

# Replace the forecast every 5 minutes. Use the same value for the download.
uv run python -m cli prefetch 2025-03-03T04:00 2025-03-10T04:00 --region NSW1 \
    --forecast-refresh 5
uv run python -m examples.run my_policy:MyPolicy NSW1 2025-03-03 2025-03-10 forecast \
    --forecast-refresh 5
```

Run `uv run python -m examples.run --help` to see every flag.

### Use Python instead of the CLI

Call `score` from your own code to get the full result. The `records` list has one entry for each episode, with every interval of the trajectory.

```python
from datetime import datetime
from backtest.score import score
from my_policy import MyPolicy

start, end = datetime(2025, 3, 3, 4), datetime(2025, 3, 10, 4)
result = score(MyPolicy(), "NSW1", start, end, "forecast", endEmpty=True)
result["summary"]    # agentProfit, lpProfit, regret, captured
result["records"]    # one entry for each episode
result["settings"]   # the values that the score depends on
```

The `score` function also takes `cycleBudget`, `terminalPrice`, `initialSoc`, `battery` and `forecastEvery`. The `cycleBudget` value is the limit for one episode, and an episode is one day by default. Set `cycleBudget=None` to remove the limit.

### Limits

- The policy picks one of three actions in each interval.
- The window starts and ends at 04:00, the NEM trading day boundary.
- The window holds whole days.

## Trained agents

A finished agent lives under `trainedPolicies/`. There is one layout:

```
trainedPolicies/<region>/<algo>-<trainMode>-s<seed>/
    agent.zip      the weights
    scaler.json    the price transform the weights were trained under
    policy.json    which command wrote it, from which window, and when
```

The directory name is the label the dashboard shows. `trainedPolicies/` lives in the
Hugging Face dataset
[`toms55/IntelligentDispatchExperiments`](https://huggingface.co/datasets/toms55/IntelligentDispatchExperiments),
alongside `results/` and `oracleResults/`. See [Experiment data](#experiment-data)
below for how to fetch it.

`cli train` and `cli rolling` publish their own agents at the end of a run, straight
into `trainedPolicies/`, so most of the time you do not need the steps below at all.
`cli train` publishes under `--train-mode cold` by default (a fresh start); `cli
rolling` publishes the last fold, which is `warm` (continued from an earlier
checkpoint). Use `cli publish` below only for an agent trained before this rule, or
for a checkpoint that arrived from another machine without going through either
command.

To add such an agent:

1. Find the checkpoint directory. It holds `agent.zip` and `scaler.json`. This is a
   directory `cli train`/`cli rolling` wrote under their own `--out` (default
   `runs/`), such as `runs/maskableppo_seed0`, not a path you create yourself.
2. Publish it. Name the region and the seed the checkpoint was trained on.

```bash
uv run python -m cli publish runs/maskableppo_seed4 \
    --region NSW1 --seeds 4
```

To add several seeds in one call, give one directory per seed, in the order of
`--seeds`:

```bash
uv run python -m cli publish \
    runs/maskableppo_seed0 \
    runs/maskableppo_seed4 \
    --region NSW1 --seeds 0 4
```

`--algo` and `--train-mode` set the rest of the label. They default to `maskableppo`
and `warm`.

Never build the path by hand. `rl/policies.py` is the only definition of it, and
`cli publish` is the only way to write one. The command copies `scaler.json` beside the
weights, because the transform is pinned to a market price cap. Weights read under a
different cap mean something else.

To remove an agent, delete its directory.

Publish a finished agent to the dataset with `hf upload` (see
[Experiment data](#experiment-data)) to move it to the dashboard machine.

The current forecast screen lists the agents in `trainedPolicies/` and nothing else. An
agent that is not published there cannot run live.

## The Dashboard

If you would like to run it locally, the command is:

```bash
uv run python -m visualise.dashboard --results results/backtestResults/<run>
```

`<run>` is a result name without a region suffix, e.g. `rolling2023_profit`, not a
file name: `results/backtestResults/` holds one CSV, one JSON manifest and a matching
Parquet file in `results/actions/` per run per region, named
`<run>_<region>.{csv,json}` and `<run>_<region>.parquet`. The dashboard needs all
three, for at least one region, or it refuses to start with a `FileNotFoundError`
naming the missing one. A finished backtest gets there through `uv run python -m
backtest.publish results/<run>`, which the folds job runs for you.

I host it with some models I've trained here: https://huggingface.co/spaces/toms55/IntelligentDispatch

## Experiment data

`results/`, `trainedPolicies/` and `oracleResults/` are not tracked by git. They live in
the Hugging Face dataset
[`toms55/IntelligentDispatchExperiments`](https://huggingface.co/datasets/toms55/IntelligentDispatchExperiments).
Fetch them with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
which is not one of this project's own dependencies and needs its own install:

```bash
uv tool install huggingface_hub[cli]
```

Then:

```bash
hf download toms55/IntelligentDispatchExperiments --repo-type=dataset --local-dir . \
    --include 'results/*' 'trainedPolicies/*' 'oracleResults/*'
```

Name the three directories. The dataset also holds its own `README.md` and
`.gitattributes` at the top level, and a download without `--include` writes both into
the repo root. That overwrites this file and adds LFS rules for `*.zip` and `*.parquet`
that this repository does not use.

Push new or changed results back up the same way:

```bash
hf upload toms55/IntelligentDispatchExperiments results results --repo-type=dataset
hf upload toms55/IntelligentDispatchExperiments trainedPolicies trainedPolicies --repo-type=dataset
hf upload toms55/IntelligentDispatchExperiments oracleResults oracleResults --repo-type=dataset
```

## Data

`nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first run
for a date range is slow and network-bound; later runs are local. The directory is
gitignored and safe to delete.

## Documentation

| Document | What it holds |
| --- | --- |
| [docs/methodology.md](docs/methodology.md) | The experiment design |
| [docs/architecture.md](docs/architecture.md) | The package tour, module by module |
| [docs/conventions.md](docs/conventions.md) | The rules a change has to keep |
| [docs/gotchas.md](docs/gotchas.md) | The traps, each one measured |
| [docs/ablation.md](docs/ablation.md) | Results of the 44 published ablation runs |

## Licence

MIT. See [LICENSE](LICENSE).

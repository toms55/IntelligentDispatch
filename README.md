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

Tests:

```bash
uv run pytest
```

## Score your own policy

You can test your own dispatch policy against the exact oracle. The oracle is the most profit that any policy can earn on the same prices. The score shows the share of that profit that your policy earns. The score uses the same battery model and rules as the published backtests.

### Write the policy

A policy is a Python class. It needs a `predict` method and a `reads_mask` attribute. The `predict` method gets the observation and returns an action and `None`. Set `reads_mask` to `True` if `predict` uses the `action_masks` argument. A `reset` method is optional. The runner calls `reset` before each episode.

The observation has 62 values. The file `rl/features.py` names the position of each value. The price values are divided by the market price cap. Each action runs at the full rated power for one 5-minute interval.

Save this class in a file named `my_policy.py` in the repository root:

```python
from rl.env import CHARGE, IDLE, DISCHARGE
from rl.features import priceSlice, socIndex


class MyPolicy:
    reads_mask = False

    def reset(self):uv run python -m visualise.dashboard --results results/backtestResults/rolling2023_profit
        pass

    def predict(self, observation, deterministic=True, action_masks=None):
        soc = observation[socIndex]
        prices = observation[priceSlice]   # now, then each later interval
        if prices[0] <= prices.min() and soc < 1.0:
            return CHARGE, None
        if prices[0] >= prices.max() and soc > 0.0:
            return DISCHARGE, None
        return IDLE, None
```

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

The directory name is the label the dashboard shows. `trainedPolicies/` is tracked by
git, so an agent travels with a commit.

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

`trainedPolicies/` is tracked by git. Commit a finished agent to move it to the
dashboard machine.

The current forecast screen lists the agents in `trainedPolicies/` and nothing else. An
agent that is not published there cannot run live.

## The Dashboard

If you would like to run it locally, the command is:

```bash
uv run python -m visualise.dashboard --results results/backtestResults/pathToYourBacktest
```

`results/backtestResults/` holds the backtests the dashboard draws, one CSV per run per
region, and git tracks it. The dispatch each day draws comes from the matching Parquet
file in `results/actions/`. A finished backtest gets there through `uv run python -m
backtest.publish results/<run>`, which the folds job runs for you.

I host it with some models I've trained here: https://huggingface.co/spaces/toms55/IntelligentDispatch

## Data

`nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first run
for a date range is slow and network-bound; later runs are local. The directory is
gitignored and safe to delete.

## Licence

MIT. See [LICENSE](LICENSE).

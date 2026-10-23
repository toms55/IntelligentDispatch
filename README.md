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

## Hosting the dashboard

The dashboard runs on Google Cloud Run. The service scales to zero, so an idle
dashboard costs nothing, and it reaches NEMWeb, which the current forecast screen
needs. Anyone with the link can open it.

Cloud Run rather than a Hugging Face Space: a Space that runs compute now needs a paid
Hugging Face plan, and only static Spaces are free. A Dash application is not static.
`scripts/publishSpace.sh` still works, and pushes the same staged directory, if you
hold that plan.

### What gets staged

`scripts/stageDashboard.sh` builds the directory both hosts deploy. It is not the
repository. It holds four things, and nothing else:

1. the packages the two screens import
2. one results directory: the fold rows the backtest screen replays
3. `trainedPolicies/`: one published agent per region
4. `deploy/`: the Dockerfile, the requirements and the card

The oracle, the solver and the AEMO archive readers stay out, because no screen solves.
`deploy/requirements.txt` therefore drops pulp, highspy, nemosis and nemseer, and takes
torch from the CPU index. Stage 4 of the script builds the app with those four names
blocked, so an import that reaches one of them fails at home rather than on the host.

### Deploy it

1. Install the Google Cloud CLI, then log in: `gcloud auth login`.
2. Create a project and attach a billing account to it. Cloud Run needs one even
   inside the free tier.
3. Add the hard stop, once. This is the part that makes the cost safe:

```bash
bash scripts/setupBillingGuard.sh --project <project-id> --budget 1
```

4. Stage, build and deploy:

```bash
bash scripts/deployCloudRun.sh --project <project-id>
```

The script prints the URL. The defaults take the 5 agents of seed 0, one per region,
from `models/rolling2023_profit_fc4h`, and the records from
`results/rolling2023_profit_all`. `--models`, `--seed` and `--results` change that. The
forecast screen offers one agent per region plus the threshold rule, so every region in
the dropdown can run.

A better agent reaches the host the same way: train it, then run the script again.

### What it costs

Almost nothing, but not exactly nothing, so here are the real numbers.

The free tier is 2 million requests, 180,000 vCPU-seconds and 360,000 GiB-seconds a
month. The service holds one instance at most, scales to zero, and allocates CPU during
a request only. A dashboard that gets visitors rather than traffic uses a low single
digit percentage of that. For the compute to cost anything, the app has to serve
requests for about 50 hours in one month.

Artifact Registry is the charge that is not zero. It gives 0.5 GB free, and the image
holds torch, so expect a few cents a month. Stage 3 of the deploy script deletes every
image but the live one to hold that down.

`scripts/setupBillingGuard.sh` is the stop. A Google budget only sends mail, so the
guard adds a Pub/Sub topic and the function in `deploy/billingGuard/`, which removes
the billing account from the project once the cost reaches the budget. That ends every
charge, and takes the dashboard down with it. A cost appears before its notification
does, so the project can run a little past the budget. Set the budget below what you
are willing to spend, not at it.

### Two limits

- The service sleeps when nobody visits, and the next request wakes it. That takes
  about a minute.
- The upload is about 75 MB, and most of it is the per-fold trajectories. A second
  results directory doubles it.

The app needs about 606 MB of memory to serve, which is why the service asks for 2 GiB.
torch is roughly 475 MB of that, and only the forecast screen needs it. A 512 MB host
can therefore run the backtest screen alone, and not both.

`deploy/app.py` is the entry point the host serves. It reads `DASHBOARD_REGION`,
`DASHBOARD_RESULTS` and `DASHBOARD_POLICIES`, because a host sets environment variables
and does not pass arguments. The container binds the port in `PORT`, which Cloud Run
chooses. Locally, run the module instead:

```bash
uv run python -m visualise.dashboard --results results/rolling2023_profit_all
```

## Data

`nemosis` downloads AEMO archive files into `data_cache/` on first use, so the first run
for a date range is slow and network-bound; later runs are local. The directory is
gitignored and safe to delete.

## Licence

MIT. See [LICENSE](LICENSE).

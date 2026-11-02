#!/usr/bin/env bash
# Stage the hosted dashboard, and push it to a Hugging Face Space.
#
#     bash scripts/publishSpace.sh --space toms55/IntelligentDispatch --push
#
# The Space runs on CPU Basic hardware, which carries no hourly cost. Creating a Space
# that runs compute needs a PRO plan, because only static Spaces are free, and a Dash
# application is not static. The Space runs the Dockerfile in `deploy/`, keeps the
# upload in a git repository, and reaches NEMWeb, which the current forecast screen
# needs.
#
# The upload holds four things and nothing else:
#
#   1. the packages the two screens import
#   2. the runs named by --results: the day rows the backtest screen replays, and their
#      actions exports, which hold the dispatch it draws
#   3. `trainedPolicies/`: the agents the forecast screen can load
#   4. the Dockerfile, the requirements and the Space card
#
# Both screens stage the same models. Stage 2 carries one seed per train mode and
# region, the one `config.best_seed_by_run` names, and stage 3 carries the agents of
# those same models, through `experiments.runnable`. So a model on the forecast screen
# has its numbers on the backtest screen at the same seed, and the thesis reports the
# seed spread from the full set that stays here, not from the Space.
#
# Name a run once per run to stage several:
#
#     bash scripts/publishSpace.sh --results results/backtestResults/rolling2023_profit \
#                                  --results results/backtestResults/rolling2023_profit_fc5m
#
# Both screens offer every staged run, and they open on the first one named.
# $DASHBOARD_RESULTS reorders or narrows the staged runs in the Space settings, with no
# republish.
#
# The training code is in the upload because the forecast screen runs an agent. The
# oracle, the archive readers and the oracle cache are not, because no screen solves.
set -euo pipefail

# The Space this repository publishes. Override it with --space, or with $DASHBOARD_SPACE
# for a test Space. Every caller reads it from here, so the name is in one place.
SPACE=${DASHBOARD_SPACE:-toms55/IntelligentDispatch}
# The runs to stage, in order. The first is the one the Space opens. --results replaces
# this default on its first use and appends on every use after it. The default is
# `config.dashboard_runs`, which is the same list the local dashboard offers, so the two
# cannot drift apart. A run stays off that list until its manifest states what it
# tested: see the comment there.
read -r -a RUNS <<< "$(bash "$(dirname "$0")/dashboardRuns.sh" | tr '\n' ' ')"
runsGiven=no
# The run's actions exports stay in results/actions/, which git tracks; see
# backtest/publish.py.
ACTIONS=results/actions
# The agents come from the published layout, not from a training output directory, so the
# Space runs the same bytes this machine's dashboard runs. Stage 3 stages the agents the
# forecast screen offers for the staged runs, so there is nothing to name here: the set
# is read straight off $POLICIES through `experiments.runnable`.
POLICIES=trainedPolicies
REGION=NSW1
STAGE=runs/space
PUSH=no
# The repository runs under `nix develop`, where `uv run` finds the pinned interpreter.
# Override it to use another one.
PYTHON=${PYTHON:-"uv run python"}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --space)      SPACE="$2"; shift 2 ;;
    --results)    if [[ "$runsGiven" == no ]]; then RUNS=("$2"); runsGiven=yes;
                  else RUNS+=("$2"); fi; shift 2 ;;
    --policies)   POLICIES="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    --stage)      STAGE="$2"; shift 2 ;;
    --push)       PUSH=yes; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n========== %s ==========\n' "$1"; }

# ------------------------------------------------------------- 0. the token
# The write token for the Space. Keep it out of the repository and out of the shell
# history. Three ways to supply it, in the order the script looks:
#
#   1. $HF_TOKEN, read from a secret store by the shell that calls this script:
#        HF_TOKEN=$(secret-tool lookup service huggingface) bash scripts/pushMain.sh
#   2. the file the Hugging Face CLI writes, after `hf auth login`
#   3. this script asks, on the terminal. The password it wants is the write token.
#
# This runs before stage 1, because a token problem must not cost a whole stage build.
# Never hand the push to git with no token. git then asks through SSH_ASKPASS, and a
# machine with no working askpass program pushes with empty credentials instead. Hugging
# Face answers that anonymous push with "You have read access but not the required
# permissions", which reads like a token with the wrong role and is not one.
readToken() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  local tokenFile="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
  if [[ -r "$tokenFile" ]]; then
    HF_TOKEN=$(tr -d '[:space:]' < "$tokenFile")
    export HF_TOKEN
    echo "token from $tokenFile"
    return 0
  fi
  # Read from the terminal, not from stdin. The pre-push hook gets its list of refs on
  # stdin, so a read there would eat them.
  # Open /dev/tty to test it. A readable mode bit is not enough: a job with no
  # controlling terminal still sees the device and cannot open it.
  HF_TOKEN=""
  export HF_TOKEN
  if { : < /dev/tty; } 2>/dev/null; then
    echo "no \$HF_TOKEN and no $tokenFile."
    printf 'Hugging Face write token for %s (it does not echo): ' "$SPACE"
    IFS= read -r -s HF_TOKEN < /dev/tty || true
    printf '\n'
    HF_TOKEN=$(printf '%s' "$HF_TOKEN" | tr -d '[:space:]')
  fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "no write token for spaces/$SPACE, so nothing is pushed." >&2
    echo "Supply one of these:" >&2
    echo "  export HF_TOKEN=<write token>, then run this again" >&2
    echo "  hf auth login, which writes $tokenFile" >&2
    exit 1
  fi
}

# A token that only reads authenticates and then fails at the upload. Ask the Space's
# own LFS endpoint first, because that endpoint is the one that refuses, and its answer
# covers a classic token and a fine-grained token alike. The object is a dummy, so this
# uploads nothing.
#
# The token goes to curl through stdin, so it stays off the command line.
checkToken() {
  if ! command -v curl >/dev/null; then
    return 0
  fi
  local status
  status=$(
    printf 'header = "Authorization: Bearer %s"\n' "$HF_TOKEN" | curl -s --config - \
      -o /dev/null -w '%{http_code}' \
      -X POST "https://huggingface.co/spaces/$SPACE.git/info/lfs/objects/batch" \
      -H "Accept: application/vnd.git-lfs+json" \
      -H "Content-Type: application/vnd.git-lfs+json" \
      --data '{"operation":"upload","transfers":["basic"],"ref":{"name":"refs/heads/main"},
               "objects":[{"oid":"0000000000000000000000000000000000000000000000000000000000000000","size":1}]}' \
      || true
  )
  # Only an authentication answer stops the run. Any other status is the endpoint
  # talking about the dummy object, which says nothing about the token.
  case "$status" in
    401)
      echo "Hugging Face rejected the token for spaces/$SPACE." >&2
      echo "Check that you copied all of it, and that it is not revoked." >&2
      exit 1 ;;
    403)
      echo "the token reaches Hugging Face, and it cannot write to spaces/$SPACE." >&2
      echo "Make a new token at https://huggingface.co/settings/tokens:" >&2
      echo "  a classic token needs the write role," >&2
      echo "  a fine-grained token needs write access to contents for" >&2
      echo "  spaces/$SPACE, which you select in its repository list." >&2
      exit 1 ;;
    404)
      echo "Hugging Face cannot see spaces/$SPACE with this token." >&2
      exit 1 ;;
    000)
      echo "cannot reach huggingface.co, so the token is unchecked." >&2 ;;
    *)
      echo "token writes to spaces/$SPACE" ;;
  esac
}

if [[ "$PUSH" == yes && -n "$SPACE" ]]; then
  say "0/5 token"
  readToken
  checkToken
fi

# ---------------------------------------------------------------- 1. the code
say "1/5 code"
rm -rf "$STAGE"
mkdir -p "$STAGE"
for item in config.py utils data degradation rl backtest visualise deploy; do
  cp -r "$item" "$STAGE/"
done
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name 'test_*.py' -delete
# The Space builds from its own root, so the Dockerfile and the card move up.
mv "$STAGE/deploy/Dockerfile" "$STAGE/Dockerfile"
mv "$STAGE/deploy/space.md" "$STAGE/README.md"
printf '__pycache__/\ndata_cache/\n' > "$STAGE/.gitignore"
# A Space refuses a binary file that git tracks directly, so the two binary kinds in
# the upload go through LFS. This file has to exist before the first `git add`, or the
# filter never runs. The agent weights are `.zip`, the actions exports `.parquet`.
# Everything else in the upload is text.
printf '*.zip filter=lfs diff=lfs merge=lfs -text\n' > "$STAGE/.gitattributes"
printf '*.parquet filter=lfs diff=lfs merge=lfs -text\n' >> "$STAGE/.gitattributes"

# ------------------------------------------------------------- 2. the results
say "2/5 results from ${#RUNS[@]} run(s), one seed per model"
# The same layout every clone has, so the app finds a run without being told. A run is
# one CSV per region, <run>_<region>.csv, and its actions export one Parquet,
# <run>_<region>.parquet.
#
# Only the rows of the seed `config.best_seed_by_run` names go up, per train mode and
# region. That is the seed the dashboard shows, through `experiments.bestSeed`, so
# dropping the others removes nothing from the screen and changes no number on it. It
# also keeps the actions exports small enough to stage several runs.
mkdir -p "$STAGE/results/actions"

staged=0
for RESULTS in "${RUNS[@]}"; do
  run=$(basename "$RESULTS")
  published=$(dirname "$RESULTS")
  mkdir -p "$STAGE/$published"
  # One interpreter per run, not per region: it reads the published rows, keeps the
  # pinned seed of every train mode, and writes the CSV and the Parquet together, so
  # the two cannot disagree about which seed the Space holds.
  found=$($PYTHON - "$published" "$run" "$ACTIONS" "$STAGE" <<'PYTHON'
import shutil
import sys
from pathlib import Path

import pandas as pd

from backtest import records
from visualise.dashboard import data, experiments

published, run, actionsRoot, stage = sys.argv[1:5]
resultsDir = Path(published) / run
kept = 0
for region in data.availableRegions(resultsDir):
    days = pd.read_csv(data.publishedFile(resultsDir, region))
    # The seed of each train mode, which is what the dropdown offers for this run.
    seeds = {mode: experiments.bestSeed(run, mode, region, days.seed.unique())
             for mode in sorted(set(days.trainMode))}
    # Vectorised, not `apply`: the actions export is half a million rows per region.
    wanted = days.seed == days.trainMode.map(seeds)
    if not wanted.any():
        sys.exit(f"{resultsDir} {region} holds no row of {seeds}, so the Space would "
                 f"draw an empty screen. Its seeds: "
                 f"{sorted(days.seed.unique().tolist())}")
    out = Path(stage) / published / f"{run}_{region}.csv"
    days[wanted].to_csv(out, index=False)
    # The manifest and the description travel with the rows. `experiments.catalogue`
    # reads them to describe the run, so a staged run without its JSON is a run the
    # screens cannot offer.
    context = records.contextPath(published, run, region)
    shutil.copy2(context, records.contextPath(Path(stage) / published, run, region))
    # The export holds the dispatch the day view draws, the forecast the agent acted on
    # and the refresh tile's check. It is Parquet, a few MB a seed.
    #
    # An export written by `python -m backtest.actions` holds seed 0 only, because it
    # comes from the fold trajectories and those record one seed. Asking it for another
    # seed stops the publish here. The day view must not fall back to seed 0 under
    # another seed's name, and a Space whose calendar draws but whose days are all
    # empty is worse than one that was never pushed.
    source = Path(actionsRoot) / f"{run}_{region}.parquet"
    if source.exists():
        frame = pd.read_parquet(source)
        rows = frame[frame.seed == frame.trainMode.map(seeds)]
        if rows.empty:
            sys.exit(f"{source} holds no row of {seeds}, so every day view would be "
                     f"empty. Its seeds: {sorted(frame.seed.unique().tolist())}\n"
                     "Either pin a seed the export holds in config.best_seed_by_run, "
                     "or re-run the\nbacktest, which logs every seed as each fold "
                     "finishes.")
        rows.to_parquet(Path(stage) / "results" / "actions" /
                        f"{run}_{region}.parquet", index=False,
                        compression="zstd", compression_level=9)
    kept += 1
print(kept)
PYTHON
)
  [[ ${found:-0} -gt 0 ]] || { echo "no published run at $RESULTS" >&2; exit 1; }
  echo "  $run: $found regions"
  staged=$((staged + 1))
done
# Every staged run is pinned into the image, so both screens offer all of them and a
# reader compares the experiments in one dropdown. The screens open on the first. A
# Space variable of the same name reorders or narrows them without a republish.
STAGED_RUNS=$(IFS=,; printf '%s' "${RUNS[*]}")
DEFAULT_RUN=${RUNS[0]}
printf '\nENV DASHBOARD_RESULTS=%s\n' "$STAGED_RUNS" >> "$STAGE/Dockerfile"
echo "$staged run(s), $(du -sh "$STAGE/results" | cut -f1), opening on $(basename "$DEFAULT_RUN")"

# -------------------------------------------------------------- 3. the agents
# The agents the forecast screen offers for the staged runs, and no others. That list
# comes from `experiments.runnable`, which keeps an agent only when
# `rl.policies.loadable` can run it and a staged experiment scored it, at the seed
# `config.best_seed_by_run` pins. So every model on the Space has its numbers on the
# backtest screen, at the same seed, and the weights that go up are a handful per region
# rather than every seed of every experiment ever trained.
#
# `loadable` reads each agent's recorded `observationSize` against
# `config.observation_size` and drops anything trained under another observation layout,
# so a staged agent and the live screen's environment always agree. Staging a
# hand-picked list here was the earlier bug: `policies.publish` stamps the *current*
# `config.observation_size` onto whatever it copies, so re-publishing something
# `loadable` had already excluded silently relabelled it as compatible and the live
# screen crashed on the first rollout. Filtering through `runnable` first is what keeps
# that stamp honest.
say "3/5 agents from $POLICIES, the models the forecast screen offers"
$PYTHON - "$POLICIES" "$STAGE/trainedPolicies" "${RUNS[@]}" <<'PYTHON'
import functools
import sys
from pathlib import Path

from rl import policies
from visualise.dashboard import data, experiments

source, root = sys.argv[1:3]
runs = [Path(item).name for item in sys.argv[3:]]
results = Path(sys.argv[3]).parent


# One read per run and region. `runnable` loads every staged run to find the models of
# one region, and each load checks that run's actions export, which is tens of
# megabytes, so the uncached loop reads every export once per region.
@functools.lru_cache(maxsize=64)
def read(run, region):
    return data.loadRun(region, results / run)


staged = 0
for region in policies.regions(source):
    for item in experiments.runnable(region, results, source, loader=read,
                                     runs=tuple(runs)):
        facts = policies.describe(item.directory)
        label = item.directory.name
        algo, trainMode, seed = facts["algo"], facts["trainMode"], int(facts["seed"])
        middle = label[len(algo) + 1:-len(f"-s{seed}")]
        tag = middle[len(trainMode) + 1:] if middle.startswith(f"{trainMode}-") else None
        # `experiments.policyRun`, not the recorded field: an agent published before
        # that field existed carries only its checkpoint path, and re-publishing
        # replaces that path with this one. The resolved link has to be written out, or
        # the staged agent loses the experiment it belongs to and the Space offers
        # nothing.
        published = policies.publish(item.directory, region, algo, trainMode, seed,
                                     root=root, tag=tag,
                                     rawRun=experiments.policyRun(item.directory),
                                     publishedBy="scripts/publishSpace.sh")
        print(f"  {region}: {published.name} for {item.entry.run}")
        staged += 1
if staged == 0:
    raise SystemExit(f"{source} holds no agent of a staged run that "
                     "`policies.loadable` accepts under today's "
                     "config.observation_size, so the forecast screen would offer "
                     "nothing. Publish a freshly trained agent, or retrain one under "
                     "the current observation layout.")
PYTHON

# --------------------------------------------------------------- 4. the check
say "4/5 smoke test with the solver and archive packages blocked"
# The image carries no pulp, no highspy and no nemosis. If any import on the dashboard
# path reaches one of them, the Space fails on its first request rather than here, so
# the check runs now, with those names blocked. It runs an agent over a made up price
# path rather than the AEMO one, so the check needs no network.
DASHBOARD_RESULTS="$STAGED_RUNS" \
DASHBOARD_POLICIES=trainedPolicies \
DASHBOARD_REGION="$REGION" \
  $PYTHON - "$STAGE" <<'PYTHON'
import builtins
import os
import sys
from pathlib import Path

stage = Path(sys.argv[1]).resolve()
os.chdir(stage)

blocked = ("pulp", "highspy", "nemosis", "nemseer", "matplotlib")
realImport = builtins.__import__


def guard(name, *args, **kwargs):
    if name.split(".")[0] in blocked:
        raise AssertionError(
            f"the dashboard path imports {name}, which the image does not install. "
            f"Add it to deploy/requirements.txt, or keep it off the path."
        )
    return realImport(name, *args, **kwargs)


builtins.__import__ = guard
from deploy import app  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

region = os.environ["DASHBOARD_REGION"]
# `app.resultsDirs` is the staged list the image names, parsed by `deploy/app.py`, so
# the check reads the runs the Space will read and not a second copy of that list.
runs = app.resultsDirs
root, names = app.dashboard.staged(runs)
models = app.dashboard.liveModels(region, app.policyRoot, runs=names, root=root)
for run in runs:
    print(f"  regions of {run.name}: {app.data.availableRegions(run)}")
print(f"  models on the forecast screen: {[item.entry.label for item in models]}")
# One name per model a staged run scored and the staged weights can run, so the count
# is the staged runs times their train modes.
if not models:
    raise AssertionError(f"{app.policyRoot} published no agent of "
                         f"{', '.join(run.name for run in runs)} for {region}")

# A day of prices with one peak, so a working agent charges and discharges once.
index = pd.date_range("2026-01-01 04:05", periods=288, freq="5min")
prices = pd.Series(60 + 240 * np.sin(np.linspace(0, 2 * np.pi, 288)), index=index)
for model in models:
    frame = app.live.suggest(prices, model, region, root=app.policyRoot)
    taken = sorted(set(frame.action))
    print(f"  {model.entry.label}: {len(frame)} intervals, actions {taken}, "
          f"profit ${frame.profit.iloc[-1]:,.2f}")
builtins.__import__ = realImport
PYTHON

# ---------------------------------------------------------------- 5. the push
say "5/5 upload"
echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"
if [[ -z "$SPACE" ]]; then
  echo "no --space given, so nothing is pushed. Run it again with:"
  echo "  bash scripts/publishSpace.sh --space <user>/<name> --push"
  exit 0
fi

# A Space is a git repository. Push the stage as one commit, so an update replaces the
# agents and the results rather than growing the history with every version of them.
(
  cd "$STAGE"
  git init -q -b main
  # `--local` keeps the filter in this throwaway repository, and out of the user's
  # global git configuration.
  git lfs install --local >/dev/null
  git add -A
  git -c user.email=publish@localhost -c user.name=publish commit -q \
    -m "dashboard: ${#RUNS[@]} run(s) in one dropdown, opening on $(basename "$DEFAULT_RUN"), one seed per model"
  git remote add space "https://huggingface.co/spaces/$SPACE"
  if [[ "$PUSH" == yes ]]; then
    echo "pushing to https://huggingface.co/spaces/$SPACE"
    # The token stays in the environment of this one command. It is not in the remote
    # URL, not in any git configuration file, and not on the command line, so it cannot
    # reach the repository or a shell history. Stage 0 has a token by now, so block
    # every prompt: a prompt here means the helper failed, and an unanswered prompt
    # pushes with empty credentials.
    env -u GIT_ASKPASS -u SSH_ASKPASS HF_TOKEN="$HF_TOKEN" GIT_TERMINAL_PROMPT=0 git \
      -c credential.helper='!f() { test "$1" = get && printf "username=hf\npassword=%s\n" "$HF_TOKEN"; }; f' \
      push --force space main
  else
    echo "committed. To upload:"
    echo "  cd $STAGE && git push --force space main"
  fi
)

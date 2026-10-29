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
#   3. `trainedPolicies/`: one published agent per region, for the forecast screen
#   4. the Dockerfile, the requirements and the Space card
#
# Both 2 and 3 carry one seed per region, the one `config.best_seed` names. Every seed
# stays here, because the thesis reports the spread over seeds. The Space shows the one,
# so a reader sees one dispatch per region rather than five that differ only by seed.
#
# Name a run once per run to stage several:
#
#     bash scripts/publishSpace.sh --results results/backtestResults/rolling2023_profit \
#                                  --results results/backtestResults/rolling2023_profit_fc5m
#
# The screen draws one run at a time, so the first one named is the one the Space opens.
# $DASHBOARD_RESULTS switches to another staged run in the Space settings, with no
# republish.
#
# The training code is in the upload because the forecast screen runs an agent. The
# oracle, the archive readers and the oracle cache are not, because no screen solves.
set -euo pipefail

# The Space this repository publishes. Override it with --space, or with $DASHBOARD_SPACE
# for a test Space. Every caller reads it from here, so the name is in one place.
SPACE=${DASHBOARD_SPACE:-toms55/IntelligentDispatch}
# The runs to stage, in order. The first is the one the Space opens. --results replaces
# this default on its first use and appends on every use after it.
RUNS=(results/backtestResults/rolling2023_profit_fc5m)
runsGiven=no
# The run's actions exports stay in results/actions/, which git tracks; see
# backtest/publish.py.
ACTIONS=results/actions
# The agents come from the published layout, not from a training output directory, so the
# Space runs the same bytes this machine's dashboard runs. `config.best_seed` picks which
# one per region.
POLICIES=trainedPolicies
ALGO=maskableppo
TRAIN_MODE=warm
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
    --algo)       ALGO="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
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

# The seed per region, from `config.best_seed`. Read once, because stage 2 and stage 3
# must stage the same seed: a result row the forecast screen cannot reproduce with the
# agent beside it would be two different models under one name.
declare -A BEST_SEED=()
while read -r region seed; do
  BEST_SEED["$region"]=$seed
done < <(bash "$(dirname "$0")/bestSeeds.sh")

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
# filter never runs. The agent weights are `.zip`, the gzipped actions exports `.gz`.
# Everything else in the upload is text.
printf '*.zip filter=lfs diff=lfs merge=lfs -text\n' > "$STAGE/.gitattributes"
printf '*.gz filter=lfs diff=lfs merge=lfs -text\n' >> "$STAGE/.gitattributes"

# ------------------------------------------------------------- 2. the results
say "2/5 results from ${#RUNS[@]} run(s), one seed per region"
# The same layout every clone has, so the app finds a run without being told. A run is
# one CSV per region, <run>_<region>.csv, and so is its actions export.
#
# Only the rows of `config.best_seed[region]` go up. Nothing on either screen pools
# seeds: every figure comes from `data.modelRows` for one dropdown label, so dropping
# the other seeds removes entries from the dropdown and changes no number in the ones
# that stay. It also keeps the actions exports small enough to stage several runs.
mkdir -p "$STAGE/results/actions"

# Keep the header, then the rows whose `seed` column equals $2. The column is found by
# name, because the two kinds of CSV order their columns differently. Both are plain
# CSV with no quoted field, and a quote would break a split on commas, so one appearing
# stops the publish rather than writing a file with rows silently dropped.
seedRows() {
  awk -F, -v want="$2" '
    /"/ { bad = "quoted field, which this filter cannot split"; exit 1 }
    NR == 1 {
      for (i = 1; i <= NF; i++) if ($i == "seed") column = i
      if (!column) { bad = "no seed column"; exit 1 }
      print; next
    }
    $column == want { kept++; print }
    END {
      if (!bad && !kept) bad = "no seed " want " rows"
      if (bad) { print bad " in " FILENAME > "/dev/stderr"; exit 1 }
    }
  ' "$1"
}

staged=0
for RESULTS in "${RUNS[@]}"; do
  run=$(basename "$RESULTS")
  published=$(dirname "$RESULTS")
  mkdir -p "$STAGE/$published"
  found=0
  for path in "$published"/"$run"_*.csv; do
    [[ -e "$path" ]] || continue
    # Only this run, not a longer name that starts with it (rolling2023_profit_fc4h_...).
    region=$(basename "$path" .csv); region=${region#"$run"_}
    [[ "$region" == *_* ]] && continue
    seed=${BEST_SEED[$region]:-}
    if [[ -z "$seed" ]]; then
      echo "  skip $region: config.best_seed names no seed for it"
      continue
    fi
    seedRows "$path" "$seed" > "$STAGE/$published/$run"_"$region.csv"
    # The export holds the dispatch the day view draws, the forecast the agent acted on
    # and the refresh tile's check. gzip cuts each region from tens of MB to a few, and
    # pandas reads it the same way.
    #
    # An export written by `python -m backtest.actions` holds seed 0 only, because it
    # comes from the fold trajectories and those record one seed. Asking it for another
    # seed stops the publish here. The day view must not fall back to seed 0 under
    # another seed's name, and a Space whose calendar draws but whose days are all
    # empty is worse than one that was never pushed.
    actionsCsv="$ACTIONS/$run"_"$region.csv"
    if [[ -e "$actionsCsv" ]]; then
      if ! seedRows "$actionsCsv" "$seed" 2>/dev/null \
           | gzip -c > "$STAGE/results/actions/$run"_"$region.csv.gz"; then
        echo "$actionsCsv holds no seed $seed, so the day view of $region would be empty." >&2
        echo "Its seeds: $(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="seed") c=i; next} {print $c}' \
                            "$actionsCsv" | sort -u | tr '\n' ' ')" >&2
        echo "Either set config.best_seed[$region] to a seed the export holds, or" >&2
        echo "re-run the backtest, which logs every seed as each fold finishes." >&2
        exit 1
      fi
    fi
    found=$((found + 1))
  done
  [[ $found -gt 0 ]] || { echo "no published run at $RESULTS" >&2; exit 1; }
  echo "  $run: $found regions"
  staged=$((staged + 1))
done
# The screen draws one run, and `data.findResults` refuses to choose between several, so
# the first run named is pinned into the image. A Space variable of the same name picks
# another staged run without a republish.
DEFAULT_RUN=${RUNS[0]}
printf '\nENV DASHBOARD_RESULTS=%s\n' "$DEFAULT_RUN" >> "$STAGE/Dockerfile"
echo "$staged run(s), $(du -sh "$STAGE/results" | cut -f1), opening on $(basename "$DEFAULT_RUN")"

# -------------------------------------------------------------- 3. the agents
# One agent per region, the seed `config.best_seed` names, so every region in the
# dropdown can run the forecast screen. This machine keeps every seed under $POLICIES
# for the thesis, and the Space gets the one, matching the rows stage 2 staged.
# `rl/policies.py` is the only code that writes this layout, so the publishing goes
# through it rather than through cp.
say "3/5 agents from $POLICIES, the best seed per region"
$PYTHON - "$POLICIES" "$STAGE/trainedPolicies" "$ALGO" "$TRAIN_MODE" <<'PYTHON'
import sys
from pathlib import Path

import config
from rl import policies

source, root, algo, trainMode = sys.argv[1:5]
for region, seed in sorted(config.best_seed.items()):
    label = policies.label(algo, trainMode, seed)
    directory = policies.available(region, source).get(label)
    if directory is None:
        raise SystemExit(
            f"{source}/{region} holds no {label}, so the forecast screen would offer "
            f"{region} a seed the backtest rows do not show. Publish it with "
            f"`python -m cli publish`, or correct config.best_seed."
        )
    published = policies.publish(directory, region, algo, trainMode, seed, root=root,
                                 publishedBy="scripts/publishSpace.sh")
    print(f"  {region}: {published.name}")
PYTHON

# --------------------------------------------------------------- 4. the check
say "4/5 smoke test with the solver and archive packages blocked"
# The image carries no pulp, no highspy and no nemosis. If any import on the dashboard
# path reaches one of them, the Space fails on its first request rather than here, so
# the check runs now, with those names blocked. It runs an agent over a made up price
# path rather than the AEMO one, so the check needs no network.
DASHBOARD_RESULTS="$DEFAULT_RUN" \
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
models = app.dashboard.liveModels(region, app.policyRoot)
print(f"  regions on the backtest screen: {app.data.availableRegions(app.resultsDir)}")
print(f"  models on the forecast screen: {models}")
# The list holds one name per published agent and nothing else, so one agent per region
# is a full list. It was longer when the screen still offered models it could not load.
if not models:
    raise AssertionError(f"{app.policyRoot} published no agent for {region}")

# A day of prices with one peak, so a working agent charges and discharges once.
index = pd.date_range("2026-01-01 04:05", periods=288, freq="5min")
prices = pd.Series(60 + 240 * np.sin(np.linspace(0, 2 * np.pi, 288)), index=index)
for label in models:
    frame = app.live.suggest(prices, label, region, root=app.policyRoot)
    taken = sorted(set(frame.action))
    print(f"  {label}: {len(frame)} intervals, actions {taken}, "
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
    -m "dashboard: $(basename "$DEFAULT_RUN") of ${#RUNS[@]} run(s), best seed per region"
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

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
#   2. one results directory: the fold rows the backtest screen replays
#   3. `trainedPolicies/`: one published agent per region, for the forecast screen
#   4. the Dockerfile, the requirements and the Space card
#
# The training code is in the upload because the forecast screen runs an agent. The
# oracle, the archive readers and the oracle cache are not, because no screen solves.
set -euo pipefail

SPACE=""
RESULTS=results/rolling2023_profit_all
MODELS=models/rolling2023_profit_fc4h
ALGO=maskableppo
TRAIN_MODE=warm
SEED=0
REGION=NSW1
STAGE=runs/space
PUSH=no
# The repository runs under `nix develop`, where `uv run` finds the pinned interpreter.
# Override it to use another one.
PYTHON=${PYTHON:-"uv run python"}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --space)      SPACE="$2"; shift 2 ;;
    --results)    RESULTS="$2"; shift 2 ;;
    --models)     MODELS="$2"; shift 2 ;;
    --algo)       ALGO="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    --stage)      STAGE="$2"; shift 2 ;;
    --push)       PUSH=yes; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n========== %s ==========\n' "$1"; }

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
# filter never runs. The agent weights are `.zip`, the per-fold trajectories `.parquet`.
# Everything else in the upload is text.
printf '*.zip filter=lfs diff=lfs merge=lfs -text\n' > "$STAGE/.gitattributes"
printf '*.parquet filter=lfs diff=lfs merge=lfs -text\n' >> "$STAGE/.gitattributes"

# ------------------------------------------------------------- 2. the results
say "2/5 results from $RESULTS"
test -d "$RESULTS" || { echo "no such results directory: $RESULTS" >&2; exit 1; }
target="$STAGE/results/$(basename "$RESULTS")"
mkdir -p "$target"
# Only what the backtest screen opens: the day rows, the trajectories and the manifest.
(cd "$RESULTS" && find . -name 'fold*.csv' -o -name 'fold*.parquet' \
  -o -name 'manifest.json') | while read -r path; do
  mkdir -p "$target/$(dirname "$path")"
  cp "$RESULTS/$path" "$target/$path"
done
echo "$(find "$target" -type f | wc -l) files, $(du -sh "$target" | cut -f1)"

# -------------------------------------------------------------- 3. the agents
# One agent per region, all of one seed, so every region in the dropdown can run the
# forecast screen. `rl/policies.py` is the only code that writes this layout, so the
# publishing goes through it rather than through cp.
say "3/5 agents from $MODELS, seed $SEED"
$PYTHON - "$MODELS" "$STAGE/trainedPolicies" "$ALGO" "$TRAIN_MODE" "$SEED" <<'PYTHON'
import sys
from pathlib import Path

from rl import policies

source, root, algo, trainMode, seed = sys.argv[1:6]
source = Path(source)
for region in sorted(child.name for child in source.iterdir() if child.is_dir()):
    checkpoint = source / region / f"seed{seed}"
    if not (checkpoint / policies.agentFile).exists():
        print(f"  skip {region}: no seed {seed}")
        continue
    directory = policies.publish(checkpoint, region, algo, trainMode, seed, root=root,
                                 run=source.name, publishedBy="scripts/publishSpace.sh")
    print(f"  {region}: {directory.name}")
PYTHON

# --------------------------------------------------------------- 4. the check
say "4/5 smoke test with the solver and archive packages blocked"
# The image carries no pulp, no highspy and no nemosis. If any import on the dashboard
# path reaches one of them, the Space fails on its first request rather than here, so
# the check runs now, with those names blocked. It runs an agent over a made up price
# path rather than the AEMO one, so the check needs no network.
DASHBOARD_RESULTS="results/$(basename "$RESULTS")" \
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
if len(models) < 2:
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

# The write token for the Space. Keep it out of the repository and out of the shell
# history. Three ways to supply it, in the order the script looks:
#
#   1. $HF_TOKEN, read from a secret store by the shell that calls this script:
#        HF_TOKEN=$(secret-tool lookup service huggingface) bash scripts/pushMain.sh
#   2. the file the Hugging Face CLI writes, after `hf auth login`
#   3. nothing, and git asks. The password it wants is the write token.
if [[ -z "${HF_TOKEN:-}" ]]; then
  tokenFile="${HF_TOKEN_PATH:-${HF_HOME:-$HOME/.cache/huggingface}/token}"
  if [[ -r "$tokenFile" ]]; then
    HF_TOKEN=$(tr -d '[:space:]' < "$tokenFile")
    export HF_TOKEN
    echo "token from $tokenFile"
  fi
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
    -m "dashboard: $(basename "$RESULTS"), agents from $(basename "$MODELS") seed $SEED"
  git remote add space "https://huggingface.co/spaces/$SPACE"
  if [[ "$PUSH" == yes ]]; then
    echo "pushing to https://huggingface.co/spaces/$SPACE"
    # The token stays in the environment of this one command. It is not in the remote
    # URL, not in any git configuration file, and not on the command line, so it cannot
    # reach the repository or a shell history. With no token, git asks for one, and the
    # password it asks for is a write token rather than an account password.
    if [[ -n "${HF_TOKEN:-}" ]]; then
      HF_TOKEN="$HF_TOKEN" git \
        -c credential.helper='!f() { test "$1" = get && printf "username=hf\npassword=%s\n" "$HF_TOKEN"; }; f' \
        push --force space main
    else
      git push --force space main
    fi
  else
    echo "committed. To upload:"
    echo "  cd $STAGE && git push --force space main"
  fi
)

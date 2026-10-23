#!/usr/bin/env bash
# Stage the hosted dashboard into one directory, and check it.
#
#     bash scripts/stageDashboard.sh --stage runs/space
#
# Both hosts build from the directory this writes, so the dashboard has one definition
# and the two hosts cannot drift apart. `scripts/publishSpace.sh` pushes the directory
# to a Hugging Face Space. `scripts/deployCloudRun.sh` builds it on Google Cloud Run.
#
# The staged directory holds four things and nothing else:
#
#   1. the packages the two screens import
#   2. one results directory: the fold rows the backtest screen replays
#   3. `trainedPolicies/`: one published agent per region, for the forecast screen
#   4. the Dockerfile, the requirements and the card
#
# The training code is in the directory because the forecast screen runs an agent. The
# oracle, the archive readers and the oracle cache are not, because no screen solves.
set -euo pipefail

RESULTS=results/rolling2023_profit_all
MODELS=models/rolling2023_profit_fc4h
ALGO=maskableppo
TRAIN_MODE=warm
SEED=0
REGION=NSW1
STAGE=runs/space
# The repository runs under `nix develop`, where `uv run` finds the pinned interpreter.
# Override it to use another one.
PYTHON=${PYTHON:-"uv run python"}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --results)    RESULTS="$2"; shift 2 ;;
    --models)     MODELS="$2"; shift 2 ;;
    --algo)       ALGO="$2"; shift 2 ;;
    --train-mode) TRAIN_MODE="$2"; shift 2 ;;
    --seed)       SEED="$2"; shift 2 ;;
    --region)     REGION="$2"; shift 2 ;;
    --stage)      STAGE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n========== %s ==========\n' "$1"; }

# ---------------------------------------------------------------- 1. the code
say "1/4 code"
rm -rf "$STAGE"
mkdir -p "$STAGE"
for item in config.py utils data degradation rl backtest visualise deploy; do
  cp -r "$item" "$STAGE/"
done
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$STAGE" -name 'test_*.py' -delete
# Both hosts build from the staged root, so the Dockerfile and the card move up.
mv "$STAGE/deploy/Dockerfile" "$STAGE/Dockerfile"
mv "$STAGE/deploy/space.md" "$STAGE/README.md"
printf '__pycache__/\ndata_cache/\n' > "$STAGE/.gitignore"
# Cloud Build reads `.gcloudignore` and skips `.gitignore` when this file exists. The
# two hold the same names, so the Cloud Build upload and the Space commit carry the
# same files.
printf '__pycache__/\ndata_cache/\n' > "$STAGE/.gcloudignore"

# ------------------------------------------------------------- 2. the results
say "2/4 results from $RESULTS"
test -d "$RESULTS" || { echo "no such results directory: $RESULTS" >&2; exit 1; }
target="$STAGE/results/$(basename "$RESULTS")"
mkdir -p "$target"
# Only what the backtest screen opens: the day rows, the trajectories and the manifest.
find "$RESULTS" -name 'fold*.csv' -o -name 'fold*.parquet' -o -name 'manifest.json' \
  | while read -r path; do
      relative="${path#"$RESULTS"/}"
      mkdir -p "$target/$(dirname "$relative")"
      cp "$path" "$target/$relative"
    done
echo "$(find "$target" -type f | wc -l) files, $(du -sh "$target" | cut -f1)"

# -------------------------------------------------------------- 3. the agents
# One agent per region, all of one seed, so every region in the dropdown can run the
# forecast screen. `rl/policies.py` is the only code that writes this layout, so the
# publishing goes through it rather than through cp.
say "3/4 agents from $MODELS, seed $SEED"
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
                                 run=source.name,
                                 publishedBy="scripts/stageDashboard.sh")
    print(f"  {region}: {directory.name}")
PYTHON

# --------------------------------------------------------------- 4. the check
say "4/4 smoke test with the solver and archive packages blocked"
# The image carries no pulp, no highspy and no nemosis. If any import on the dashboard
# path reaches one of them, the host fails on its first request rather than here, so
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

echo
echo "staged $(du -sh "$STAGE" | cut -f1) in $STAGE"

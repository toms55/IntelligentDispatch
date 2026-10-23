#!/usr/bin/env bash
# Push the hosted dashboard to a Hugging Face Space.
#
#     bash scripts/publishSpace.sh --space tomstorey/nem-bess --push
#
# `scripts/stageDashboard.sh` builds the directory this uploads. Every argument except
# `--space` and `--push` goes straight to that script.
#
# A Space that runs compute needs a paid Hugging Face plan. Static Spaces are free, and
# a Dash application is not static, so this path costs money. Google Cloud Run runs the
# same staged directory inside its free tier. See `scripts/deployCloudRun.sh`.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SPACE=""
STAGE=runs/space
PUSH=no
RESULTS=results/rolling2023_profit_all
MODELS=models/rolling2023_profit_fc4h
SEED=0
stageArgs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --space)   SPACE="$2"; shift 2 ;;
    --push)    PUSH=yes; shift ;;
    --stage)   STAGE="$2"; stageArgs+=(--stage "$2"); shift 2 ;;
    --results) RESULTS="$2"; stageArgs+=(--results "$2"); shift 2 ;;
    --models)  MODELS="$2"; stageArgs+=(--models "$2"); shift 2 ;;
    --seed)    SEED="$2"; stageArgs+=(--seed "$2"); shift 2 ;;
    *)         stageArgs+=("$1" "$2"); shift 2 ;;
  esac
done

bash "$here/stageDashboard.sh" "${stageArgs[@]}"

printf '\n========== upload ==========\n'
if [[ -z "$SPACE" ]]; then
  echo "no --space given, so nothing is pushed. Run it again with:"
  echo "  bash scripts/publishSpace.sh --space <user>/<name> --push"
  exit 0
fi

# A Space is a git repository. Push the stage as one commit, so an update replaces the
# agents and the results rather than growing the history with every version of them.
cd "$STAGE"
git init -q -b main
git add -A
git -c user.email=publish@localhost -c user.name=publish commit -q \
  -m "dashboard: $(basename "$RESULTS"), agents from $(basename "$MODELS") seed $SEED"
git remote add space "https://huggingface.co/spaces/$SPACE"
if [[ "$PUSH" == yes ]]; then
  echo "pushing to https://huggingface.co/spaces/$SPACE"
  git push --force space main
else
  echo "committed. To upload:"
  echo "  cd $STAGE && git push --force space main"
fi

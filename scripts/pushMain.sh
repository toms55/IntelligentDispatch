#!/usr/bin/env bash
# Push main to GitHub, then publish the same code to the Hugging Face Space.
#
#     bash scripts/pushMain.sh
#     bash scripts/pushMain.sh --space toms55/IntelligentDispatch
#
# The Space is a separate git repository, so a push to GitHub never reaches it. Use
# this script instead of `git push origin main`, so the two stay at the same code.
#
# The Space upload is staged from the working tree, not from a commit, so the script
# refuses to run unless main is checked out and the tree is clean. Every argument it
# does not read goes through to `scripts/publishSpace.sh`.
set -euo pipefail

SPACE=${DASHBOARD_SPACE:-toms55/IntelligentDispatch}
REMOTE=origin
passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --space)  SPACE="$2"; shift 2 ;;
    --remote) REMOTE="$2"; shift 2 ;;
    *) passthrough+=("$1"); shift ;;
  esac
done

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != main ]]; then
  echo "on $branch, not main. Check out main first." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "the working tree is not clean. The Space stages from it, so commit first." >&2
  exit 1
fi

printf '\n########## main -> %s ##########\n' "$REMOTE"
git push "$REMOTE" main

printf '\n########## main -> the Space ##########\n'
bash scripts/publishSpace.sh --space "$SPACE" --push "${passthrough[@]}"

#!/usr/bin/env bash
# Push main to GitHub, then publish the same code to the Hugging Face Space.
#
#     bash scripts/pushMain.sh
#     bash scripts/pushMain.sh --space <user>/<name>
#
# The Space is a separate git repository, so a push to GitHub never reaches it. Run
# `scripts/installHooks.sh` once and a plain `git push` publishes the Space as well.
# This script is the same job on demand, and it publishes whether or not the inputs
# have changed.
#
# The Space upload is staged from the working tree, not from a commit, so the script
# refuses to run unless main is checked out and the tree is clean. Every argument it
# does not read goes through to `scripts/publishSpace.sh`.
#
# The Space needs a Hugging Face write token. Keep it in a secret store, and hand it
# over for the one command:
#
#     HF_TOKEN=$(secret-tool lookup service huggingface) bash scripts/pushMain.sh
#
# Or run `hf auth login` once, and the script reads the token file the CLI writes. Stage
# 5 of `scripts/publishSpace.sh` holds the token handling. Never write a token into a
# file in this repository, and never put one in a remote URL.
set -euo pipefail

REMOTE=origin
passthrough=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    *) passthrough+=("$1"); shift ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"

branch=$(git rev-parse --abbrev-ref HEAD)
if [[ "$branch" != main ]]; then
  echo "on $branch, not main. Check out main first." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "the working tree is not clean. The Space stages from it, so commit first." >&2
  exit 1
fi

# The pre-push hook publishes the Space itself, and this script publishes below, so one
# push would do the work twice.
printf '\n########## main -> %s ##########\n' "$REMOTE"
NO_SPACE_PUBLISH=1 git push "$REMOTE" main

printf '\n########## main -> the Space ##########\n'
fingerprint=$(bash scripts/spaceFingerprint.sh)
bash scripts/publishSpace.sh --push "${passthrough[@]}"
# Record what went up, so the hook knows the Space is level and skips the next push.
printf '%s\n' "$fingerprint" > "$(bash scripts/spaceFingerprint.sh --path)"

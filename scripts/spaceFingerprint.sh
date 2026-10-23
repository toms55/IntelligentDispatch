#!/usr/bin/env bash
# Print a fingerprint of everything the Space upload is built from.
#
#     bash scripts/spaceFingerprint.sh          the fingerprint
#     bash scripts/spaceFingerprint.sh --path   where the last published one is kept
#
# The pre-push hook publishes the Space only when this value moves, so a push that
# changes nothing the Space serves stays as fast as a plain push. The fingerprint
# covers both kinds of input:
#
#   1. the tracked code the upload copies, as the git tree hash of each item
#   2. the results and the agent weights, which git does not track at all, as the name,
#      the size and the modification time of every file that goes up
#
# The second kind is why a git diff cannot drive this. `results/` and `models/` are
# gitignored, so a new training run or a new backtest changes the Space without
# changing one tracked file.
#
# The defaults name the same results, models and seed as `scripts/publishSpace.sh`.
# Keep the two in step.
set -euo pipefail

RESULTS=${RESULTS:-results/rolling2023_profit_all}
MODELS=${MODELS:-models/rolling2023_profit_fc4h}
SEED=${SEED:-0}

# The record of the last publish. It sits in the git directory, because it is state of
# this one clone and not content of the repository. `--git-common-dir` keeps one record
# for the main checkout and every worktree, which is right: they publish one Space.
if [[ "${1:-}" == --path ]]; then
  echo "$(git rev-parse --git-common-dir)/spacePublished"
  exit 0
fi

# Name, size and modification time of each path on stdin. BSD find has no -printf and
# BSD stat takes other flags, so perl, which both systems ship, reads them.
describe() {
  perl -nle '@s = stat; print "$_ $s[7] $s[9]"'
}

{
  # Stage 1 of publishSpace.sh copies these eight items.
  for item in config.py utils data degradation rl backtest visualise deploy; do
    git rev-parse "HEAD:$item"
  done
  # Stage 2 stages these three names, and stage 3 one seed of the weights.
  if [[ -d "$RESULTS" ]]; then
    find "$RESULTS" -type f \( -name 'fold*.csv' -o -name 'fold*.parquet' \
      -o -name 'manifest.json' \) | sort | describe
  fi
  if [[ -d "$MODELS" ]]; then
    find "$MODELS" -type f -path "*seed$SEED*" | sort | describe
  fi
} | sha256sum | cut -d' ' -f1

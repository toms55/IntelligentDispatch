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
#   2. the results and the agent weights, as the name, the size and the modification
#      time of every file that goes up
#
# The second kind is why a git diff cannot drive this. `models/` is gitignored, so a
# new training run changes the Space without changing one tracked file, and a new
# backtest changes `results/backtestResults/` before anyone commits it.
#
# The defaults name the same results, models and seed as `scripts/publishSpace.sh`.
# Keep the two in step.
set -euo pipefail

# The runs the upload stages, in the same order and with the same default as
# publishSpace.sh. $RUNS overrides it, space separated.
read -r -a RUNS <<< "${RUNS:-results/backtestResults/rolling2023_profit_fc5m}"
ACTIONS=${ACTIONS:-results/actions}
POLICIES=${POLICIES:-trainedPolicies}
ALGO=${ALGO:-maskableppo}
TRAIN_MODE=${TRAIN_MODE:-warm}

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
  # Stage 2 stages each named run's CSV and its actions export, one per region, and
  # stage 3 the best seed's weights. The sources are the whole files, not the filtered
  # rows, so a change to any seed of a staged run moves this. That is the safe
  # direction: it republishes a little too often, never too rarely.
  for results in "${RUNS[@]}"; do
    run=$(basename "$results")
    # The run's CSV, and its export as Parquet. The live CSV a backtest appends to sits
    # beside the export, and is not staged.
    { find "$(dirname "$results")" -maxdepth 1 -type f -name "${run}_*.csv"
      find "$ACTIONS" -maxdepth 1 -type f -name "${run}_*.parquet"
    } 2>/dev/null | grep -E "/${run}_[A-Z]+[0-9]\.(csv|parquet)$" | sort | describe || true
  done
  # Which agent each region stages comes from `config.best_seed`, whose tree hash is
  # already above, so only the weights themselves are described here. The other seeds
  # stay out: republishing one must not look like a change to the Space.
  while read -r region seed; do
    directory="$POLICIES/$region/$ALGO-$TRAIN_MODE-s$seed"
    [[ -d "$directory" ]] && find "$directory" -type f | sort | describe
  done < <(bash "$(dirname "$0")/bestSeeds.sh")
} | sha256sum | cut -d' ' -f1

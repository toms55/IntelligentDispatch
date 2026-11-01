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
# The defaults name the same results as `scripts/publishSpace.sh`. Keep the two in step.
# The agents need no such list here: both scripts read the same live set, through
# `rl.policies.loadable`, so there is nothing to keep in step for stage 3.
set -euo pipefail

# The runs the upload stages, in the same order and with the same default as
# publishSpace.sh. $RUNS overrides it, space separated.
defaultRuns="results/backtestResults/rolling2023_start_at_0_SoC_five_min_refresh"
defaultRuns+=" results/backtestResults/rolling2023_dqn_start_at_0_SoC_five_min_refresh"
defaultRuns+=" results/backtestResults/rolling2023_price_scale_start_at_0_SoC_five_min_refresh"
defaultRuns+=" results/backtestResults/rolling2023_price_history_start_at_0_SoC_five_min_refresh"
defaultRuns+=" results/backtestResults/rolling2023_weekday_season_start_at_0_SoC_five_min_refresh"
read -r -a RUNS <<< "${RUNS:-$defaultRuns}"
ACTIONS=${ACTIONS:-results/actions}
POLICIES=${POLICIES:-trainedPolicies}
PYTHON=${PYTHON:-"uv run python"}

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
  # Stage 2 stages each named run's CSV and its actions export, one per region. The
  # sources are the whole files, not the filtered rows, so a change to any seed of a
  # staged run moves this. That is the safe direction: it republishes a little too
  # often, never too rarely.
  for results in "${RUNS[@]}"; do
    run=$(basename "$results")
    # The run's CSV, and its export as Parquet. The live CSV a backtest appends to sits
    # beside the export, and is not staged.
    { find "$(dirname "$results")" -maxdepth 1 -type f -name "${run}_*.csv"
      find "$ACTIONS" -maxdepth 1 -type f -name "${run}_*.parquet"
    } 2>/dev/null | grep -E "/${run}_[A-Z]+[0-9]\.(csv|parquet)$" | sort | describe || true
  done
  # Stage 3 of publishSpace.sh stages every agent `rl.policies.loadable` accepts, for
  # every region. That set moves whenever a new agent is published or the observation
  # layout in `config` changes, neither of which is a tracked diff, so it is read fresh
  # here rather than named.
  $PYTHON -c "
from rl import policies
for region in policies.regions('$POLICIES'):
    for directory in policies.loadable(region, '$POLICIES').values():
        print(directory)
" | sort | while read -r directory; do
    find "$directory" -type f | sort | describe
  done
} | sha256sum | cut -d' ' -f1

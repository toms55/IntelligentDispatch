#!/usr/bin/env bash
# Print the published runs the dashboard offers, one path per line, in order.
#
#     bash scripts/dashboardRuns.sh
#     results/backtestResults/rolling2023_ppo_daily_1_cycle_end_empty_5min_forecast
#     ...
#
# `config.dashboard_runs` is the one definition, so the dashboard, the Space upload and
# the Space fingerprint all stage the same experiments. This reads it with sed rather
# than with python, because `scripts/spaceFingerprint.sh` runs inside the pre-push hook,
# and the hook has to stay fast enough that a push which changes nothing still feels
# like a plain push. An interpreter start costs more than the rest of the fingerprint.
set -euo pipefail

CONFIG=${CONFIG:-config.py}
ROOT=${ROOT:-results/backtestResults}

# The block `dashboard_runs = (` ... `)`, one quoted name per line. An empty result is a
# format change in config.py, not an empty list, so it stops here rather than staging
# nothing by accident.
runs=$(sed -n '/^dashboard_runs = (/,/^)/p' "$CONFIG" \
       | sed -n 's/^ *"\([A-Za-z0-9_]*\)" *, *$/\1/p')

if [[ -z "$runs" ]]; then
  echo "no dashboard_runs list in $CONFIG, so the Space upload has no run to stage." >&2
  echo "Expected a block: dashboard_runs = (" >&2
  echo '    "rolling2023_...",' >&2
  echo ")" >&2
  exit 1
fi

while read -r run; do
  echo "$ROOT/$run"
done <<< "$runs"

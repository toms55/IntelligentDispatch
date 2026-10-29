#!/usr/bin/env bash
# Print the seed the hosted dashboard shows, one "REGION SEED" line per region.
#
#     bash scripts/bestSeeds.sh
#     NSW1 4
#     QLD1 0
#     ...
#
# `config.best_seed` is the one definition. This reads it with sed rather than with
# python, because `scripts/spaceFingerprint.sh` runs inside the pre-push hook, and the
# hook has to stay fast enough that a push which changes nothing still feels like a
# plain push. An interpreter start costs more than the rest of the fingerprint.
#
# The map selects two things in the upload: which result rows go up, and which agent.
# Every seed stays in `trainedPolicies/` and in `results/`, so nothing here narrows what
# a local dashboard offers or what the thesis can report.
set -euo pipefail

CONFIG=${CONFIG:-config.py}

# One line, `best_seed = {"NSW1": 4, ...}`. Take the braces, split the pairs, drop the
# quotes. An empty result is a format change in config.py, not an empty map, so it stops
# here rather than staging every seed by accident.
seeds=$(sed -n 's/^best_seed = {\(.*\)}$/\1/p' "$CONFIG" \
        | tr ',' '\n' \
        | sed -n 's/^ *"\([A-Z0-9]*\)" *: *\([0-9]*\) *$/\1 \2/p')

if [[ -z "$seeds" ]]; then
  echo "no best_seed map in $CONFIG, so the Space upload cannot pick a seed." >&2
  echo "Expected one line: best_seed = {\"NSW1\": 4, ...}" >&2
  exit 1
fi

echo "$seeds"

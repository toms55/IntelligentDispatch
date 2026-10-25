#!/usr/bin/env bash
# One-region experiment: train on 2024, score on 2025. Leaner than overnight.sh -- no
# validation split, just the train/test pair.
#
#     bash scripts/run_experiment.sh NSW1 2>&1 | tee runs/NSW1/run.log
#
# Resumable: the oracle cache is keyed on its own parameters. Writes/overwrites
# results/<REGION>.md on success and pushes it, so that file always holds the most
# recent experiment for that region, never a history of past ones.
set -euo pipefail

REGION="${1:?usage: run_experiment.sh REGION, e.g. NSW1}"

WORKERS=6
TIMESTEPS=300000
SEEDS="0 1 2 3 4"
GAP=0.02
LIMIT=300

# Every boundary is 04:00, the NEM trading day boundary: PREDISPATCH only reaches 288
# intervals from a 04:00 run, so 04:00 is the one run time where exactly one episode fits.
TRAIN="train=2024-01-01T04:00..2025-01-01T04:00"
TEST="test=2025-01-01T04:00..2026-01-01T04:00"

# The AEMC market price cap rises each 1 July, and the observation price scale must stay
# fixed from training through to scoring. Pin it to the end of the test window so prices
# above the training year's cap still resolve.
CAP_DATE="2026-01-01T04:00"

OUT="runs/${REGION}"
RESULT="results/${REGION}.md"
mkdir -p "$OUT" results

say() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }

say "1/3 train MaskablePPO, 5 seeds, calendar aging ON in the environment"
uv run python -u -m cli train --region "$REGION" --split "$TRAIN" \
  --algo maskableppo --seeds $SEEDS \
  --timesteps "$TIMESTEPS" --cap-date "$CAP_DATE" --out "$OUT" --verbose

say "2/3 test bound: continuous, no calendar, 365 windows"
uv run python -u -m cli baseline --region "$REGION" --split "$TEST" \
  --continuous --no-calendar \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --verbose

say "3/3 test 2025: realised prices, then the AEMO forecast"
# --no-calendar applies to both the oracle lookup and the environment here, so the agent
# is scored on the same terms the bound was solved under.
uv run python -u -m cli eval --region "$REGION" --split "$TEST" \
  --algo maskableppo --seeds $SEEDS \
  --baseline-oracle continuous --no-calendar --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out "$OUT" --result-file "$RESULT"
uv run python -u -m cli eval --region "$REGION" --split "$TEST" \
  --algo maskableppo --seeds $SEEDS \
  --baseline-oracle continuous --no-calendar --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out "$OUT" --result-file "$RESULT" --forecast

say "committing results"
git add "$RESULT"
git commit -m "results: ${REGION} $(date -Iseconds)"
git push

say "done"

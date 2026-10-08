#!/usr/bin/env bash
# First RL experiment: train on 2024, score on 2025.
#
#     bash scripts/overnight.sh 2>&1 | tee runs/overnight.log
#
# Resumable: the oracle cache is keyed on its own parameters.
#
# Three oracle configurations: reward baseline (continuous, no calendar -- cheap, only
# needs to be a constant), validation bound (discrete, calendar -- the fair bound), test
# bound (continuous, no calendar -- doubly loose, chosen on cost; overstates regret).
#
# Measured solve costs, 288 intervals, one SA1 day, single thread: continuous/no-calendar
# 19s, continuous/calendar 44-190s (day-dependent), discrete/no-calendar 82s at 0.5% gap.
set -euo pipefail

WORKERS=6
TIMESTEPS=300000
SEEDS="0 1 2 3 4"
GAP=0.02
LIMIT=300

# Train on 2024 up to November, validate on the tail of 2024, test on 2025. Every
# boundary is 04:00, the NEM trading day boundary, and that is load-bearing: PREDISPATCH
# only reaches 288 intervals from a 04:00 run, so 04:00 is the one run time where exactly
# one episode fits.
TRAIN="train=2024-01-01T04:00..2024-11-01T04:00"
VAL="val=2024-11-01T04:00..2025-01-01T04:00"
TEST="test=2025-01-01T04:00..2026-01-01T04:00"

# The AEMC market price cap rises each 1 July, and the observation price scale must stay
# fixed from training through to scoring. Pin it to the end of the test window so prices
# above the 2024-25 cap of 17500 still resolve. Drop this to pin it to the end of the
# training window instead.
CAP_DATE="2026-01-01T04:00"

mkdir -p runs

say() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }

say "1/6 training baseline: continuous, no calendar, 305 windows"
uv run python -u -m cli baseline --split "$TRAIN" --continuous --no-calendar \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --verbose

say "2/6 train MaskablePPO, 5 seeds, calendar aging ON in the environment"
uv run python -u -m cli train --split "$TRAIN" --algo maskableppo --seeds $SEEDS \
  --timesteps "$TIMESTEPS" --cap-date "$CAP_DATE" \
  --baseline-oracle continuous --baseline-no-calendar \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --out runs --verbose

say "3/6 validation bound: discrete, calendar, 61 windows"
uv run python -u -m cli baseline --split "$VAL" --gap "$GAP" \
  --time-limit "$LIMIT" --workers "$WORKERS" --verbose

say "4/6 validation Nov-Dec 2024: realised prices, then the AEMO forecast"
uv run python -u -m cli eval --split "$VAL" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle discrete --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs
uv run python -u -m cli eval --split "$VAL" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle discrete --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs --forecast

say "5/6 test bound: continuous, no calendar, 365 windows"
uv run python -u -m cli baseline --split "$TEST" --continuous --no-calendar \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --verbose

say "6/6 test 2025: realised prices, then the AEMO forecast"
# --no-calendar applies to both the oracle lookup and the environment here, so the agent
# is scored on the same terms the bound was solved under.
uv run python -u -m cli eval --split "$TEST" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle continuous --no-calendar --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs
uv run python -u -m cli eval --split "$TEST" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle continuous --no-calendar --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs --forecast

say "done"

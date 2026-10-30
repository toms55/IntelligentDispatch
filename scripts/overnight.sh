#!/usr/bin/env bash
# First RL experiment: train on 2024, score on 2025.
#
#     bash scripts/overnight.sh 2>&1 | tee runs/overnight.log
#
# Resumable: the oracle cache is keyed on its own parameters.
#
# Two oracle configurations, both for scoring only (the reward is raw profit): validation
# bound (discrete, calendar -- the fair bound), test bound (continuous, no calendar --
# doubly loose, chosen on cost; overstates regret).
#
# Measured solve costs, 288 intervals, one SA1 day, single thread: continuous/no-calendar
# 19s, continuous/calendar 44-190s (day-dependent), discrete/no-calendar 82s at 0.5% gap.
set -euo pipefail

WORKERS=6
TIMESTEPS=300000
SEEDS="0 1 2 3 4"
GAP=0.02
LIMIT=300
# The cycle warranty, and what the pack still holds at the episode's close is worth. The
# budget is weekly, so an episode is a week: see config.episode_intervals. The default
# region is the one config.default_region names, so the terminal price is read for it.
CYCLES=7.0
TERMINAL_PRICE=$(uv run python -c "import config; print(config.terminal_energy_price[config.default_region])")
# A budgeted agent means something other than the unbudgeted agent of the same name, so
# it publishes beside it rather than over it.
TAG=efc7
ORACLE="--cycles $CYCLES --terminal-price $TERMINAL_PRICE"

# Train on 2024 up to November, validate on the tail of 2024, test on 2025. Every
# boundary is 04:00, the NEM trading day boundary, and that is load-bearing: PREDISPATCH
# only reaches 288 intervals from a 04:00 run, so 04:00 is the one run time where exactly
# one trading day of forecast fits. An episode is now a week, and it still opens at
# 04:00, so the lookahead still stops at the end of the day it is in.
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

say "1/5 train MaskablePPO, 5 seeds, calendar aging ON in the environment"
uv run python -u -m cli train --split "$TRAIN" --algo maskableppo --seeds $SEEDS \
  --timesteps "$TIMESTEPS" --cap-date "$CAP_DATE" --out runs --verbose \
  $ORACLE --policy-tag "$TAG"

say "2/5 validation bound: discrete, calendar, one record per weekly episode"
uv run python -u -m cli baseline --split "$VAL" --gap "$GAP" \
  --time-limit "$LIMIT" --workers "$WORKERS" --verbose $ORACLE

say "3/5 validation Nov-Dec 2024: realised prices, then the AEMO forecast"
uv run python -u -m cli eval --split "$VAL" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle discrete --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs $ORACLE
uv run python -u -m cli eval --split "$VAL" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle discrete --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs --forecast $ORACLE

# A budgeted continuous window carries the cycle row through every interval of the week,
# which is a far harder MILP than the day it replaced. Expect this stage to be the long
# one again.
say "4/5 test bound: continuous, no calendar, one record per weekly episode"
uv run python -u -m cli baseline --split "$TEST" --continuous --no-calendar \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --verbose $ORACLE

say "5/5 test 2025: realised prices, then the AEMO forecast"
# --no-calendar applies to both the oracle lookup and the environment here, so the agent
# is scored on the same terms the bound was solved under.
uv run python -u -m cli eval --split "$TEST" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle continuous --no-calendar --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs $ORACLE
uv run python -u -m cli eval --split "$TEST" --algo maskableppo --seeds $SEEDS \
  --baseline-oracle continuous --no-calendar --gap "$GAP" --time-limit "$LIMIT" \
  --workers "$WORKERS" --out runs --forecast $ORACLE

say "done"

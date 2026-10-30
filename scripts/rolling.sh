#!/usr/bin/env bash
# Rolling-origin backtest over two years of NSW1, 52-week train, 1-week test.
#
#     bash scripts/rolling.sh 2>&1 | tee runs/rolling.log
#
# Resumable at both stages. The oracle cache is keyed on its own parameters, and a fold
# is skipped once its rows are in the fold CSV.
set -euo pipefail

REGION=NSW1
SPAN="all=2024-01-01T04:00..2026-01-01T04:00"
WORKERS=6
GAP=0.02
LIMIT=1800
TRAIN_WEEKS=52
CAP_DATE="2026-01-01T04:00"
# The cycle warranty, and what the pack still holds at the week's close is worth. The
# budget is weekly, so an episode is a week: see config.episode_intervals. The terminal
# price is the region's own, out of config.terminal_energy_price, because the median RRP
# differs by several dollars between regions.
CYCLES=7.0
TERMINAL_PRICE=$(uv run python -c "import config; print(config.terminal_energy_price['$REGION'])")
# Agents and results under a budget mean something other than the unbudgeted ones of the
# same name, so they publish beside them rather than over them.
TAG=efc7
# The results directory names the run, and `backtest.publish` drops the region from it,
# so the five regional jobs of one experiment share one published name.
RUN="rolling_efc${CYCLES%.*}"

mkdir -p runs

say() { printf '\n========== %s  (%s) ==========\n' "$1" "$(date +%H:%M)"; }

# Stage 1 used to be the long one. The discrete windows now go to optimisation/dp.py,
# which solves them exactly by dynamic programming, so 731 windows take seconds rather
# than the 10 to 14 hours the MILP needed at six workers.
#
# --gap, --time-limit and --workers no longer do anything to a discrete window. They
# stay on the command line because the flags are shared with the continuous mode, and
# because --gap is part of the cache key.
# One record per episode now, so a 104-week span is 104 weekly solves rather than 731
# daily ones. A weekly discrete window is about 15 s against 25 s unbudgeted, because the
# rent bisection solves the same state space several times over.
say "1/2 oracle cache: discrete, no calendar, one record per weekly episode"
uv run python -u -m cli baseline --split "$SPAN" --region "$REGION" --no-calendar \
  --cycles "$CYCLES" --terminal-price "$TERMINAL_PRICE" \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --verbose

# The discrete oracle is only the dual bound that `captured` divides by; the reward is
# raw profit. --allow-solve is still passed, because a gap in the cache must solve rather
# than stop the run at this point.
say "2/2 rolling backtest: 52 folds, warm chain, realised prices then the AEMO forecast"
uv run python -u -m cli rolling --split "$SPAN" --region "$REGION" \
  --train-weeks "$TRAIN_WEEKS" --test-weeks 1 --step-weeks 1 \
  --algo maskableppo --cap-date "$CAP_DATE" \
  --baseline-oracle discrete --no-calendar \
  --cycles "$CYCLES" --terminal-price "$TERMINAL_PRICE" \
  --policy-tag "$TAG" --results "results/${RUN}_${REGION}" \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" \
  --price-source both --allow-solve

say "done"

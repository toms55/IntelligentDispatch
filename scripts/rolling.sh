#!/usr/bin/env bash
# Rolling-origin backtest over 2023-01 to 2026-09 (143 weekly folds, as the 2023 runs),
# 52-week train, 1-week test. REGION=SA1 bash scripts/rolling.sh for another region,
# ALGO=dqn for another algorithm, STAGE=oracle or STAGE=folds for one stage only.
# CYCLES (a number, or none), HORIZON, TERMINAL_PRICE, REFRESH (minutes), END_EMPTY=1,
# TAG and RUN override
# the efc7 defaults below, e.g. the daily sell-off run:
#
#     HORIZON=288 CYCLES=1.0 TERMINAL_PRICE=0 END_EMPTY=1 REFRESH=5 \
#       TAG=daily_empty_efc1 RUN=rolling_daily_empty_efc1 bash scripts/rolling.sh
#
#     bash scripts/rolling.sh 2>&1 | tee runs/rolling.log
#
# Resumable at both stages. The oracle cache is keyed on its own parameters, and a fold
# is skipped once its rows are in the fold CSV.
set -euo pipefail

REGION=${REGION:-NSW1}
ALGO=${ALGO:-maskableppo}
STAGE=${STAGE:-all}
# 195 whole weeks from the 2023 runs' start, so the last episode is a full week.
SPAN="all=2023-01-01T04:00..2026-09-27T04:00"
WORKERS=1
GAP=0.02
LIMIT=1800
TRAIN_WEEKS=52
CAP_DATE="2026-10-01T04:00"
# The cycle warranty, and what the pack still holds at the week's close is worth. The
# budget is weekly, so an episode is a week: see config.episode_intervals. The terminal
# price is the region's own, out of config.terminal_energy_price, because the median RRP
# differs by several dollars between regions.
CYCLES=${CYCLES:-7.0}
# CYCLES=none is no budget, the soc0 setup.
if [ "$CYCLES" = none ]; then CYCLES_ARG=--no-cycles; else CYCLES_ARG="--cycles $CYCLES"; fi
HORIZON=${HORIZON:-2016}
TERMINAL_PRICE=${TERMINAL_PRICE:-$(uv run python -c "import config; print(config.terminal_energy_price['$REGION'])")}
# Unset keeps config.forecast_refresh_intervals, which is what efc7 ran under.
REFRESH_ARG=${REFRESH:+--forecast-refresh $REFRESH}
END_EMPTY_ARG=${END_EMPTY:+--end-empty}
# Agents and results under a budget mean something other than the unbudgeted ones of the
# same name, so they publish beside them rather than over them.
TAG=${TAG:-efc7}
# The results directory names the run, and `backtest.publish` drops the region from it,
# so the five regional jobs of one experiment share one published name.
# PPO keeps the plain name; another algorithm adds its own.
RUN=${RUN:-"rolling_efc${CYCLES%.*}"}
[ "$ALGO" = maskableppo ] || RUN="${RUN}_${ALGO}"

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
if [ "$STAGE" != folds ]; then
say "1/2 oracle cache: discrete, no calendar, one record per weekly episode"
uv run python -u -m cli baseline --split "$SPAN" --region "$REGION" --no-calendar \
  --horizon "$HORIZON" $CYCLES_ARG --terminal-price "$TERMINAL_PRICE" $END_EMPTY_ARG \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" --verbose
fi
[ "$STAGE" = oracle ] && exit 0

# The discrete oracle is only the dual bound that `captured` divides by; the reward is
# raw profit. --allow-solve is still passed, because a gap in the cache must solve rather
# than stop the run at this point.
say "2/2 rolling backtest: 52 folds, warm chain, realised prices then the AEMO forecast"
uv run python -u -m cli rolling --split "$SPAN" --region "$REGION" \
  --train-weeks "$TRAIN_WEEKS" --test-weeks 1 --step-weeks 1 \
  --algo "$ALGO" --cap-date "$CAP_DATE" \
  --baseline-oracle discrete --no-calendar \
  --horizon "$HORIZON" $CYCLES_ARG --terminal-price "$TERMINAL_PRICE" $END_EMPTY_ARG \
  $REFRESH_ARG --policy-tag "$TAG" --results "results/${RUN}_${REGION}" --out "runs/${RUN}_${REGION}" \
  --gap "$GAP" --time-limit "$LIMIT" --workers "$WORKERS" \
  --price-source both --allow-solve

say "done"

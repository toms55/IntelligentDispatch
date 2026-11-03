#!/bin/bash
# The MDP ablation study: one run per variant and algorithm, each the pooled-regions agent
# (backtest/pooled.py) in the daily end-empty setup, so a variant differs from the
# published reference in the flags below and nothing else. See ABLATION_PLAN.md.
#
# HOW TO SUBMIT (from the repo root, on the PBS login node)
#
#   1. See the variants:        bash scripts/ablation.sh list
#   2. Dry run one variant:     bash scripts/ablation.sh plan deg_flat
#      This prints every qsub it would make, and submits nothing. `<id>` stands in for
#      the job ids that the later jobs wait on.
#   3. Submit it:               SUBMIT=1 bash scripts/ablation.sh plan deg_flat
#      Several at once:         SUBMIT=1 bash scripts/ablation.sh plan deg_flat look_1h
#      Everything (~220 seed-jobs of up to 48 h): leave the variant names off. Agree the
#      grid before you do.
#   4. Watch it:                qstat -u $USER
#   5. When the merge job ends, its results are in results/backtestResults/. Commit them:
#                               git add results/backtestResults results/actions
#
# Narrower runs, set on the same command line:
#
#     ALGOS="dqn"               which algorithms (default: both, where the variant applies)
#     SEEDS="0 1 2"             which seeds (default 0 1 2 3 4); the merge uses the same list
#     REGIONS="NSW1 TAS1"       which regions (default all five); every job uses the same list
#
#     SEEDS="0 1 2" REGIONS="NSW1 TAS1" SUBMIT=1 bash scripts/ablation.sh plan lev_9
#
# Before submitting: run from a checkout with `.venv` built (`uv sync`) that nobody edits
# while the jobs queue or run, and check each run name is free (the check is in
# scripts/pooled.pbs). The baseline oracle corpus must already hold the test episodes.
# Nothing here submits unless SUBMIT=1.
#
# A "problem" variant (kind in `list`) changes the dispatch problem, so it needs its own
# oracle records first: one oracle job per region, and its seed jobs wait on all of them.
# An "agent" variant only changes what the agent sees or is paid, and reuses the baseline
# corpus. Job graph per variant and algorithm:
#
#     [oracle x regions]  ->  [seed job x seeds]  ->  [merge + publish]
#
# qsub -v splits on commas, so no variant text or flag may contain one.
#
# Run names are rolling2023_ablate_<variant>; pooled.pbs adds _dqn for DQN. Ablation runs
# publish no agents to trainedPolicies/ (an ablation's observation, action set or
# out-of-bounds penalty is not the dashboard's), and backtest.publish writes their rows
# to results/backtestResults/.
set -euo pipefail
cd "$(dirname "$0")/.."

ALGOS=${ALGOS:-"maskableppo dqn"}
SEEDS=${SEEDS:-"0 1 2 3 4"}
REGIONS=${REGIONS:-"NSW1 QLD1 SA1 TAS1 VIC1"}

# name | kind | algorithms ("both" or one) | flags | what it tests
VARIANTS=(
  "deg_full|problem|both|--degradation full|Schade cycling plus calendar wear"
  "deg_flat|problem|both|--degradation flat|flat per-MWh wear at the Schade full-discharge rate"
  "deg_none|problem|both|--degradation none|no wear cost at all"
  "look_spot|agent|both|--lookahead 1|spot price only"
  "look_1h|agent|both|--lookahead 12|one hour ahead"
  "look_4h|agent|both|--lookahead 48|four hours ahead"
  "look_12h|agent|both|--lookahead 144|twelve hours ahead"
  "time_sincos|agent|both|--time-encoding sincos|sine and cosine of the time of day"
  "time_none|agent|both|--time-encoding none|no time of day"
  "hist_1h|agent|both|--history 2|one hour of settled prices"
  "hist_4h|agent|both|--history 8|four hours of settled prices"
  "hist_24h|agent|both|--history 48|a day of settled prices (zero before the episode)"
  "lev_5|problem|both|--action-levels 2|five power levels"
  "lev_9|problem|both|--action-levels 4|nine power levels"
  "lev_17|problem|both|--action-levels 8|seventeen power levels"
  "mask_hidden|agent|maskableppo|--hide-mask|MaskablePPO without the action mask"
  "pen_0|agent|dqn|--invalid-penalty 0|no out-of-bounds penalty"
  "pen_0.1|agent|dqn|--invalid-penalty 0.1|a tenth of the pinned penalty"
  "pen_10|agent|dqn|--invalid-penalty 10|ten times the pinned penalty"
  "shaping|agent|both|--shaping-bonus 0.05|buy-low/sell-high bonus; reward only"
  "px_knee100|agent|both|--price-knee 100|asinh knee at 100"
  "px_knee1000|agent|both|--price-knee 1000|asinh knee at 1000"
  "px_linear|agent|both|--price-transform linear|price over the cap"
  "px_clip|agent|both|--price-transform clip|10th to 90th percentile band then asinh"
)

field() { echo "$1" | cut -d'|' -f"$2"; }

if [ "${1:-}" = list ]; then
  for v in "${VARIANTS[@]}"; do
    printf '%-12s %-8s %-11s %-30s %s\n' "$(field "$v" 1)" "$(field "$v" 2)" \
      "$(field "$v" 3)" "$(field "$v" 4)" "$(field "$v" 5)"
  done
  exit 0
fi
[ "${1:-}" = plan ] || { sed -n 2,12p "$0"; exit 2; }
shift
WANT=("$@")

submit() {  # echo an id: a real one when submitting, a placeholder otherwise
  if [ "${SUBMIT:-0}" = 1 ]; then eval "$@"; else printf '%s\n' "$*" >&2; echo "<id>"; fi
}

for v in "${VARIANTS[@]}"; do
  name=$(field "$v" 1); kind=$(field "$v" 2); only=$(field "$v" 3); flags=$(field "$v" 4)
  if [ ${#WANT[@]} -gt 0 ] && [[ ! " ${WANT[*]} " =~ " $name " ]]; then continue; fi
  echo "# ---- $name ($kind): $(field "$v" 5)"

  # An agent variant reads the baseline corpus. BASE_AFTER holds the ids of the jobs that
  # solve it (scripts/ablationOracle.pbs with no EXTRA), colon separated, when it is not
  # already in data_cache/lp_baseline.
  dependency=""
  if [ "$kind" = agent ] && [ -n "${BASE_AFTER:-}" ]; then
    dependency="-W depend=afterok:$BASE_AFTER"
  fi
  # SKIP_ORACLE names variants whose oracle is already in data_cache/lp_baseline: their
  # seeds are submitted with no oracle job in front of them.
  if [ "$kind" = problem ] && [[ " ${SKIP_ORACLE:-} " != *" $name "* ]]; then
    ids=()
    for region in $REGIONS; do
      ids+=("$(submit "qsub -N ab-or-$name-$region -v REGION=$region,EXTRA=\"$flags\" scripts/ablationOracle.pbs")")
    done
    dependency="-W depend=afterok:$(IFS=:; echo "${ids[*]}")"
  fi

  for algo in $ALGOS; do
    [ "$only" = both ] || [ "$only" = "$algo" ] || continue
    run_name="rolling2023_ablate_$name"
    seedIds=()
    for seed in $SEEDS; do
      seedIds+=("$(submit "qsub -N ab-$name-$algo-s$seed $dependency -v SEED=$seed,ALGO=$algo,RUN=$run_name,TAG=ablate_$name,REGIONS=\"$REGIONS\",EXTRA=\"$flags\" scripts/pooled.pbs")")
    done
    desc="MDP ablation $name: $(field "$v" 5). Pooled-regions $algo; daily; 1 EFC; end empty; otherwise the reference run."
    mergeId=$(submit "qsub -N ab-$name-$algo-merge -W depend=afterok:$(IFS=:; echo "${seedIds[*]}") -v MERGE=1,ALGO=$algo,RUN=$run_name,SEEDS=\"$SEEDS\",REGIONS=\"$REGIONS\",DESCRIPTION=\"$desc\" scripts/pooled.pbs")
    if [ "${SUBMIT:-0}" = 1 ]; then echo "$name $algo: merge job $mergeId"; fi
  done
done

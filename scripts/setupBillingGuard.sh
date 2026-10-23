#!/usr/bin/env bash
# Put a hard stop on the spending of the dashboard project. Run this once.
#
#     bash scripts/setupBillingGuard.sh --project nem-bess-dashboard --budget 1
#
# Google budgets send notifications. They do not stop spending. This script builds the
# part that does:
#
#   1. a Pub/Sub topic the budget publishes to
#   2. the function in `deploy/billingGuard/`, which removes the billing account from
#      the project once the cost reaches the budget
#   3. a budget scoped to that one project, wired to the topic
#
# The function needs the Billing Account Administrator role, because detaching a
# billing account is a change to the billing account. The script grants it to the
# service account of the function and to nothing else.
#
# A cost appears before its notification does, so the project can run a little past the
# budget before the guard fires. Set the budget below what you are willing to spend.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROJECT=""
BILLING_ACCOUNT=""
BUDGET=1
REGION_GCP=australia-southeast1
TOPIC=billing-guard
FUNCTION=billing-guard

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)         PROJECT="$2"; shift 2 ;;
    --billing-account) BILLING_ACCOUNT="$2"; shift 2 ;;
    --budget)          BUDGET="$2"; shift 2 ;;
    --gcp-region)      REGION_GCP="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "give a project: bash scripts/setupBillingGuard.sh --project <id>" >&2
  exit 2
fi
command -v gcloud >/dev/null || { echo "gcloud is not installed" >&2; exit 2; }

say() { printf '\n========== %s ==========\n' "$1"; }

if [[ -z "$BILLING_ACCOUNT" ]]; then
  BILLING_ACCOUNT=$(gcloud beta billing projects describe "$PROJECT" \
    --format 'value(billingAccountName)' | sed 's|billingAccounts/||')
fi
if [[ -z "$BILLING_ACCOUNT" ]]; then
  echo "$PROJECT has no billing account. Attach one first." >&2
  exit 1
fi
echo "project $PROJECT on billing account $BILLING_ACCOUNT, budget $BUDGET"

# ------------------------------------------------------------ 1. the services
say "1/5 services"
gcloud services enable cloudbilling.googleapis.com billingbudgets.googleapis.com \
  pubsub.googleapis.com cloudfunctions.googleapis.com run.googleapis.com \
  cloudbuild.googleapis.com eventarc.googleapis.com --project "$PROJECT"

# --------------------------------------------------------------- 2. the topic
say "2/5 topic $TOPIC"
gcloud pubsub topics create "$TOPIC" --project "$PROJECT" 2>/dev/null \
  || echo "  the topic is already there"

# ------------------------------------------------------------ 3. the function
say "3/5 function $FUNCTION"
gcloud functions deploy "$FUNCTION" \
  --project "$PROJECT" \
  --region "$REGION_GCP" \
  --gen2 \
  --runtime python311 \
  --source "$here/../deploy/billingGuard" \
  --entry-point stopBilling \
  --trigger-topic "$TOPIC" \
  --memory 256Mi \
  --max-instances 1 \
  --set-env-vars "GUARDED_PROJECT=$PROJECT"

# ---------------------------------------------------------------- 4. the role
# Detaching a billing account is a change to the billing account, not to the project,
# so the role goes on the billing account.
say "4/5 Billing Account Administrator for the function"
account=$(gcloud functions describe "$FUNCTION" --project "$PROJECT" \
  --region "$REGION_GCP" --gen2 --format 'value(serviceConfig.serviceAccountEmail)')
echo "  service account: $account"
gcloud beta billing accounts add-iam-policy-binding "$BILLING_ACCOUNT" \
  --member "serviceAccount:$account" \
  --role roles/billing.admin

# -------------------------------------------------------------- 5. the budget
say "5/5 budget"
# Thresholds below 1.0 only warn. The function acts when the cost reaches the budget,
# so the 0.5 and 0.9 rows are the warning mail and the 1.0 row is the stop.
gcloud billing budgets create \
  --billing-account "$BILLING_ACCOUNT" \
  --display-name "dashboard hard stop" \
  --budget-amount "${BUDGET}AUD" \
  --filter-projects "projects/$(gcloud projects describe "$PROJECT" \
      --format 'value(projectNumber)')" \
  --threshold-rule percent=0.5 \
  --threshold-rule percent=0.9 \
  --threshold-rule percent=1.0 \
  --all-updates-rule-pubsub-topic "projects/$PROJECT/topics/$TOPIC"

printf '\nthe guard is on.\n'
echo "$PROJECT loses its billing account once it costs $BUDGET AUD in a month."
echo "That stops the dashboard as well. To test it without waiting, publish a"
echo "message by hand:"
echo
echo "  gcloud pubsub topics publish $TOPIC --project $PROJECT \\"
echo "    --message '{\"costAmount\":99,\"budgetAmount\":1,\"currencyCode\":\"AUD\"}'"

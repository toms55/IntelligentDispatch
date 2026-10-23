#!/usr/bin/env bash
# Build the hosted dashboard and run it on Google Cloud Run.
#
#     bash scripts/deployCloudRun.sh --project nem-bess-dashboard
#
# `scripts/stageDashboard.sh` builds the directory this uploads. Every argument it does
# not recognise goes straight to that script.
#
# Cloud Run is the host because the dashboard fits inside its free tier and the service
# scales to zero. A Hugging Face Space that runs compute needs a paid plan.
#
# The flags below are the cost ceiling, and each one matters:
#
#   --max-instances 1    one container at most, so a traffic spike cannot multiply the
#                        bill. The dashboard holds no shared state, so one is enough.
#   --min-instances 0    scale to zero. An idle service costs nothing for compute.
#   --no-cpu-throttling  off. CPU is allocated during a request only, which is what
#                        makes an idle service free.
#   --concurrency 20     one container serves 20 callers at once. Eight gunicorn
#                        threads carry them.
#
# Image storage is the one charge that is not zero. Artifact Registry gives 0.5 GB
# free, and this image holds torch, so expect a few cents a month. Stage 3 deletes
# every image but the live one to hold that down.
#
# Run `scripts/setupBillingGuard.sh` once to add the automatic stop.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PROJECT=""
SERVICE=nem-bess-dashboard
REGION_GCP=australia-southeast1
MEMORY=2Gi
CPU=1
STAGE=runs/cloudrun
REGION=NSW1
RESULTS=results/rolling2023_profit_all
stageArgs=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project)    PROJECT="$2"; shift 2 ;;
    --service)    SERVICE="$2"; shift 2 ;;
    --gcp-region) REGION_GCP="$2"; shift 2 ;;
    --memory)     MEMORY="$2"; shift 2 ;;
    --cpu)        CPU="$2"; shift 2 ;;
    --stage)      STAGE="$2"; stageArgs+=(--stage "$2"); shift 2 ;;
    --region)     REGION="$2"; stageArgs+=(--region "$2"); shift 2 ;;
    --results)    RESULTS="$2"; stageArgs+=(--results "$2"); shift 2 ;;
    *)            stageArgs+=("$1" "$2"); shift 2 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "give a project: bash scripts/deployCloudRun.sh --project <id>" >&2
  exit 2
fi
command -v gcloud >/dev/null || { echo "gcloud is not installed" >&2; exit 2; }

say() { printf '\n========== %s ==========\n' "$1"; }

bash "$here/stageDashboard.sh" "${stageArgs[@]}" --stage "$STAGE"

# ------------------------------------------------------------ 1. the services
say "1/3 Cloud Build and Cloud Run on project $PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com --project "$PROJECT"

# ------------------------------------------------------------- 2. the deploy
# Cloud Build reads the Dockerfile at the root of the staged directory. The container
# listens on the port Cloud Run passes in `PORT`, which the Dockerfile reads.
say "2/3 build and deploy $SERVICE in $REGION_GCP"
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" \
  --region "$REGION_GCP" \
  --source "$STAGE" \
  --allow-unauthenticated \
  --memory "$MEMORY" \
  --cpu "$CPU" \
  --min-instances 0 \
  --max-instances 1 \
  --concurrency 20 \
  --timeout 300 \
  --no-cpu-throttling \
  --set-env-vars "DASHBOARD_REGION=$REGION,DASHBOARD_RESULTS=results/$(basename "$RESULTS"),DASHBOARD_POLICIES=trainedPolicies"

# -------------------------------------------------------------- 3. the tidy
# Every deploy writes a new image. Artifact Registry charges for what it stores, so
# only the live one stays.
say "3/3 delete the images this deploy replaced"
repository="$REGION_GCP-docker.pkg.dev/$PROJECT/cloud-run-source-deploy/$SERVICE"
live=$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
  --region "$REGION_GCP" --format 'value(spec.template.spec.containers[0].image)')
echo "live image: $live"
gcloud artifacts docker images list "$repository" --project "$PROJECT" \
  --format 'value(version)' --include-tags 2>/dev/null | while read -r digest; do
    [[ "$live" == *"$digest"* ]] && continue
    echo "  delete $digest"
    gcloud artifacts docker images delete "$repository@$digest" \
      --project "$PROJECT" --delete-tags --quiet || true
  done

url=$(gcloud run services describe "$SERVICE" --project "$PROJECT" \
  --region "$REGION_GCP" --format 'value(status.url)')
printf '\nthe dashboard is at %s\n' "$url"
echo "anyone with the link can open it. The first request after an idle period"
echo "starts the container, and that takes about a minute."

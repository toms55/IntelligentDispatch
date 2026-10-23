"""Turn billing off on the project when the budget is reached.

A Cloud Billing budget publishes to a Pub/Sub topic on every threshold it crosses.
This function reads those messages and, once the cost reaches the budget, removes the
billing account from the project. That stops every charge, because a project with no
billing account runs nothing.

`scripts/setupBillingGuard.sh` deploys it and wires the budget to it.

Two things to know before you rely on it:

1. A cost appears before the notification does. The dashboard can therefore run a
   little past the budget before this fires. Set the budget below what you are willing
   to spend, not at it.
2. Removing the billing account stops the Cloud Run service as well. The link goes
   dead until you attach billing again by hand. That is the point: the guard trades
   the dashboard for the certainty of no further charge.
"""
import base64
import json
import os

import functions_framework
from googleapiclient import discovery

# The project the guard switches off. `scripts/setupBillingGuard.sh` sets it, so the
# guard cannot disable a project it was not deployed for.
guardedProject = os.environ["GUARDED_PROJECT"]

billing = discovery.build("cloudbilling", "v1", cache_discovery=False)


@functions_framework.cloud_event
def stopBilling(event):
    """Read one budget notification, and act if the cost reached the budget."""
    payload = base64.b64decode(event.data["message"]["data"]).decode()
    notice = json.loads(payload)
    cost = float(notice.get("costAmount", 0.0))
    budget = float(notice.get("budgetAmount", 0.0))
    name = notice.get("budgetDisplayName", "the budget")
    currency = notice.get("currencyCode", "")

    if cost < budget:
        print(f"{name}: {cost:.2f} of {budget:.2f} {currency}. Nothing to do.")
        return

    project = f"projects/{guardedProject}"
    info = billing.projects().getBillingInfo(name=project).execute()
    if not info.get("billingAccountName"):
        print(f"{guardedProject} already has no billing account.")
        return

    print(f"{name}: {cost:.2f} of {budget:.2f} {currency}. "
          f"Removing the billing account from {guardedProject}.")
    # An empty name is how the API detaches the account.
    billing.projects().updateBillingInfo(
        name=project, body={"billingAccountName": ""}).execute()
    print(f"billing is off for {guardedProject}. The dashboard is now down. "
          f"Attach a billing account in the console to bring it back.")

"""The WSGI entry point a host runs. `python -m visualise.dashboard` for a server.

`visualise/dashboard/__main__.py` owns the app. This module only turns the three
command line arguments into environment variables, because a host sets environment
variables and does not pass arguments.

    DASHBOARD_REGION     the region the first screen opens on (default NSW1)
    DASHBOARD_RESULTS    the backtest records to read (default: the one run in backtestResults/)
    DASHBOARD_POLICIES   where the published agents are (default: trainedPolicies)

Serve it with gunicorn, not with the Dash development server:

    gunicorn --chdir . --bind 0.0.0.0:7860 deploy.app:server
"""
import os

from visualise.dashboard import __main__ as dashboard
from visualise.dashboard import data, live

region = os.environ.get("DASHBOARD_REGION") or "NSW1"
resultsDir = data.findResults(os.environ.get("DASHBOARD_RESULTS") or None)
policyRoot = os.environ.get("DASHBOARD_POLICIES") or str(live.defaultCheckpoints)

print(f"reading {resultsDir}", flush=True)
print(f"policies in {policyRoot}", flush=True)

app = dashboard.build(region, resultsDir, policyRoot)

# What gunicorn binds to. Dash keeps the Flask application here.
server = app.server

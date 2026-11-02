"""The WSGI entry point a host runs. `python -m visualise.dashboard` for a server.

`visualise/dashboard/__main__.py` owns the app. This module only turns the three
command line arguments into environment variables, because a host sets environment
variables and does not pass arguments.

    DASHBOARD_REGION     the region the first screen opens on (default NSW1)
    DASHBOARD_RESULTS    the published runs to show, comma separated, the screens
                         opening on the first (default: every experiment
                         config.dashboard_runs names)
    DASHBOARD_POLICIES   where the published agents are (default: trainedPolicies)

Serve it with gunicorn, not with the Dash development server:

    gunicorn --chdir . --bind 0.0.0.0:7860 deploy.app:server
"""
import os
import re

from visualise.dashboard import __main__ as dashboard
from visualise.dashboard import data, live

region = os.environ.get("DASHBOARD_REGION") or "NSW1"
# The Space names its staged runs here, so the screens offer those experiments and no
# others, and they open on the first one named. Commas or spaces separate them. Without
# the variable the app offers the whole pinned catalogue, which is what a local
# dashboard does.
named = re.split(r"[,\s]+", (os.environ.get("DASHBOARD_RESULTS") or "").strip())
staged = [item for item in named if item]
resultsDirs = data.findResults(staged) if staged else None
policyRoot = os.environ.get("DASHBOARD_POLICIES") or str(live.defaultCheckpoints)

reading = resultsDirs or ("config.dashboard_runs",)
print(f"reading {', '.join(str(item) for item in reading)}", flush=True)
print(f"policies in {policyRoot}", flush=True)

app = dashboard.build(region, resultsDirs, policyRoot)

# What gunicorn binds to. Dash keeps the Flask application here.
server = app.server

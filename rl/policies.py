"""`trainedPolicies/`: the one place a finished agent lives.

Training and the dashboard run on different machines, so the agents have to travel.
This module is the only definition of how they are laid out, and every writer and
reader goes through it.

    trainedPolicies/<region>/<algo>-<trainMode>-s<seed>/
        agent.zip      the weights
        scaler.json    the price transform the weights were trained under
        policy.json    where the agent came from, and when

The directory name is the label the dashboard shows, so a reader needs no index file
to list what is on the machine. `policy.json` is read by a person, not by the loader.
Copy a region directory to another machine and the dashboard picks it up.

Working state stays where it is. The rolling backtest still writes one checkpoint per
fold under `--out`, because it warm starts the next fold from the last one. Only the
final agent of a run is published here.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

defaultRoot = Path("trainedPolicies")
agentFile = "agent.zip"
scalerFile = "scaler.json"
metadataFile = "policy.json"


def label(algo, trainMode, seed):
    """The name of one published agent. This is what the dashboard dropdown shows."""
    return f"{algo}-{trainMode}-s{int(seed)}"


def policyDir(region, algo, trainMode, seed, root=defaultRoot):
    return Path(root) / region / label(algo, trainMode, seed)


def publish(checkpoint, region, algo, trainMode, seed, root=defaultRoot, **facts):
    """Copy a finished checkpoint into `trainedPolicies/` and return its directory.

    `checkpoint` is a training output directory holding `agent.zip` and `scaler.json`.
    Both are copied, because the transform is part of what the weights mean. Anything
    in `facts` is written into `policy.json` beside them.

    A second publish of the same name replaces the files. That is what a later run of
    the same experiment should do.
    """
    checkpoint = Path(checkpoint)
    missing = [name for name in (agentFile, scalerFile)
               if not (checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{checkpoint} holds no {' or '.join(missing)}, so it is not a finished "
            f"checkpoint and cannot be published"
        )
    directory = policyDir(region, algo, trainMode, seed, root)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (agentFile, scalerFile):
        shutil.copy2(checkpoint / name, directory / name)
    (directory / metadataFile).write_text(json.dumps({
        "region": region, "algo": algo, "trainMode": trainMode, "seed": int(seed),
        "label": label(algo, trainMode, seed),
        "source": str(checkpoint),
        "publishedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **facts,
    }, indent=2, default=str) + "\n")
    return directory


def available(region, root=defaultRoot):
    """Every published agent for `region`, as {label: directory}.

    A directory without its weights is left out. A copy that stopped halfway must not
    show up in a dropdown as something the dashboard can run.
    """
    directory = Path(root) / region
    if not directory.is_dir():
        return {}
    return {child.name: child for child in sorted(directory.iterdir())
            if (child / agentFile).exists()}


def regions(root=defaultRoot):
    """Every region that holds at least one published agent."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(child.name for child in root.iterdir() if available(child.name, root))


def describe(directory):
    """The facts written at publish time, or an empty dict for an older directory."""
    path = Path(directory) / metadataFile
    return json.loads(path.read_text()) if path.exists() else {}

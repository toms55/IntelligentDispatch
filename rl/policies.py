"""`trainedPolicies/`: the one place a finished agent lives.

Training and the dashboard run on different machines, so the agents have to travel.
This module is the only definition of how they are laid out, and every writer and
reader goes through it.

    trainedPolicies/<region>/<algo>-<trainMode>[-<tag>]-s<seed>/
        agent.zip      the weights
        scaler.json    the price transform the weights were trained under
        policy.json    where the agent came from, and when

The directory name is the label the dashboard shows, so a reader needs no index file
to list what is on the machine. The optional tag keeps a run whose agents mean something
else, such as a different episode start, beside the ones already published rather than
over them. `policy.json` is read by a person, not by the loader.
Copy a region directory to another machine and the dashboard picks it up.

Working state stays where it is. The rolling backtest still writes one checkpoint per
fold under `--out`, because it warm starts the next fold from the last one. Only the
final agent of a run is published here.
"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import config
from rl import algos

defaultRoot = Path("trainedPolicies")
agentFile = "agent.zip"
scalerFile = "scaler.json"
metadataFile = "policy.json"


def label(algo, trainMode, seed, tag=None):
    """The name of one published agent. This is what the dashboard dropdown shows."""
    middle = f"{trainMode}-{tag}" if tag else trainMode
    return f"{algo}-{middle}-s{int(seed)}"


def policyDir(region, algo, trainMode, seed, root=defaultRoot, tag=None):
    return Path(root) / region / label(algo, trainMode, seed, tag)


def publish(checkpoint, region, algo, trainMode, seed, root=defaultRoot, tag=None,
            **facts):
    """Copy a finished checkpoint into `trainedPolicies/` and return its directory.

    `checkpoint` is a training output directory holding `agent.zip` and `scaler.json`.
    Both are copied, because the transform is part of what the weights mean. Anything
    in `facts` is written into `policy.json` beside them.

    A second publish of the same name replaces the files. That is what a later run of
    the same experiment should do. A `tag` gives a different experiment its own name.
    """
    checkpoint = Path(checkpoint)
    missing = [name for name in (agentFile, scalerFile)
               if not (checkpoint / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{checkpoint} holds no {' or '.join(missing)}, so it is not a finished "
            f"checkpoint and cannot be published"
        )
    directory = policyDir(region, algo, trainMode, seed, root, tag)
    directory.mkdir(parents=True, exist_ok=True)
    for name in (agentFile, scalerFile):
        shutil.copy2(checkpoint / name, directory / name)
    (directory / metadataFile).write_text(json.dumps({
        "region": region, "algo": algo, "trainMode": trainMode, "seed": int(seed),
        "label": label(algo, trainMode, seed, tag),
        "source": str(checkpoint),
        # What the weights expect to read, and the budget they were trained under. The
        # forecast screen builds its environment from config, so an agent of another
        # layout cannot be loaded into it and must not be offered: see `loadable`.
        "observationSize": config.observation_size,
        "cycleBudget": config.cycle_budget,
        # Whether this algorithm read the action mask. Recorded so a later build that
        # moves the mask on or off an algorithm can drop the agent rather than run it
        # under the wrong policy: see `loadable` and `maskedDqnPolicies`.
        "readsMask": algos.supportsMasking(algo),
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


# The observation layout of every agent published before `publish` recorded one: SoC,
# 58 price features and the trading-day clock. Those agents carry no `observationSize`,
# and this is what their weights expect.
unrecordedObservationSize = 60

# Published agents whose DQN read the action mask when it acted. `rl/algos.py` here has
# no such algorithm: DQN relies on the environment's clipping, and
# `config.invalid_action_penalty` measures what that costs. The weights load, because a
# masked DQN is a plain DQN with another `predict`, so nothing refuses them and the
# dashboard would run them unmasked under their own name. That is a wrong dispatch run,
# not a missing one, so `loadable` leaves them out.
#
# A fixed list, named by label without the seed suffix, because the mask is a property
# of the algorithm that trained them and no published `policy.json` records it. Every
# publish from now on records `readsMask`, so this list is closed and never grows.
maskedDqnPolicies = frozenset({
    "dqn-warm-daily_dump_soc0_maskdqn",
    "dqn-pooled-pooled_daily_empty_efc1",
    "dqn-finetune-pooled_daily_empty_efc1",
})


def readsMaskedDqn(label):
    """Whether this label names one of `maskedDqnPolicies`, at any seed."""
    return label.rsplit("-s", 1)[0] in maskedDqnPolicies


def loadable(region, root=defaultRoot):
    """The published agents this build can actually run, as {label: directory}.

    An agent is loadable when it was published under the observation layout now in
    `config`, and when the algorithm that trained it is the algorithm this build runs
    under that name. A 60-slot agent cannot be loaded into a 62-slot environment, and
    offering it in a dropdown offers a model the screen is certain to fail on.

    An agent published before the layout was recorded states none, so it is read as
    `unrecordedObservationSize`, which is the layout every one of them was trained under.

    A masked-DQN agent is dropped as well. See `maskedDqnPolicies` for why that one needs
    a list rather than a recorded fact.
    """
    out = {}
    for label, directory in available(region, root).items():
        if readsMaskedDqn(label):
            continue
        facts = describe(directory)
        size = facts.get("observationSize", unrecordedObservationSize)
        if int(size) != config.observation_size:
            continue
        # Recorded from this module's `publish`. An agent whose algorithm read the mask
        # under a name that reads none here means something else than it would now.
        readsMask = facts.get("readsMask")
        if readsMask is not None and bool(readsMask) != algos.supportsMasking(facts["algo"]):
            continue
        out[label] = directory
    return out


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

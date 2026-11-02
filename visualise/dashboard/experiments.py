"""The experiments the dashboard offers, and what each one tested.

An experiment is one published run of one region: the day rows, the manifest the run was
produced under, and the one-line description of what it tested. `config.dashboard_runs`
is the list, and this module is the only reader of it.

Both screens read the same catalogue, which is what keeps them honest against each other.
The backtest screen offers an entry per experiment, train mode and price source, at one
seed. The forecast screen offers the agents those same experiments trained, at the same
seed, and holds back any agent with no experiment behind it. So a reader cannot run a
model on one screen and fail to find its numbers on the other.

Three things travel with an entry and nothing else may state them:

1. The settings box. `settings()` reads the manifest, so the screen states the episode,
   the cycle budget, the forecast refresh and the training budget as the backtest ran
   them.
2. The oracle. Every published row carries the bound of its own run, so selecting an
   experiment selects its oracle with it. Nothing here solves and nothing re-derives a
   bound.
3. The environment the forecast screen runs. `envSettings()` comes from the manifest, not
   from `config` and not from `policy.json`. The agents were trained on another machine
   whose `config` has moved since, and `policy.json` records that machine's values.
"""
import functools
import json
from pathlib import Path
from typing import NamedTuple, Optional

import config
from backtest import records
from rl import policies
from visualise.dashboard import data

# The region a pooled agent is published under in `trainedPolicies/`, and the name the
# pooled training directory carries in place of a region.
pooledRegion = "POOLED"

# What a train mode means, for the settings box.
trainModes = {
    "warm": "each fold continues the agent the fold before it trained",
    "cold": "each fold trains a new agent from scratch",
    "pooled": "one shared agent trained on every region end to end",
    "finetune": "a copy of the shared agent, fine-tuned on this region",
}

maskNote = {True: "reads the action mask", False: "no action mask, the environment clips"}


class Experiment(NamedTuple):
    """One published run of one region."""
    run: str
    region: str
    description: Optional[str]
    manifest: dict
    resultsDir: Path
    # Where the run was scored, which is how a published agent finds its experiment.
    # See `rawRun`.
    rawResults: str = ""


class Entry(NamedTuple):
    """One model of one experiment, at the seed the dashboard shows.

    `priceSource` is None on the forecast screen, which has one source and runs the agent
    itself. The backtest screen scores the same agent twice, so there it is part of the
    entry.
    """
    run: str
    region: str
    algo: str
    trainMode: str
    seed: int
    priceSource: Optional[str] = None

    @property
    def label(self):
        """The dropdown text. Unique by construction, so it is also the dropdown value.

        It opens with the old `algo-trainMode-sN[-priceSource]` name, because that is how
        a published agent is named, and closes with the experiment the numbers came from.
        Everything else sits in the settings box.
        """
        source = f"-{self.priceSource}" if self.priceSource else ""
        return f"{self.algo}-{self.trainMode}-s{int(self.seed)}{source} | {shortRun(self.run)}"


def shortRun(run):
    """A run name without the span every run shares, for a label that has to fit."""
    return run[len("rolling2023_"):] if run.startswith("rolling2023_") else run


def contextOf(run, region, root=data.backtestRoot):
    """The published context of one run and region: its description and its manifest."""
    path = records.contextPath(root, run, region)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist, so what {run} tested is not recorded and the "
            f"dashboard cannot describe it. Publish it again with "
            f"`python -m backtest.publish --context-only --description ...`")
    return json.loads(path.read_text())


def load(run, region, root=data.backtestRoot):
    """One experiment, or a FileNotFoundError naming what is missing."""
    if not data.publishedFile(Path(root) / run, region).exists():
        raise FileNotFoundError(
            f"{run} is in config.dashboard_runs but holds no {region} CSV under {root}")
    context = contextOf(run, region, root)
    return Experiment(run=run, region=region, description=context.get("description"),
                      manifest=context.get("manifest") or {},
                      resultsDir=Path(root) / run,
                      rawResults=str(context.get("rawResults") or ""))


def catalogue(region, runs=None, root=data.backtestRoot):
    """Every pinned experiment published for `region`, in the pinned order.

    A pinned run that never scored this region is left out, not raised on: the regions a
    run holds are its own, and a reader of another region still gets the rest. A run that
    did score it and carries no manifest raises, because that is a published run missing
    a file rather than a region it never covered.
    """
    out = []
    for run in (config.dashboard_runs if runs is None else runs):
        if not data.publishedFile(Path(root) / run, region).exists():
            continue
        out.append(load(run, region, root))
    return out


def regions(runs=None, root=data.backtestRoot):
    """Every region any pinned run was published for."""
    found = set()
    for run in (config.dashboard_runs if runs is None else runs):
        found.update(data.availableRegions(Path(root) / run))
    return sorted(found)


def bestSeed(run, trainMode, region, seeds=None):
    """The seed the dashboard shows for one model, from `config.best_seed_by_run`.

    Pinned, never ranked here, so the thesis quotes one file and a reader sees the same
    seed every time.

    `seeds` are the seeds the rows hold, and they are only read for a run the map does
    not name: the dashboard then shows the lowest of them and `pinned()` reports that it
    was not chosen. Every run in `config.dashboard_runs` is named in the map, which
    `test_experiments.py` holds it to, so this fallback never fires on a published run.
    A missing entry with no rows to fall back on raises.
    """
    picked = (config.best_seed_by_run.get(run) or {}).get(trainMode, {})
    if region in picked:
        return int(picked[region])
    if seeds is not None and len(seeds):
        return int(min(seeds))
    raise KeyError(
        f"config.best_seed_by_run names no seed for {run} {trainMode} {region}. "
        f"Run `uv run python scripts/rankSeeds.py {run}` and pin what it prints")


def pinned(run, trainMode, region):
    """Whether `config.best_seed_by_run` names this model's seed."""
    return region in (config.best_seed_by_run.get(run) or {}).get(trainMode, {})


def seedCount(days, algo, trainMode):
    """How many seeds that model scored, which the settings box states beside the one."""
    rows = days[(days.algo == algo) & (days.trainMode == trainMode)]
    return int(rows.seed.nunique())


def entries(experiment, days, priceSources=True):
    """The models of one experiment, at the pinned seed, in a stable order.

    `days` are the published rows, so only a model the run actually scored is offered.
    `priceSources=False` drops the price source, which is what the forecast screen wants:
    it runs the agent against the AEMO path and has one source.
    """
    if days.empty:
        return []
    pairs = (days[["algo", "trainMode"]].drop_duplicates()
             .sort_values(["algo", "trainMode"]).to_numpy().tolist())
    sources = (sorted(set(days.priceSource)) if priceSources else [None])
    out = []
    for algo, trainMode in pairs:
        held = days[(days.algo == algo) & (days.trainMode == trainMode)].seed.unique()
        seed = bestSeed(experiment.run, trainMode, experiment.region, held)
        for source in sources:
            rows = days[(days.algo == algo) & (days.trainMode == trainMode)
                        & (days.seed == seed)]
            if priceSources:
                rows = rows[rows.priceSource == source]
            if rows.empty:
                continue
            out.append(Entry(experiment.run, experiment.region, algo, trainMode, seed,
                             source))
    return out


# --- What an experiment tested -----------------------------------------------


def _intervals(count):
    """A count of 5-minute intervals as the time it covers."""
    if count is None:
        return "not recorded"
    hours = count * 5 / 60
    if count % 288 == 0:
        days = count // 288
        return f"{count:,} intervals, {days} trading day" + ("s" if days > 1 else "")
    if hours >= 1 and float(hours).is_integer():
        return f"{count:,} intervals, {int(hours)} hours"
    return f"{count:,} intervals, {count * 5} minutes"


def _refresh(every):
    """How often the backtest pulled a new AEMO forecast on a test day."""
    if every is None:
        return "not recorded"
    if every == 1:
        return "every 5 minutes, each dispatch interval"
    minutes = every * 5
    if minutes % 60:
        return f"every {minutes} minutes"
    hours = minutes // 60
    return f"every {hours} hour" + ("s" if hours > 1 else "")


def _budget(manifest):
    budget = manifest.get("cycleBudget")
    if budget is None:
        return "none, the agent may cycle as often as it pays to"
    return f"{float(budget):.2f} equivalent full cycles per episode"


def _closing(manifest):
    """What the episode does with the energy still in the pack at its close."""
    if manifest.get("endEmpty"):
        return "the pack is forced empty by the close"
    price = manifest.get("terminalPrice")
    if price:
        return f"closing stock valued at ${float(price):,.2f}/MWh"
    return "unvalued, so closing stock is worth nothing"


def _training(manifest):
    warm = manifest.get("warmTimesteps")
    cold = manifest.get("coldTimesteps")
    weeks = manifest.get("trainWeeks")
    parts = []
    if weeks:
        parts.append(f"{int(weeks)} weeks of realised prices per fold")
    pooled = manifest.get("pooled") or {}
    # A pooled run states its own three budgets, and its warm and cold counts are the
    # shared agent's, so naming them twice would read as four numbers.
    if warm and cold and not pooled:
        parts.append(f"{int(warm):,} warm steps, {int(cold):,} cold")
    if pooled:
        parts.append(f"pooled {int(pooled.get('pooledWarm', 0)):,} warm / "
                     f"{int(pooled.get('pooledCold', 0)):,} cold over "
                     f"{len(pooled.get('regions', []))} regions, then "
                     f"{int(pooled.get('fineTune', 0)):,} per region")
    return ", ".join(parts) or "not recorded"


def _span(manifest):
    start, end = manifest.get("start"), manifest.get("end")
    if not start or not end:
        return "not recorded"
    test, step = manifest.get("testWeeks"), manifest.get("stepWeeks")
    rolls = f", {int(test)}-week folds stepped {int(step)} week" if test and step else ""
    return f"{str(start)[:10]} to {str(end)[:10]}{rolls}"


def _oracle(manifest, days=None):
    """Which perfect-foresight bound the rows carry.

    The bound is in the rows, one per episode, so an experiment and its oracle are
    selected together. This line names the problem that oracle solved.
    """
    mode = ("rated throughput or nothing, solved exactly by value iteration"
            if manifest.get("discretePower")
            else "continuous power, solved as a MILP")
    out = [mode]
    if manifest.get("cycleBudget") is not None:
        out.append("budget priced, not capped")
    if days is not None and "oracleGap" in days and days.oracleGap.notna().any():
        worst = float(days.oracleGap.max())
        out.append(f"widest gap {worst * 100:.2f}%" if worst else "no gap")
    return "; ".join(out)


def settings(experiment, entry=None, days=None):
    """What the experiment tested, as (name, value) lines for the box on screen.

    Every value comes from the published manifest, so the box describes the backtest that
    was run rather than the settings this checkout happens to hold now. One function, so
    both screens describe a model the same way.
    """
    manifest = experiment.manifest
    lines = [("Experiment", experiment.run),
             ("What it tested", experiment.description or "not recorded")]
    if entry is not None:
        # Imported here, not at the top: this module is read by the backtest screen as
        # well, and `rl.algos` drags in torch and stable-baselines3 to answer one
        # question. `algos` is still the only definition of which algorithm masks.
        from rl import algos
        lines.append(("Algorithm",
                      f"{entry.algo}, {maskNote[algos.supportsMasking(entry.algo)]}"))
        lines.append(("Train mode",
                      f"{entry.trainMode}: {trainModes.get(entry.trainMode, 'not recorded')}"))
        seeds = manifest.get("seeds") or []
        of = f" of {seedCount(days, entry.algo, entry.trainMode)}" if days is not None \
            else (f" of {len(seeds)}" if seeds else "")
        why = ("the highest total profit on forecast prices"
               if pinned(entry.run, entry.trainMode, entry.region)
               else "not pinned for this run, so the lowest seed is shown")
        lines.append(("Seed", f"s{int(entry.seed)}{of}, {why}"))
        if entry.priceSource:
            lines.append(("Prices observed", "AEMO forecast" if entry.priceSource ==
                          "forecast" else "realised, perfect knowledge of the price"))
    lines += [
        ("Episode", _intervals(manifest.get("horizon"))),
        ("Cycle budget", _budget(manifest)),
        ("Episode close", _closing(manifest)),
        ("Start state of charge", f"{float(manifest.get('initialSoc', 0)) * 100:.0f}%"),
        ("Forecast refresh", _refresh(manifest.get("forecastEvery"))),
        ("Training", _training(manifest)),
        ("Test span", _span(manifest)),
        ("Oracle", _oracle(manifest, days)),
    ]
    return lines


def envSettings(experiment):
    """The environment one episode of this experiment ran under, for `live.suggest`.

    From the manifest, because the agents were trained on another machine and this
    checkout's `config` has moved since. `policy.json` is no better: it records that
    machine's `config` at publish time, which for the daily 1-cycle agents says a weekly
    budget of 7.
    """
    manifest = experiment.manifest
    budget = manifest.get("cycleBudget")
    return {
        "horizon": int(manifest.get("horizon") or config.episode_intervals),
        "cycleBudget": None if budget is None else float(budget),
        "endEmpty": bool(manifest.get("endEmpty")),
        "terminalPrice": float(manifest.get("terminalPrice") or 0.0),
        "initialSoc": float(manifest.get("initialSoc", config.initial_soc)),
    }


# --- The link from a published agent to its experiment -----------------------


def rawRun(experiment):
    """The raw results directory the experiment was published from, without its region.

    This is the one name both ends share. The manifest records where the run was scored,
    `policy.json` records where each agent was trained, and the published run name is
    neither of them: `backtest.publish --name` renames a run on its way out.
    """
    actions = experiment.manifest.get("actionsFile")
    name = Path(actions).stem if actions else Path(experiment.rawResults).name
    return _stripRegion(name)


# The regions the project runs, from the one per-region constant in config.py.
knownRegions = tuple(config.terminal_energy_price) + (pooledRegion,)


def _stripRegion(name):
    """A run directory name without the region it was run for.

    A pooled run is written under POOLED rather than a region, because one agent covers
    all five.
    """
    for region in knownRegions:
        if name.endswith(f"_{region}"):
            return name[:-len(region) - 1]
    return name


def policyRun(directory):
    """The raw run one published agent was trained by, without its region.

    `rawRun` in `policy.json` states it, which is what every agent published since this
    link existed records. An agent published before that is read from its `source`, the
    checkpoint it was copied from: `runs/<raw run>_<region>/<region>/fold<N>/...`.
    """
    facts = policies.describe(directory)
    recorded = facts.get("rawRun")
    if recorded:
        return _stripRegion(str(recorded))
    source = str(facts.get("source", ""))
    if "runs/" not in source:
        return None
    return _stripRegion(source.split("runs/")[-1].split("/")[0]) or None


def loadableAgents(region, root=policies.defaultRoot):
    """Every published agent of `region` this build can run, as {label: directory}.

    `rl/policies.py` owns the layout. Training publishes its final agent there, and the
    forecast screen runs it, so a machine that only runs the dashboard needs that one
    directory copied across and nothing else.

    `loadable`, not `available`: the screen builds its observation from `config`, so an
    agent trained under another observation layout cannot be run and must not be
    offered.
    """
    return policies.loadable(region, root)


@functools.lru_cache(maxsize=16)
def _agentIndex(region, root):
    """{(rawRun, algo, trainMode, seed): directory} for every loadable agent of `region`."""
    out = {}
    for directory in loadableAgents(region, root).values():
        facts = policies.describe(directory)
        raw = policyRun(directory)
        if raw is None:
            continue
        out[(raw, facts.get("algo"), facts.get("trainMode"),
             int(facts.get("seed", -1)))] = directory
    return out


def agentOf(entry, experiment, root=policies.defaultRoot):
    """The published agent one entry names, or None when this machine holds no such agent.

    None is a held-back model, not an error. The forecast screen drops it, so it never
    offers a model it cannot run, and the backtest screen keeps it: its numbers are
    published whether or not the weights travelled.
    """
    index = _agentIndex(entry.region, str(root))
    return index.get((rawRun(experiment), entry.algo, entry.trainMode, int(entry.seed)))


class Runnable(NamedTuple):
    """One model the forecast screen can run: what to run, under which settings.

    `experiment` carries the manifest the agent was trained and scored under, which is
    what the screen builds its environment from. See `envSettings`.
    """
    entry: Entry
    experiment: Experiment
    directory: Path


def runnable(region, root=data.backtestRoot, policyRoot=policies.defaultRoot,
             loader=None, runs=None):
    """Every model the forecast screen can run, as a list of `Runnable`.

    One per experiment and train mode, at the pinned seed, with no price source: the
    screen runs the agent against the AEMO path published now. `loader` is the screen's
    own cache of loaded runs, so opening the forecast tab re-reads nothing.
    """
    out = []
    read = loader or (lambda run, region_: data.loadRun(region_, Path(root) / run))
    for experiment in catalogue(region, runs, root):
        days = read(experiment.run, region).days
        for entry in entries(experiment, days, priceSources=False):
            directory = agentOf(entry, experiment, policyRoot)
            if directory is not None:
                out.append(Runnable(entry, experiment, directory))
    return out

"""Tests for visualise/dashboard/experiments.py.

Two kinds. Most build a published run under tmp_path and read it back. The last class
reads the repository itself, because the catalogue, the published runs and the published
agents are all tracked, so what they say about each other is testable and is the one
thing a reader of the dashboard depends on.
"""
import functools
import json

import pytest

import config
from rl import policies
from visualise.dashboard import data, experiments
from visualise.dashboard.test_app import publish
from visualise.dashboard.test_data import REGION, dayRows, publishRun


@pytest.fixture
def pinned(monkeypatch):
    """`rolling` as the one dashboard experiment, with its seed pinned."""
    monkeypatch.setattr(config, "dashboard_runs", ("rolling",))
    monkeypatch.setattr(config, "best_seed_by_run",
                        {"rolling": {"warm": {REGION: 0}}})


@pytest.fixture
def published(tmp_path):
    return publishRun(tmp_path, dayRows(), every=288)


class TestCatalogue:
    # The catalogue is the pinned list, not whatever is on disk. A run nobody pinned is
    # published, in the thesis, and off the screen.
    def test_it_offers_the_pinned_runs_only(self, tmp_path, pinned):
        publishRun(tmp_path, dayRows(), name="rolling")
        publishRun(tmp_path, dayRows(), name="unpinned")
        found = experiments.catalogue(REGION, root=tmp_path / "backtestResults")
        assert [item.run for item in found] == ["rolling"]

    # A pinned run that never scored this region is left out, not raised on.
    def test_a_region_the_run_never_scored_is_left_out(self, tmp_path, pinned):
        publishRun(tmp_path, dayRows(), name="rolling")
        assert experiments.catalogue("SA1", root=tmp_path / "backtestResults") == []

    # The description and the manifest come off disk, so the box describes the backtest
    # that ran rather than the settings this checkout holds now.
    def test_it_reads_the_published_context(self, published, pinned):
        item = experiments.load("rolling", REGION, published.parent)
        assert item.description == "a test run"
        assert item.manifest["cycleBudget"] == 1.0

    # A run with no context JSON cannot be described, and says which file is missing.
    def test_a_run_without_its_context_names_the_file(self, tmp_path, pinned):
        run = publishRun(tmp_path, dayRows(), name="rolling", context=False)
        with pytest.raises(FileNotFoundError, match="rolling_NSW1.json"):
            experiments.load("rolling", REGION, run.parent)

    # The regions come from the published files, never from config.py.
    def test_the_regions_come_from_disk(self, published, pinned):
        assert experiments.regions(root=published.parent) == [REGION]


class TestEntries:
    # One entry per train mode and price source, at the pinned seed and no other.
    def test_it_offers_one_seed(self, published, pinned):
        rows = dayRows()
        for row in rows:
            row["seed"] = 3
        publishRun(published.parent.parent, dayRows() + rows, name="rolling")
        item = experiments.load("rolling", REGION, published.parent)
        days = data.loadRun(REGION, published).days
        found = experiments.entries(item, days)
        assert [entry.seed for entry in found] == [0]

    # The label names the model and the experiment, because two experiments can hold
    # the same model.
    def test_the_label_names_the_experiment(self, published, pinned):
        item = experiments.load("rolling", REGION, published.parent)
        days = data.loadRun(REGION, published).days
        entry = experiments.entries(item, days)[0]
        assert entry.label == "maskableppo-warm-s0-realised | rolling"

    # The forecast screen has one price source, so its entries carry none.
    def test_the_forecast_screen_drops_the_price_source(self, published, pinned):
        item = experiments.load("rolling", REGION, published.parent)
        days = data.loadRun(REGION, published).days
        entry = experiments.entries(item, days, priceSources=False)[0]
        assert entry.priceSource is None
        assert entry.label == "maskableppo-warm-s0 | rolling"

    # A model the run never scored is never offered, whatever the manifest claims.
    def test_only_a_scored_model_is_offered(self, published, pinned):
        item = experiments.load("rolling", REGION, published.parent)
        days = data.loadRun(REGION, published).days
        assert {entry.trainMode for entry in experiments.entries(item, days)} == {"warm"}


class TestBestSeed:
    # Pinned, never ranked at draw time, so a reader sees the same seed every time.
    def test_it_reads_the_pinned_map(self, monkeypatch):
        monkeypatch.setattr(config, "best_seed_by_run", {"r": {"warm": {REGION: 3}}})
        assert experiments.bestSeed("r", "warm", REGION) == 3
        assert experiments.pinned("r", "warm", REGION)

    # A run nobody pinned still draws, on the lowest seed it holds, and `pinned` says
    # the seed was not chosen so the box can state it.
    def test_an_unpinned_run_falls_back_to_the_lowest_seed(self, monkeypatch):
        monkeypatch.setattr(config, "best_seed_by_run", {})
        assert experiments.bestSeed("r", "warm", REGION, [4, 2, 3]) == 2
        assert not experiments.pinned("r", "warm", REGION)

    # With nothing pinned and no rows to fall back on, it names the script to run.
    def test_nothing_to_show_names_the_script(self, monkeypatch):
        monkeypatch.setattr(config, "best_seed_by_run", {})
        with pytest.raises(KeyError, match="rankSeeds"):
            experiments.bestSeed("r", "warm", REGION)


class TestSettings:
    def lines(self, published, **kwargs):
        item = experiments.load("rolling", REGION, published.parent)
        days = data.loadRun(REGION, published).days
        entry = experiments.entries(item, days)[0]
        return dict(experiments.settings(item, entry, days, **kwargs))

    # Every setting comes from the manifest, so the box states the backtest that ran.
    def test_it_describes_the_episode(self, published, pinned):
        got = self.lines(published)
        assert got["Episode"] == "288 intervals, 1 trading day"
        assert got["Cycle budget"] == "1.00 equivalent full cycles per episode"
        assert got["Episode close"] == "the pack is forced empty by the close"
        assert got["Start state of charge"] == "0%"

    # The refresh is the pull the backtest made, which `data.loadRun` has verified.
    def test_it_states_the_forecast_refresh(self, published, pinned):
        assert self.lines(published)["Forecast refresh"] == "every 24 hours"

    # The seed line says which seed of how many, because the thesis quotes the spread.
    def test_it_states_the_seed_and_the_count(self, published, pinned):
        assert self.lines(published)["Seed"].startswith("s0 of 1")

    # Which bound the rows carry, and how tight it is. Selecting an experiment selects
    # its oracle, so the box names the problem that oracle solved.
    def test_it_names_the_oracle(self, published, pinned):
        got = self.lines(published)["Oracle"]
        assert "value iteration" in got and "widest gap 1.00%" in got

    # An experiment with no budget says so rather than printing a zero.
    def test_no_budget_reads_as_none(self, tmp_path, pinned):
        run = publishRun(tmp_path, dayRows(), name="rolling", every=288,
                         settings={"cycleBudget": None, "endEmpty": False})
        item = experiments.load("rolling", REGION, run.parent)
        got = dict(experiments.settings(item))
        assert got["Cycle budget"].startswith("none")
        assert got["Episode close"].startswith("unvalued")


class TestEnvSettings:
    # The forecast screen runs the episode the experiment ran, from the manifest.
    def test_it_comes_from_the_manifest(self, published, pinned, monkeypatch):
        monkeypatch.setattr(config, "cycle_budget", 7.0)
        item = experiments.load("rolling", REGION, published.parent)
        assert experiments.envSettings(item) == {
            "horizon": 288, "cycleBudget": 1.0, "endEmpty": True,
            "terminalPrice": 0.0, "initialSoc": 0.0}


class TestTheAgentLink:
    # The link is the run that trained the agent, which `policy.json` records.
    def test_it_finds_the_agent_of_an_experiment(self, published, pinned, tmp_path):
        policyRoot = publish(tmp_path, rawRun="rolling")
        found = experiments.runnable(REGION, published.parent, policyRoot,
                                     runs=("rolling",))
        assert [item.entry.label for item in found] == [
            "maskableppo-warm-s0 | rolling"]

    # An agent of another run is not offered under this experiment's name.
    def test_an_agent_of_another_run_is_not_offered(self, published, pinned, tmp_path):
        policyRoot = publish(tmp_path, rawRun="somethingelse")
        assert experiments.runnable(REGION, published.parent, policyRoot,
                                    runs=("rolling",)) == []

    # An agent published before the link existed is read from its checkpoint path.
    def test_an_older_agent_is_read_from_its_source(self, tmp_path):
        directory = tmp_path / REGION / "maskableppo-warm-s0"
        directory.mkdir(parents=True)
        (directory / policies.metadataFile).write_text(json.dumps(
            {"source": "runs/rolling_efc7_NSW1/NSW1/fold142/warm/maskableppo_seed0"}))
        assert experiments.policyRun(directory) == "rolling_efc7"

    # A pooled agent is trained under POOLED rather than under a region.
    def test_a_pooled_source_drops_the_pooled_name(self, tmp_path):
        directory = tmp_path / REGION / "maskableppo-pooled-s0"
        directory.mkdir(parents=True)
        (directory / policies.metadataFile).write_text(json.dumps(
            {"source": "runs/rolling_pooled_POOLED/POOLED/fold142/pooled/x_seed0"}))
        assert experiments.policyRun(directory) == "rolling_pooled"

    # An agent with no link at all is held back rather than guessed at.
    def test_an_agent_with_no_link_is_dropped(self, tmp_path):
        directory = tmp_path / REGION / "maskableppo-warm-s0"
        directory.mkdir(parents=True)
        (directory / policies.metadataFile).write_text(json.dumps({"source": "/tmp/x"}))
        assert experiments.policyRun(directory) is None


@functools.lru_cache(maxsize=64)
def _run(run, region):
    """One published region of the repository, read once for the whole module.

    Every test below reads every pinned run over five regions, and each one is ten
    thousand CSV rows beside a 30 MB actions export. Reading each once turns a hundred
    reads into twenty five and keeps the suite quick enough to run on every change. The
    production screens cache the same way: the app holds one loader for the session.
    """
    return data.loadRun(region, data.backtestRoot / run)


def _days(run, region):
    return _run(run, region).days


@functools.lru_cache(maxsize=8)
def _runnable(region):
    """The models the forecast screen offers for `region`, through the cache above."""
    return tuple(experiments.runnable(region, loader=_run))


class TestThisRepository:
    """What the tracked catalogue, runs and agents say about each other.

    These read the repository, not a fixture. Every input is tracked by git, so the
    answers are the same on every machine, and they are the two promises the dashboard
    makes: every experiment it offers is published, and every model the forecast screen
    offers has its numbers on the backtest screen.
    """

    @pytest.mark.parametrize("run", config.dashboard_runs)
    def test_every_pinned_run_is_published(self, run):
        regions = data.availableRegions(data.backtestRoot / run)
        assert regions, f"{run} is pinned but published for no region"
        for region in regions:
            experiments.load(run, region)

    # Every model of a pinned run has its seed pinned, so nothing on the screen falls
    # back to the lowest seed.
    @pytest.mark.parametrize("run", config.dashboard_runs)
    def test_every_model_has_a_pinned_seed(self, run):
        for region in data.availableRegions(data.backtestRoot / run):
            for mode in sorted(set(_days(run, region).trainMode)):
                assert experiments.pinned(run, mode, region), (
                    f"config.best_seed_by_run names no seed for {run} {mode} {region}; "
                    f"run scripts/rankSeeds.py")

    # The promise the whole catalogue exists for: a reader who runs a model on the
    # forecast screen can read its backtest.
    @pytest.mark.parametrize("region", experiments.regions())
    def test_every_runnable_model_has_an_experiment(self, region):
        for item in _runnable(region):
            assert item.entry.run in config.dashboard_runs
            days = _days(item.entry.run, region)
            assert not data.entryRows(days, item.entry.algo, item.entry.trainMode,
                                      item.entry.seed).empty

    # Every pinned experiment can be run live in every region it scored, so the two
    # screens offer the same models and not two different lists.
    @pytest.mark.parametrize("region", experiments.regions())
    def test_every_experiment_has_its_agents(self, region):
        offered = {(item.entry.run, item.entry.trainMode)
                   for item in _runnable(region)}
        for item in experiments.catalogue(region):
            for mode in sorted(set(_days(item.run, region).trainMode)):
                assert (item.run, mode) in offered, (
                    f"{item.run} {mode} has no published agent for {region}; copy it "
                    f"from the machine that trained it")

    # The agents the forecast screen runs are published under the layout config states,
    # and a published run's observation layout cannot have moved under them.
    @pytest.mark.parametrize("region", experiments.regions())
    def test_every_runnable_agent_matches_the_observation_layout(self, region):
        for item in _runnable(region):
            facts = policies.describe(item.directory)
            assert int(facts.get("observationSize", policies.unrecordedObservationSize)
                       ) == config.observation_size

    # Every agent on disk that the screen does not offer is one of two things: an agent
    # of a run the catalogue drops, or another seed of a run it keeps. Nothing else is
    # held back, so no model disappears for a reason nobody stated.
    @pytest.mark.parametrize("region", experiments.regions())
    def test_held_back_agents_are_another_run_or_another_seed(self, region):
        offered = {item.directory for item in _runnable(region)}
        kept = {experiments.rawRun(item): item
                for item in experiments.catalogue(region)}
        held = set(experiments.loadableAgents(region).values()) - offered
        assert held, "nothing is held back, so this test proves nothing"
        for directory in held:
            facts = policies.describe(directory)
            item = kept.get(experiments.policyRun(directory))
            if item is None:
                continue
            seed = experiments.bestSeed(item.run, facts["trainMode"], region)
            assert int(facts["seed"]) != seed, (
                f"{directory} is the pinned seed of {item.run} and is not offered")

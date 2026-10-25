"""Tests for rl/policies.py: the trainedPolicies/ layout."""
import json

import pytest

from rl import policies

REGION = "NSW1"


def checkpoint(directory, weights=b"weights", scaler=None):
    """A finished training output directory: the weights and the transform."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / policies.agentFile).write_bytes(weights)
    (directory / policies.scalerFile).write_text(
        json.dumps(scaler or {"cap": 23200.0, "floor": -1000.0, "knee": 300.0}))
    return directory


class TestPublish:
    # The directory name is the label the dashboard shows, so it is built in one place.
    def test_the_label_names_the_directory(self, tmp_path):
        directory = policies.publish(checkpoint(tmp_path / "fold51"), REGION,
                                     "maskableppo", "warm", 0, root=tmp_path / "out")
        assert directory.name == "maskableppo-warm-s0"
        assert directory.parent.name == REGION

    # The transform travels with the weights. Weights read under a different market
    # price cap mean something other than what they learned.
    def test_the_scaler_travels_with_the_weights(self, tmp_path):
        source = checkpoint(tmp_path / "fold51", scaler={"cap": 600.0, "floor": -1000.0,
                                                         "knee": 300.0})
        directory = policies.publish(source, REGION, "maskableppo", "warm", 0,
                                     root=tmp_path / "out")
        assert (directory / policies.agentFile).read_bytes() == b"weights"
        assert json.loads((directory / policies.scalerFile).read_text())["cap"] == 600.0

    # A tag gives another experiment its own directory, so it cannot replace the first.
    def test_a_tag_publishes_beside_the_untagged_agent(self, tmp_path):
        plain = policies.publish(checkpoint(tmp_path / "a", weights=b"old"), REGION,
                                 "maskableppo", "warm", 0, root=tmp_path / "out")
        tagged = policies.publish(checkpoint(tmp_path / "b", weights=b"new"), REGION,
                                  "maskableppo", "warm", 0, root=tmp_path / "out",
                                  tag="start_at_0_SoC")
        assert tagged.name == "maskableppo-warm-start_at_0_SoC-s0"
        assert (plain / policies.agentFile).read_bytes() == b"old"
        assert policies.describe(tagged)["label"] == tagged.name
        assert set(policies.available(REGION, tmp_path / "out")) == {plain.name, tagged.name}

    # policy.json says where the agent came from, because the weights do not.
    def test_it_records_where_the_agent_came_from(self, tmp_path):
        directory = policies.publish(checkpoint(tmp_path / "fold51"), REGION,
                                     "maskableppo", "warm", 3, root=tmp_path / "out",
                                     command="rolling", fold=51)
        facts = policies.describe(directory)
        assert facts["region"] == REGION and facts["seed"] == 3
        assert facts["command"] == "rolling" and facts["fold"] == 51
        assert facts["publishedAt"]

    # A later run of the same experiment replaces what it published before.
    def test_publishing_again_replaces_the_weights(self, tmp_path):
        first = checkpoint(tmp_path / "fold51", weights=b"old")
        policies.publish(first, REGION, "maskableppo", "warm", 0, root=tmp_path / "out")
        second = checkpoint(tmp_path / "fold52", weights=b"new")
        directory = policies.publish(second, REGION, "maskableppo", "warm", 0,
                                     root=tmp_path / "out")
        assert (directory / policies.agentFile).read_bytes() == b"new"

    # Half a checkpoint is not publishable, and the message says which half is missing.
    @pytest.mark.parametrize("drop", [policies.agentFile, policies.scalerFile])
    def test_an_unfinished_checkpoint_is_refused(self, tmp_path, drop):
        source = checkpoint(tmp_path / "fold51")
        (source / drop).unlink()
        with pytest.raises(FileNotFoundError, match=drop):
            policies.publish(source, REGION, "maskableppo", "warm", 0,
                             root=tmp_path / "out")


class TestAvailable:
    def test_it_lists_what_was_published(self, tmp_path):
        for seed in (0, 1):
            policies.publish(checkpoint(tmp_path / f"s{seed}"), REGION, "maskableppo",
                             "warm", seed, root=tmp_path / "out")
        found = policies.available(REGION, tmp_path / "out")
        assert sorted(found) == ["maskableppo-warm-s0", "maskableppo-warm-s1"]

    # A copy that stopped halfway must not show up as something the dashboard can run.
    def test_a_directory_without_weights_is_left_out(self, tmp_path):
        (tmp_path / "out" / REGION / "a2c-cold-s1").mkdir(parents=True)
        assert policies.available(REGION, tmp_path / "out") == {}

    def test_an_unknown_region_is_empty(self, tmp_path):
        assert policies.available("TAS1", tmp_path / "out") == {}

    def test_regions_lists_only_those_holding_an_agent(self, tmp_path):
        policies.publish(checkpoint(tmp_path / "s0"), REGION, "maskableppo", "warm", 0,
                         root=tmp_path / "out")
        (tmp_path / "out" / "VIC1" / "a2c-cold-s0").mkdir(parents=True)
        assert policies.regions(tmp_path / "out") == [REGION]

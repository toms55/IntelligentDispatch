"""The problem-side ablations: the environment and the oracle must stay one problem.

The agent-side ablations leave the oracle alone. These change the wear cost and the action
set, so the oracle is solved under the same change and the replay below holds the two to
1e-9, as rl/test_env.py::TestMatchesTheOracle does for the baseline.
"""
from datetime import datetime

import pytest

from degradation.schade import SchadeDegradation
from optimisation import dp
from rl import baseline
from rl.ablation import Ablation
from rl.env import BessEnv
from rl.test_env import (MIXED_PRICES, PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST,
                         actionsFrom, priceSeries)
from utils import nem_interval_hours


@pytest.fixture
def paper():
    return SchadeDegradation(PAPER_MW, PAPER_MWH, PAPER_REPLACEMENT_COST)


class TestWearCostMatchesTheOracle:
    @pytest.mark.parametrize("variant", [
        Ablation(degradation="none"),
        Ablation(degradation="flat"),
        Ablation(degradation="flat", flatCost=40.0),
    ])
    def test_profit_and_soc_agree(self, paper, variant):
        costing = variant.costing(paper, nem_interval_hours)
        schedule = dp.solve(MIXED_PRICES, paper, initialSoc=0.0, calendar=False,
                            costing=costing)
        env = BessEnv(priceSeries(MIXED_PRICES), paper, horizon=len(MIXED_PRICES),
                      calendar=False, rewardScale=1.0, ablation=variant)
        env.reset(seed=0)
        total = 0.0
        for action in actionsFrom(schedule):
            _, _, _, _, info = env.step(action)
            total += info["profit"]
        assert total == pytest.approx(schedule.profit, abs=1e-9)
        assert info["soc"] == pytest.approx(schedule.soc[-1], abs=1e-9)

    def test_free_wear_is_never_worse_than_priced_wear(self, paper):
        free = dp.solve(MIXED_PRICES, paper, initialSoc=0.0, calendar=False,
                        costing=("none", 0.0))
        priced = dp.solve(MIXED_PRICES, paper, initialSoc=0.0, calendar=False)
        assert free.degradationCost == 0.0
        assert free.profit >= priced.profit

    def test_the_default_flat_rate_prices_a_full_discharge_like_schade(self, paper):
        _, rate = Ablation(degradation="flat").costing(paper, nem_interval_hours)
        qCd, _, _ = paper.step(1.0, 0.0, nem_interval_hours)
        assert rate * paper.MWh == pytest.approx(paper.cost(qCd))

    def test_the_shared_state_space_is_not_written(self, paper):
        space = dp.stateSpace(paper, 0.0, nem_interval_hours, len(MIXED_PRICES))
        before = space.cycle.copy()
        dp.solve(MIXED_PRICES, paper, initialSoc=0.0, calendar=False,
                 costing=("none", 0.0))
        assert (space.cycle == before).all()

    def test_the_baseline_has_no_costing(self, paper):
        assert Ablation().costing(paper, nem_interval_hours) is None
        assert Ablation(degradation="full").costing(paper, nem_interval_hours) is None


LONG_PRICES = [60 + 40 * ((i * 7) % 13) - 55 * (i % 5 == 0) + 400 * (i % 29 == 0)
               for i in range(60)]


class TestActionLevelsMatchTheOracle:
    """The same replay with a finer action set: the chosen action indices go straight
    into the environment, so the two must agree on what each index does."""

    @pytest.mark.parametrize("levels", [1, 2, 4])
    @pytest.mark.parametrize("endEmpty", [False, True])
    def test_replay(self, paper, levels, endEmpty):
        schedule = dp.solve(LONG_PRICES, paper, initialSoc=0.0, calendar=False,
                            endEmpty=endEmpty, levels=levels)
        env = BessEnv(priceSeries(LONG_PRICES), paper, horizon=len(LONG_PRICES),
                      calendar=False, rewardScale=1.0, endEmpty=endEmpty,
                      ablation=Ablation(actionLevels=levels))
        assert env.action_space.n == 2 * levels + 1
        env.reset(seed=0)
        total = 0.0
        for action in schedule.actions:
            assert env.action_masks()[action], "the oracle chose a masked action"
            _, _, _, _, info = env.step(action)
            assert not info["outOfBounds"]
            total += info["profit"]
        assert total == pytest.approx(schedule.profit, abs=1e-9)
        assert info["soc"] == pytest.approx(schedule.soc[-1], abs=1e-9)
        if endEmpty:
            assert info["soc"] == pytest.approx(0.0, abs=1e-9)

    def test_a_cycle_budget_binds_the_same_way(self, paper):
        schedule = dp.solve(LONG_PRICES, paper, initialSoc=0.0, calendar=False,
                            cycles=0.5, levels=2)
        env = BessEnv(priceSeries(LONG_PRICES), paper, horizon=len(LONG_PRICES),
                      calendar=False, rewardScale=1.0, cycleBudget=0.5,
                      ablation=Ablation(actionLevels=2))
        env.reset(seed=0)
        for action in schedule.actions:
            _, _, _, _, info = env.step(action)
        assert info["cycles"] == pytest.approx(schedule.cyclesUsed, abs=1e-9)
        assert info["cycles"] <= 0.5 + 1e-9

    def test_a_finer_action_set_can_only_help_the_oracle(self, paper):
        profits = [dp.solve(LONG_PRICES, paper, initialSoc=0.0, calendar=False,
                            levels=levels).profit for levels in (1, 2, 4)]
        assert profits[0] <= profits[1] + 1e-9 <= profits[2] + 2e-9

    def test_attempted_action_stays_a_direction(self, paper):
        env = BessEnv(priceSeries(LONG_PRICES), paper, horizon=len(LONG_PRICES),
                      calendar=False, ablation=Ablation(actionLevels=4))
        env.reset(seed=0)
        seen = {env.step(action)[4]["attemptedAction"] for action in (0, 2, 4, 6, 8)}
        assert seen == {0, 1, 2}

    def test_the_cache_key_separates_the_action_sets(self, paper):
        start = datetime(2024, 1, 10, 4, 5)
        keys = {baseline.cacheKey("SA1", start, 288, False, True, paper, 0.0, 0.02,
                                  levels=levels) for levels in (1, 2, 4, 8)}
        assert len(keys) == 4
        assert (baseline.cacheKey("SA1", start, 288, False, True, paper, 0.0, 0.02)
                == baseline.cacheKey("SA1", start, 288, False, True, paper, 0.0, 0.02,
                                     levels=1))


class TestCacheKey:
    def test_each_cost_has_its_own_record(self, paper):
        start = datetime(2024, 1, 10, 4, 5)
        keys = {baseline.cacheKey("SA1", start, 288, False, True, paper, 0.0, 0.02,
                                  costing=cost)
                for cost in (None, ("none", 0.0), ("flat", 5.0), ("flat", 6.0))}
        assert len(keys) == 4

    def test_the_baseline_key_is_unchanged(self, paper):
        start = datetime(2024, 1, 10, 4, 5)
        assert (baseline.cacheKey("SA1", start, 288, False, True, paper, 0.0, 0.02)
                == baseline.cacheKey("SA1", start, 288, False, True, paper, 0.0, 0.02,
                                     costing=None))

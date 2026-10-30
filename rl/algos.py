"""The --algo registry.

MaskablePPO is the only SB3-family algorithm that reads an action mask. DQN and A2C
rely on the environment's clipping instead.

Every algorithm runs undiscounted: an episode is one week of profit, and every interval
of it counts the same.

The rollout buffer and the replay warm-up are pinned to one trading day, not to the
episode. An episode became a week when the cycle budget arrived, and scaling these with
it would have changed every hyperparameter at the same time as the problem. They hold the
values every published agent trained under.
"""
from stable_baselines3 import A2C, DQN
from stable_baselines3.common.utils import ConstantSchedule
from sb3_contrib import MaskablePPO

import config

maskingAlgorithms = frozenset({"maskableppo"})


def build(name, env, seed=0, **overrides):
    name = name.lower()
    if name not in builders:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(builders)}")
    return builders[name](env, seed, overrides)


def supportsMasking(name):
    return name.lower() in maskingAlgorithms


def _maskablePpo(env, seed, overrides):
    settings = dict(policy="MlpPolicy", env=env, seed=seed, gamma=config.gamma,
                    n_steps=config.observation_intervals * 2, verbose=0)
    settings.update(overrides)
    return MaskablePPO(**settings)


def _dqn(env, seed, overrides):
    settings = dict(policy="MlpPolicy", env=env, seed=seed, gamma=config.gamma,
                    learning_starts=config.observation_intervals, verbose=0)
    settings.update(overrides)
    return DQN(**settings)


def _a2c(env, seed, overrides):
    settings = dict(policy="MlpPolicy", env=env, seed=seed, gamma=config.gamma,
                    verbose=0)
    settings.update(overrides)
    return A2C(**settings)


builders = {
    "maskableppo": _maskablePpo,
    "dqn": _dqn,
    "a2c": _a2c,
}

assert set(builders) == set(config.algorithms), (sorted(builders), sorted(config.algorithms))


def holdExploration(agent):
    """Keep a warm-started chain at the epsilon it reached.

    SB3 recomputes exploration_rate from _current_progress_remaining, which every
    learn() call resets, so an off-policy agent would restart at exploration_initial_eps
    (1.0) on each fold and re-randomise itself. On-policy algorithms have no schedule.
    """
    # ConstantSchedule, not a closure over `agent`: the schedule is pickled into the
    # checkpoint, and a closure would drag the whole agent and its open monitor file in.
    if hasattr(agent, "exploration_schedule"):
        agent.exploration_schedule = ConstantSchedule(agent.exploration_final_eps)


def load(name, path, env=None):
    classes = {"maskableppo": MaskablePPO, "dqn": DQN, "a2c": A2C}
    return classes[name.lower()].load(path, env=env)

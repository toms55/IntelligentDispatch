"""The --algo registry.

MaskablePPO is the only SB3-family algorithm that reads an action mask. DQN and A2C
rely on the environment's clipping instead.

Every algorithm runs undiscounted: the reward subtracts a per-interval LP baseline,
and that subtraction is only policy-invariant at gamma = 1.
"""
from stable_baselines3 import A2C, DQN
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
                    n_steps=config.episode_intervals * 2, verbose=0)
    settings.update(overrides)
    return MaskablePPO(**settings)


def _dqn(env, seed, overrides):
    settings = dict(policy="MlpPolicy", env=env, seed=seed, gamma=config.gamma,
                    learning_starts=config.episode_intervals, verbose=0)
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


def load(name, path, env=None):
    classes = {"maskableppo": MaskablePPO, "dqn": DQN, "a2c": A2C}
    return classes[name.lower()].load(path, env=env)

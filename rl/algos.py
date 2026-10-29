"""The --algo registry.

MaskablePPO is the only SB3-family algorithm that reads an action mask. DQN and A2C
rely on the environment's clipping instead.

Every algorithm runs undiscounted: an episode is one trading day of profit, and every
interval of it counts the same.
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
                    n_steps=config.episode_intervals * 2, verbose=0)
    settings.update(overrides)
    return MaskablePPO(**settings)


def _dqn(env, seed, overrides):
    """DQN runs on the SB3 defaults, with two exceptions, and the thesis quotes them
    from here.

    `buffer_size` is set because the buffer travels between folds on disk; see
    config.replay_buffer_size. `learning_starts` is one episode, so a fold fills a day
    of transitions before it updates.

    Everything else is the library default, stated here because the comparison against
    MaskablePPO has to name it: learning_rate 1e-4 (PPO uses 3e-4), batch_size 32,
    train_freq every 4 steps, gradient_steps 1, target_update_interval 10000 steps,
    tau 1.0, exploration 1.0 down to 0.05 over the first 10% of each learn() call, and
    net_arch [64, 64], which is the same network MaskablePPO builds.

    The budget is equal in environment steps, not in updates. Over a 30000-step warm
    fold DQN takes 30000/4 = 7500 gradient steps, while MaskablePPO takes
    (30000/576 rollouts) x 10 epochs x (576/64 minibatches) = 4680. Neither algorithm is
    tuned, on any window.
    """
    settings = dict(policy="MlpPolicy", env=env, seed=seed, gamma=config.gamma,
                    learning_starts=config.episode_intervals,
                    buffer_size=config.replay_buffer_size, verbose=0)
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


def isOffPolicy(agent):
    """Whether this agent learns from a replay buffer, and so carries one between folds.

    `save()` excludes the buffer, so rl/train.py writes and reads it separately. Asking
    here keeps the SB3 attribute test in one place, as holdExploration does.
    """
    return hasattr(agent, "replay_buffer")


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

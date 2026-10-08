"""Train one agent on one seed. The seed loop lives in the caller."""
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

import config
from rl import algos
from rl.env import BessEnv


def trainOne(algoName, prices, model, baselines=None, seed=0,
             timesteps=100_000, calendar=True, horizon=config.episode_intervals,
             rewardScale=config.reward_scale, outDir="runs", **overrides):
    """Train `algoName` on `prices` and write a checkpoint. Returns (agent, path)."""
    directory = Path(outDir) / f"{algoName}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        BessEnv(prices, model, baseline=baselines, calendar=calendar, horizon=horizon,
                rewardScale=rewardScale, shuffle=True),
        filename=str(directory / "monitor.csv"),
    )
    agent = algos.build(algoName, env, seed=seed, **overrides)
    agent.learn(total_timesteps=timesteps, progress_bar=False)

    path = directory / "agent"
    agent.save(str(path))
    return agent, path


def trainSeeds(algoName, prices, model, baselines=None, seeds=config.seeds, **kwargs):
    """Every seed in turn. Returns the checkpoint path for each."""
    return {
        seed: trainOne(algoName, prices, model, baselines=baselines, seed=seed, **kwargs)[1]
        for seed in seeds
    }

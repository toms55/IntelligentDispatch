"""Train one agent on one seed. The seed loop lives in the caller."""
import json
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

import config
from rl import algos
from rl.env import BessEnv
from rl.features import PriceScaler, scalerFor

scalerFile = "scaler.json"


def saveScaler(directory, scaler):
    """Write the price transform beside the checkpoint."""
    (Path(directory) / scalerFile).write_text(json.dumps(scaler.asDict()))


def loadScaler(directory):
    """Read back the transform a checkpoint was trained under.

    Raise rather than fall back. Rebuilding the scaler from the evaluation prices would
    pin a different market price cap, and the agent would then read an observation that
    means something other than what it learned.
    """
    path = Path(directory) / scalerFile
    if not path.exists():
        raise FileNotFoundError(
            f"no {scalerFile} at {path}: the checkpoint predates the dated price cap, "
            f"so the transform it trained under is unknown. Retrain it."
        )
    return PriceScaler(**json.loads(path.read_text()))


def trainOne(algoName, prices, model, baselines=None, seed=0,
             timesteps=100000, calendar=True, horizon=config.episode_intervals,
             rewardScale=config.reward_scale, outDir="runs", scaler=None, **overrides):
    """Train `algoName` on `prices` and write a checkpoint. Returns (agent, path)."""
    directory = Path(outDir) / f"{algoName}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    scaler = scaler if scaler is not None else scalerFor(prices.index)
    saveScaler(directory, scaler)

    env = Monitor(
        BessEnv(prices, model, baseline=baselines, calendar=calendar, horizon=horizon,
                rewardScale=rewardScale, shuffle=True, scaler=scaler),
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

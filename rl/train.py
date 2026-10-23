"""Train one agent on one seed. The seed loop lives in the caller."""
import json
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

import config
from rl import algos
from rl.env import BessEnv
from rl.features import PriceScaler, scalerFor
from rl.policies import scalerFile


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


def resumeFrom(algoName, warmFrom, env, scaler, seed):
    """Pick up the previous fold's checkpoint instead of starting from fresh weights.

    The scaler check is the one that matters. The transform is pinned to a market price
    cap, so a parent trained under a different cap means every observation value denotes
    a different price and the chain is meaningless.
    """
    warmFrom = Path(warmFrom)
    previous = loadScaler(warmFrom)
    if previous != scaler:
        raise ValueError(
            f"cannot warm start from {warmFrom}: it trained under a price cap of "
            f"{previous.cap:.0f}, this run uses {scaler.cap:.0f}"
        )
    # load() restores the optimiser state as well as the weights, and already raises
    # naming both spaces if the observation layout moved.
    agent = algos.load(algoName, str(warmFrom / "agent"), env=env)
    agent.set_random_seed(seed)
    algos.holdExploration(agent)
    return agent


def trainOne(algoName, prices, model, seed=0,
             timesteps=100000, calendar=True, horizon=config.episode_intervals,
             rewardScale=config.reward_scale, outDir="runs", scaler=None,
             warmFrom=None, forecasts=None, **overrides):
    """Train `algoName` on `prices` and write a checkpoint. Returns (agent, path).

    `warmFrom` is a previous checkpoint directory to continue from, which is how a
    rolling backtest chains one fold to the next. `forecasts` is one entry per window of
    `prices`, as BessEnv takes it: the agent then trains on what AEMO published, and is
    still paid the realised price.
    """
    if warmFrom is not None and overrides:
        raise ValueError(
            f"algos.load takes no hyperparameter channel, so a warm start would drop "
            f"these overrides silently: {sorted(overrides)}"
        )
    directory = Path(outDir) / f"{algoName}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    scaler = scaler if scaler is not None else scalerFor(prices.index)
    saveScaler(directory, scaler)

    env = Monitor(
        BessEnv(prices, model, calendar=calendar, horizon=horizon,
                rewardScale=rewardScale, shuffle=True, scaler=scaler,
                forecasts=forecasts),
        filename=str(directory / "monitor.csv"),
    )
    agent = (algos.build(algoName, env, seed=seed, **overrides) if warmFrom is None
             else resumeFrom(algoName, warmFrom, env, scaler, seed))
    agent.learn(total_timesteps=timesteps, progress_bar=False)

    path = directory / "agent"
    agent.save(str(path))
    return agent, path


def trainSeeds(algoName, prices, model, seeds=config.seeds, **kwargs):
    """Every seed in turn. Returns the checkpoint path for each."""
    return {
        seed: trainOne(algoName, prices, model, seed=seed, **kwargs)[1]
        for seed in seeds
    }

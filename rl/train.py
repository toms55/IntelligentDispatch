"""Train one agent on one seed. The seed loop lives in the caller."""
import json
from pathlib import Path

from stable_baselines3.common.monitor import Monitor

import config
from rl import algos
from rl.env import BessEnv
from rl.features import PriceScaler, scalerFor
from rl.policies import scalerFile

# The replay buffer, beside the weights. Working state: it never reaches
# trainedPolicies/, and backtest/runner.py keeps only the live end of each chain.
replayFile = "replay.pkl"


def saveScaler(directory, scaler):
    """Write the price transform beside the checkpoint."""
    (Path(directory) / scalerFile).write_text(json.dumps(scaler.asDict()))


def saveReplay(directory, agent):
    """Write an off-policy agent's replay buffer beside its weights.

    `agent.save()` excludes the buffer (SB3 lists it in `_excluded_save_params`), so
    without this a warm-started DQN keeps its weights and its optimiser but relearns
    from an empty buffer on every fold. MaskablePPO is on-policy and loses nothing, so
    the two algorithms would not mean the same thing by "warm".

    Working state, not part of a published agent: `policies.publish` copies an allowlist
    of the weights and the scaler, so this never travels.
    """
    if algos.isOffPolicy(agent):
        agent.save_replay_buffer(str(Path(directory) / replayFile))


def loadReplay(directory, agent):
    """Read the parent fold's replay buffer back into `agent`.

    Raise rather than continue, as loadScaler does. An empty buffer is exactly the
    failure this exists to remove, and it is invisible in the numbers: the fold still
    trains, it just throws away everything the chain learned before it.

    The size is checked against `config.replay_buffer_size`, which is the only place
    that states it. Neither of the two values to hand is the configured one: the buffer
    pickles its own size, and `algos.load` restores `buffer_size` onto the agent from
    the checkpoint, so both come from the chain's cold fold. Comparing them to each
    other would compare a number with itself and check nothing.
    """
    if not algos.isOffPolicy(agent):
        return
    path = Path(directory) / replayFile
    if not path.exists():
        raise FileNotFoundError(
            f"no {replayFile} at {path}: the checkpoint predates the travelling replay "
            f"buffer, so the experience the chain reached is gone. Start a fresh run."
        )
    agent.load_replay_buffer(str(path))
    found = agent.replay_buffer.buffer_size
    if found != config.replay_buffer_size:
        raise ValueError(
            f"{path} holds a buffer of {found} transitions, and this run is configured "
            f"for {config.replay_buffer_size}. The chain was started under the other "
            f"size and cannot adopt this one, because the size is fixed when the buffer "
            f"is built. Start a fresh run, or put the old size back in config.py."
        )


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
    loadReplay(warmFrom, agent)
    return agent


def trainOne(algoName, prices, model, seed=0,
             timesteps=100000, calendar=True, horizon=config.episode_intervals,
             rewardScale=config.reward_scale, outDir="runs", scaler=None,
             warmFrom=None, forecasts=None,
             invalidPenalty=config.invalid_action_penalty, **overrides):
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
                forecasts=forecasts, invalidPenalty=invalidPenalty),
        filename=str(directory / "monitor.csv"),
        # One column per episode saying how often the agent chose to charge a full pack
        # or discharge an empty one, so a run shows whether the penalty bites while it
        # trains. Monitor copies this key out of the final step's info, so BessEnv sets
        # it flat on every truncation.
        info_keywords=("outOfBoundsTotal",),
    )
    agent = (algos.build(algoName, env, seed=seed, **overrides) if warmFrom is None
             else resumeFrom(algoName, warmFrom, env, scaler, seed))
    agent.learn(total_timesteps=timesteps, progress_bar=False)

    path = directory / "agent"
    agent.save(str(path))
    saveReplay(directory, agent)
    return agent, path


def trainSeeds(algoName, prices, model, seeds=config.seeds, **kwargs):
    """Every seed in turn. Returns the checkpoint path for each."""
    return {
        seed: trainOne(algoName, prices, model, seed=seed, **kwargs)[1]
        for seed in seeds
    }

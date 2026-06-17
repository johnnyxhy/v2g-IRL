"""
Trajectory runner for the V2G-IRL demo.

Given a loaded model + env factory, runs one deterministic rollout
starting from the supplied initial_values and returns the SoC history
and journey timing metadata.
"""

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv


def run_rl_trajectory(
    model,
    env_factory,
    initial_values: dict,
    reward_net=None,
    shaping_net=None,
    deterministic: bool = False,
) -> dict:
    """Run a single episode and return trajectory data.

    Parameters
    ----------
    model          : loaded SBX PPO / SAC policy
    env_factory    : callable() → wrapped gym.Env
    initial_values : dict matching the environment's set_initial_states() signature
    reward_net     : optional RewardNet (attached to env if env supports it)
    shaping_net    : optional ShapingNet (attached to env if env supports it)
    deterministic  : use deterministic policy actions

    Returns
    -------
    dict with keys:
        soc_history           : list[float]  (0–1 range, 96 timesteps)
        out_start_timestep    : int
        return_start_timestep : int
        out_duration          : int
        return_duration       : int
        feature_expectation   : list[float]
    """
    vec_env = DummyVecEnv([env_factory])
    raw_env = vec_env.envs[0].unwrapped

    # Set initial conditions
    raw_env.set_initial_states(initial_values)

    # Attach optional reward / shaping nets (Deep MaxEnt / AIRL envs)
    if reward_net is not None and hasattr(raw_env, "set_reward_net"):
        raw_env.set_reward_net(reward_net)
    if shaping_net is not None and hasattr(raw_env, "set_shaping_net"):
        raw_env.set_shaping_net(shaping_net)

    obs = vec_env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _rewards, done, info = vec_env.step(action)
        if done[0]:
            result_info = info[0]
            break

    vec_env.close()

    return {
        "soc_history":           result_info.get("soc_history", []),
        "out_start_timestep":    result_info.get("out_start_timestep", 0),
        "return_start_timestep": result_info.get("return_start_timestep", 0),
        "out_duration":          result_info.get("out_duration", 0),
        "return_duration":       result_info.get("return_duration", 0),
        "feature_expectation":   result_info.get("feature_expectation", []),
    }


def compute_metrics(soc_rl: list, soc_expert: list) -> dict:
    """Compute DTW and MAE between two SoC trajectories.

    Both inputs are expected in the same range (0–1).
    """
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    from irl.utils.tools import compute_dtw

    a = np.asarray(soc_rl, dtype=np.float32)
    b = np.asarray(soc_expert, dtype=np.float32)

    # Resample to same length for MAE
    target = max(len(a), len(b))

    def _resample(v, n):
        if len(v) == n:
            return v
        src = np.linspace(0, 1, len(v))
        dst = np.linspace(0, 1, n)
        return np.interp(dst, src, v).astype(np.float32)

    a_r = _resample(a, target)
    b_r = _resample(b, target)

    mae = float(np.mean(np.abs(a_r - b_r)))
    dtw = float(compute_dtw(b, a))

    return {"mae": mae, "dtw": dtw}

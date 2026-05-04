"""
Comparison evaluation: Linear MaxEnt (V2GEnv-discrete + PPO) vs
Deep MaxEnt (V2GDeepEnv-discrete + PPO).

For each expert trajectory in the segment, the script runs both models,
selects the best-DTW rollout from each, and overlays:
  - Expert SoC  (reference, from the linear MaxEnt dataset)
  - Linear MaxEnt agent SoC
  - Deep MaxEnt agent SoC
on a single figure.
"""

import numpy as np
import gymnasium as gym
from sbx import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.MaxEnt.MaxEnt_discrete import FlattenNormalizeObsWrapper as LinearFlattenWrapper
from irl.DeepMaxEnt.DeepMaxEnt_discrete import FlattenNormalizeObsWrapper as DeepFlattenWrapper
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet
from irl.Adversarial.Adversarial_discrete import (
    ShapingNet as AIRLShapingNet,
    FlattenNormalizeObsWrapper as AIRLFlattenWrapper,
)
from irl.utils.tools import compute_dtw
import torch
import json

# --- Register environments ---
gym.register(
    id='V2GEnv-discrete',
    entry_point="irl.envs.V2GEnv_discrete:V2GEnv",
    max_episode_steps=96,
)
gym.register(
    id='V2GDeepEnv-discrete',
    entry_point="irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
    max_episode_steps=96,
)

energy_price_profile = np.array([
    0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24, 0.26,
    0.27, 0.28, 0.30, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
    0.41, 0.42, 0.43, 0.44, 0.44, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.47,
    0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.42,
    0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32, 0.31, 0.30, 0.28,
    0.27, 0.26, 0.24, 0.23, 0.22, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
    0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07
])


def find_expert_indexes(expert_data, segment):
    return [i for i, traj in enumerate(expert_data) if traj['segment'] == segment]


def _resample_trajectory(values, target_len):
    values = np.asarray(values, dtype=np.float32)
    if len(values) == target_len:
        return values
    if len(values) <= 1:
        return np.full(target_len, values[0] if len(values) == 1 else 0.0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, values).astype(np.float32)


def _compute_is(all_soc_histories, expert_soc, target_len, alpha=0.2):
    soc_stack = np.stack(all_soc_histories, axis=0)
    expert_resampled = _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
    lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
    upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
    width = upper_soc - lower_soc
    undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
    overshoot = np.maximum(expert_resampled - upper_soc, 0.0)
    return float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))


def load_reward_net(path, obs_dim=7, action_dim=1, hidden_dim=64):
    reward_net = RewardNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    reward_net.load_state_dict(torch.load(path, weights_only=True))
    reward_net.eval()
    return reward_net


def load_shaping_net(path, obs_dim=7, hidden_dim=32):
    shaping_net = AIRLShapingNet(obs_dim=obs_dim, hidden_dim=hidden_dim)
    shaping_net.load_state_dict(torch.load(path, weights_only=True))
    shaping_net.eval()
    return shaping_net


def _best_mae_rollout(vec_env, model, expert_soc, initial_states, n_rollouts, deterministic):
    """Run n_rollouts, return the soc_history + journey info from the best-MAE rollout."""
    vec_env.envs[0].unwrapped.set_initial_states(initial_states)
    target_len = len(expert_soc)

    best_mae = float('inf')
    best_dtw = float('inf')
    best_soc = None
    best_info = None
    all_soc_histories = []

    for _ in range(n_rollouts):
        obs = vec_env.reset()

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, info = vec_env.step(action)
            if done[0]:
                break

        soc_history = info[0]["soc_history"]
        all_soc_histories.append(_resample_trajectory(soc_history, target_len))
        mae = float(np.mean(np.abs(
            _resample_trajectory(soc_history, target_len)
            - _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
        )))
        dtw = compute_dtw(
            np.array(expert_soc, dtype=np.float32),
            np.array(soc_history, dtype=np.float32),
        )
        if mae < best_mae:
            best_mae = mae
            best_soc = soc_history
            best_info = info[0]
        if dtw < best_dtw:
            best_dtw = dtw

    return best_soc, best_mae, best_dtw, best_info, all_soc_histories


def plot_comparison(
    expert_soc,
    maxent_soc,
    deep_soc,
    airl_soc,
    expert_index,
    out_start,
    out_dur,
    return_start,
    return_dur,
    maxent_mae,
    deep_mae,
    airl_mae,
):
    plt.figure(figsize=(12, 5))

    # Journey shading
    plt.axvspan(out_start, out_start + out_dur, color='red', alpha=0.15, label='Out Journey')
    plt.axvspan(return_start, return_start + return_dur, color='blue', alpha=0.15, label='Return Journey')

    # SoC curves
    plt.plot(range(len(expert_soc)), expert_soc,
             label='Expert SoC', color='black', linestyle='--', linewidth=1.5)
    plt.plot(range(len(maxent_soc)), maxent_soc,
             label=f'Linear MaxEnt (MAE={maxent_mae:.4f})', color='tab:blue', linewidth=1.5)
    plt.plot(range(len(deep_soc)), deep_soc,
             label=f'Deep MaxEnt (MAE={deep_mae:.4f})', color='tab:orange', linewidth=1.5)
    plt.plot(range(len(airl_soc)), airl_soc,
             label=f'AIRL (MAE={airl_mae:.4f})', color='tab:red', linewidth=1.5)

    # Energy price (secondary reference)
    n = max(len(maxent_soc), len(deep_soc), len(airl_soc), len(expert_soc))
    price = energy_price_profile[:n]
    plt.plot(range(len(price)), price, color='green', alpha=0.5, linewidth=1.0, label='Energy Price')

    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.title(
        f'MaxEnt vs Deep MaxEnt vs AIRL (Discrete) \u2014 Trajectory {expert_index} '
        f'(Linear MAE={maxent_mae:.4f}, Deep MAE={deep_mae:.4f}, AIRL MAE={airl_mae:.4f})'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Configuration                                                       #
    # ------------------------------------------------------------------ #
    # Linear MaxEnt model
    maxent_experiment    = "MaxEnt/discrete/MaxEntIRL_discrete_profit_exp1"
    maxent_epoch         = 20
    maxent_data_path     = "data/processed_trajectories_discrete_profit.json"

    # Deep MaxEnt model
    deep_experiment      = "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_profit_exp1"
    deep_epoch           = 20
    deep_hidden_dim      = 32
    deep_data_path       = "data/processed_trajectories_deep_discrete_profit.json"

    # AIRL model
    airl_experiment      = "Adversarial/discrete/Adversarial_discrete_exp5"
    airl_epoch           = 20
    airl_reward_hidden   = 32
    airl_shaping_hidden  = 32
    airl_data_path       = "data/processed_trajectories_airl_discrete.json"

    segment              = "Male 50-59"
    n_rollouts           = 20
    n_figures            = 5    # how many comparison plots to show
    eval_ratio           = 1.0  # fraction of segment trajectories to evaluate
    deterministic_policy = False
    # ------------------------------------------------------------------ #

    # Load expert data
    with open(maxent_data_path, "r") as f:
        maxent_expert_data = json.load(f)
    with open(deep_data_path, "r") as f:
        deep_expert_data = json.load(f)
    with open(airl_data_path, "r") as f:
        airl_expert_data = json.load(f)

    # Load models
    maxent_model  = PPO.load(f"./models/{maxent_experiment}/ppo_epoch{maxent_epoch}")
    deep_model    = PPO.load(f"./models/{deep_experiment}/ppo_epoch{deep_epoch}")
    airl_model    = PPO.load(f"./models/{airl_experiment}/ppo_epoch{airl_epoch}")
    deep_reward_net = load_reward_net(
        f"./models/{deep_experiment}/reward_net_epoch{deep_epoch}.pt",
        hidden_dim=deep_hidden_dim,
    )
    airl_reward_net = load_reward_net(
        f"./models/{airl_experiment}/reward_net_epoch{airl_epoch}.pt",
        hidden_dim=airl_reward_hidden,
    )
    airl_shaping_net = load_shaping_net(
        f"./models/{airl_experiment}/shaping_net_epoch{airl_epoch}.pt",
        hidden_dim=airl_shaping_hidden,
    )

    # Build envs (each with its own wrapper)
    maxent_vec_env = DummyVecEnv([lambda: LinearFlattenWrapper(gym.make('V2GEnv-discrete'))])
    deep_vec_env   = DummyVecEnv([lambda: DeepFlattenWrapper(gym.make('V2GDeepEnv-discrete'))])
    airl_vec_env   = DummyVecEnv([lambda: AIRLFlattenWrapper(gym.make('V2GDeepEnv-discrete'))])

    # Set reward/shaping nets once (persist across resets)
    deep_vec_env.envs[0].unwrapped.set_reward_net(deep_reward_net)
    airl_vec_env.envs[0].unwrapped.set_reward_net(airl_reward_net)
    airl_vec_env.envs[0].unwrapped.set_shaping_net(airl_shaping_net)

    # Find matching trajectory indexes within the segment
    maxent_indexes = find_expert_indexes(maxent_expert_data, segment)
    deep_indexes   = find_expert_indexes(deep_expert_data,   segment)
    airl_indexes   = find_expert_indexes(airl_expert_data,   segment)

    n_common = min(len(maxent_indexes), len(deep_indexes), len(airl_indexes))
    n_eval   = max(1, int(n_common * eval_ratio))
    maxent_indexes = maxent_indexes[:n_eval]
    deep_indexes   = deep_indexes[:n_eval]
    airl_indexes   = airl_indexes[:n_eval]

    print(f"Segment '{segment}': {n_common} trajectories available, evaluating {n_eval}")

    all_maxent_mae = []
    all_deep_mae   = []
    all_airl_mae   = []
    all_maxent_dtw = []
    all_deep_dtw   = []
    all_airl_dtw   = []
    all_maxent_is  = []
    all_deep_is    = []
    all_airl_is    = []

    for i, (m_idx, d_idx, a_idx) in enumerate(zip(maxent_indexes, deep_indexes, airl_indexes)):
        print(f"\nTrajectory {i+1}/{n_eval}  (maxent_idx={m_idx}, deep_idx={d_idx}, airl_idx={a_idx})")

        expert_soc = maxent_expert_data[m_idx]['soc_history']
        target_len = len(expert_soc)

        # --- Linear MaxEnt rollout ---
        maxent_soc, maxent_mae, maxent_dtw, maxent_info, maxent_socs = _best_mae_rollout(
            maxent_vec_env,
            maxent_model,
            expert_soc,
            initial_states=maxent_expert_data[m_idx]['initial_values'],
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
        )

        # --- Deep MaxEnt rollout ---
        deep_soc, deep_mae, deep_dtw, _, deep_socs = _best_mae_rollout(
            deep_vec_env,
            deep_model,
            expert_soc,
            initial_states=deep_expert_data[d_idx]['initial_values'],
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
        )

        # --- AIRL rollout ---
        airl_soc, airl_mae, airl_dtw, _, airl_socs = _best_mae_rollout(
            airl_vec_env,
            airl_model,
            expert_soc,
            initial_states=airl_expert_data[a_idx]['initial_values'],
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
        )

        maxent_is = _compute_is(maxent_socs, expert_soc, target_len)
        deep_is   = _compute_is(deep_socs,   expert_soc, target_len)
        airl_is   = _compute_is(airl_socs,   expert_soc, target_len)

        all_maxent_mae.append(maxent_mae)
        all_deep_mae.append(deep_mae)
        all_airl_mae.append(airl_mae)
        all_maxent_dtw.append(maxent_dtw)
        all_deep_dtw.append(deep_dtw)
        all_airl_dtw.append(airl_dtw)
        all_maxent_is.append(maxent_is)
        all_deep_is.append(deep_is)
        all_airl_is.append(airl_is)

        print(
            f"  Linear MaxEnt best MAE: {maxent_mae:.4f}, best DTW: {maxent_dtw:.3f}, IS: {maxent_is:.3f} | "
            f"Deep MaxEnt best MAE: {deep_mae:.4f}, best DTW: {deep_dtw:.3f}, IS: {deep_is:.3f} | "
            f"AIRL best MAE: {airl_mae:.4f}, best DTW: {airl_dtw:.3f}, IS: {airl_is:.3f}"
        )

        if i < n_figures:
            plot_comparison(
                expert_soc=expert_soc,
                maxent_soc=maxent_soc,
                deep_soc=deep_soc,
                airl_soc=airl_soc,
                expert_index=m_idx,
                out_start=maxent_info["out_start_timestep"],
                out_dur=maxent_info["out_duration"],
                return_start=maxent_info["return_start_timestep"],
                return_dur=maxent_info["return_duration"],
                maxent_mae=maxent_mae,
                deep_mae=deep_mae,
                airl_mae=airl_mae,
            )

    print(f"\n{'='*60}")
    print(f"Summary over {n_eval} trajectories (segment: {segment}):")
    print(f"  Linear MaxEnt  Avg Best MAE: {np.mean(all_maxent_mae):.4f} \u00b1 {np.std(all_maxent_mae):.4f}  |  Avg Best DTW: {np.mean(all_maxent_dtw):.3f} \u00b1 {np.std(all_maxent_dtw):.3f}  |  Avg IS: {np.mean(all_maxent_is):.3f} \u00b1 {np.std(all_maxent_is):.3f}")
    print(f"  Deep MaxEnt    Avg Best MAE: {np.mean(all_deep_mae):.4f} \u00b1 {np.std(all_deep_mae):.4f}  |  Avg Best DTW: {np.mean(all_deep_dtw):.3f} \u00b1 {np.std(all_deep_dtw):.3f}  |  Avg IS: {np.mean(all_deep_is):.3f} \u00b1 {np.std(all_deep_is):.3f}")
    print(f"  AIRL           Avg Best MAE: {np.mean(all_airl_mae):.4f} \u00b1 {np.std(all_airl_mae):.4f}  |  Avg Best DTW: {np.mean(all_airl_dtw):.3f} \u00b1 {np.std(all_airl_dtw):.3f}  |  Avg IS: {np.mean(all_airl_is):.3f} \u00b1 {np.std(all_airl_is):.3f}")
    print(f"{'='*60}")

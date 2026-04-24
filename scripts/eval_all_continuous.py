"""
Comparison evaluation: Linear MaxEnt (V2GEnv-profit + SAC) vs
Deep MaxEnt (V2GDeepEnv-profit + SAC).

For each expert trajectory in the segment, the script runs both models,
selects the best-DTW rollout from each, and overlays:
  - Expert SoC  (reference, from the linear MaxEnt dataset)
  - Linear MaxEnt agent SoC
  - Deep MaxEnt agent SoC
on a single figure.
"""

import numpy as np
import gymnasium as gym
from sbx import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.utils.tools import compute_dtw
import torch
import json

# --- Register environments ---
gym.register(
    id='V2GEnv-profit',
    entry_point="irl.envs.V2GEnv_profit:V2GEnv",
    max_episode_steps=96,
)
gym.register(
    id='V2GDeepEnv-profit',
    entry_point="irl.envs.V2GDeepEnv_profit:V2GDeepEnv",
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


def load_reward_net(path, obs_dim=7, action_dim=1, hidden_dim=64):
    reward_net = RewardNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    reward_net.load_state_dict(torch.load(path, weights_only=True))
    reward_net.eval()
    return reward_net


def _best_dtw_rollout(vec_env, model, expert_soc, initial_states,
                      n_rollouts, deterministic, noise_scale,
                      set_reward_fn=None):
    """Run n_rollouts, return the soc_history + journey info from the best-DTW rollout."""
    if set_reward_fn is not None:
        set_reward_fn(vec_env)

    vec_env.envs[0].unwrapped.set_initial_states(initial_states)

    best_dtw = float('inf')
    best_soc = None
    best_info = None

    for _ in range(n_rollouts):
        obs = vec_env.reset()

        while True:
            if deterministic or noise_scale >= 1.0:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                det_action, _ = model.predict(obs, deterministic=True)
                stoch_action, _ = model.predict(obs, deterministic=False)
                action = det_action + noise_scale * (stoch_action - det_action)
            obs, _, done, info = vec_env.step(action)
            if done[0]:
                break

        soc_history = info[0]["soc_history"]
        dtw = compute_dtw(
            np.array(expert_soc, dtype=np.float32),
            np.array(soc_history, dtype=np.float32),
        )
        if dtw < best_dtw:
            best_dtw = dtw
            best_soc = soc_history
            best_info = info[0]

    return best_soc, best_dtw, best_info


def plot_comparison(
    expert_soc,
    maxent_soc,
    deep_soc,
    expert_index,
    out_start,
    out_dur,
    return_start,
    return_dur,
    maxent_dtw,
    deep_dtw,
):
    plt.figure(figsize=(12, 5))

    # Journey shading
    plt.axvspan(out_start, out_start + out_dur, color='red', alpha=0.15, label='Out Journey')
    plt.axvspan(return_start, return_start + return_dur, color='blue', alpha=0.15, label='Return Journey')

    # SoC curves
    plt.plot(range(len(expert_soc)), expert_soc,
             label='Expert SoC', color='black', linestyle='--', linewidth=1.5)
    plt.plot(range(len(maxent_soc)), maxent_soc,
             label=f'Linear MaxEnt (DTW={maxent_dtw:.2f})', color='tab:blue', linewidth=1.5)
    plt.plot(range(len(deep_soc)), deep_soc,
             label=f'Deep MaxEnt (DTW={deep_dtw:.2f})', color='tab:orange', linewidth=1.5)

    # Energy price (secondary reference)
    n = max(len(maxent_soc), len(deep_soc), len(expert_soc))
    price = energy_price_profile[:n]
    plt.plot(range(len(price)), price, color='green', alpha=0.5, linewidth=1.0, label='Energy Price')

    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.title(
        f'MaxEnt vs Deep MaxEnt \u2014 Trajectory {expert_index} '
        f'(Linear DTW={maxent_dtw:.2f}, Deep DTW={deep_dtw:.2f})'
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ------------------------------------------------------------------ #
    #  Configuration                                                       #
    # ------------------------------------------------------------------ #
    # Linear MaxEnt model
    maxent_model_path    = "./models/MaxEntIRL_profit_v7_exp2/maxent_irl_epoch50"
    maxent_data_path     = "data/processed_trajectories_profit.json"

    # Deep MaxEnt model
    deep_experiment      = "Deep_profit_sum_0.01reg_50charge_5.0gradclip_continued"
    deep_epoch           = 20
    deep_hidden_dim      = 32
    deep_data_path       = "data/processed_trajectories_deep_profit.json"

    segment              = "Male 50-59"
    n_rollouts           = 20
    n_figures            = 5    # how many comparison plots to show
    eval_ratio           = 0.2  # fraction of segment trajectories to evaluate
    deterministic_policy = False
    noise_scale          = 0.5  # 0.0 = deterministic, 1.0 = full stochastic
    # ------------------------------------------------------------------ #

    # Load expert data
    with open(maxent_data_path, "r") as f:
        maxent_expert_data = json.load(f)
    with open(deep_data_path, "r") as f:
        deep_expert_data = json.load(f)

    # Load models
    maxent_model = SAC.load(maxent_model_path)
    deep_model   = SAC.load(f"./models/{deep_experiment}/sac_epoch{deep_epoch}")
    reward_net   = load_reward_net(
        f"./models/{deep_experiment}/reward_net_epoch{deep_epoch}.pt",
        hidden_dim=deep_hidden_dim,
    )

    # Build envs
    maxent_vec_env = DummyVecEnv([lambda: gym.make('V2GEnv-profit')])
    deep_vec_env   = DummyVecEnv([lambda: gym.make('V2GDeepEnv-profit')])

    # Set the deep reward net once (persists across resets)
    deep_vec_env.envs[0].unwrapped.set_reward_net(reward_net)

    # Find matching trajectory indexes within the segment
    maxent_indexes = find_expert_indexes(maxent_expert_data, segment)
    deep_indexes   = find_expert_indexes(deep_expert_data,   segment)

    n_common = min(len(maxent_indexes), len(deep_indexes))
    n_eval   = max(1, int(n_common * eval_ratio))
    maxent_indexes = maxent_indexes[:n_eval]
    deep_indexes   = deep_indexes[:n_eval]

    print(f"Segment '{segment}': {n_common} trajectories available, evaluating {n_eval}")

    all_maxent_dtw = []
    all_deep_dtw   = []

    for i, (m_idx, d_idx) in enumerate(zip(maxent_indexes, deep_indexes)):
        print(f"\nTrajectory {i+1}/{n_eval}  (maxent_idx={m_idx}, deep_idx={d_idx})")

        expert_soc = maxent_expert_data[m_idx]['soc_history']

        # --- Linear MaxEnt rollout ---
        maxent_soc, maxent_dtw, maxent_info = _best_dtw_rollout(
            maxent_vec_env,
            maxent_model,
            expert_soc,
            initial_states=maxent_expert_data[m_idx]['initial_values'],
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
            noise_scale=noise_scale,
        )

        # --- Deep MaxEnt rollout ---
        deep_soc, deep_dtw, _ = _best_dtw_rollout(
            deep_vec_env,
            deep_model,
            expert_soc,
            initial_states=deep_expert_data[d_idx]['initial_values'],
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
            noise_scale=noise_scale,
        )

        all_maxent_dtw.append(maxent_dtw)
        all_deep_dtw.append(deep_dtw)

        print(
            f"  Linear MaxEnt best DTW: {maxent_dtw:.3f} | "
            f"Deep MaxEnt best DTW: {deep_dtw:.3f}"
        )

        if i < n_figures:
            plot_comparison(
                expert_soc=expert_soc,
                maxent_soc=maxent_soc,
                deep_soc=deep_soc,
                expert_index=m_idx,
                out_start=maxent_info["out_start_timestep"],
                out_dur=maxent_info["out_duration"],
                return_start=maxent_info["return_start_timestep"],
                return_dur=maxent_info["return_duration"],
                maxent_dtw=maxent_dtw,
                deep_dtw=deep_dtw,
            )

    print(f"\n{'='*60}")
    print(f"Summary over {n_eval} trajectories (segment: {segment}):")
    print(f"  Linear MaxEnt  Avg Best DTW: {np.mean(all_maxent_dtw):.3f} ± {np.std(all_maxent_dtw):.3f}")
    print(f"  Deep MaxEnt    Avg Best DTW: {np.mean(all_deep_dtw):.3f} ± {np.std(all_deep_dtw):.3f}")
    print(f"{'='*60}")

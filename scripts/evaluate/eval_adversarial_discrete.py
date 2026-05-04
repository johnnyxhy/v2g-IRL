"""
Evaluation script for AIRL (Adversarial IRL) discrete experiments.

Loads a trained SBX PPO policy and the AIRL reward/shaping networks from
a given experiment folder and epoch, then evaluates over expert trajectories
from the AIRL-format JSON, reporting DTW and Interval Score.

Usage:
    Edit the configuration block at the bottom and run:
        python scripts/eval_adversarial_discrete.py
"""
import json
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from sbx import PPO

from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.Adversarial.Adversarial_discrete import ShapingNet, FlattenNormalizeObsWrapper
from irl.utils.tools import compute_dtw

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
    0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07,
])


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def load_networks(folder, epoch, reward_hidden_dim=32, shaping_hidden_dim=32,
                  obs_dim=7, action_dim=1):
    reward_net = RewardNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=reward_hidden_dim)
    reward_net.load_state_dict(
        torch.load(f"./models/{folder}/reward_net_epoch{epoch}.pt", weights_only=True)
    )
    reward_net.eval()

    shaping_net = ShapingNet(obs_dim=obs_dim, hidden_dim=shaping_hidden_dim)
    shaping_net.load_state_dict(
        torch.load(f"./models/{folder}/shaping_net_epoch{epoch}.pt", weights_only=True)
    )
    shaping_net.eval()
    return reward_net, shaping_net


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


def compute_g_score(traj, reward_net):
    """Sum of g_theta(s,a)*delta_t over an AIRL expert trajectory."""
    obs_scales = torch.tensor(PROFIT_OBS_SCALES, dtype=torch.float32)
    sap = traj['state_action_pairs']
    obs  = torch.tensor(np.array(sap['observations'], dtype=np.float32)) / obs_scales
    acts = torch.tensor(np.array(sap['actions'],      dtype=np.float32))
    dts  = torch.tensor(np.array(sap['delta_ts'],     dtype=np.float32))
    with torch.no_grad():
        return (reward_net(obs, acts) * dts).sum().item()


# ------------------------------------------------------------------ #
#  Single-trajectory evaluation                                        #
# ------------------------------------------------------------------ #

def evaluate(vec_env, model, expert_data, expert_index,
             reward_net=None, n_rollouts=10, deterministic=False, plot=True):
    traj = expert_data[expert_index]
    initial_states = traj['initial_values']
    expert_soc = traj['soc_history']
    target_len = len(expert_soc)

    vec_env.envs[0].unwrapped.set_initial_states(initial_states)
    if reward_net is not None:
        vec_env.envs[0].unwrapped.set_reward_net(reward_net)

    all_dtw = []
    all_rewards = []
    all_feat_exp = []
    all_soc_histories = []
    all_mae = []

    best_dtw = float('inf')
    best_soc_history = None
    best_info = None
    best_mae_val = float('inf')
    best_mae_soc_history = None
    best_mae_info = None

    for _ in range(n_rollouts):
        obs = vec_env.reset()
        accumulated_reward = 0.0
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = vec_env.step(action)
            accumulated_reward += float(rewards[0])
            if done[0]:
                break

        soc_history = info[0]['soc_history']
        feat_exp = np.array(info[0]['feature_expectation'], dtype=np.float32)

        dtw_distance = compute_dtw(
            np.array(expert_soc, dtype=np.float32),
            np.array(soc_history, dtype=np.float32),
        )
        all_dtw.append(dtw_distance)
        all_rewards.append(accumulated_reward)
        all_feat_exp.append(feat_exp)
        all_soc_histories.append(_resample_trajectory(soc_history, target_len))
        mae_this = float(np.mean(np.abs(
            _resample_trajectory(soc_history, target_len)
            - _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
        )))
        all_mae.append(mae_this)

        if dtw_distance < best_dtw:
            best_dtw = dtw_distance
            best_soc_history = soc_history
            best_info = info[0]

        if mae_this < best_mae_val:
            best_mae_val = mae_this
            best_mae_soc_history = soc_history
            best_mae_info = info[0]

    dtw_mean = float(np.mean(all_dtw))
    dtw_std  = float(np.std(all_dtw))
    mae_mean = float(np.mean(all_mae))
    mae_std  = float(np.std(all_mae))
    reward_mean = float(np.mean(all_rewards))
    reward_std  = float(np.std(all_rewards))
    feat_mean = np.mean(np.stack(all_feat_exp), axis=0)

    soc_stack = np.stack(all_soc_histories)
    min_soc = np.min(soc_stack, axis=0)
    max_soc = np.max(soc_stack, axis=0)

    # 80% Interval Score (α=0.2)
    alpha = 0.2
    lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
    upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
    expert_resampled = _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
    width = upper_soc - lower_soc
    undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
    overshoot  = np.maximum(expert_resampled - upper_soc,  0.0)
    interval_score = float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))

    expert_g = compute_g_score(traj, reward_net) if reward_net is not None else None
    reward_str = f"AgentReward={reward_mean:.3f}±{reward_std:.3f}"
    if expert_g is not None:
        reward_str += f", ExpertG={expert_g:.3f}"

    print(
        f"Traj {expert_index} over {n_rollouts} rollouts: "
        f"DTW={dtw_mean:.3f}\u00b1{dtw_std:.3f}, BestDTW={best_dtw:.3f}, "
        f"MAE={mae_mean:.4f}\u00b1{mae_std:.4f}, BestMAE={best_mae_val:.4f}, "
        f"IS={interval_score:.3f}, {reward_str}"
    )

    if plot and best_mae_info is not None:
        _plot(
            best_mae_soc_history, expert_soc, expert_index,
            best_mae_info['out_start_timestep'], best_mae_info['return_start_timestep'],
            best_mae_info['out_duration'], best_mae_info['return_duration'],
            best_mae_val, mae_std,
            min_soc=min_soc, max_soc=max_soc,
        )

    return dtw_mean, interval_score, mae_mean, best_dtw, best_mae_val, all_dtw, all_mae


# ------------------------------------------------------------------ #
#  Plotting                                                            #
# ------------------------------------------------------------------ #

def _plot(soc_history, expert_soc, expert_index,
          out_start, return_start, out_dur, return_dur,
          best_mae_val, mae_std,
          min_soc=None, max_soc=None):
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()

    ax1.plot(range(len(soc_history)), soc_history, label=f'Best MAE={best_mae_val:.4f}', color='tab:blue')
    if min_soc is not None and max_soc is not None:
        ax1.fill_between(range(len(min_soc)), min_soc, max_soc,
                         color='tab:blue', alpha=0.2, label='Agent SoC Range')
    ax1.plot(range(len(expert_soc)), expert_soc,
             label='Expert SoC', linestyle='--', color='tab:orange')

    ax1.axvspan(out_start, out_start + out_dur,
                color='red', alpha=0.2, label='Out Journey')
    ax1.axvspan(return_start, return_start + return_dur,
                color='purple', alpha=0.15, label='Return Journey')

    # Energy price on a twin axis
    ax2 = ax1.twinx()
    price_len = min(len(soc_history), len(energy_price_profile))
    ax2.plot(range(price_len), energy_price_profile[:price_len],
             color='green', linewidth=0.8, alpha=0.6, label='Energy Price')
    ax2.set_ylabel('Energy Price (£/kWh)', color='green', fontsize=8)
    ax2.tick_params(axis='y', labelcolor='green', labelsize=7)
    ax2.set_ylim(0, 0.6)

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('State of Charge (SoC)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

    plt.title(
        f'AIRL Discrete — Best MAE={best_mae_val:.4f} ± {mae_std:.4f} — Traj {expert_index}'
    )
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # ---- Configuration ----
    experiment_folder  = "Adversarial/discrete/Adversarial_discrete_exp5"  # folder under models/
    epoch_to_load      = 30                            # which epoch checkpoint to load
    expert_data_path   = "data/processed_trajectories_airl_discrete.json"
    segment            = "Male 50-59"
    reward_hidden_dim  = 32    # must match cfg.reward_hidden_dim used during training
    shaping_hidden_dim = 32    # must match cfg.shaping_hidden_dim used during training
    n_rollouts         = 20    # stochastic rollouts per trajectory
    n_figures          = 10    # how many SoC plots to display
    eval_ratio         = 1.0   # fraction of segment trajectories to evaluate
    deterministic      = False # True = greedy policy, False = stochastic

    # ---------------------------------------------------------------
    with open(expert_data_path, "r") as f:
        expert_data = json.load(f)

    reward_net, shaping_net = load_networks(
        experiment_folder, epoch_to_load,
        reward_hidden_dim=reward_hidden_dim,
        shaping_hidden_dim=shaping_hidden_dim,
    )

    model = PPO.load(f"./models/{experiment_folder}/ppo_epoch{epoch_to_load}")

    vec_env = DummyVecEnv([
        lambda: FlattenNormalizeObsWrapper(gym.make('V2GDeepEnv-discrete'))
    ])
    # Set AIRL reward components in the env so rollout rewards reflect f(s,a,s')
    vec_env.envs[0].unwrapped.set_reward_net(reward_net)
    vec_env.envs[0].unwrapped.set_shaping_net(shaping_net)

    expert_indexes = find_expert_indexes(expert_data, segment)
    n_eval = max(1, int(len(expert_indexes) * eval_ratio))
    expert_indexes = expert_indexes[:n_eval]
    print(f"Evaluating {n_eval} trajectories for segment '{segment}' "
          f"from {experiment_folder} epoch {epoch_to_load}")

    all_rollout_dtw = []
    all_rollout_mae = []
    all_is = []

    for i, idx in enumerate(expert_indexes):
        print(f"\nTrajectory {idx} ({i+1}/{len(expert_indexes)})...")
        dtw_mean_t, is_score, mae_mean_t, best_dtw, best_mae, traj_dtw, traj_mae = evaluate(
            vec_env, model, expert_data, idx,
            reward_net=reward_net,
            n_rollouts=n_rollouts,
            deterministic=deterministic,
            plot=(i < n_figures),
        )
        all_rollout_dtw.extend(traj_dtw)
        all_rollout_mae.extend(traj_mae)
        all_is.append(is_score)

    print(f"\n{'='*60}")
    print(f"Summary over {len(expert_indexes)} trajectories x {n_rollouts} rollouts [{segment}]:")
    print(f"  Mean DTW : {np.mean(all_rollout_dtw):.3f} ± {np.std(all_rollout_dtw):.3f}")
    print(f"  Mean MAE : {np.mean(all_rollout_mae):.4f} ± {np.std(all_rollout_mae):.4f}")
    print(f"  Avg IS   : {np.mean(all_is):.3f} ± {np.std(all_is):.3f}")
    print(f"  Median DTW : {np.median(all_rollout_dtw):.3f}")
    print(f"  Median MAE : {np.median(all_rollout_mae):.4f}")
    print(f"{'='*60}")

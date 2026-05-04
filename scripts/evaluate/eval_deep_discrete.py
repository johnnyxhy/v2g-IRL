import numpy as np
import gymnasium as gym
from sbx import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.DeepMaxEnt.DeepMaxEnt_discrete import FlattenNormalizeObsWrapper
from irl.utils.tools import compute_dtw
import torch
import json

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


def load_reward_net(path, obs_dim=7, action_dim=1, hidden_dim=64):
    reward_net = RewardNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    reward_net.load_state_dict(torch.load(path, weights_only=True))
    reward_net.eval()
    return reward_net


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


def plot_expert_trajectory(soc_history, expert_soc, expert_index,
                           out_start, return_start, out_dur, return_dur,
                           best_mae_val, mae_std,
                           min_soc=None, max_soc=None):
    plt.figure()
    plt.plot(range(len(soc_history)), soc_history, label=f'Best MAE={best_mae_val:.4f}', color='tab:blue')
    if min_soc is not None and max_soc is not None:
        plt.fill_between(range(len(min_soc)), min_soc, max_soc,
                         color='tab:blue', alpha=0.2, label='Agent SoC Range')
    plt.plot(range(len(expert_soc)), expert_soc, label='Expert SoC', linestyle='--', color='tab:orange')
    plt.axvspan(out_start, out_start + out_dur,
                color='red', alpha=0.3, label='Out Journey')
    plt.axvspan(return_start, return_start + return_dur,
                color='blue', alpha=0.3, label='Return Journey')
    energy_price = energy_price_profile[:len(soc_history)]
    plt.plot(range(len(energy_price)), energy_price, label='Energy Price', color='green')
    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.legend()
    plt.title(
        f'Deep MaxEnt Discrete IRL \u2014 Best MAE={best_mae_val:.4f} \u00b1 {mae_std:.4f} \u2014 Traj {expert_index}'
    )
    plt.show()


def plot_expert_trajectory_rollouts(
    mean_soc,
    min_soc,
    max_soc,
    expert_soc,
    expert_index,
    out_start_timestep,
    return_start_timestep,
    out_duration,
    return_duration,
    dtw_mean,
    dtw_std,
):
    """Plot expert SoC vs rollout mean SoC with min-max range."""
    timesteps = range(len(mean_soc))
    plt.figure()
    plt.plot(timesteps, mean_soc, label='Agent Mean SoC', color='tab:blue')
    plt.fill_between(timesteps, min_soc, max_soc, color='tab:blue', alpha=0.2, label='Agent SoC Range')
    plt.plot(range(len(expert_soc)), expert_soc, label='Expert SoC', linestyle='--', color='tab:orange')
    plt.axvspan(out_start_timestep, out_start_timestep + out_duration,
                color='red', alpha=0.2, label='Out Journey')
    plt.axvspan(return_start_timestep, return_start_timestep + return_duration,
                color='blue', alpha=0.15, label='Return Journey')
    energy_price = energy_price_profile[:len(mean_soc)]
    plt.plot(range(len(energy_price)), energy_price, label='Energy Price', color='green')
    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.legend()
    plt.title(
        f'Deep MaxEnt Discrete IRL \u2014 DTW mean\u00b1std: {dtw_mean:.2f}\u00b1{dtw_std:.2f} \u2014 Trajectory {expert_index}'
    )
    plt.show()


def compute_expert_reward(traj, reward_net):
    """Compute the reward net score for an expert trajectory."""
    obs_scales = torch.tensor(PROFIT_OBS_SCALES, dtype=torch.float32)
    obs  = torch.tensor(np.array(traj['state_action_pairs']['observations'], dtype=np.float32)) / obs_scales
    acts = torch.tensor(np.array(traj['state_action_pairs']['actions'],      dtype=np.float32))
    dts  = torch.tensor(np.array(traj['state_action_pairs']['delta_ts'],     dtype=np.float32))
    with torch.no_grad():
        r = (reward_net(obs, acts) * dts).sum().item()
    return r


def evaluate(vec_env, model, expert_data, expert_index, reward_net=None, n_rollouts=10, deterministic=False, plot=True):
    initial_states = expert_data[expert_index]['initial_values']
    vec_env.envs[0].unwrapped.set_initial_states(initial_states)
    if reward_net is not None:
        vec_env.envs[0].unwrapped.set_reward_net(reward_net)

    expert_soc = expert_data[expert_index]['soc_history']
    target_len = len(expert_soc)

    all_dtw = []
    all_rewards = []
    all_feat_exp = []
    all_soc_histories = []
    all_mae = []
    all_charge_prices = []
    all_discharge_prices = []

    best_dtw = float('inf')
    best_soc_history = None
    best_out_start = best_return_start = best_out_dur = best_return_dur = None
    best_feat_exp = None
    best_mae_val = float('inf')
    best_mae_soc_history = None
    best_mae_info = None

    for _ in range(n_rollouts):
        obs = vec_env.reset()
        accumulated_reward = 0.0
        rollout_charge_prices = []
        rollout_discharge_prices = []

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)

            current_timestep = vec_env.envs[0].unwrapped.timestep
            current_price = energy_price_profile[min(int(current_timestep), 95)]
            action_idx = int(action[0])
            if action_idx > 10:
                rollout_charge_prices.append(current_price)
            elif action_idx < 10:
                rollout_discharge_prices.append(current_price)

            obs, rewards, done, info = vec_env.step(action)
            accumulated_reward += rewards[0]
            if done[0]:
                soc_history = info[0]["soc_history"]
                feat_exp = np.array(info[0]["feature_expectation"], dtype=np.float32)
                break

        all_rewards.append(accumulated_reward)
        all_feat_exp.append(feat_exp)
        all_soc_histories.append(_resample_trajectory(soc_history, target_len))
        if rollout_charge_prices:
            all_charge_prices.append(float(np.mean(rollout_charge_prices)))
        if rollout_discharge_prices:
            all_discharge_prices.append(float(np.mean(rollout_discharge_prices)))

        dtw_distance = compute_dtw(
            np.array(expert_soc, dtype=np.float32),
            np.array(soc_history, dtype=np.float32),
        )
        mae_this = float(np.mean(np.abs(
            _resample_trajectory(soc_history, target_len)
            - _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
        )))
        all_mae.append(mae_this)
        all_dtw.append(dtw_distance)

        if dtw_distance < best_dtw:
            best_dtw = dtw_distance
            best_soc_history = soc_history
            best_out_start = info[0]["out_start_timestep"]
            best_return_start = info[0]["return_start_timestep"]
            best_out_dur = info[0]["out_duration"]
            best_return_dur = info[0]["return_duration"]
            best_feat_exp = feat_exp

        if mae_this < best_mae_val:
            best_mae_val = mae_this
            best_mae_soc_history = soc_history
            best_mae_info = info[0]

    dtw_mean = float(np.mean(all_dtw))
    dtw_std = float(np.std(all_dtw))
    mae_mean = float(np.mean(all_mae))
    mae_std  = float(np.std(all_mae))
    reward_mean = float(np.mean(all_rewards))
    reward_std = float(np.std(all_rewards))
    feat_mean = np.mean(np.stack(all_feat_exp, axis=0), axis=0)
    soc_stack = np.stack(all_soc_histories, axis=0)
    min_soc = np.min(soc_stack, axis=0)
    max_soc = np.max(soc_stack, axis=0)

    agent_avg_charge_price = float(np.mean(all_charge_prices)) if all_charge_prices else float('nan')
    agent_avg_discharge_price = float(np.mean(all_discharge_prices)) if all_discharge_prices else float('nan')

    expert_obs = np.array(expert_data[expert_index]['state_action_pairs']['observations'], dtype=np.float32)
    expert_acts = np.array(expert_data[expert_index]['state_action_pairs']['actions'], dtype=np.float32).flatten()
    expert_prices = expert_obs[:, 3]  # energy_price is observation index 3
    expert_charge_mask = expert_acts > 0
    expert_discharge_mask = expert_acts < 0
    expert_avg_charge_price = float(np.mean(expert_prices[expert_charge_mask])) if expert_charge_mask.any() else float('nan')
    expert_avg_discharge_price = float(np.mean(expert_prices[expert_discharge_mask])) if expert_discharge_mask.any() else float('nan')

    expert_reward = compute_expert_reward(expert_data[expert_index], reward_net) if reward_net is not None else None

    # Interval Score — 80% quantile interval (α=0.2, multiplier=2/α=10)
    alpha = 0.2
    lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
    upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
    expert_resampled = _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
    width = upper_soc - lower_soc
    undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
    overshoot = np.maximum(expert_resampled - upper_soc, 0.0)
    interval_score = float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))

    reward_str = f"AgentReward={reward_mean:.3f}\u00b1{reward_std:.3f}"
    if expert_reward is not None:
        reward_str += f", ExpertReward={expert_reward:.3f}, Gap={expert_reward - reward_mean:.3f}"
    charge_price_str = f"ChargePrice: Expert={expert_avg_charge_price:.4f}, Agent={agent_avg_charge_price:.4f}"
    discharge_price_str = f"DischargePrice: Expert={expert_avg_discharge_price:.4f}, Agent={agent_avg_discharge_price:.4f}"
    print(
        f"Traj {expert_index} over {n_rollouts} rollouts: "
        f"DTW={dtw_mean:.3f}\u00b1{dtw_std:.3f}, BestDTW={best_dtw:.3f}, "
        f"MAE={mae_mean:.4f}\u00b1{mae_std:.4f}, BestMAE={best_mae_val:.4f}, "
        f"IS={interval_score:.3f}, {reward_str}, "
        f"{charge_price_str}, {discharge_price_str}"
    )

    if plot and best_mae_info is not None:
        plot_expert_trajectory(
            best_mae_soc_history, expert_soc, expert_index,
            best_mae_info['out_start_timestep'], best_mae_info['return_start_timestep'],
            best_mae_info['out_duration'], best_mae_info['return_duration'],
            best_mae_val, mae_std,
            min_soc=min_soc, max_soc=max_soc,
        )

    return dtw_mean, interval_score, mae_mean, best_dtw, best_mae_val, all_dtw, all_mae, agent_avg_charge_price, agent_avg_discharge_price, expert_avg_charge_price, expert_avg_discharge_price


if __name__ == "__main__":
    # --- Configuration ---
    experiment_folder = "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male4049"  # Must match folder_name used during training
    epoch_to_load = 20
    segment = "Male 40-49"
    hidden_dim = 32  # Must match reward_hidden_dim used during training
    n_rollouts = 20
    n_figures = 10              # number of example figures to display
    eval_ratio = 1.0           # fraction of segment trajectories to evaluate (1.0 = all)
    deterministic_policy = False

    with open("data/processed_trajectories_deep_discrete_pricediff.json", "r") as f:
        expert_data = json.load(f)

    vec_env = DummyVecEnv([lambda: FlattenNormalizeObsWrapper(gym.make('V2GDeepEnv-discrete'))])

    # The PPO at epoch N was trained under reward_net from epoch N-1, so load
    # the reward net one epoch earlier for a fair evaluation.
    reward_epoch = max(1, epoch_to_load - 1)
    model = PPO.load(f"./models/{experiment_folder}/ppo_epoch{epoch_to_load}")
    reward_net = load_reward_net(
        f"./models/{experiment_folder}/reward_net_epoch{reward_epoch}.pt",
        hidden_dim=hidden_dim,
    )
    print(f"Loaded PPO epoch {epoch_to_load} with reward net epoch {reward_epoch}")

    expert_indexes = find_expert_indexes(expert_data, segment)
    n_eval = max(1, int(len(expert_indexes) * eval_ratio))
    expert_indexes = expert_indexes[:n_eval]
    print(f"Found {len(expert_indexes)} trajectories for segment '{segment}' (evaluating {n_eval})")

    all_rollout_dtw = []
    all_rollout_mae = []
    all_is = []
    all_agent_charge_prices = []
    all_agent_discharge_prices = []
    all_expert_charge_prices = []
    all_expert_discharge_prices = []

    for i, idx in enumerate(expert_indexes):
        print(f"\nEvaluating trajectory {idx} ({i+1}/{len(expert_indexes)})...")
        dtw_mean_t, is_score, mae_mean_t, best_dtw, best_mae, traj_dtw, traj_mae, agent_cp, agent_dp, expert_cp, expert_dp = evaluate(
            vec_env,
            model,
            expert_data,
            idx,
            reward_net=reward_net,
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
            plot=(i < n_figures),
        )
        all_rollout_dtw.extend(traj_dtw)
        all_rollout_mae.extend(traj_mae)
        all_is.append(is_score)
        if not np.isnan(agent_cp): all_agent_charge_prices.append(agent_cp)
        if not np.isnan(agent_dp): all_agent_discharge_prices.append(agent_dp)
        if not np.isnan(expert_cp): all_expert_charge_prices.append(expert_cp)
        if not np.isnan(expert_dp): all_expert_discharge_prices.append(expert_dp)

    print(f"\n{'='*60}")
    print(f"Summary over {len(expert_indexes)} trajectories x {n_rollouts} rollouts [{segment}]:")
    print(f"  Mean DTW : {np.mean(all_rollout_dtw):.3f} \u00b1 {np.std(all_rollout_dtw):.3f}")
    print(f"  Mean MAE : {np.mean(all_rollout_mae):.4f} \u00b1 {np.std(all_rollout_mae):.4f}")
    print(f"  Avg IS   : {np.mean(all_is):.3f} \u00b1 {np.std(all_is):.3f}")
    print(f"  Median DTW : {np.median(all_rollout_dtw):.3f}")
    print(f"  Median MAE : {np.median(all_rollout_mae):.4f}")
    agent_cp_mean = np.mean(all_agent_charge_prices) if all_agent_charge_prices else float('nan')
    agent_dp_mean = np.mean(all_agent_discharge_prices) if all_agent_discharge_prices else float('nan')
    expert_cp_mean = np.mean(all_expert_charge_prices) if all_expert_charge_prices else float('nan')
    expert_dp_mean = np.mean(all_expert_discharge_prices) if all_expert_discharge_prices else float('nan')
    print(f"  Avg Charge Price   : Expert={expert_cp_mean:.4f}, Agent={agent_cp_mean:.4f}")
    print(f"  Avg Discharge Price: Expert={expert_dp_mean:.4f}, Agent={agent_dp_mean:.4f}")
    print(f"{'='*60}")

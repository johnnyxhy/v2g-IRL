import numpy as np
import gymnasium as gym
from sbx import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.utils.tools import compute_dtw
import json

gym.register(
    id='V2GEnv-profit',
    entry_point="irl.envs.V2GEnv_profit:V2GEnv",
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
    """Resample a 1D trajectory to target_len using linear interpolation."""
    values = np.asarray(values, dtype=np.float32)
    if len(values) == target_len:
        return values
    if len(values) <= 1:
        return np.full(target_len, values[0] if len(values) == 1 else 0.0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, values).astype(np.float32)


def plot_expert_trajectory(soc_history, expert_soc, expert_index,
                           out_start_timestep, return_start_timestep,
                           out_duration, return_duration, best_mae_val, mae_std,
                           min_soc=None, max_soc=None, interval_score=None):
    plt.figure()
    plt.plot(range(len(soc_history)), soc_history, label=f'Best MAE={best_mae_val:.4f}', color='tab:blue')
    if min_soc is not None and max_soc is not None:
        plt.fill_between(
            range(len(min_soc)),
            min_soc,
            max_soc,
            color='tab:blue',
            alpha=0.2,
            label='Agent SoC Range'
        )
    plt.plot(range(len(expert_soc)), expert_soc, label='Expert SoC', linestyle='--', color='tab:orange')
    plt.axvspan(out_start_timestep, out_start_timestep + out_duration,
                color='red', alpha=0.3, label='Out Journey')
    plt.axvspan(return_start_timestep, return_start_timestep + return_duration,
                color='blue', alpha=0.3, label='Return Journey')

    energy_price = energy_price_profile[:len(soc_history)]
    plt.plot(range(len(energy_price)), energy_price, label='Energy Price', color='green')

    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.legend()
    title = f'Linear MaxEnt IRL \u2014 Best MAE={best_mae_val:.4f} \u00b1 {mae_std:.4f}'
    if interval_score is not None:
        title += f' \u2014 IS: {interval_score:.3f}'
    title += f' \u2014 Trajectory {expert_index}'
    plt.title(title)
    plt.show()


def evaluate(vec_env, model, expert_data, expert_index, n_rollouts=10, deterministic=False, noise_scale=1.0, plot=True):
    """Run n evaluation episodes and plot the single rollout with the lowest DTW.
    
    noise_scale: float in [0, 1]. Controls stochasticity when deterministic=False.
        1.0 = full stochastic, 0.0 = deterministic, values in between blend.
    """
    initial_states = expert_data[expert_index]['initial_values']
    vec_env.envs[0].unwrapped.set_initial_states(initial_states)

    all_dtw = []
    all_mae = []
    all_rewards = []
    all_feat_exp = []
    all_soc_histories = []

    best_dtw = float('inf')
    best_soc_history = None
    best_out_start = None
    best_return_start = None
    best_out_dur = None
    best_return_dur = None
    best_feat_exp = None
    best_mae_val = float('inf')
    best_mae_soc_history = None
    best_mae_info = None

    expert_soc = expert_data[expert_index]['soc_history']
    target_len = len(expert_soc)

    for _ in range(n_rollouts):
        obs = vec_env.reset()
        accumulated_reward = 0.0

        while True:
            if deterministic or noise_scale >= 1.0:
                action, _ = model.predict(obs, deterministic=deterministic)
            else:
                det_action, _ = model.predict(obs, deterministic=True)
                stoch_action, _ = model.predict(obs, deterministic=False)
                action = det_action + noise_scale * (stoch_action - det_action)
            obs, rewards, done, info = vec_env.step(action)
            accumulated_reward += rewards[0]

            if done[0]:
                soc_history = info[0]["soc_history"]
                feat_exp = np.array(info[0]["feature_expectation"], dtype=np.float32)
                break

        all_rewards.append(accumulated_reward)
        all_feat_exp.append(feat_exp)
        all_soc_histories.append(_resample_trajectory(soc_history, target_len))

        dtw_distance = compute_dtw(
            np.array(expert_soc, dtype=np.float32),
            np.array(soc_history, dtype=np.float32),
        )
        mae_this = float(np.mean(np.abs(
            _resample_trajectory(soc_history, target_len)
            - _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
        )))
        all_dtw.append(dtw_distance)
        all_mae.append(mae_this)

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
    mae_std = float(np.std(all_mae))
    reward_mean = float(np.mean(all_rewards))
    reward_std = float(np.std(all_rewards))
    feat_mean = np.mean(np.stack(all_feat_exp, axis=0), axis=0)
    soc_stack = np.stack(all_soc_histories, axis=0)
    min_soc = np.min(soc_stack, axis=0)
    max_soc = np.max(soc_stack, axis=0)

    # Interval Score: penalizes both width and misses (lower is better)
    expert_soc_arr = np.array(expert_soc, dtype=np.float32)
    expert_resampled = _resample_trajectory(expert_soc_arr, target_len)
    width = max_soc - min_soc
    undershoot = np.maximum(min_soc - expert_resampled, 0.0)
    overshoot = np.maximum(expert_resampled - max_soc, 0.0)
    interval_score = float(np.mean(width + 2.0 * undershoot + 2.0 * overshoot))

    print(
        f"Traj {expert_index} over {n_rollouts} rollouts: "
        f"DTW={dtw_mean:.3f}\u00b1{dtw_std:.3f}, BestDTW={best_dtw:.3f}, "
        f"MAE={mae_mean:.4f}\u00b1{mae_std:.4f}, BestMAE={best_mae_val:.4f}, "
        f"IS={interval_score:.3f}, "
        f"AccReward={reward_mean:.3f}\u00b1{reward_std:.3f}"
    )

    if plot and best_mae_info is not None:
        plot_expert_trajectory(
            best_mae_soc_history,
            expert_soc,
            expert_index,
            best_mae_info["out_start_timestep"],
            best_mae_info["return_start_timestep"],
            best_mae_info["out_duration"],
            best_mae_info["return_duration"],
            best_mae_val,
            mae_std,
            min_soc=min_soc,
            max_soc=max_soc,
            interval_score=interval_score,
        )

    return dtw_mean, interval_score, mae_mean, best_dtw, best_mae_val, all_dtw, all_mae


if __name__ == "__main__":
    # --- Configuration ---
    model_path = "./models/MaxEnt/continuous/MaxEntIRL_profit_v7_exp2/maxent_irl_epoch50"
    expert_data_path = "data/processed_trajectories_profit.json"
    segment = "Male 50-59"
    n_rollouts = 20
    n_figures = 5              # number of example figures to display
    eval_ratio = 1.0           # fraction of segment trajectories to evaluate (1.0 = all)
    deterministic_policy = False
    noise_scale = 0.5           # 0.0 = deterministic, 1.0 = full stochastic

    # Load expert data
    with open(expert_data_path, "r") as f:
        expert_data = json.load(f)

    # Load trained model
    model = SAC.load(model_path)

    vec_env = DummyVecEnv([lambda: gym.make('V2GEnv-profit')])

    expert_indexes = find_expert_indexes(expert_data, segment)
    n_eval = max(1, int(len(expert_indexes) * eval_ratio))
    expert_indexes = expert_indexes[:n_eval]
    print(f"Found {len(expert_indexes)} trajectories for segment '{segment}' (evaluating {n_eval})")

    all_rollout_dtw = []
    all_rollout_mae = []
    all_is = []

    for i, idx in enumerate(expert_indexes):
        print(f"\nEvaluating trajectory {idx} ({i+1}/{len(expert_indexes)})...")
        dtw_mean_t, is_score, mae_mean_t, best_dtw, best_mae, traj_dtw, traj_mae = evaluate(
            vec_env,
            model,
            expert_data,
            idx,
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
            noise_scale=noise_scale,
            plot=(i < n_figures),
        )
        all_rollout_dtw.extend(traj_dtw)
        all_rollout_mae.extend(traj_mae)
        all_is.append(is_score)

    print(f"\n{'='*60}")
    print(f"Summary over {len(expert_indexes)} trajectories x {n_rollouts} rollouts [{segment}]:")
    print(f"  Mean DTW : {np.mean(all_rollout_dtw):.3f} \u00b1 {np.std(all_rollout_dtw):.3f}")
    print(f"  Mean MAE : {np.mean(all_rollout_mae):.4f} \u00b1 {np.std(all_rollout_mae):.4f}")
    print(f"  Avg IS   : {np.mean(all_is):.3f} \u00b1 {np.std(all_is):.3f}")
    print(f"  Median DTW : {np.median(all_rollout_dtw):.3f}")
    print(f"  Median MAE : {np.median(all_rollout_mae):.4f}")
    print(f"{'='*60}")

        

    

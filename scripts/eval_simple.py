import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.utils.tools import compute_dtw
import json

gym.register(
    id='V2GEnv-simple',
    entry_point="irl.envs.V2GEnv_simple:V2GEnv",
    max_episode_steps=96,
)


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
                           out_duration, return_duration, dtw_distance,
                           min_soc=None, max_soc=None,
                           dtw_mean=None, dtw_std=None, interval_score=None):
    plt.figure()
    plt.plot(range(len(soc_history)), soc_history, label='Agent SoC')
    if min_soc is not None and max_soc is not None:
        plt.fill_between(
            range(len(min_soc)),
            min_soc,
            max_soc,
            color='tab:blue',
            alpha=0.2,
            label='Agent SoC Range'
        )
    plt.plot(range(len(expert_soc)), expert_soc, label='Expert SoC', linestyle='--')
    plt.axvspan(out_start_timestep, out_start_timestep + out_duration,
                color='red', alpha=0.3, label='Out Journey')
    plt.axvspan(return_start_timestep, return_start_timestep + return_duration,
                color='blue', alpha=0.3, label='Return Journey')

    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.legend()
    if dtw_mean is not None and dtw_std is not None:
        title = f'Linear MaxEnt IRL \u2014 Best DTW: {dtw_distance:.2f} (mean\u00b1std: {dtw_mean:.2f}\u00b1{dtw_std:.2f})'
        if interval_score is not None:
            title += f' \u2014 IS: {interval_score:.3f}'
        title += f' \u2014 Trajectory {expert_index}'
        plt.title(title)
    else:
        plt.title(f'Linear MaxEnt IRL \u2014 DTW: {dtw_distance:.2f} \u2014 Trajectory {expert_index}')
    plt.show()


def evaluate(vec_env, model, expert_data, expert_index, reward_weights, n_rollouts=10, deterministic=False, plot=True):
    """Run n evaluation episodes and plot the single rollout with the lowest DTW."""
    initial_states = expert_data[expert_index]['initial_values']
    vec_env.envs[0].unwrapped.set_initial_states(initial_states)
    vec_env.envs[0].unwrapped.set_reward_weights(reward_weights)

    expert_soc = expert_data[expert_index]['soc_history']
    target_len = len(expert_soc)

    all_dtw = []
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

    for _ in range(n_rollouts):
        obs = vec_env.reset()
        accumulated_reward = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = vec_env.step(action)
            accumulated_reward += rewards[0]

            if done[0]:
                soc_history = info[0]["soc_history"]
                feat_exp = np.array(info[0]["feature_expectation"], dtype=np.float32)
                out_start = info[0]["out_start_timestep"]
                return_start = info[0]["return_start_timestep"]
                out_dur = info[0]["out_duration"]
                return_dur = info[0]["return_duration"]
                break

        all_rewards.append(accumulated_reward)
        all_feat_exp.append(feat_exp)
        all_soc_histories.append(_resample_trajectory(soc_history, target_len))

        dtw_distance = compute_dtw(
            np.array(expert_soc, dtype=np.float32),
            np.array(soc_history, dtype=np.float32),
        )
        all_dtw.append(dtw_distance)

        if dtw_distance < best_dtw:
            best_dtw = dtw_distance
            best_soc_history = soc_history
            best_out_start = out_start
            best_return_start = return_start
            best_out_dur = out_dur
            best_return_dur = return_dur
            best_feat_exp = feat_exp

    dtw_mean = float(np.mean(all_dtw))
    dtw_std = float(np.std(all_dtw))
    reward_mean = float(np.mean(all_rewards))
    reward_std = float(np.std(all_rewards))
    feat_mean = np.mean(np.stack(all_feat_exp, axis=0), axis=0)

    soc_stack = np.stack(all_soc_histories, axis=0)
    min_soc = np.min(soc_stack, axis=0)
    max_soc = np.max(soc_stack, axis=0)

    # Interval Score (80% quantile interval, α=0.2)
    alpha = 0.2
    lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
    upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
    expert_resampled = _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
    width = upper_soc - lower_soc
    undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
    overshoot = np.maximum(expert_resampled - upper_soc, 0.0)
    interval_score = float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))

    print(
        f"Traj {expert_index} over {n_rollouts} rollouts: "
        f"DTW={dtw_mean:.3f}\u00b1{dtw_std:.3f}, "
        f"BestDTW={best_dtw:.3f}, "
        f"IS={interval_score:.3f}, "
        f"AccReward={reward_mean:.3f}\u00b1{reward_std:.3f}, "
        f"FeatExpMean={feat_mean}, FeatExpBest={best_feat_exp}"
    )

    if plot:
        plot_expert_trajectory(
            best_soc_history,
            expert_soc,
            expert_index,
            best_out_start,
            best_return_start,
            best_out_dur,
            best_return_dur,
            best_dtw,
            min_soc=min_soc,
            max_soc=max_soc,
            dtw_mean=dtw_mean,
            dtw_std=dtw_std,
            interval_score=interval_score,
        )

    return best_dtw, interval_score


if __name__ == "__main__":
    # --- Configuration ---
    experiment_folder = "MaxEntIRL_simple_probabilistic_exp1"
    epoch_to_load = 20
    expert_data_path = "data/processed_trajectories_simple_probabilistic.json"
    segment = "Male 50-59"
    n_rollouts = 20
    n_figures = 5              # number of example figures to display
    eval_ratio = 1.0           # fraction of segment trajectories to evaluate (1.0 = all)
    deterministic_policy = False

    reward_weights = np.array([-7.522875, -5.11008, -5.3194823, -0.85412014], dtype=np.float32)

    # Load expert data
    with open(expert_data_path, "r") as f:
        expert_data = json.load(f)

    vec_env = DummyVecEnv([lambda: gym.make('V2GEnv-simple')])

    # Load trained model
    model = PPO.load(f"./models/{experiment_folder}/maxent_irl_simple_epoch{epoch_to_load}")

    expert_indexes = find_expert_indexes(expert_data, segment)
    n_eval = max(1, int(len(expert_indexes) * eval_ratio))
    expert_indexes = expert_indexes[:n_eval]
    print(f"Found {len(expert_indexes)} trajectories for segment '{segment}' (evaluating {n_eval})")

    all_best_dtw = []
    all_is = []

    for i, idx in enumerate(expert_indexes):
        print(f"\nEvaluating trajectory {idx} ({i+1}/{len(expert_indexes)})...")
        best_dtw, interval_score = evaluate(
            vec_env,
            model,
            expert_data,
            idx,
            reward_weights=reward_weights,
            n_rollouts=n_rollouts,
            deterministic=deterministic_policy,
            plot=(i < n_figures),
        )
        all_best_dtw.append(best_dtw)
        all_is.append(interval_score)

    print(f"\n{'='*60}")
    print(f"Summary over {len(all_best_dtw)} trajectories:")
    print(f"  Avg Best DTW: {np.mean(all_best_dtw):.3f} \u00b1 {np.std(all_best_dtw):.3f}")
    print(f"  Avg IS:       {np.mean(all_is):.3f} \u00b1 {np.std(all_is):.3f}")
    print(f"{'='*60}")
    
    
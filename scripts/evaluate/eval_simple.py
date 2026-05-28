import numpy as np
import gymnasium as gym
from sbx import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.MaxEnt.MaxEnt_simple import FlattenNormalizeObsWrapper
from irl.utils.tools import compute_dtw
import json

# ── Plot style ────────────────────────────────────────────────────────────────
AGENT_COLOR  = "#0000FF"
EXPERT_COLOR = "#000000"
RANGE_COLOR  = "tab:blue"
GRID_COLOR   = "#E0E0E0"
SPINE_COLOR  = "#000000"
FIG_SIZE     = (6, 4)

plt.rcParams.update({
    "font.size":          10,
    "axes.titlesize":     9,
    "axes.titleweight":   "regular",
    "axes.labelsize":     9,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.edgecolor":     SPINE_COLOR,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.fontsize":    8,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   SPINE_COLOR,
    "figure.dpi":         150,
})
# ─────────────────────────────────────────────────────────────────────────────

gym.register(
    id='V2GEnv-simple',
    entry_point="irl.envs.V2GEnv_simple:V2GEnv",
    max_episode_steps=96,
)

FEAT_NAMES = ['charged', 'discharged', 'need_tgt', 'exceed_tgt', 'fail']


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
                           out_duration, return_duration, best_mae_val, best_mae_dtw,
                           min_soc=None, max_soc=None, interval_score=None):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(range(len(soc_history)), soc_history,
            label=f'Agent', color=AGENT_COLOR)
    if min_soc is not None and max_soc is not None:
        ax.fill_between(range(len(min_soc)), min_soc, max_soc,
                        color=RANGE_COLOR, alpha=0.2, label='Agent SoC Range')
    ax.plot(range(len(expert_soc)), expert_soc,
            label='Expert SoC', linestyle='--', color=EXPERT_COLOR)
    ax.axvspan(out_start_timestep, out_start_timestep + out_duration,
               color='red', alpha=0.2, label='Outbound Journey')
    ax.axvspan(return_start_timestep, return_start_timestep + return_duration,
               color='blue', alpha=0.15, label='Return Journey')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State of Charge (SoC)')
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend()
    title = f'Simple MaxEnt IRL \u2014 MAE={best_mae_val:.4f} | DTW={best_mae_dtw:.3f}'
    if interval_score is not None:
        title += f' | IS={interval_score:.3f}'
    title += f' \u2014 Trajectory {expert_index}'
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_expert_trajectory_rollouts(
    mean_soc, min_soc, max_soc,
    expert_soc, expert_index,
    out_start_timestep, return_start_timestep,
    out_duration, return_duration,
    dtw_mean, dtw_std,
):
    """Plot expert SoC vs rollout mean SoC with min-max range."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    timesteps = range(len(mean_soc))
    ax.plot(timesteps, mean_soc, label='Agent Mean SoC', color=AGENT_COLOR, linewidth=1.8)
    ax.fill_between(timesteps, min_soc, max_soc,
                    color=AGENT_COLOR, alpha=0.15, label='Agent SoC Range')
    ax.plot(range(len(expert_soc)), expert_soc,
            label='Expert SoC', linestyle='--', color=EXPERT_COLOR, linewidth=1.8)
    ax.axvspan(out_start_timestep, out_start_timestep + out_duration,
               color='#FF4444', alpha=0.15, label='Outbound Journey')
    ax.axvspan(return_start_timestep, return_start_timestep + return_duration,
               color='#4444FF', alpha=0.15, label='Return Journey')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('State of Charge (SoC)')
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend()
    ax.set_title(
        f'Simple MaxEnt IRL \u2014 DTW mean\u00b1std: {dtw_mean:.2f}\u00b1{dtw_std:.2f}'
        f' \u2014 Trajectory {expert_index}'
    )
    plt.tight_layout()
    plt.show()


def evaluate(vec_env, model, expert_data, expert_index, n_rollouts=10, deterministic=False, plot=True):
    """Run n evaluation episodes; plot the rollout with the lowest MAE."""
    initial_states = expert_data[expert_index]['initial_values']
    vec_env.envs[0].unwrapped.set_initial_states(initial_states)

    expert_soc = expert_data[expert_index]['soc_history']
    target_len = len(expert_soc)

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
            best_mae_dtw = dtw_distance
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

    # Interval Score (80% quantile interval, α=0.2)
    alpha = 0.2
    lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
    upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
    expert_resampled = _resample_trajectory(np.array(expert_soc, dtype=np.float32), target_len)
    width = upper_soc - lower_soc
    undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
    overshoot = np.maximum(expert_resampled - upper_soc, 0.0)
    interval_score = float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))

    expert_feat = np.array(expert_data[expert_index]['feature_expectation'], dtype=np.float32)

    print(
        f"Traj {expert_index} over {n_rollouts} rollouts: "
        f"DTW={dtw_mean:.3f}\u00b1{dtw_std:.3f}, BestDTW={best_dtw:.3f}, "
        f"MAE={mae_mean:.4f}\u00b1{mae_std:.4f}, BestMAE={best_mae_val:.4f}, "
        f"IS={interval_score:.3f}, "
        f"AccReward={reward_mean:.3f}\u00b1{reward_std:.3f}"
    )
    feat_parts = ", ".join(
        f"{n}={e:.4f}|{a:.4f}" for n, e, a in zip(FEAT_NAMES, expert_feat, feat_mean)
    )
    print(f"  Features [expert|agent]: {feat_parts}")

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
            best_mae_dtw,
            min_soc=min_soc,
            max_soc=max_soc,
            interval_score=interval_score,
        )

    return dtw_mean, interval_score, mae_mean, best_dtw, best_mae_val, all_dtw, all_mae


if __name__ == "__main__":
    # --- Configuration ---
    experiment_folder = "MaxEnt/simple/MaxEntIRL_simple_male5059"
    epoch_to_load = 10
    expert_data_path = "data/processed_trajectories_simple.json"
    segment = "Male 50-59"
    n_rollouts = 30
    n_figures = 10              # number of example figures to display (last n)
    eval_ratio = 1.0           # fraction of segment trajectories to evaluate (1.0 = all)
    plot_only = True           # if True, only evaluate the last n_figures trajectories
    deterministic_policy = False

    # Load expert data
    with open(expert_data_path, "r") as f:
        expert_data = json.load(f)

    vec_env = DummyVecEnv([lambda: FlattenNormalizeObsWrapper(gym.make('V2GEnv-simple'))])

    # Load trained model
    model = PPO.load(f"./models/{experiment_folder}/ppo_epoch{epoch_to_load}")

    expert_indexes = find_expert_indexes(expert_data, segment)
    n_eval = max(1, int(len(expert_indexes) * eval_ratio))
    expert_indexes = expert_indexes[:n_eval]
    if plot_only:
        expert_indexes = expert_indexes[-n_figures:]
    print(f"Found {len(expert_indexes)} trajectories for segment '{segment}' (evaluating {len(expert_indexes)})")

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
            plot=(plot_only or i >= len(expert_indexes) - n_figures),
        )
        all_rollout_dtw.extend(traj_dtw)
        all_rollout_mae.extend(traj_mae)
        all_is.append(is_score)

    print(f"\n{'='*60}")
    print(f"Summary over {len(expert_indexes)} trajectories x {n_rollouts} rollouts [{segment}]:")
    print(f"  Mean DTW   : {np.mean(all_rollout_dtw):.3f} \u00b1 {np.std(all_rollout_dtw):.3f}")
    print(f"  Mean MAE   : {np.mean(all_rollout_mae):.4f} \u00b1 {np.std(all_rollout_mae):.4f}")
    print(f"  Avg IS     : {np.mean(all_is):.3f} \u00b1 {np.std(all_is):.3f}")
    print(f"  Median DTW : {np.median(all_rollout_dtw):.3f}")
    print(f"  Median MAE : {np.median(all_rollout_mae):.4f}")
    print(f"{'='*60}")

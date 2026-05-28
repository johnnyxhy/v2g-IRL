import numpy as np
import matplotlib.pyplot as plt
import math
import os
import csv
import gymnasium as gym

from irl.utils.tools import compute_dtw, compute_mae, resample_trajectory

# ------------------------------------------------------------------ #
#  Utilities                                                         #
# ------------------------------------------------------------------ #

def logsumexp_feat_exp(rewards_arr, feat_exp_arr):
    """Return the importance-weighted feature expectation using LogSumExp weights."""
    max_r = np.max(rewards_arr)
    exp_w = np.exp(rewards_arr - max_r)
    return np.dot(exp_w / np.sum(exp_w), feat_exp_arr)


def compute_log_likelihood(reward_weights, expert_feat_exp, traj_feat_exp_arr):
    """
    Estimate log p(τ_expert) = R(τ_expert) - log Z, where Z is approximated
    from rollout rewards.
    """
    r_expert = float(np.dot(reward_weights, expert_feat_exp))
    r_rollouts = np.array([float(np.dot(reward_weights, fe)) for fe in traj_feat_exp_arr])
    log_Z = np.max(r_rollouts) + np.log(np.sum(np.exp(r_rollouts - np.max(r_rollouts))))
    return r_expert - log_Z


def clip_gradient(grad, clip_norm):
    """Clip gradient by global norm. Returns (clipped_grad, norm)."""
    norm = np.linalg.norm(grad)
    if clip_norm is not None and norm > clip_norm:
        grad = grad * (clip_norm / norm)
    return grad, norm


def linear_lr(lr_start, lr_end, epoch, n_epochs):
    """Linear learning rate schedule: lr_start -> lr_end over n_epochs."""
    if n_epochs > 1:
        return lr_start + (lr_end - lr_start) * (epoch / (n_epochs - 1))
    return lr_start


def append_metrics_csv(path, epoch, train_l2, train_dtw, train_mae, train_ll,
                        val_l2, val_dtw, val_mae, lr, grad_norm, has_val):
    """Append one epoch row to the metrics CSV, writing the header on first call."""
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['epoch', 'train_l2', 'train_dtw', 'train_mae', 'log_likelihood',
                             'val_l2', 'val_dtw', 'val_mae', 'lr', 'grad_norm'])
        writer.writerow([
            epoch,
            round(train_l2, 6), round(train_dtw, 6), round(train_mae, 6), round(train_ll, 6),
            round(val_l2, 6) if has_val else '',
            round(val_dtw, 6) if has_val else '',
            round(val_mae, 6) if has_val else '',
            round(lr, 8),
            round(float(grad_norm), 6),
        ])


# ------------------------------------------------------------------ #
#  Observation wrapper                                               #
# ------------------------------------------------------------------ #

class FlattenNormalizeObsWrapper(gym.ObservationWrapper):
    """
    Converts the Dict observation space to a normalized flat Box.
    Required because SBX's PPO only supports MlpPolicy (no MultiInputPolicy).
    """
    def __init__(self, env, obs_scales):
        super().__init__(env)
        self._scales = np.array(obs_scales, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(obs_scales),), dtype=np.float32
        )

    def observation(self, obs):
        raw = np.concatenate([
            np.asarray(obs['timestep']).flatten(),
            np.asarray(obs['soc']).flatten(),
            np.asarray(obs['soc_gap']).flatten(),
            np.asarray(obs['energy_price']).flatten(),
            np.asarray(obs['battery_capacity']).flatten(),
            np.asarray(obs['time_to_next_journey']).flatten(),
            np.asarray(obs['current_charger_power']).flatten(),
        ]).astype(np.float32)
        return raw / self._scales


# ------------------------------------------------------------------ #
#  Base trainer                                                      #
# ------------------------------------------------------------------ #

class MaxEntIRLTrainerBase:
    """
    Shared base for all MaxEnt IRL trainers.

    Handles dataset splitting, metric tracking, test-set evaluation,
    and reward-weights plotting. Subclasses provide train() and
    _plot_results() for their specific policy algorithm and environment.
    """

    def __init__(self, initial_reward_weights, expert_trajectories, env_name, cfg):
        self.cfg = cfg
        self.env_name = env_name
        self.expert_trajectories = expert_trajectories
        self.train_set, self.val_set, self.test_set = expert_trajectories.split_dataset(
            cfg.train_ratio, cfg.segment
        )
        self.reward_weights = initial_reward_weights.copy()

        # Tracking
        self.train_l2_loss = []
        self.train_dtw_distance = []
        self.train_mae = []
        self.train_log_likelihood = []
        self.val_l2_loss = []
        self.val_dtw_distance = []
        self.val_mae = []
        self.reward_weights_history = []

        os.makedirs(f"./models/{cfg.folder_name}", exist_ok=True)

    def _evaluate_test_set(self, rollout_env, model, n_rollouts=30):
        """
        Evaluate the learned policy on held-out test trajectories.

        Reports per-trajectory DTW, MAE, and Winkler Interval Score (IS)
        across n_rollouts stochastic rollouts per trajectory.
        """
        print(f"\n{'='*60}")
        print(f"Test Set Evaluation — {len(self.test_set)} trajectories x {n_rollouts} rollouts")
        print(f"{'='*60}")

        alpha = 0.2
        all_dtw, all_mae, all_is = [], [], []
        rollout_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)

        for i, traj in enumerate(self.test_set):
            expert_soc = np.array(traj.soc_history, dtype=np.float32)
            target_len = len(expert_soc)
            expert_resampled = resample_trajectory(expert_soc, target_len)

            traj_dtw, traj_mae, soc_histories = [], [], []

            for _ in range(n_rollouts):
                rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
                obs = rollout_env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=False)
                    obs, _, dones, infos = rollout_env.step(action)
                    done = bool(dones[0])
                agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                traj_dtw.append(compute_dtw(expert_soc, agent_soc))
                traj_mae.append(compute_mae(expert_soc, agent_soc))
                soc_histories.append(resample_trajectory(agent_soc, target_len))

            soc_stack = np.stack(soc_histories, axis=0)
            lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
            upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
            width = upper_soc - lower_soc
            undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
            overshoot = np.maximum(expert_resampled - upper_soc, 0.0)
            is_score = float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))

            print(f"  Traj {i+1}/{len(self.test_set)} (ep {traj.episodeID}): "
                  f"DTW={float(np.mean(traj_dtw)):.3f}, MAE={float(np.mean(traj_mae)):.4f}, IS={is_score:.3f}")
            all_dtw.extend(traj_dtw)
            all_mae.extend(traj_mae)
            all_is.append(is_score)

        print(f"\n--- Test Summary ({len(self.test_set)} trajectories x {n_rollouts} rollouts) ---")
        print(f"  Mean DTW : {np.mean(all_dtw):.3f} \u00b1 {np.std(all_dtw):.3f}")
        print(f"  Mean MAE : {np.mean(all_mae):.4f} \u00b1 {np.std(all_mae):.4f}")
        print(f"  Mean IS  : {np.mean(all_is):.3f} \u00b1 {np.std(all_is):.3f}")
        print(f"  Median DTW : {np.median(all_dtw):.3f}")
        print(f"  Median MAE : {np.median(all_mae):.4f}")
        print(f"{'='*60}")

    def _plot_weights_evolution(self, feature_names=None, include_weight_index=False):
        """Plot reward weight trajectories across epochs and save to CSV."""
        cfg = self.cfg
        weights_arr = np.array(self.reward_weights_history)
        n_weights = weights_arr.shape[1]
        epochs = range(1, cfg.n_epochs + 1)

        if feature_names is None:
            feature_names = [f'feature_{i+1}' for i in range(n_weights)]

        weights_csv_path = f'./models/{cfg.folder_name}/reward_weights_evolution.csv'
        with open(weights_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch'] + feature_names)
            for epoch_idx, row in enumerate(weights_arr, start=1):
                writer.writerow([epoch_idx] + [round(float(v), 6) for v in row])

        ncols = 2
        nrows = math.ceil(n_weights / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True)
        axes_flat = np.array(axes).flatten()
        for i in range(n_weights):
            axes_flat[i].plot(epochs, weights_arr[:, i], color=f'C{i}', linewidth=2)
            title = f'Weight {i+1}: {feature_names[i]}' if include_weight_index else feature_names[i]
            axes_flat[i].set_title(title)
            axes_flat[i].set_ylabel('Value')
            axes_flat[i].grid(True, linestyle='--', alpha=0.5)
        for j in range(n_weights, len(axes_flat)):
            axes_flat[j].axis('off')
        for j in range(n_weights):
            if j + ncols >= n_weights:
                axes_flat[j].set_xlabel('Epoch')
        plt.suptitle('Reward Weights Evolution', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f'./models/{cfg.folder_name}/reward_weights_evolution.png')
        plt.close()

        np.savetxt(f'./models/{cfg.folder_name}/final_reward_weights.txt',
                   self.reward_weights, fmt='%.6f')

import torch
import torch.nn as nn
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from irl.utils.tools import compute_dtw, compute_mae, resample_trajectory


# ------------------------------------------------------------------ #
#  Reward network                                                    #
# ------------------------------------------------------------------ #

# Order: [timestep, soc, soc_gap, energy_price, battery_capacity, time_to_next_journey, current_charger_power]
OBS_SCALES = np.array([96.0, 1.0, 1.0, 0.47, 2.0, 96.0, 22.0], dtype=np.float32)


class RewardNet(nn.Module):
    """
    Neural network that approximates the reward function R_θ(s, a).
    Takes normalized observation and action as input, outputs an unbounded scalar reward.

    Architecture: two hidden layers with ReLU activations, linear output.
    Unbounded output gives PPO a strong learning signal; L2 regularization (AdamW
    weight_decay) prevents reward divergence.
    """

    def __init__(self, obs_dim=7, action_dim=1, hidden_dim=32, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        """
        Args:
            obs: (batch, obs_dim) normalized observations
            action: (batch, action_dim) or (batch,) actions
        Returns:
            (batch,) scalar rewards
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


def flatten_obs_dict(obs_dict, scales=OBS_SCALES):
    """
    Flatten a Dict observation (from DummyVecEnv or raw env) to a normalized 1D numpy array.

    Args:
        obs_dict: Dict with keys matching the V2G observation space
        scales: normalization divisors for each observation dimension

    Returns:
        np.ndarray of shape (obs_dim,) with normalized values
    """
    raw = np.array([
        float(obs_dict['timestep'].flatten()[0]),
        float(obs_dict['soc'].flatten()[0]),
        float(obs_dict['soc_gap'].flatten()[0]),
        float(obs_dict['energy_price'].flatten()[0]),
        float(obs_dict['battery_capacity'].flatten()[0]),
        float(obs_dict['time_to_next_journey'].flatten()[0]),
        float(obs_dict['current_charger_power'].flatten()[0]),
    ], dtype=np.float32)
    return raw / scales

# ------------------------------------------------------------------ #
#  Base trainer                                                      #
# ------------------------------------------------------------------ #

class BaseDeepMaxEntTrainer:
    """
    Shared base for Deep MaxEnt IRL trainers (discrete and continuous).

    Handles reward network setup, optimizer/scheduler, metric tracking,
    the IRL gradient update, validation, logging, test-set evaluation,
    and plotting. Subclasses must implement _do_rollout() and train(),
    and may override _process_rollout_action() and plot_title_prefix.
    """

    plot_title_prefix: str = "Deep MaxEnt IRL"

    def __init__(self,
                 train_set: list,
                 val_set: list,
                 env_name: str,
                 cfg,
                 test_set: list = None):
        self.cfg = cfg
        self.env_name = env_name
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set if test_set is not None else []

        self.obs_scales = torch.tensor(OBS_SCALES, dtype=torch.float32)

        self.reward_net = RewardNet(
            obs_dim=cfg.reward_obs_dim,
            action_dim=cfg.reward_action_dim,
            hidden_dim=cfg.reward_hidden_dim,
        )
        if cfg.pretrained_reward_net_path is not None:
            self.reward_net.load_state_dict(
                torch.load(cfg.pretrained_reward_net_path, weights_only=True)
            )
            print(f"Loaded pretrained reward network from {cfg.pretrained_reward_net_path}")

        self.reward_optimizer = torch.optim.AdamW(
            self.reward_net.parameters(), lr=cfg.reward_lr,
            weight_decay=cfg.reward_l2_reg,
        )
        self.reward_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.reward_optimizer, T_max=cfg.n_epochs, eta_min=cfg.reward_lr_end
        )

        # Tracking
        self.train_dtw_distance = []
        self.train_mae = []
        self.train_feat_l2 = []
        self.val_dtw_distance = []
        self.val_mae = []
        self.val_feat_l2 = []
        self.train_reward_loss = []
        self.train_log_likelihood = []

        os.makedirs(f"./models/{cfg.folder_name}", exist_ok=True)

    # -------------------------------------------------------------- #
    #  IRL gradient update                                           #
    # -------------------------------------------------------------- #

    def _process_rollout_action(self, act_arr: np.ndarray) -> torch.Tensor:
        """Convert rollout actions to a tensor for reward computation. Override to normalize."""
        return torch.tensor(act_arr, dtype=torch.float32)

    def _irl_gradient_update(self, rollout_env, model):
        """
        Run one IRL gradient update over all training trajectories.

        Returns:
            (avg_dtw, avg_mae, avg_feat_l2, avg_reward_loss, avg_log_likelihood)
        """
        cfg = self.cfg
        self.reward_optimizer.zero_grad()
        N = len(self.train_set)
        avg_dtw = avg_mae = avg_feat_l2 = avg_reward_loss = avg_log_likelihood = 0.0

        for traj in tqdm(self.train_set, desc="IRL gradient"):
            expert_obs  = torch.tensor(traj.observations, dtype=torch.float32) / self.obs_scales
            expert_act  = torch.tensor(traj.actions,  dtype=torch.float32)
            expert_dt   = torch.tensor(traj.delta_ts, dtype=torch.float32)
            expert_feat = traj.feature_expectation
            expert_reward = (self.reward_net(expert_obs, expert_act) * expert_dt).sum()

            rollout_obs_all, rollout_act_all, rollout_dt_all = [], [], []

            for _ in range(cfg.rollout_samples):
                obs_list, act_list, dt_list, _, soc_hist, feat_exp = self._do_rollout(
                    rollout_env, model, traj
                )
                rollout_obs_all.append(obs_list)
                rollout_act_all.append(act_list)
                rollout_dt_all.append(dt_list)

                expert_soc = np.array(traj.soc_history, dtype=np.float32)
                agent_soc  = np.array(soc_hist, dtype=np.float32)
                avg_dtw     += compute_dtw(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                avg_mae     += compute_mae(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                avg_feat_l2 += np.linalg.norm(expert_feat - feat_exp) / (N * cfg.rollout_samples)

            # Compute log Z = logsumexp(R(τ_k)) over rollouts.
            # ∇θ logsumexp = Σ_k softmax(r_k) * ∇θ r_k, which is the correct MaxEnt
            # partition function gradient.
            rollout_rewards = []
            for k in range(cfg.rollout_samples):
                obs_t = torch.tensor(np.array(rollout_obs_all[k]), dtype=torch.float32)
                act_t = self._process_rollout_action(np.array(rollout_act_all[k]))
                dt_t  = torch.tensor(np.array(rollout_dt_all[k]),  dtype=torch.float32)
                rollout_rewards.append((self.reward_net(obs_t, act_t) * dt_t).sum())

            log_Z = torch.logsumexp(torch.stack(rollout_rewards), dim=0)
            avg_log_likelihood += (expert_reward.detach() - log_Z.detach()).item() / N
            loss_traj = -(expert_reward - log_Z) / N
            loss_traj.backward()
            avg_reward_loss += loss_traj.item()

        if cfg.reward_grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.reward_net.parameters(), max_norm=cfg.reward_grad_clip
            )
        self.reward_optimizer.step()
        self.reward_scheduler.step()

        return avg_dtw, avg_mae, avg_feat_l2, avg_reward_loss, avg_log_likelihood

    # -------------------------------------------------------------- #
    #  Validation                                                    #
    # -------------------------------------------------------------- #

    def _run_validation(self, rollout_env, model):
        """Run one validation pass. Returns (val_dtw, val_mae, val_feat_l2)."""
        cfg = self.cfg
        val_dtw = val_mae = val_feat_l2 = 0.0
        if not (cfg.validation and len(self.val_set) > 0):
            return val_dtw, val_mae, val_feat_l2
        print("Validating...")
        M = len(self.val_set)
        for traj in tqdm(self.val_set, desc="Validation"):
            _, _, _, _, soc_hist, feat_exp = self._do_rollout(rollout_env, model, traj)
            expert_soc = np.array(traj.soc_history, dtype=np.float32)
            agent_soc  = np.array(soc_hist, dtype=np.float32)
            val_dtw     += compute_dtw(expert_soc, agent_soc) / M
            val_mae     += compute_mae(expert_soc, agent_soc) / M
            val_feat_l2 += np.linalg.norm(traj.feature_expectation - feat_exp) / M
        return val_dtw, val_mae, val_feat_l2

    # -------------------------------------------------------------- #
    #  Logging                                                       #
    # -------------------------------------------------------------- #

    def _log_and_save_epoch(self, epoch, avg_dtw, avg_mae, avg_feat_l2,
                             avg_reward_loss, avg_log_likelihood,
                             val_dtw, val_mae, val_feat_l2, current_lr):
        """Append metrics to tracking lists, print epoch summary, and write CSV row."""
        cfg = self.cfg
        has_val = cfg.validation and len(self.val_set) > 0

        self.train_dtw_distance.append(avg_dtw)
        self.train_mae.append(avg_mae)
        self.train_feat_l2.append(avg_feat_l2)
        self.train_reward_loss.append(avg_reward_loss)
        self.train_log_likelihood.append(avg_log_likelihood)
        if has_val:
            self.val_dtw_distance.append(val_dtw)
            self.val_mae.append(val_mae)
            self.val_feat_l2.append(val_feat_l2)

        print(f"--- Epoch {epoch+1}/{cfg.n_epochs} Summary ---")
        print(f"Reward Loss: {avg_reward_loss:.4f}, Log-Likelihood: {avg_log_likelihood:.4f}, LR: {current_lr:.6f}")
        print(f"Train DTW: {avg_dtw:.4f}, Train MAE: {avg_mae:.4f}, Train Feat L2: {avg_feat_l2:.4f}")
        if has_val:
            print(f"Val DTW: {val_dtw:.4f}, Val MAE: {val_mae:.4f}, Val Feat L2: {val_feat_l2:.4f}")

        metrics_path = f"./models/{cfg.folder_name}/metrics.csv"
        write_header = not os.path.exists(metrics_path)
        with open(metrics_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch', 'train_dtw', 'train_mae', 'train_feat_l2', 'reward_loss',
                                 'log_likelihood', 'val_dtw', 'val_mae', 'val_feat_l2', 'lr'])
            writer.writerow([
                epoch + 1,
                round(avg_dtw, 6), round(avg_mae, 6), round(avg_feat_l2, 6),
                round(avg_reward_loss, 6), round(avg_log_likelihood, 6),
                round(val_dtw, 6) if has_val else '',
                round(val_mae, 6) if has_val else '',
                round(val_feat_l2, 6) if has_val else '',
                round(current_lr, 8),
            ])

    # -------------------------------------------------------------- #
    #  Test set evaluation                                           #
    # -------------------------------------------------------------- #

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

        for i, traj in enumerate(self.test_set):
            expert_soc = np.array(traj.soc_history, dtype=np.float32)
            target_len = len(expert_soc)
            expert_resampled = resample_trajectory(expert_soc, target_len)

            traj_dtw, traj_mae, soc_histories = [], [], []

            rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
            for _ in range(n_rollouts):
                _, _, _, _, soc_hist, _ = self._do_rollout(rollout_env, model, traj)
                agent_soc = np.array(soc_hist, dtype=np.float32)
                traj_dtw.append(compute_dtw(expert_soc, agent_soc))
                traj_mae.append(compute_mae(expert_soc, agent_soc))
                soc_histories.append(resample_trajectory(agent_soc, target_len))

            soc_stack = np.stack(soc_histories, axis=0)
            lower_soc = np.percentile(soc_stack, 100 * alpha / 2, axis=0).astype(np.float32)
            upper_soc = np.percentile(soc_stack, 100 * (1 - alpha / 2), axis=0).astype(np.float32)
            width      = upper_soc - lower_soc
            undershoot = np.maximum(lower_soc - expert_resampled, 0.0)
            overshoot  = np.maximum(expert_resampled - upper_soc, 0.0)
            is_score   = float(np.mean(width + (2.0 / alpha) * (undershoot + overshoot)))

            dtw_mean = float(np.mean(traj_dtw))
            mae_mean = float(np.mean(traj_mae))
            print(f"  Traj {i+1}/{len(self.test_set)} (ep {traj.episodeID}): "
                  f"DTW={dtw_mean:.3f}, MAE={mae_mean:.4f}, IS={is_score:.3f}")

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

    # -------------------------------------------------------------- #
    #  Plotting                                                      #
    # -------------------------------------------------------------- #

    def _plot_results(self):
        cfg = self.cfg
        pfx = self.plot_title_prefix
        epochs = range(1, cfg.n_epochs + 1)

        plt.figure(1)
        plt.plot(epochs, self.train_dtw_distance, label='Train DTW')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_dtw_distance, label='Val DTW')
            plt.legend()
        plt.title(f'{pfx} — DTW Distance')
        plt.xlabel('Epoch'); plt.ylabel('Average DTW Distance'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/dtw_distance.png')

        plt.figure(2)
        plt.plot(epochs, self.train_mae, label='Train MAE')
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae, label='Val MAE')
            plt.legend()
        plt.title(f'{pfx} — SoC MAE')
        plt.xlabel('Epoch'); plt.ylabel('Average MAE'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/mae.png')

        plt.figure(3)
        plt.plot(epochs, self.train_feat_l2, label='Train Feature L2')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_feat_l2, label='Val Feature L2')
            plt.legend()
        plt.title(f'{pfx} — Feature L2 Loss')
        plt.xlabel('Epoch'); plt.ylabel('Average Feature L2'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/feature_l2.png')

        plt.figure(4)
        plt.plot(epochs, self.train_reward_loss)
        plt.title(f'{pfx} — Reward Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss (neg log-likelihood)'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/reward_loss.png')

        plt.figure(5)
        plt.plot(epochs, self.train_log_likelihood)
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Converged (LL=0)')
        plt.title(f'{pfx} — Expert Log-Likelihood')
        plt.xlabel('Epoch'); plt.ylabel('Avg log p(τ_expert)'); plt.grid()
        plt.legend()
        plt.savefig(f'./models/{cfg.folder_name}/log_likelihood.png')

        plt.show()

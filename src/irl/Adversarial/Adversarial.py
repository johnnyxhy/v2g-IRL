import numpy as np
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym

from irl.utils.tools import compute_dtw, compute_mae, resample_trajectory

# Order: [timestep, soc, soc_gap, energy_price, battery_capacity, time_to_next_journey, current_charger_power]
OBS_SCALES = np.array([96.0, 1.0, 1.0, 0.47, 2.0, 96.0, 22.0], dtype=np.float32)

# ------------------------------------------------------------------ #
#  Observation wrapper: flatten + normalize Dict obs for MlpPolicy   #
# ------------------------------------------------------------------ #

class FlattenNormalizeObsWrapper(gym.ObservationWrapper):
    """
    Converts the Dict observation space to a normalized flat Box.
    Normalization uses OBS_SCALES (shared with the env and discriminator).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_dim = len(OBS_SCALES)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._scales = np.array(OBS_SCALES, dtype=np.float32)

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
#  Reward network g_φ(s)                                            #
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Shaping network h_φ(s)                                            #
# ------------------------------------------------------------------ #

class ShapingNet(nn.Module):
    """
    Neural network approximating the AIRL potential function h_φ(s).

    Takes only the (normalized) observation as input and outputs an unbounded
    scalar.  At convergence the shaping term γ^Δt · h_φ(s') − h_φ(s) cancels
    out of the recovered reward g_θ(s,a), leaving g_θ = r* + const.
    """

    def __init__(self, obs_dim: int = 7, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# ------------------------------------------------------------------ #
#  Base Adversarial IRL Trainer                                      #
# ------------------------------------------------------------------ #

class BaseAdversarialTrainer:
    """
    Shared base for Adversarial IRL trainers (discrete and continuous).

    Handles discriminator network setup, optimizer/scheduler, metric tracking,
    _compute_f, test-set evaluation, and plotting.  Subclasses must implement
    train(), _do_rollout(), _get_log_prob(), and _bc_pretrain().
    """

    plot_title_prefix: str = "Adversarial IRL"

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
        self.device = torch.device("cpu")

        # ---- Networks ----
        self.reward_net = RewardNet(
            obs_dim=cfg.reward_obs_dim,
            action_dim=cfg.reward_action_dim,
            hidden_dim=cfg.reward_hidden_dim,
        )
        self.shaping_net = ShapingNet(
            obs_dim=cfg.reward_obs_dim,
            hidden_dim=cfg.shaping_hidden_dim,
        )

        if cfg.pretrained_reward_net_path is not None:
            self.reward_net.load_state_dict(
                torch.load(cfg.pretrained_reward_net_path, weights_only=True)
            )
            print(f"Loaded pretrained reward net from {cfg.pretrained_reward_net_path}")

        if cfg.pretrained_shaping_net_path is not None:
            self.shaping_net.load_state_dict(
                torch.load(cfg.pretrained_shaping_net_path, weights_only=True)
            )
            print(f"Loaded pretrained shaping net from {cfg.pretrained_shaping_net_path}")

        # ---- Single optimiser for both discriminator networks ----
        self.disc_optimizer = torch.optim.AdamW(
            list(self.reward_net.parameters()) + list(self.shaping_net.parameters()),
            lr=cfg.disc_lr,
            weight_decay=cfg.disc_l2_reg,
        )
        self.disc_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.disc_optimizer, T_max=cfg.n_epochs, eta_min=cfg.disc_lr_end
        )

        # ---- Tracking ----
        self.train_dtw_distance = []
        self.train_mae = []
        self.train_feat_l2 = []
        self.val_dtw_distance = []
        self.val_mae = []
        self.val_feat_l2 = []
        self.train_disc_loss = []
        self.train_expert_acc = []
        self.train_policy_acc = []

        os.makedirs(f"./models/{cfg.folder_name}", exist_ok=True)
        print(
            f"Adversarial Trainer: {len(self.train_set)} train, "
            f"{len(self.val_set)} val, {len(self.test_set)} test trajectories."
        )

    # -------------------------------------------------------------- #
    #  Discriminator helpers                                         #
    # -------------------------------------------------------------- #

    def _compute_f(
        self,
        obs: torch.Tensor,
        act_norm: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        delta_ts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the AIRL reward shaping function:

            f(s, a, s', Δt, done)
                = g_θ(s, a)
                + γ^Δt · (1 − done) · h_φ(s')
                − h_φ(s)
        """
        if act_norm.dim() == 1:
            act_norm = act_norm.unsqueeze(-1)

        g    = self.reward_net(obs, act_norm)
        h_s  = self.shaping_net(obs)
        h_sp = self.shaping_net(next_obs)
        gamma_dt = self.cfg.policy_gamma ** delta_ts

        return g + gamma_dt * (1.0 - dones) * h_sp - h_s

    # -------------------------------------------------------------- #
    #  Test set evaluation                                             #
    # -------------------------------------------------------------- #

    def _evaluate_test_set(self, rollout_env, model, n_rollouts=30):
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
                result = self._do_rollout(rollout_env, model, traj)
                soc_hist = result[-2]  # soc_history is second-to-last in both subclass return tuples
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
            print(
                f"  Traj {i+1}/{len(self.test_set)} (ep {traj.episodeID}): "
                f"DTW={dtw_mean:.3f}, MAE={mae_mean:.4f}, IS={is_score:.3f}"
            )

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
    #  Plotting                                                        #
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
        plt.title(f'{pfx} - DTW Distance')
        plt.xlabel('Epoch'); plt.ylabel('Average DTW Distance'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/dtw_distance.png')
        plt.close()

        plt.figure(2)
        plt.plot(epochs, self.train_mae, label='Train MAE')
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae, label='Val MAE')
            plt.legend()
        plt.title(f'{pfx} - SoC MAE')
        plt.xlabel('Epoch'); plt.ylabel('Average MAE'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/mae.png')
        plt.close()

        plt.figure(3)
        plt.plot(epochs, self.train_feat_l2, label='Train Feature L2')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_feat_l2, label='Val Feature L2')
            plt.legend()
        plt.title(f'{pfx} - Feature L2')
        plt.xlabel('Epoch'); plt.ylabel('Average Feature L2'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/feature_l2.png')
        plt.close()

        plt.figure(4)
        plt.plot(epochs, self.train_disc_loss)
        plt.title(f'{pfx} - Discriminator BCE Loss')
        plt.xlabel('Epoch'); plt.ylabel('Disc Loss'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/disc_loss.png')
        plt.close()

        plt.figure(5)
        plt.plot(epochs, self.train_expert_acc, label='Expert Acc (D→1)')
        plt.plot(epochs, self.train_policy_acc, label='Policy Acc (D→0)')
        plt.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='Chance (0.5)')
        plt.title(f'{pfx} -Discriminator Accuracy')
        plt.xlabel('Epoch'); plt.ylabel('Fraction correctly classified'); plt.grid()
        plt.legend()
        plt.savefig(f'./models/{cfg.folder_name}/disc_accuracy.png')
        plt.close()

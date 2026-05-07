import numpy as np
import json
import csv
import gymnasium as gym
from dataclasses import dataclass, field
from sbx import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, flatten_obs_dict, PROFIT_OBS_SCALES
from irl.utils.tools import compute_dtw
from irl.utils.variable_dt_buffer import VariableDtReplayBuffer
from tqdm import tqdm
import os

import torch
import jax.numpy as jnp


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

class DeepMaxEntConfig:
    """Configuration for Deep MaxEnt IRL training."""
    device: str = 'cuda'

    # IRL outer loop
    n_epochs: int = 10
    reward_lr: float = 1e-3           # Adam LR for reward network
    reward_lr_end: float = 1e-4       # Final LR (cosine annealing)
    rollout_samples: int = 20

    # SAC inner loop
    policy_train_steps_per_iter: int = 5_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 0.99
    policy_batch_size: int = 64

    # Data
    train_ratio: float = 0.8
    segment: str = None

    # Reward network architecture
    reward_hidden_dim: int = 64
    reward_obs_dim: int = 7
    reward_action_dim: int = 1

    # Warm-start from a previous run
    pretrained_reward_net_path: str = None   # e.g. "models/exp1/reward_net_epoch10.pt"

    # Action magnitude penalty
    action_penalty_coeff: float = 0.0   # λ for -λ·a² in env reward

    # Reward net regularization
    reward_grad_clip: float = 5.0
    reward_l2_reg: float = 0.01

    # Saving
    folder_name: str = "DeepMaxEntIRL_profit"
    validation: bool = False


# ------------------------------------------------------------------ #
#  Expert data loading                                                 #
# ------------------------------------------------------------------ #

@dataclass
class DeepExpertTrajectory:
    episodeID: int
    segment: str
    initial_values: dict
    soc_history: list
    observations: np.ndarray      # (n_actions, obs_dim)  raw, unnormalized
    actions: np.ndarray           # (n_actions, 1)
    delta_ts: np.ndarray          # (n_actions,) number of env timesteps per action
    feature_expectation: np.ndarray   # (n_features,) for monitoring


def load_deep_expert_data(json_path, segment=None, train_ratio=0.8):
    """
    Load expert trajectories from the deep-format JSON file.

    Returns:
        (train_set, val_set): lists of DeepExpertTrajectory
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    trajectories = []
    for traj in data:
        sap = traj['state_action_pairs']
        n_actions = len(sap['actions'])
        expert = DeepExpertTrajectory(
            episodeID=traj['episodeID'],
            segment=traj['segment'],
            initial_values=traj['initial_values'],
            soc_history=traj['soc_history'],
            observations=np.array(sap['observations'], dtype=np.float32),
            actions=np.array(sap['actions'], dtype=np.float32),
            delta_ts=np.array(sap['delta_ts'], dtype=np.float32) if 'delta_ts' in sap
                     else np.ones(n_actions, dtype=np.float32),
            feature_expectation=np.array(traj['feature_expectation'], dtype=np.float32),
        )
        trajectories.append(expert)

    if segment is not None:
        trajectories = [t for t in trajectories if segment in t.segment]

    n_train = int(len(trajectories) * train_ratio)
    return trajectories[:n_train], trajectories[n_train:]


# ------------------------------------------------------------------ #
#  Trainer                                                             #
# ------------------------------------------------------------------ #

class DeepMaxEntIRLTrainer:
    """
    Deep Maximum Entropy Inverse Reinforcement Learning trainer.

    Replaces the linear reward w·φ with a neural-network reward R_θ(s,a).
    The gradient is:
        ∇θ L = E_expert[∇θ R_θ(s,a)] − E_π[∇θ R_θ(s,a)]
    where the policy term is estimated via importance-weighted rollouts.
    """

    def __init__(self,
                 train_set: list,
                 val_set: list,
                 env_name: str,
                 cfg: DeepMaxEntConfig):
        self.cfg = cfg
        self.env_name = env_name
        self.train_set = train_set
        self.val_set = val_set

        # Observation normalization scales
        self.obs_scales = torch.tensor(PROFIT_OBS_SCALES, dtype=torch.float32)

        # Build reward network
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

        print(f"Deep MaxEnt IRL Trainer: {len(self.train_set)} train, {len(self.val_set)} val trajectories.")

    # -------------------------------------------------------------- #
    #  Main training loop                                              #
    # -------------------------------------------------------------- #

    def train(self):
        cfg = self.cfg

        # Training env
        env = gym.make(self.env_name)
        env = Monitor(env, filename=f"./models/{cfg.folder_name}/monitor.csv")
        vec_env = DummyVecEnv([lambda: env])

        # Rollout env
        rollout_env = gym.make(self.env_name)
        rollout_env = DummyVecEnv([lambda: rollout_env])

        # SAC policy
        model = SAC(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=0,
            learning_rate=cfg.policy_train_lr,
            gamma=cfg.policy_gamma,
            device=cfg.device,
            batch_size=cfg.policy_batch_size,
            seed=42,
            tensorboard_log=f"./models/{cfg.folder_name}/tensorboard/",
            replay_buffer_class=VariableDtReplayBuffer,
            replay_buffer_kwargs={'base_gamma': cfg.policy_gamma},
            buffer_size=100_000,
            learning_starts=1_000,
            ent_coef='auto',
            train_freq=1,
            gradient_steps=1,
        )

        for epoch in range(cfg.n_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs}: Starting training...")

            # ---------- 1. SAC policy training ----------
            # Set reward net in training & rollout envs
            vec_env.envs[0].unwrapped.set_reward_net(self.reward_net)
            vec_env.envs[0].unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
            vec_env.envs[0].unwrapped.set_initial_states(None)
            rollout_env.envs[0].unwrapped.set_reward_net(self.reward_net)
            rollout_env.envs[0].unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)

            # Reset SAC entropy coefficient (log(0.1) ≈ -2.3026 for more exploration)
            if hasattr(model, 'ent_coef_state'):
                model.ent_coef_state = model.ent_coef_state.replace(
                    params={'log_ent_coef': jnp.array(-2.3026)}
                )

            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=False,
                log_interval=500,
            )

            # ---------- 2. IRL gradient update ----------
            self.reward_optimizer.zero_grad()
            N = len(self.train_set)

            avg_dtw = 0.0
            avg_mae = 0.0
            avg_feat_l2 = 0.0
            avg_reward_loss = 0.0
            avg_log_likelihood = 0.0

            for traj in tqdm(self.train_set, desc="IRL gradient"):
                # --- Expert state-action pairs (normalized) ---
                expert_obs = torch.tensor(traj.observations, dtype=torch.float32) / self.obs_scales
                expert_act = torch.tensor(traj.actions, dtype=torch.float32)
                expert_feat = traj.feature_expectation

                # Expert reward: R_net(s,a) weighted by delta_t (matches SAC's cumulative reward)
                expert_dt = torch.tensor(traj.delta_ts, dtype=torch.float32)
                expert_reward = (self.reward_net(expert_obs, expert_act) * expert_dt).sum()

                # --- Agent rollouts ---
                rollout_obs_all = []
                rollout_act_all = []
                rollout_env_rewards = []

                rollout_dt_all = []

                for _ in range(cfg.rollout_samples):
                    obs_list, act_list, dt_list, env_reward, soc_hist, feat_exp = self._do_rollout(
                        rollout_env, model, traj
                    )
                    rollout_obs_all.append(obs_list)
                    rollout_act_all.append(act_list)
                    rollout_dt_all.append(dt_list)
                    rollout_env_rewards.append(env_reward)

                    # DTW / MAE monitoring
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(soc_hist, dtype=np.float32)
                    avg_dtw += compute_dtw(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                    avg_mae += self._compute_mae(expert_soc, agent_soc) / (N * cfg.rollout_samples)

                    # Feature L2 monitoring
                    avg_feat_l2 += np.linalg.norm(expert_feat - feat_exp) / (N * cfg.rollout_samples)

                # Compute log Z = logsumexp(R(τ_k)) over rollouts.
                # ∇θ logsumexp = Σ_k softmax(r_k) * ∇θ r_k, which is the correct MaxEnt
                # partition function gradient. Using logsumexp directly also gives the
                # correct loss value: loss = -(R(τ_expert) - log Z).
                rollout_rewards = []
                for k in range(cfg.rollout_samples):
                    obs_t = torch.tensor(np.array(rollout_obs_all[k]), dtype=torch.float32)
                    act_t = torch.tensor(np.array(rollout_act_all[k]), dtype=torch.float32)
                    dt_t  = torch.tensor(np.array(rollout_dt_all[k]),  dtype=torch.float32)
                    r_k = (self.reward_net(obs_t, act_t) * dt_t).sum()
                    rollout_rewards.append(r_k)

                # log Z: differentiable; gradient = softmax-weighted ∇θ R(τ_k)
                log_Z = torch.logsumexp(torch.stack(rollout_rewards), dim=0)

                # Log-likelihood: R(τ_expert) - log Z, where Z is estimated from rollouts only.
                # Can be positive when expert reward exceeds all rollout rewards.
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

            current_lr = self.reward_scheduler.get_last_lr()[0]

            # ---------- 3. Validation ----------
            val_dtw = 0.0
            val_mae = 0.0
            val_feat_l2 = 0.0
            if cfg.validation and len(self.val_set) > 0:
                print("Validating...")
                M = len(self.val_set)
                for traj in tqdm(self.val_set, desc="Validation"):
                    _, _, _, _, soc_hist, feat_exp = self._do_rollout(
                        rollout_env, model, traj, deterministic=False
                    )
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(soc_hist, dtype=np.float32)
                    val_dtw += compute_dtw(expert_soc, agent_soc) / M
                    val_mae += self._compute_mae(expert_soc, agent_soc) / M
                    val_feat_l2 += np.linalg.norm(traj.feature_expectation - feat_exp) / M

            # ---------- 4. Logging ----------
            self.train_dtw_distance.append(avg_dtw)
            self.train_mae.append(avg_mae)
            self.train_feat_l2.append(avg_feat_l2)
            self.train_reward_loss.append(avg_reward_loss)
            self.train_log_likelihood.append(avg_log_likelihood)
            if cfg.validation and len(self.val_set) > 0:
                self.val_dtw_distance.append(val_dtw)
                self.val_mae.append(val_mae)
                self.val_feat_l2.append(val_feat_l2)

            print(f"--- Epoch {epoch+1}/{cfg.n_epochs} Summary ---")
            print(f"Reward Loss: {avg_reward_loss:.4f}, Log-Likelihood: {avg_log_likelihood:.4f}, LR: {current_lr:.6f}")
            print(f"Train DTW: {avg_dtw:.4f}, Train MAE: {avg_mae:.4f}, Train Feat L2: {avg_feat_l2:.4f}")
            if cfg.validation and len(self.val_set) > 0:
                print(f"Val DTW: {val_dtw:.4f}, Val MAE: {val_mae:.4f}, Val Feat L2: {val_feat_l2:.4f}")

            # Save metrics to CSV (append — survives checkpoint continuations)
            metrics_path = f"./models/{cfg.folder_name}/metrics.csv"
            write_header = not os.path.exists(metrics_path)
            with open(metrics_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(['epoch', 'train_dtw', 'train_mae', 'train_feat_l2', 'reward_loss',
                                     'log_likelihood', 'val_dtw', 'val_mae', 'val_feat_l2', 'lr'])
                writer.writerow([
                    epoch + 1,
                    round(avg_dtw, 6),
                    round(avg_mae, 6),
                    round(avg_feat_l2, 6),
                    round(avg_reward_loss, 6),
                    round(avg_log_likelihood, 6),
                    round(val_dtw, 6) if (cfg.validation and len(self.val_set) > 0) else '',
                    round(val_mae, 6) if (cfg.validation and len(self.val_set) > 0) else '',
                    round(val_feat_l2, 6) if (cfg.validation and len(self.val_set) > 0) else '',
                    round(current_lr, 8),
                ])

            model.save(f"./models/{cfg.folder_name}/sac_epoch{epoch+1}")
            torch.save(self.reward_net.state_dict(), f"./models/{cfg.folder_name}/reward_net_epoch{epoch+1}.pt")

        self.__plot_results()
        print("Training completed.")

    # -------------------------------------------------------------- #
    #  Helpers                                                         #
    # -------------------------------------------------------------- #

    @staticmethod
    def _compute_mae(soc_a: np.ndarray, soc_b: np.ndarray) -> float:
        """MAE between two SoC trajectories, resampling to the longer length."""
        n = max(len(soc_a), len(soc_b))
        if len(soc_a) != n:
            soc_a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_a)), soc_a)
        if len(soc_b) != n:
            soc_b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_b)), soc_b)
        return float(np.mean(np.abs(soc_a - soc_b)))

    def _do_rollout(self, rollout_env, model, traj, deterministic=False):
        """
        Run one episode from the expert's initial state.

        Returns:
            obs_list:   list of normalized obs arrays (T, obs_dim)
            act_list:   list of action arrays (T, action_dim)
            traj_reward: scalar total env reward
            soc_history: list of SoC values
            feature_expectation: np.ndarray
        """
        rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
        obs = rollout_env.reset()

        obs_list = []
        act_list = []
        dt_list = []
        traj_reward = 0.0
        done = False

        while not done:
            obs_flat = flatten_obs_dict(obs)  # already normalized
            action, _ = model.predict(obs, deterministic=deterministic)

            obs_list.append(obs_flat)
            act_list.append(action[0].copy())  # (1,) array

            obs, reward, dones, infos = rollout_env.step(action)
            traj_reward += reward[0]
            dt_list.append(infos[0]['delta_t'])

            done = bool(dones[0])

        soc_history = infos[0]['soc_history']
        feature_expectation = np.array(infos[0]['feature_expectation'], dtype=np.float32)

        return obs_list, act_list, dt_list, traj_reward, soc_history, feature_expectation

    # -------------------------------------------------------------- #
    #  Plotting                                                        #
    # -------------------------------------------------------------- #

    def __plot_results(self):
        cfg = self.cfg
        epochs = range(1, cfg.n_epochs + 1)

        # DTW distance
        plt.figure(1)
        plt.plot(epochs, self.train_dtw_distance, label='Train DTW')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_dtw_distance, label='Val DTW')
            plt.legend()
        plt.title('Deep MaxEnt IRL — DTW Distance')
        plt.xlabel('Epoch')
        plt.ylabel('Average DTW Distance')
        plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/dtw_distance.png')

        # MAE
        plt.figure(2)
        plt.plot(epochs, self.train_mae, label='Train MAE')
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae, label='Val MAE')
            plt.legend()
        plt.title('Deep MaxEnt IRL — SoC MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Average MAE')
        plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/mae.png')

        # Feature L2
        plt.figure(3)
        plt.plot(epochs, self.train_feat_l2, label='Train Feature L2')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_feat_l2, label='Val Feature L2')
            plt.legend()
        plt.title('Deep MaxEnt IRL — Feature L2 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average Feature L2')
        plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/feature_l2.png')

        # Reward loss
        plt.figure(4)
        plt.plot(epochs, self.train_reward_loss)
        plt.title('Deep MaxEnt IRL — Reward Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (neg log-likelihood)')
        plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/reward_loss.png')

        # Log-likelihood
        plt.figure(5)
        plt.plot(epochs, self.train_log_likelihood)
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Converged (LL=0)')
        plt.title('Deep MaxEnt IRL — Expert Log-Likelihood')
        plt.xlabel('Epoch')
        plt.ylabel('Avg log p(τ_expert)')
        plt.grid()
        plt.legend()
        plt.savefig(f'./models/{cfg.folder_name}/log_likelihood.png')

        plt.show()

import numpy as np
import json
import csv
import gymnasium as gym
from dataclasses import dataclass
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.utils.tools import compute_dtw
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import torch


# ------------------------------------------------------------------ #
#  Observation wrapper: flatten + normalize Dict obs for MlpPolicy    #
# ------------------------------------------------------------------ #

class FlattenNormalizeObsWrapper(gym.ObservationWrapper):
    """
    Converts the Dict observation space to a normalized flat Box.
    Required because SBX's PPO only supports MlpPolicy (no MultiInputPolicy).
    The normalization matches PROFIT_OBS_SCALES (shared with the discrete env).
    """
    def __init__(self, env):
        super().__init__(env)
        obs_dim = len(PROFIT_OBS_SCALES)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._scales = np.array(PROFIT_OBS_SCALES, dtype=np.float32)

    def observation(self, obs):
        raw = np.concatenate([
            np.asarray(obs['timestep']).flatten(),
            np.asarray(obs['soc']).flatten(),
            np.asarray(obs['soc_target']).flatten(),
            np.asarray(obs['energy_price']).flatten(),
            np.asarray(obs['battery_capacity']).flatten(),
            np.asarray(obs['time_to_next_journey']).flatten(),
            np.asarray(obs['current_charger_power']).flatten(),
        ]).astype(np.float32)
        return raw / self._scales


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

class DeepMaxEntDiscreteConfig:
    """Configuration for Deep MaxEnt IRL training with PPO inner loop (discrete env)."""

    # IRL outer loop
    n_epochs: int = 10
    reward_lr: float = 1e-3
    reward_lr_end: float = 1e-4
    rollout_samples: int = 10

    # PPO inner loop (SBX — does not support clip_range / gae_lambda / device)
    policy_train_steps_per_iter: int = 100_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 1.0
    policy_batch_size: int = 64
    policy_n_steps: int = 2048
    policy_n_epochs: int = 10
    policy_ent_coef: float = 0.01

    # Data
    train_ratio: float = 0.8
    segment: str = None

    # Reward network architecture
    reward_hidden_dim: int = 64
    reward_obs_dim: int = 7
    reward_action_dim: int = 1

    # Warm-start from a previous run
    pretrained_reward_net_path: str = None

    # Action magnitude penalty
    action_penalty_coeff: float = 0.0

    # Reward scaling for PPO (does not affect IRL gradient)
    reward_scale: float = 1.0

    # Number of parallel envs in DummyVecEnv for PPO training.
    # Profiling shows 4 is the sweet spot (1.82x speedup); 8 gives no further gain.
    n_envs: int = 4

    # Recreate PPO model from scratch at the start of every epoch.
    # Clears Adam momentum (which accumulated for the old reward landscape) so PPO
    # gets a fresh optimisation start each time the reward function changes.
    reset_ppo_each_epoch: bool = True

    # Reward net regularization
    reward_grad_clip: float = 5.0
    reward_l2_reg: float = 0.01

    # Saving
    folder_name: str = "DeepMaxEntIRL_discrete"
    validation: bool = False


# ------------------------------------------------------------------ #
#  Expert data loading                                                 #
# ------------------------------------------------------------------ #

@dataclass
class DeepDiscreteExpertTrajectory:
    episodeID: int
    segment: str
    initial_values: dict
    soc_history: list
    observations: np.ndarray      # (n_actions, obs_dim) raw, unnormalized
    actions: np.ndarray           # (n_actions, 1) normalized to [-1, 1]: (discrete-10)/10
    delta_ts: np.ndarray          # (n_actions,) number of env timesteps per action
    feature_expectation: np.ndarray   # (n_features,)


def load_deep_discrete_expert_data(json_path, segment=None, train_ratio=0.8):
    with open(json_path, 'r') as f:
        data = json.load(f)

    trajectories = []
    for traj in data:
        expert = DeepDiscreteExpertTrajectory(
            episodeID=traj['episodeID'],
            segment=traj['segment'],
            initial_values=traj['initial_values'],
            soc_history=traj['soc_history'],
            observations=np.array(traj['state_action_pairs']['observations'], dtype=np.float32),
            actions=np.array(traj['state_action_pairs']['actions'], dtype=np.float32),
            delta_ts=np.array(traj['state_action_pairs']['delta_ts'], dtype=np.float32),
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

class DeepMaxEntDiscreteTrainer:
    """
    Deep Maximum Entropy IRL with PPO inner-loop for V2GDeepEnv_discrete.

    The reward is a neural network R_θ(s, a) scoring (normalized_obs, action).
    The IRL gradient maximises the likelihood of expert trajectories under the
    MaxEnt distribution:

        ∇θ L = E_expert[∇θ R_θ(s,a)] − E_π[softmax-weighted ∇θ R_θ(s,a)]

    Key differences from the profit SAC variant:
    - PPO inner loop: on-policy, no replay buffer / entropy reset needed.
    - SBX PPO: MlpPolicy only; obs wrapper flattens + normalizes the Dict space.
    """

    def __init__(self,
                 train_set: list,
                 val_set: list,
                 env_name: str,
                 cfg: DeepMaxEntDiscreteConfig):
        self.cfg = cfg
        self.env_name = env_name
        self.train_set = train_set
        self.val_set = val_set

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

        print(f"Deep MaxEnt Discrete Trainer: {len(self.train_set)} train, {len(self.val_set)} val trajectories.")

    # -------------------------------------------------------------- #
    #  Main training loop                                              #
    # -------------------------------------------------------------- #

    def train(self):
        cfg = self.cfg

        # Create about.md description file
        about_path = f"./models/{cfg.folder_name}/about.md"
        if not os.path.exists(about_path):
            with open(about_path, 'w') as f:
                f.write(cfg.description)

        # Training env — N parallel envs wrapped to flatten+normalize obs for MlpPolicy
        def _make_train_env():
            e = gym.make(self.env_name)
            return FlattenNormalizeObsWrapper(e)

        vec_env = DummyVecEnv([_make_train_env for _ in range(cfg.n_envs)])
        vec_env = VecMonitor(vec_env, filename=f"./models/{cfg.folder_name}/monitor")

        # Separate single env for IRL rollouts — also wrapped so model.predict works
        rollout_env = gym.make(self.env_name)
        rollout_env = FlattenNormalizeObsWrapper(rollout_env)
        rollout_env = DummyVecEnv([lambda: rollout_env])

        def _make_ppo(epoch_idx: int):
            """Create a fresh PPO model. Called once per epoch when reset_ppo_each_epoch=True."""
            return PPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=0,
                learning_rate=cfg.policy_train_lr,
                gamma=cfg.policy_gamma,
                batch_size=cfg.policy_batch_size,
                n_steps=cfg.policy_n_steps,
                n_epochs=cfg.policy_n_epochs,
                ent_coef=cfg.policy_ent_coef,
                seed=42 + epoch_idx,
                tensorboard_log=f"./models/{cfg.folder_name}/tensorboard/",
            )

        # SBX PPO — MlpPolicy only; clip_range/gae_lambda/device not supported by SBX
        model = _make_ppo(0)

        for epoch in range(cfg.n_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs}: Starting training...")

            # ---------- 1. PPO policy training ----------
            if cfg.reset_ppo_each_epoch and epoch > 0:
                model = _make_ppo(epoch)

            for e in vec_env.unwrapped.envs:
                e.unwrapped.set_reward_net(self.reward_net)
                e.unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
                e.unwrapped.set_reward_scale(cfg.reward_scale)
                e.unwrapped.set_initial_states(None)
            rollout_env.envs[0].unwrapped.set_reward_net(self.reward_net)
            rollout_env.envs[0].unwrapped.set_action_penalty_coeff(cfg.action_penalty_coeff)
            rollout_env.envs[0].unwrapped.set_reward_scale(cfg.reward_scale)

            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=cfg.reset_ppo_each_epoch,
                log_interval=10,
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
                # Expert state-action pairs:
                #   - obs: normalized by PROFIT_OBS_SCALES
                #   - act: already normalized to [-1, 1] by the expert loader
                #   - delta_t: number of env timesteps this action consumed
                expert_obs = torch.tensor(traj.observations, dtype=torch.float32) / self.obs_scales
                expert_act = torch.tensor(traj.actions, dtype=torch.float32)  # already [-1, 1]
                expert_dt  = torch.tensor(traj.delta_ts,   dtype=torch.float32)  # (n_actions,)
                expert_feat = traj.feature_expectation

                # Expert reward: R_net(s,a) weighted by delta_t (matches PPO's cumulative reward)
                expert_reward = (self.reward_net(expert_obs, expert_act) * expert_dt).sum()

                # Agent rollouts
                rollout_obs_all = []
                rollout_act_all = []
                rollout_dt_all  = []

                for _ in range(cfg.rollout_samples):
                    obs_list, act_list, dt_list, _, soc_hist, feat_exp = self._do_rollout(
                        rollout_env, model, traj
                    )
                    rollout_obs_all.append(obs_list)
                    rollout_act_all.append(act_list)
                    rollout_dt_all.append(dt_list)

                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(soc_hist, dtype=np.float32)
                    avg_dtw += compute_dtw(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                    avg_mae += self._compute_mae(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                    avg_feat_l2 += np.linalg.norm(expert_feat - feat_exp) / (N * cfg.rollout_samples)

                # Compute log Z = logsumexp(R(τ_k)) over rollouts.
                # ∇θ logsumexp = Σ_k softmax(r_k) * ∇θ r_k, which is the correct MaxEnt
                # partition function gradient. Using logsumexp directly (rather than a
                # detached-softmax weighted sum) also gives the correct loss *value*:
                # loss = -(R(τ_expert) - log Z).
                rollout_rewards = []
                for k in range(cfg.rollout_samples):
                    obs_t = torch.tensor(np.array(rollout_obs_all[k]), dtype=torch.float32)
                    # Rollout acts are raw discrete 0-20; normalize to [-1, 1] to match expert
                    act_t = (torch.tensor(np.array(rollout_act_all[k]), dtype=torch.float32) - 10.0) / 10.0
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
                    _, _, _, _, soc_hist, feat_exp = self._do_rollout(rollout_env, model, traj)
                    expert_soc_v = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc_v = np.array(soc_hist, dtype=np.float32)
                    val_dtw += compute_dtw(expert_soc_v, agent_soc_v) / M
                    val_mae += self._compute_mae(expert_soc_v, agent_soc_v) / M
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

            # ---------- 5. Save metrics to CSV (append — survives checkpoint continuations) ----------
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

            model.save(f"./models/{cfg.folder_name}/ppo_epoch{epoch+1}")
            torch.save(self.reward_net.state_dict(), f"./models/{cfg.folder_name}/reward_net_epoch{epoch+1}.pt")

        self._plot_results()
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
        rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
        obs = rollout_env.reset()

        obs_list = []
        act_list = []
        dt_list  = []
        traj_reward = 0.0

        while True:
            # obs is (1, obs_dim) from DummyVecEnv, already normalized by wrapper
            obs_flat = obs[0].copy()
            action, _ = model.predict(obs, deterministic=deterministic)

            obs_list.append(obs_flat)
            act_list.append(action[0].copy())

            obs, reward, dones, infos = rollout_env.step(action)
            traj_reward += reward[0]
            dt_list.append(infos[0]['delta_t'])

            if bool(dones[0]):
                break

        soc_history = infos[0]['soc_history']
        feature_expectation = np.array(infos[0]['feature_expectation'], dtype=np.float32)
        return obs_list, act_list, dt_list, traj_reward, soc_history, feature_expectation

    # -------------------------------------------------------------- #
    #  Plotting                                                        #
    # -------------------------------------------------------------- #

    def _plot_results(self):
        cfg = self.cfg
        epochs = range(1, cfg.n_epochs + 1)

        plt.figure(1)
        plt.plot(epochs, self.train_dtw_distance, label='Train DTW')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_dtw_distance, label='Val DTW')
            plt.legend()
        plt.title('Deep MaxEnt Discrete IRL — DTW Distance')
        plt.xlabel('Epoch'); plt.ylabel('Average DTW Distance'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/dtw_distance.png')

        plt.figure(2)
        plt.plot(epochs, self.train_mae, label='Train MAE')
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae, label='Val MAE')
            plt.legend()
        plt.title('Deep MaxEnt Discrete IRL — SoC MAE')
        plt.xlabel('Epoch'); plt.ylabel('Average MAE'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/mae.png')

        plt.figure(3)
        plt.plot(epochs, self.train_feat_l2, label='Train Feature L2')
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_feat_l2, label='Val Feature L2')
            plt.legend()
        plt.title('Deep MaxEnt Discrete IRL — Feature L2 Loss')
        plt.xlabel('Epoch'); plt.ylabel('Average Feature L2'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/feature_l2.png')

        plt.figure(4)
        plt.plot(epochs, self.train_reward_loss)
        plt.title('Deep MaxEnt Discrete IRL — Reward Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss (neg log-likelihood)'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/reward_loss.png')

        plt.figure(5)
        plt.plot(epochs, self.train_log_likelihood)
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Converged (LL=0)')
        plt.title('Deep MaxEnt Discrete IRL — Expert Log-Likelihood')
        plt.xlabel('Epoch'); plt.ylabel('Avg log p(τ_expert)'); plt.grid()
        plt.legend()
        plt.savefig(f'./models/{cfg.folder_name}/log_likelihood.png')


        plt.show()

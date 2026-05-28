import numpy as np
import json
import gymnasium as gym
from dataclasses import dataclass
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from irl.DeepMaxEnt.DeepMaxEnt import BaseDeepMaxEntTrainer, flatten_obs_dict
from irl.utils.variable_dt_buffer import VariableDtReplayBuffer
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


def load_deep_expert_data(json_path, segment=None, train_ratio=0.8, val_ratio=0.1):
    """
    Load expert trajectories from the deep-format JSON file.

    Returns:
        (train_set, val_set, test_set): lists of DeepExpertTrajectory
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
    n_val = int(len(trajectories) * val_ratio)
    return trajectories[:n_train], trajectories[n_train:n_train + n_val], trajectories[n_train + n_val:]


# ------------------------------------------------------------------ #
#  Trainer                                                             #
# ------------------------------------------------------------------ #

class DeepMaxEntIRLTrainer(BaseDeepMaxEntTrainer):
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
                 cfg: DeepMaxEntConfig,
                 test_set: list = None):
        super().__init__(train_set, val_set, env_name, cfg, test_set)
        print(f"Deep MaxEnt IRL Trainer: {len(self.train_set)} train, {len(self.val_set)} val, {len(self.test_set)} test trajectories.")

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
            avg_dtw, avg_mae, avg_feat_l2, avg_reward_loss, avg_log_likelihood = \
                self._irl_gradient_update(rollout_env, model)

            # ---------- 3. Validation ----------
            val_dtw, val_mae, val_feat_l2 = self._run_validation(rollout_env, model)

            # ---------- 4-5. Logging + CSV ----------
            current_lr = self.reward_scheduler.get_last_lr()[0]
            self._log_and_save_epoch(epoch, avg_dtw, avg_mae, avg_feat_l2,
                                     avg_reward_loss, avg_log_likelihood,
                                     val_dtw, val_mae, val_feat_l2, current_lr)

            model.save(f"./models/{cfg.folder_name}/sac_epoch{epoch+1}")
            torch.save(self.reward_net.state_dict(), f"./models/{cfg.folder_name}/reward_net_epoch{epoch+1}.pt")

        self._plot_results()

        if self.test_set:
            self._evaluate_test_set(rollout_env, model, n_rollouts=30)

        print("Training completed.")

    # -------------------------------------------------------------- #
    #  Helpers                                                         #
    # -------------------------------------------------------------- #

    def _do_rollout(self, rollout_env, model, traj, deterministic=False):
        rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
        obs = rollout_env.reset()

        obs_list, act_list, dt_list = [], [], []
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

import numpy as np
import gymnasium as gym
from sbx import SAC
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from irl.dataset.expert_dataset import ExpertDataset
from irl.utils.tools import compute_dtw, compute_mae
from irl.utils.variable_dt_buffer import VariableDtReplayBuffer
from tqdm import tqdm
import jax.numpy as jnp

from irl.MaxEnt.MaxEnt import (
    MaxEntIRLTrainerBase,
    logsumexp_feat_exp, compute_log_likelihood,
    clip_gradient, linear_lr, append_metrics_csv,
)


# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #

class MaxEntConfig:
    """Configuration class for MaxEnt IRL (continuous env)."""

    device: str = 'cpu'

    # IRL outer loop
    n_epochs: int = 10
    reward_lr: float = 0.01
    reward_lr_end: float = 0.0        # Final reward LR for linear decay (0 = decay to zero)
    rollout_samples: int = 20
    grad_clip_norm: float = 5.0       # Max gradient norm for clipping (None to disable)

    # SAC inner loop (SBX)
    policy_train_steps_per_iter: int = 5_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 0.99
    policy_batch_size: int = 64

    # Data
    train_ratio: float = 0.8    
    segment: str = None

    # Saving
    folder_name: str = "MaxEntIRL_continuous"
    validation: bool = False



# ------------------------------------------------------------------ #
#  Trainer                                                           #
# ------------------------------------------------------------------ #

class MaxEntIRLTrainer_Continuous(MaxEntIRLTrainerBase):
    """
    MaxEnt IRL with SAC as the inner-loop policy optimizer for the
    continuous V2GEnv_continuous environment.

    The reward is linear in hand-crafted features:
        R(s, a) = w · φ(s, a)

    The IRL gradient is:
        ∇w = (1/N) Σ_i [ φ_expert^(i) − Σ_j softmax(R(τ_j)) · φ(τ_j) ]
    """

    def __init__(self,
                 initial_reward_weights: np.ndarray,
                 expert_trajectories: ExpertDataset,
                 env_name: str,
                 cfg: MaxEntConfig):
        super().__init__(initial_reward_weights, expert_trajectories, env_name, cfg)
        print(f"MaxEnt IRL Trainer initialized with {len(self.train_set)} training, "
              f"{len(self.val_set)} validation, {len(self.test_set)} test samples.")

    # -------------------------------------------------------------- #
    #  Main training loop                                            #
    # -------------------------------------------------------------- #

    def train(self):
        """Train the MaxEnt IRL model."""
        cfg = self.cfg

        # Create training environment
        env = gym.make(self.env_name)
        env = Monitor(env, filename=f"./models/{cfg.folder_name}/monitor.csv")
        vec_env = DummyVecEnv([lambda: env])

        # Create rollout env
        rollout_base = gym.make(self.env_name)
        rollout_env = DummyVecEnv([lambda: rollout_base])

        model = SAC(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=0,
            learning_rate=cfg.policy_train_lr,
            gamma=cfg.policy_gamma,
            device=cfg.device,
            batch_size=cfg.policy_batch_size,
            tensorboard_log=f"./models/{cfg.folder_name}/tensorboard/",

            # --- Custom Buffer ---
            replay_buffer_class=VariableDtReplayBuffer,
            replay_buffer_kwargs={'base_gamma': cfg.policy_gamma},

            # --- SAC Specific Parameters ---
            buffer_size=100_000,
            learning_starts=1_000,
            ent_coef='auto',          # Learned entropy coefficient, reset each epoch
            train_freq=1,
            gradient_steps=1,
        )

        for epoch in range(cfg.n_epochs):
            print(f"Epoch {epoch+1}/{cfg.n_epochs}: Starting training iteration...")

            # --- LOGGING ----
            avg_expert_feat_exp = np.zeros_like(self.reward_weights)
            avg_traj_feat_exp = np.zeros_like(self.reward_weights)
            avg_episode_length = 0.0
            avg_dtw_distance = 0.0
            avg_mae = 0.0

            # --- TRAINING ----

            # Train SAC policy with current reward weights
            vec_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)
            vec_env.envs[0].unwrapped.set_initial_states(None)

            # Reset learned entropy coefficient so auto-tuning restarts fresh each epoch
            if hasattr(model, 'ent_coef_state'):
                model.ent_coef_state = model.ent_coef_state.replace(
                    params={'log_ent_coef': jnp.array(-4.6052)}
                )

            # Clear replay buffer so SAC only trains on rewards from current weights
            model.replay_buffer.reset()
            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=False,
                log_interval=500,
            )

            # Loop through each expert trajectory
            grad = np.zeros_like(self.reward_weights)
            average_l2_loss = 0.0
            avg_log_likelihood = 0.0
            N = len(self.train_set)

            for traj in tqdm(self.train_set, desc="Processing Expert Trajectories"):
                expert_feat_exp = np.array(traj.feature_expectation, dtype=np.float32)
                n_features = expert_feat_exp.shape[0]

                traj_feat_exp_arr = np.zeros((cfg.rollout_samples, n_features), dtype=np.float32)
                rewards_arr = np.zeros(cfg.rollout_samples, dtype=np.float32)

                # Perform rollouts with current policy and expert initial states
                for i in range(cfg.rollout_samples):
                    rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
                    rollout_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)
                    obs = rollout_env.reset()
                    done = False
                    traj_reward = 0.0
                    episode_length = 0
                    while not done:
                        action, _ = model.predict(obs, deterministic=False)
                        obs, reward, dones, infos = rollout_env.step(action)
                        traj_reward += reward[0]
                        episode_length += 1
                        done = bool(dones[0])

                    traj_feat_exp_arr[i] = np.array(infos[0]['feature_expectation'], dtype=np.float32)
                    rewards_arr[i] = traj_reward
                    avg_episode_length += episode_length / (N * cfg.rollout_samples)

                    # Compute DTW / MAE distance for monitoring
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                    avg_dtw_distance += compute_dtw(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                    avg_mae += compute_mae(expert_soc, agent_soc) / (N * cfg.rollout_samples)

                # Use LogSumExp to compute importance-weighted feature expectations
                traj_feat_exp = logsumexp_feat_exp(rewards_arr, traj_feat_exp_arr)

                # Log-likelihood: R(τ_expert) - log Z, where Z is estimated from rollouts only.
                # Can be positive when expert reward exceeds all rollout rewards.
                avg_log_likelihood += compute_log_likelihood(
                    self.reward_weights, expert_feat_exp, traj_feat_exp_arr
                ) / N

                # Update gradient
                grad += (expert_feat_exp - traj_feat_exp) / N

                # Update averages for monitoring
                avg_expert_feat_exp += expert_feat_exp / N
                avg_traj_feat_exp += traj_feat_exp / N

                # Calculate L2 loss for monitoring
                average_l2_loss += np.linalg.norm(expert_feat_exp - traj_feat_exp) / N

            # Update reward weights
            grad, grad_norm = clip_gradient(grad, cfg.grad_clip_norm)
            print(f"Gradient before update: {grad} (norm: {grad_norm:.4f}, clipped: {grad_norm > (cfg.grad_clip_norm or np.inf)})")

            # Linear LR decay: lr(t) = lr_start + (lr_end - lr_start) * (epoch / (n_epochs - 1))
            current_lr = linear_lr(cfg.reward_lr, cfg.reward_lr_end, epoch, cfg.n_epochs)
            self.reward_weights = self.reward_weights + current_lr * grad
            self.reward_weights_history.append(self.reward_weights.copy())
            print(f"Reward LR: {current_lr:.6f}")

            # --- VALIDATION ----
            val_loss = val_dtw_distance = val_mae = 0.0
            if cfg.validation and len(self.val_set) > 0:
                print("Starting validation...")
                M = len(self.val_set)
                for traj in tqdm(self.val_set, desc="Validating Expert Trajectories"):
                    rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
                    obs = rollout_env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, dones, infos = rollout_env.step(action)
                        done = bool(dones[0])
                    traj_feat_exp = np.array(infos[0]['feature_expectation'], dtype=np.float32)
                    expert_feat_exp = np.array(traj.feature_expectation, dtype=np.float32)
                    val_loss += np.linalg.norm(expert_feat_exp - traj_feat_exp) / M
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                    val_dtw_distance += compute_dtw(expert_soc, agent_soc) / M
                    val_mae += compute_mae(expert_soc, agent_soc) / M

            # --- LOGGING ----
            self.train_l2_loss.append(average_l2_loss)
            self.train_dtw_distance.append(avg_dtw_distance)
            self.train_mae.append(avg_mae)
            self.train_log_likelihood.append(avg_log_likelihood)
            has_val = cfg.validation and len(self.val_set) > 0
            if has_val:
                self.val_l2_loss.append(val_loss)
                self.val_dtw_distance.append(val_dtw_distance)
                self.val_mae.append(val_mae)

            print(f"--- Epoch {epoch+1}/{cfg.n_epochs} Summary ---")
            print(f"Avg L2 Loss: {average_l2_loss:.4f}, Avg DTW Distance: {avg_dtw_distance:.4f}, "
                  f"Avg MAE: {avg_mae:.4f}, Log-Likelihood: {avg_log_likelihood:.4f}")
            print(f"Avg Expert Feature Expectation: {avg_expert_feat_exp}")
            print(f"Avg Traj   Feature Expectation: {avg_traj_feat_exp}")
            print(f"Updated Reward Weights: {self.reward_weights}")
            print(f"Avg Episode Length: {avg_episode_length:.2f}")
            if has_val:
                print(f"Validation Avg L2 Loss: {val_loss:.4f}, Validation Avg DTW Distance: {val_dtw_distance:.4f}, "
                      f"Validation Avg MAE: {val_mae:.4f}")

            append_metrics_csv(
                f"./models/{cfg.folder_name}/metrics.csv",
                epoch + 1,
                average_l2_loss, avg_dtw_distance, avg_mae, avg_log_likelihood,
                val_loss, val_dtw_distance, val_mae,
                current_lr, grad_norm, has_val,
            )

            model.save(f"./models/{cfg.folder_name}/maxent_irl_epoch{epoch+1}")

        self._plot_results()

        if self.test_set:
            self._evaluate_test_set(rollout_env, model, n_rollouts=30)

        print("Training completed")

    # -------------------------------------------------------------- #
    #  Plotting                                                      #
    # -------------------------------------------------------------- #

    def _plot_results(self):
        """Plot training results and reward weights evolution."""
        cfg = self.cfg
        epochs = range(1, cfg.n_epochs + 1)

        plt.figure(1)
        plt.plot(epochs, self.train_l2_loss)
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_l2_loss)
            plt.legend(['Train L2 Loss', 'Validation L2 Loss'])
        plt.title('MaxEnt IRL (Continuous) - L2 Loss')
        plt.xlabel('Epoch'); plt.ylabel('Average L2 Loss'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/maxent_irl_training_loss.png')

        plt.figure(2)
        plt.plot(epochs, self.train_mae)
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae)
            plt.legend(['Train MAE', 'Validation MAE'])
        plt.title('MaxEnt IRL (Continuous) - SoC MAE')
        plt.xlabel('Epoch'); plt.ylabel('Average MAE'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/maxent_irl_mae.png')

        plt.figure(3)
        plt.plot(epochs, self.train_dtw_distance)
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_dtw_distance)
            plt.legend(['Train DTW Distance', 'Validation DTW Distance'])
        plt.title('MaxEnt IRL (Continuous) - DTW Distance')
        plt.xlabel('Epoch'); plt.ylabel('Average DTW Distance'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/maxent_irl_training_dtw_distance.png')

        plt.figure(4)
        plt.plot(epochs, self.train_log_likelihood)
        plt.title('MaxEnt IRL (Continuous) - Expert Log-Likelihood')
        plt.xlabel('Epoch'); plt.ylabel('Avg log p(τ_expert)'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/maxent_irl_log_likelihood.png')

        self._plot_weights_evolution()

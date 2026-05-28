import numpy as np
import gymnasium as gym
from sbx import PPO
import matplotlib.pyplot as plt
import os
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from irl.dataset.expert_dataset import ExpertDataset
from irl.utils.tools import compute_dtw, compute_mae
from tqdm import tqdm

from irl.MaxEnt.MaxEnt import (
    FlattenNormalizeObsWrapper, MaxEntIRLTrainerBase,
    logsumexp_feat_exp, compute_log_likelihood,
    clip_gradient, linear_lr, append_metrics_csv,
)

# Normalization divisors matching the 7 obs keys:
# timestep(96), soc(1), soc_gap(1), energy_price(0.47),
# battery_capacity(2), time_to_next_journey(96), current_charger_power(22)
OBS_SCALES = [96, 1, 1, 0.47, 2, 96, 22]

FEATURE_NAMES = [
    'amount_charged',
    'amount_discharged',
    'charge_price_quality',
    'discharge_price_quality',
    'soc_below_target',
    'soc_above_target',
    'journey_failure',
]


# ------------------------------------------------------------------ #
#  Configuration                                                     #
# ------------------------------------------------------------------ #

class MaxEntConfig:
    """Configuration for MaxEnt IRL (discrete env)."""

    device: str = 'cpu'

    # IRL outer loop
    n_epochs: int = 10
    reward_lr: float = 0.1            # Initial reward weight learning rate
    reward_lr_end: float = 0.01       # Final reward LR (linear decay)
    rollout_samples: int = 20
    grad_clip_norm: float = 5.0       # Max gradient norm (None to disable)

    # PPO inner loop (SBX)
    policy_train_steps_per_iter: int = 50_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 0.99
    policy_batch_size: int = 64
    policy_n_steps: int = 2048        # Steps per PPO rollout collection
    policy_n_epochs: int = 10         # PPO gradient epochs per update
    policy_ent_coef: float = 0.01     # Entropy bonus (fixed; no auto-tune)
    policy_clip_range: float = 0.2
    policy_gae_lambda: float = 0.95

    # Data
    train_ratio: float = 0.8
    segment: str = None

    # Regularisation
    reg_lambda: float = 0.01          # L2 regularisation on reward weights

    # Saving
    folder_name: str = "MaxEntIRL_discrete_PPO"
    validation: bool = False
    reset_ppo_each_epoch: bool = False
    description: str = ""
    reward_feature_names = None


# ------------------------------------------------------------------ #
#  Trainer                                                           #
# ------------------------------------------------------------------ #

class MaxEntIRLTrainer_Discrete_PPO(MaxEntIRLTrainerBase):
    """
    MaxEnt IRL with PPO as the inner-loop policy optimizer for the
    discrete V2GEnv_discrete environment.

    The reward is linear in hand-crafted features:
        R(s, a) = w · φ(s, a)

    The IRL gradient is:
        ∇w = (1/N) Σ_i [ φ_expert^(i) − Σ_j softmax(R(τ_j)) · φ(τ_j) ]

    Feature set (7 features):
        amount_charged, amount_discharged,
        charge_price_quality, discharge_price_quality,
        soc_below_target, soc_above_target, journey_failure
    """

    def __init__(self,
                 initial_reward_weights: np.ndarray,
                 expert_trajectories: ExpertDataset,
                 env_name: str,
                 cfg: MaxEntConfig):
        super().__init__(initial_reward_weights, expert_trajectories, env_name, cfg)

        # Compute per-feature scale from expert data for gradient normalisation.
        # Divide each gradient component by the std of that feature across expert trajectories.
        all_feat = np.array([t.feature_expectation for t in self.train_set], dtype=np.float32)
        feat_std = np.std(all_feat, axis=0)
        feat_std[feat_std < 1e-6] = 1.0          # avoid division by zero
        self.feat_scale = feat_std

        print(f"MaxEnt IRL (Discrete PPO) Trainer: "
              f"{len(self.train_set)} train, {len(self.val_set)} val, {len(self.test_set)} test trajectories.")

    # -------------------------------------------------------------- #
    #  Main training loop                                            #
    # -------------------------------------------------------------- #

    def train(self):
        cfg = self.cfg

        # Write about.md description file
        about_path = f"./models/{cfg.folder_name}/about.md"
        if not os.path.exists(about_path):
            with open(about_path, 'w') as f:
                f.write(cfg.description)

        # Training env — wrapped to flatten+normalize obs for SBX MlpPolicy
        train_base_env = gym.make(self.env_name)
        train_base_env = Monitor(train_base_env, filename=f"./models/{cfg.folder_name}/monitor.csv")
        train_wrapped = FlattenNormalizeObsWrapper(train_base_env, OBS_SCALES)
        vec_env = DummyVecEnv([lambda: train_wrapped])

        # Rollout env (separate so we can set initial states per trajectory)
        rollout_base_env = gym.make(self.env_name)
        rollout_wrapped = FlattenNormalizeObsWrapper(rollout_base_env, OBS_SCALES)
        rollout_env = DummyVecEnv([lambda: rollout_wrapped])

        # PPO policy — SBX PPO only supports MlpPolicy
        def _make_ppo(epoch_idx):
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

        model = _make_ppo(0)

        for epoch in range(cfg.n_epochs):
            print(f"\nEpoch {epoch+1}/{cfg.n_epochs}: Starting training iteration...")

            # ------ 1. PPO policy training ------
            if cfg.reset_ppo_each_epoch and epoch > 0:
                model = _make_ppo(epoch)

            vec_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)
            vec_env.envs[0].unwrapped.set_initial_states(None)

            model.learn(
                total_timesteps=cfg.policy_train_steps_per_iter,
                tb_log_name=f"epoch_{epoch+1}",
                progress_bar=True,
                reset_num_timesteps=cfg.reset_ppo_each_epoch,
                log_interval=10,
            )

            # ------ 2. IRL gradient update ------
            grad = np.zeros_like(self.reward_weights)
            average_l2_loss = 0.0
            avg_expert_feat_exp = np.zeros_like(self.reward_weights)
            avg_traj_feat_exp = np.zeros_like(self.reward_weights)
            avg_dtw_distance = 0.0
            avg_mae = 0.0
            avg_episode_length = 0.0
            avg_log_likelihood = 0.0
            N = len(self.train_set)

            for traj in tqdm(self.train_set, desc="IRL gradient"):
                expert_feat_exp = np.array(traj.feature_expectation, dtype=np.float32)
                n_features = expert_feat_exp.shape[0]

                traj_feat_exp_arr = np.zeros((cfg.rollout_samples, n_features), dtype=np.float32)
                rewards_arr = np.zeros(cfg.rollout_samples, dtype=np.float32)

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

                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                    avg_dtw_distance += compute_dtw(expert_soc, agent_soc) / (N * cfg.rollout_samples)
                    avg_mae += compute_mae(expert_soc, agent_soc) / (N * cfg.rollout_samples)

                traj_feat_exp = logsumexp_feat_exp(rewards_arr, traj_feat_exp_arr)
                avg_log_likelihood += compute_log_likelihood(
                    self.reward_weights, expert_feat_exp, traj_feat_exp_arr
                ) / N

                grad += (expert_feat_exp - traj_feat_exp) / N
                avg_expert_feat_exp += expert_feat_exp / N
                avg_traj_feat_exp += traj_feat_exp / N
                average_l2_loss += np.linalg.norm(expert_feat_exp - traj_feat_exp) / N

            # Per-feature normalisation by expert feature std
            grad = grad / self.feat_scale
            grad, grad_norm = clip_gradient(grad, cfg.grad_clip_norm)
            current_lr = linear_lr(cfg.reward_lr, cfg.reward_lr_end, epoch, cfg.n_epochs)

            # Gradient ascent on reward weights (with L2 regularisation)
            self.reward_weights = self.reward_weights + current_lr * (
                grad - cfg.reg_lambda * self.reward_weights
            )
            self.reward_weights_history.append(self.reward_weights.copy())

            # ------ 3. Validation ------
            val_loss = val_dtw_distance = val_mae = 0.0
            if cfg.validation and len(self.val_set) > 0:
                print("Validating...")
                M = len(self.val_set)
                for traj in tqdm(self.val_set, desc="Validation"):
                    rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
                    rollout_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)
                    obs = rollout_env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=False)
                        obs, _, dones, infos = rollout_env.step(action)
                        done = bool(dones[0])
                    val_feat_exp = np.array(infos[0]['feature_expectation'], dtype=np.float32)
                    expert_feat_exp = np.array(traj.feature_expectation, dtype=np.float32)
                    val_loss += np.linalg.norm(expert_feat_exp - val_feat_exp) / M
                    val_dtw_distance += compute_dtw(
                        np.array(traj.soc_history, dtype=np.float32),
                        np.array(infos[0]['soc_history'], dtype=np.float32),
                    ) / M
                    val_mae += compute_mae(
                        np.array(traj.soc_history, dtype=np.float32),
                        np.array(infos[0]['soc_history'], dtype=np.float32),
                    ) / M

            # ------ 4. Logging ------
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
            print(f"Reward LR: {current_lr:.6f}, Gradient norm: {grad_norm:.4f}")
            print(f"Train L2: {average_l2_loss:.4f}, Train DTW: {avg_dtw_distance:.4f}, "
                  f"Train MAE: {avg_mae:.4f}, Log-Likelihood: {avg_log_likelihood:.4f}")
            print(f"Expert feat exp: {avg_expert_feat_exp}")
            print(f"Agent  feat exp: {avg_traj_feat_exp}")
            print(f"Updated weights: {self.reward_weights}")
            print(f"Avg episode length: {avg_episode_length:.2f}")
            if has_val:
                print(f"Val L2: {val_loss:.4f}, Val DTW: {val_dtw_distance:.4f}, Val MAE: {val_mae:.4f}")

            append_metrics_csv(
                f"./models/{cfg.folder_name}/metrics.csv",
                epoch + 1,
                average_l2_loss, avg_dtw_distance, avg_mae, avg_log_likelihood,
                val_loss, val_dtw_distance, val_mae,
                current_lr, grad_norm, has_val,
            )

            model.save(f"./models/{cfg.folder_name}/ppo_epoch{epoch+1}")

        self._plot_results()

        if self.test_set:
            self._evaluate_test_set(rollout_env, model, n_rollouts=30)

        print("Training completed.")

    # -------------------------------------------------------------- #
    #  Plotting                                                      #
    # -------------------------------------------------------------- #

    def _plot_results(self):
        cfg = self.cfg
        epochs = range(1, cfg.n_epochs + 1)

        plt.figure(1)
        plt.plot(epochs, self.train_l2_loss)
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_l2_loss)
            plt.legend(['Train L2 Loss', 'Validation L2 Loss'])
        plt.title('MaxEnt IRL (Discrete) - L2 Loss')
        plt.xlabel('Epoch'); plt.ylabel('Average L2 Loss'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/l2_loss.png')

        plt.figure(2)
        plt.plot(epochs, self.train_mae)
        if cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(epochs, self.val_mae)
            plt.legend(['Train MAE', 'Validation MAE'])
        plt.title('MaxEnt IRL (Discrete) - SoC MAE')
        plt.xlabel('Epoch'); plt.ylabel('Average MAE'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/mae.png')

        plt.figure(3)
        plt.plot(epochs, self.train_dtw_distance)
        if cfg.validation and len(self.val_set) > 0:
            plt.plot(epochs, self.val_dtw_distance)
            plt.legend(['Train DTW', 'Validation DTW'])
        plt.title('MaxEnt IRL (Discrete) - DTW Distance')
        plt.xlabel('Epoch'); plt.ylabel('Average DTW Distance'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/dtw_distance.png')

        plt.figure(4)
        plt.plot(epochs, self.train_log_likelihood)
        plt.title('MaxEnt IRL (Discrete) - Expert Log-Likelihood')
        plt.xlabel('Epoch'); plt.ylabel('Avg log p(τ_expert)'); plt.grid()
        plt.savefig(f'./models/{cfg.folder_name}/log_likelihood.png')

        n_weights = len(self.reward_weights)
        if cfg.reward_feature_names is not None and len(cfg.reward_feature_names) == n_weights:
            feature_names = list(cfg.reward_feature_names)
        elif n_weights <= len(FEATURE_NAMES):
            feature_names = FEATURE_NAMES[:n_weights]
        else:
            feature_names = FEATURE_NAMES + [f'feature_{i+1}' for i in range(len(FEATURE_NAMES), n_weights)]
        self._plot_weights_evolution(feature_names, include_weight_index=True)

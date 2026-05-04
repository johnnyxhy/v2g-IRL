import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from irl.dataset.expert_dataset_simple import ExpertDatasetSimple
from irl.utils.tools import AdamOptimizer, compute_dtw
from tqdm import tqdm


class MaxEntConfig:
    """
    Configuration class for MaxEnt IRL
    """
    device: str = 'cuda'
    n_epochs: int = 10
    reward_lr: float = 0.01
    rollout_samples: int = 20
    policy_train_steps_per_iter: int = 5_000
    policy_train_lr: float = 3e-4
    policy_gamma: float = 0.99
    policy_batch_size: int = 64
    train_ratio: float = 0.8
    segment: str = None
    folder_name: str = "MaxEntIRL_simple"
    validation: bool = False

class MaxEntIRLTrainer_Simple:
    """
    Docstring for MaxEntSimple
    """

    def __init__(self, 
                 initial_reward_weights: np.ndarray,
                 expert_trajectories: ExpertDatasetSimple,
                 env_name: str,
                 cfg: MaxEntConfig, 
                 ):
        self.cfg = cfg
        self.env_name = env_name
        self.expert_trajectories = expert_trajectories
        self.train_set, self.val_set = expert_trajectories.split_dataset(cfg.train_ratio, cfg.segment)
        
        self.reward_weights = initial_reward_weights

        # For tracking
        self.train_l2_loss = []
        self.train_dtw_distance = []
        self.train_mae = []
        self.train_log_likelihood = []
        self.val_l2_loss = []
        self.val_dtw_distance = []
        self.val_mae = []

        # Register optimiser
        self.optimizer = AdamOptimizer(params_shape=self.reward_weights.shape, lr=self.cfg.reward_lr)

        print(f"MaxEnt IRL Trainer initialized with {len(self.train_set)} training samples and {len(self.val_set)} validation samples.")


    def train(self):
        """
        Train the MaxEnt IRL model
        """

        # Create training environment
        env = gym.make(self.env_name)
        env = Monitor(env, filename=f"./models/{self.cfg.folder_name}/monitor.csv")
        vec_env = DummyVecEnv([lambda: env])

        # create rollout env 
        rollout_env = gym.make(self.env_name)
        rollout_env = DummyVecEnv([lambda: rollout_env])

        # Add Monitor

        model = PPO(
                policy="MultiInputPolicy",
                env=vec_env,
                verbose=0,
                learning_rate=self.cfg.policy_train_lr,
                gamma=self.cfg.policy_gamma,
                device=self.cfg.device,
                n_steps=2048,
                batch_size=self.cfg.policy_batch_size,
                tensorboard_log=f"./models/{self.cfg.folder_name}/tensorboard/",
            )

        for epoch in range(self.cfg.n_epochs):

            print(f"Epoch {epoch+1}/{self.cfg.n_epochs}: Starting training iteration...")
            
            # --- LOGGING ----

            # Track average expert feature and trajectory feature expectations
            avg_expert_feat_exp = np.zeros_like(self.reward_weights)
            avg_traj_feat_exp = np.zeros_like(self.reward_weights)

            # Track average episode length
            avg_episode_length = 0.0

            # Track training average DTW distance
            avg_dtw_distance = 0.0
            avg_mae = 0.0

            # --- TRAINING ----

            # Train PPO policy with current reward weights
            
            vec_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)
            vec_env.envs[0].unwrapped.set_initial_states(None)

            model.learn(total_timesteps=self.cfg.policy_train_steps_per_iter,tb_log_name=f"epoch_{epoch+1}", progress_bar=True)

            # Loop through each expert trajectory 
            grad = np.zeros_like(self.reward_weights)
            average_l2_loss = 0.0
            N = len(self.train_set)
            avg_log_likelihood = 0.0

            for traj in tqdm(self.train_set, desc="Processing Expert Trajectories"):
                # Compute feature expectations from expert trajectory
                expert_feat_exp = np.array(traj.feature_expectation, dtype=np.float32)
                traj_feat_exp = np.zeros_like(self.reward_weights)

                n_samples = self.cfg.rollout_samples
                n_features = expert_feat_exp.shape[0]

                traj_feat_exp_arr =  np.zeros((n_samples, n_features), dtype=np.float32)
                rewards_arr = np.zeros(n_samples, dtype=np.float32)

                # Perform rollouts with current policy and expert initial states
                for i in range(self.cfg.rollout_samples):
                    rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
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

                    avg_episode_length += (episode_length) / (N * self.cfg.rollout_samples)

                    # Compute DTW distance for monitoring
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                    dtw_distance = compute_dtw(expert_soc, agent_soc)
                    avg_dtw_distance += dtw_distance / (N * self.cfg.rollout_samples)
                    avg_mae += self._compute_mae(expert_soc, agent_soc) / (N * self.cfg.rollout_samples)
                
                # Use LogSumExp to compute expected feature expectations
                max_reward = np.max(rewards_arr)
                exp_weights = np.exp(rewards_arr - max_reward)
                weights = exp_weights / np.sum(exp_weights)

                traj_feat_exp = np.dot(weights, traj_feat_exp_arr)

                # Log-likelihood: R(τ_expert) - log Z, where Z is estimated from rollouts only.
                # Can be positive when expert reward exceeds all rollout rewards.
                r_expert_step = float(np.dot(self.reward_weights, expert_feat_exp))
                r_rollouts_step = np.array([float(np.dot(self.reward_weights, traj_feat_exp_arr[i]))
                                            for i in range(self.cfg.rollout_samples)])
                log_Z = np.max(r_rollouts_step) + np.log(np.sum(np.exp(r_rollouts_step - np.max(r_rollouts_step))))
                avg_log_likelihood += (r_expert_step - log_Z) / N
                
                # Update gradient
                grad += (expert_feat_exp - traj_feat_exp)

                # Update averages for monitoring
                avg_expert_feat_exp += expert_feat_exp / N
                avg_traj_feat_exp += traj_feat_exp / N

                # Calculate L2 loss for monitoring
                l2_loss = np.linalg.norm(expert_feat_exp - traj_feat_exp)
                average_l2_loss += l2_loss / N  
            
            # Update reward weights
            self.reward_weights = self.optimizer.step(self.reward_weights, grad / N)

            # --- VALIDATION ----

            # Perform validation (optional)
            if self.cfg.validation and len(self.val_set) > 0:

                print("Starting validation...")

                val_loss = 0.0
                val_dtw_distance = 0.0
                val_mae = 0.0
                M = len(self.val_set)

                for traj in tqdm(self.val_set, desc="Validating Expert Trajectories"):
                    # Compute feature expectations from expert trajectory
                    expert_feat_exp = np.array(traj.feature_expectation, dtype=np.float32)
                    traj_feat_exp = np.zeros_like(self.reward_weights)

                    n_samples = self.cfg.rollout_samples
                    n_features = expert_feat_exp.shape[0]

                    traj_feat_exp_arr =  np.zeros((n_samples, n_features), dtype=np.float32)
                    rewards_arr = np.zeros(n_samples, dtype=np.float32)

                    # Perform single deterministic rollout with current policy and expert initial states
                    rollout_env.envs[0].unwrapped.set_initial_states(traj.initial_values)
                    obs = rollout_env.reset()
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, dones, infos = rollout_env.step(action)
                        done = bool(dones[0])
                    traj_feat_exp = np.array(infos[0]['feature_expectation'], dtype=np.float32)

                    # Calculate L2 loss for monitoring
                    l2_loss = np.linalg.norm(expert_feat_exp - traj_feat_exp)
                    val_loss += l2_loss / M

                    # Compute DTW distance for monitoring
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                    dtw_distance = compute_dtw(expert_soc, agent_soc)
                    val_dtw_distance += dtw_distance / M
                    val_mae += self._compute_mae(expert_soc, agent_soc) / M
        
            # --- LOGGING ----

            # Log average L2 loss
            self.train_l2_loss.append(average_l2_loss)
            self.train_dtw_distance.append(avg_dtw_distance)
            self.train_mae.append(avg_mae)
            self.train_log_likelihood.append(avg_log_likelihood)
            if self.cfg.validation and len(self.val_set) > 0:
                self.val_l2_loss.append(val_loss)
                self.val_dtw_distance.append(val_dtw_distance)
                self.val_mae.append(val_mae)

            print(f"--- Epoch {epoch+1}/{self.cfg.n_epochs} Summary ---")
            print(f"Avg L2 Loss: {average_l2_loss:.4f}, Avg DTW Distance: {avg_dtw_distance:.4f}, Avg MAE: {avg_mae:.4f}, Log-Likelihood: {avg_log_likelihood:.4f}")
            print(f"Avg Expert Feature Expectation: {avg_expert_feat_exp}, Avg Traj Feature Expectation: {avg_traj_feat_exp}")
            print(f"Updated Reward Weights: {self.reward_weights}")
            print(f"Avg Episode Length: {avg_episode_length:.2f}")
            if self.cfg.validation and len(self.val_set) > 0:
                print(f"Validation Avg L2 Loss: {val_loss:.4f}, Validation Avg DTW Distance: {val_dtw_distance:.4f}, Validation Avg MAE: {val_mae:.4f}")

            # Save model checkpoint
            model.save(f"./models/{self.cfg.folder_name}/maxent_irl_simple_epoch{epoch+1}")
        
        # --- PLOTTING ---
        plt.figure(1)
        plt.plot(range(1, self.cfg.n_epochs + 1), self.train_l2_loss, marker='o')
        if self.cfg.validation and len(self.val_set) > 0:
            plt.plot(range(1, self.cfg.n_epochs + 1), self.val_l2_loss, marker='o')
            plt.legend(['Train L2 Loss', 'Validation L2 Loss'])

        plt.title('MaxEnt IRL L2 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average L2 Loss')
        plt.grid()
        plt.savefig(f'./models/{self.cfg.folder_name}/maxent_irl_simple_training_loss.png')

        plt.figure(2)
        plt.plot(range(1, self.cfg.n_epochs + 1), self.train_mae, marker='o')
        if self.cfg.validation and len(self.val_set) > 0 and self.val_mae:
            plt.plot(range(1, self.cfg.n_epochs + 1), self.val_mae, marker='o')
            plt.legend(['Train MAE', 'Validation MAE'])
        plt.title('MaxEnt IRL — SoC MAE')
        plt.xlabel('Epoch')
        plt.ylabel('Average MAE')
        plt.grid()
        plt.savefig(f'./models/{self.cfg.folder_name}/maxent_irl_simple_mae.png')

        plt.figure(3)
        plt.plot(range(1, self.cfg.n_epochs + 1), self.train_dtw_distance, marker='o')
        if self.cfg.validation and len(self.val_set) > 0:
            plt.plot(range(1, self.cfg.n_epochs + 1), self.val_dtw_distance, marker='o')
            plt.legend(['Train DTW Distance', 'Validation DTW Distance'])
        plt.title('MaxEnt IRL DTW Distance')
        plt.xlabel('Epoch')
        plt.ylabel('Average DTW Distance')
        plt.grid()
        plt.savefig(f'./models/{self.cfg.folder_name}/maxent_irl_simple_training_dtw_distance.png')

        plt.figure(4)
        plt.plot(range(1, self.cfg.n_epochs + 1), self.train_log_likelihood, marker='o')
        plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Converged (LL=0)')
        plt.title('MaxEnt IRL — Expert Log-Likelihood')
        plt.xlabel('Epoch')
        plt.ylabel('Avg log p(τ_expert)')
        plt.grid()
        plt.legend()
        plt.savefig(f'./models/{self.cfg.folder_name}/maxent_irl_simple_log_likelihood.png')

        plt.show()

        print("Training completed")

    @staticmethod
    def _compute_mae(soc_a: np.ndarray, soc_b: np.ndarray) -> float:
        """MAE between two SoC trajectories, resampling to the longer length."""
        n = max(len(soc_a), len(soc_b))
        if len(soc_a) != n:
            soc_a = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_a)), soc_a)
        if len(soc_b) != n:
            soc_b = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(soc_b)), soc_b)
        return float(np.mean(np.abs(soc_a - soc_b)))






    
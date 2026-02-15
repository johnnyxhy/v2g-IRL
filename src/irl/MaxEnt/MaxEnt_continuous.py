import numpy as np
import gymnasium as gym
from sbx import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from irl.dataset.expert_dataset_continuous import ExpertDatasetContinuous
from irl.utils.tools import AdamOptimizer, compute_dtw
from tqdm import tqdm
import math


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
    folder_name: str = "MaxEntIRL_continuous"
    validation: bool = False

class MaxEntIRLTrainer_Continuous:
    """
    Docstring for MaxEntContinuous
    """

    def __init__(self, 
                 initial_reward_weights: np.ndarray,
                 expert_trajectories: ExpertDatasetContinuous,
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
        self.val_l2_loss = []
        self.val_dtw_distance = []
        self.reward_weights_history = []

        # Register optimiser
        #self.optimizer = AdamOptimizer(params_shape=self.reward_weights.shape, lr=self.cfg.reward_lr)

        print(f"MaxEnt IRL Trainer initialized with {len(self.train_set)} training samples and {len(self.val_set)} validation samples.")


    def train(self):
        """
        Train the MaxEnt IRL model
        """

        # Create training environment
        env = gym.make(self.env_name)
        env = Monitor(env, filename=f"./models/{self.cfg.folder_name}/monitor.csv")
        vec_env = DummyVecEnv([lambda: env])

        # Normalise rewards for training
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

        # create rollout env 
        rollout_env = gym.make(self.env_name)
        rollout_env = DummyVecEnv([lambda: rollout_env])

        # Add Monitor

        model = SAC(
            policy="MultiInputPolicy",
            env=vec_env,
            verbose=0,
            learning_rate=self.cfg.policy_train_lr,
            gamma=self.cfg.policy_gamma,
            device=self.cfg.device,
            batch_size=self.cfg.policy_batch_size,
            tensorboard_log=f"./models/{self.cfg.folder_name}/tensorboard/",
            
            # --- SAC Specific Parameters ---
            buffer_size=100_000,  # Size of the replay buffer
            learning_starts=1_000,    # Steps to collect random data before learning starts
            ent_coef=0.1,           # Manually set the entropy coefficient 
            train_freq=1,           # Update the model every n step
            gradient_steps=1,       # How many gradient updates to do per n step
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

            # --- TRAINING ----

            # Train PPO policy with current reward weights
            
            vec_env.envs[0].unwrapped.set_reward_weights(self.reward_weights)
            vec_env.envs[0].unwrapped.set_initial_states(None)

            model.learn(total_timesteps=self.cfg.policy_train_steps_per_iter,tb_log_name=f"epoch_{epoch+1}", progress_bar=True)

            # Loop through each expert trajectory 
            grad = np.zeros_like(self.reward_weights)
            average_l2_loss = 0.0
            N = len(self.train_set)

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

                    avg_episode_length += (episode_length) / (N * self.cfg.rollout_samples)

                    # Compute DTW distance for monitoring
                    expert_soc = np.array(traj.soc_history, dtype=np.float32)
                    agent_soc = np.array(infos[0]['soc_history'], dtype=np.float32)
                    dtw_distance = compute_dtw(expert_soc, agent_soc)
                    avg_dtw_distance += dtw_distance / (N * self.cfg.rollout_samples)
                
                # Use LogSumExp to compute expected feature expectations
                max_reward = np.max(rewards_arr)
                exp_weights = np.exp(rewards_arr - max_reward)
                weights = exp_weights / np.sum(exp_weights)

                traj_feat_exp = np.dot(weights, traj_feat_exp_arr)
                
                # Update gradient
                grad += (expert_feat_exp - traj_feat_exp) / N

                # Update averages for monitoring
                avg_expert_feat_exp += expert_feat_exp / N
                avg_traj_feat_exp += traj_feat_exp / N

                # Calculate L2 loss for monitoring
                l2_loss = np.linalg.norm(expert_feat_exp - traj_feat_exp)
                average_l2_loss += l2_loss / N  
            
            # Update reward weights
            self.reward_weights += self.cfg.reward_lr * grad

            # --- VALIDATION ----

            # Perform validation (optional)
            if self.cfg.validation and len(self.val_set) > 0:

                print("Starting validation...")

                val_loss = 0.0
                val_dtw_distance = 0.0
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
        
            # --- LOGGING ----

            # Log average L2 loss
            self.train_l2_loss.append(average_l2_loss)
            self.train_dtw_distance.append(avg_dtw_distance)
            if self.cfg.validation and len(self.val_set) > 0:
                self.val_l2_loss.append(val_loss)
                self.val_dtw_distance.append(val_dtw_distance)

            self.reward_weights_history.append(self.reward_weights.copy())

            print(f"--- Epoch {epoch+1}/{self.cfg.n_epochs} Summary ---")
            print(f"Avg L2 Loss: {average_l2_loss:.4f}, Avg DTW Distance: {avg_dtw_distance:.4f}")
            print(f"Avg Expert Feature Expectation: {avg_expert_feat_exp}")
            print(f"Avg Traj   Feature Expectation: {avg_traj_feat_exp}")
            print(f"Updated Reward Weights: {self.reward_weights}")
            print(f"Avg Episode Length: {avg_episode_length:.2f}")
            if self.cfg.validation and len(self.val_set) > 0:
                print(f"Validation Avg L2 Loss: {val_loss:.4f}, Validation Avg DTW Distance: {val_dtw_distance:.4f}")

            # Save model checkpoint
            model.save(f"./models/{self.cfg.folder_name}/maxent_irl_epoch{epoch+1}")
        
        # --- PLOTTING ---
        plt.figure(1)
        plt.plot(range(1, self.cfg.n_epochs + 1), self.train_l2_loss)
        if self.cfg.validation and len(self.val_set) > 0:
            plt.plot(range(1, self.cfg.n_epochs + 1), self.val_l2_loss)
            plt.legend(['Train L2 Loss', 'Validation L2 Loss'])

        plt.title('MaxEnt IRL L2 Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Average L2 Loss')
        plt.grid()
        plt.savefig(f'./models/{self.cfg.folder_name}/maxent_irl_training_loss.png')

        plt.figure(2)
        plt.plot(range(1, self.cfg.n_epochs + 1), self.train_dtw_distance)
        if self.cfg.validation and len(self.val_set) > 0:
            plt.plot(range(1, self.cfg.n_epochs + 1), self.val_dtw_distance)
            plt.legend(['Train DTW Distance', 'Validation DTW Distance'])
        plt.title('MaxEnt IRL DTW Distance')
        plt.xlabel('Epoch')
        plt.ylabel('Average DTW Distance')
        plt.grid()
        plt.savefig(f'./models/{self.cfg.folder_name}/maxent_irl_training_dtw_distance.png')

        # Plot reward weights evolution
        reward_weights_history_arr = np.array(self.reward_weights_history)
        n_weights = reward_weights_history_arr.shape[1]
        epochs = range(1, self.cfg.n_epochs + 1)

        # Dynamic grid calculation
        ncols = 2
        nrows = math.ceil(n_weights / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows), sharex=True)
        axes_flat = axes.flatten()

        for i in range(n_weights):
            ax = axes_flat[i]
            ax.plot(epochs, reward_weights_history_arr[:, i], color=f'C{i}', linewidth=2)
            ax.set_title(f'Weight {i+1}')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.5)

        # Hide any unused subplots (e.g., if n_weights is 7 but grid is 8)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        # Ensure the bottom-most visible plots have x-axis labels
        for j in range(n_weights):
            # If the plot below this one is hidden or doesn't exist, it's a bottom plot
            if j + ncols >= n_weights:
                axes_flat[j].set_xlabel('Epoch')

        plt.suptitle('Reward Weights Evolution Across Epochs', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Save the figure
        save_path = f'./models/{self.cfg.folder_name}/maxent_irl_reward_weights_evolution_dynamic.png'
        plt.savefig(save_path)

        # Save final reward weights to a text file
        np.savetxt(f'./models/{self.cfg.folder_name}/final_reward_weights.txt', self.reward_weights, fmt='%.6f')

        plt.show()

        print("Training completed")






    
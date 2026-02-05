import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.utils.tools import compute_dtw
import json

gym.register(
    id='V2GEnv-simple',
    entry_point="irl.envs.V2GEnv_simple:V2GEnv",
    max_episode_steps=96,
)

def extract_trajectory(obs, trajectory):
    trajectory['timestep'] = np.append(trajectory['timestep'], obs['timestep'])
    trajectory['soc'] = np.append(trajectory['soc'], obs['soc'][0])
    trajectory['soc_target'] = np.append(trajectory['soc_target'], obs['soc_target'][0])
    trajectory['energy_price'] = np.append(trajectory['energy_price'], obs['energy_price'][0])
    trajectory['time_to_next_journey'] = np.append(trajectory['time_to_next_journey'], obs['time_to_next_journey'])

if __name__ == "__main__":

    # Initialize trajectory storage

    trajectories = {
        'timestep': np.array([], dtype=int),
        'soc': np.array([], dtype=np.float32),
        'soc_target': np.array([], dtype=np.float32),
        'energy_price': np.array([], dtype=np.float32),
        'time_to_next_journey': np.array([], dtype=int),
        'reward': np.array([0], dtype=np.float32),
    }

    accumulated_reward = [0.0]

    # Extract initial states from expert json
    with open("data/processed_trajectories_simple_probabilistic.json", "r") as f:
        expert_data = json.load(f)

    expert_index = 0

    # Extract the first trajectory's initial state
    initial_states = expert_data[expert_index]['initial_values']

    # Load initial reward weights
    reward_weights = np.array([-7.522875, -5.11008, -5.3194823, -0.85412014], dtype=np.float32)

    # Load the trained model
    model = PPO.load("./models/MaxEntIRL_simple_probabilistic_exp1/maxent_irl_simple_epoch20")

    vec_env = DummyVecEnv([lambda: gym.make('V2GEnv-simple')])
    vec_env.envs[0].unwrapped.set_initial_states(initial_states)
    vec_env.envs[0].unwrapped.set_reward_weights(reward_weights)

    def evaluate():
        obs = vec_env.reset()
        extract_trajectory(obs, trajectories)
        print(obs)
        
        while True:
            action, _states = model.predict(obs, deterministic=True)

            # Print every action taken
            print(f"Action taken at timestep {obs['timestep'][0]}: {action[0]}")

            obs, rewards, done, info = vec_env.step(action)
            trajectories['reward'] = np.append(trajectories['reward'], rewards[0])
            accumulated_reward.append(accumulated_reward[-1] + rewards[0])

            if done[0]:
                terminal_obs = info[0]["terminal_observation"]
                soc_history = info[0]["soc_history"]
                out_start_timestep = info[0]["out_start_timestep"]
                return_start_timestep = info[0]["return_start_timestep"]
                out_duration = info[0]["out_duration"]
                return_duration = info[0]["return_duration"]
                feature_expectation = info[0]["feature_expectation"]
                print("Length of soc history:", len(soc_history))
                print("Feature expectation:", feature_expectation)
                print("Accumulated reward:", accumulated_reward[-1])

                extract_trajectory(terminal_obs, trajectories)
                break
            else:
                extract_trajectory(obs, trajectories)
        
        # Print Expert Feature Expectation
        print("Expert Feature Expectation:", expert_data[expert_index]['feature_expectation'])

        # Compute DTW distance between expert and agent SoC trajectories
        dtw_distance = compute_dtw(
            np.array(expert_data[expert_index]['soc_history'], dtype=np.float32),
            np.array(soc_history, dtype=np.float32)
        )
        print(f"DTW Distance between expert and agent SoC trajectories: {dtw_distance}")

        # Plot SOC over timesteps
        plt.figure()
        plt.plot(range(len(soc_history)), soc_history, label='SoC')

        # Plot expert SoC
        expert_soc = expert_data[expert_index]['soc_history']
        plt.plot(range(len(expert_soc)), expert_soc, label='Expert SoC', linestyle='--')

        # Mark with colored regions the out and return journeys
        plt.axvspan(out_start_timestep, out_start_timestep + out_duration, color='red', alpha=0.3, label='Out Journey')
        plt.axvspan(return_start_timestep, return_start_timestep + return_duration, color='blue', alpha=0.3, label='Return Journey')

        plt.xlabel('Timestep')
        plt.ylabel('State of Charge (SoC)')
        plt.legend()

        plt.title('V2G Environment Evaluation with DTW Distance: {:.2f}'.format(dtw_distance))
        plt.show()

    evaluate()
    
    
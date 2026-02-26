import numpy as np
import gymnasium as gym
from sbx import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.utils.tools import compute_dtw
import json

gym.register(
    id='V2GEnv-continuous',
    entry_point="irl.envs.V2GEnv_continuous:V2GEnv",
    max_episode_steps=96,
)

def extract_trajectory(obs, trajectory):
    trajectory['timestep'] = np.append(trajectory['timestep'], obs['timestep'])
    trajectory['soc'] = np.append(trajectory['soc'], obs['soc'][0])
    trajectory['soc_target'] = np.append(trajectory['soc_target'], obs['soc_target'][0])
    #trajectory['energy_price'] = np.append(trajectory['energy_price'], obs['energy_price'][0])
    trajectory['time_to_next_journey'] = np.append(trajectory['time_to_next_journey'], obs['time_to_next_journey'])

if __name__ == "__main__":

    # Initialize trajectory storage

    trajectories = {
        'timestep': np.array([], dtype=int),
        'soc': np.array([], dtype=np.float32),
        'soc_target': np.array([], dtype=np.float32),
        #'energy_price': np.array([], dtype=np.float32),
        'time_to_next_journey': np.array([], dtype=int),
        'reward': np.array([0], dtype=np.float32),
    }

    accumulated_reward = [0.0]

    # Extract initial states from expert json
    with open("data/processed_trajectories_profit.json", "r") as f:
        expert_data = json.load(f)

    # Load the trained model
    model = SAC.load("./models/MaxEntIRL_continuous_v7_exp5(No norm)/maxent_irl_epoch5")
    
    vec_env = DummyVecEnv([lambda: gym.make('V2GEnv-continuous')])

    # Find expert indexes corresponding to segment
    def find_expert_indexes(segment):
        indexes = []
        for i, traj in enumerate(expert_data):
            if traj['segment'] == segment:
                indexes.append(i)
        return indexes

    # Plotting function to evaluate a specific expert trajectory
    def plot_expert_trajectory(soc_history, expert_index, out_start_timestep, return_start_timestep, out_duration, return_duration, dtw_distance):
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

        plt.title('V2G Environment Evaluation with DTW Distance: {:.2f} for Trajectory {}'.format(dtw_distance, expert_index))
        plt.show()

    def evaluate(expert_index):

        # Extract the first trajectory's initial state
        initial_states = expert_data[expert_index]['initial_values']
        vec_env.envs[0].unwrapped.set_initial_states(initial_states)
        vec_env.envs[0].unwrapped.set_reward_weights(np.array([3.442509, 2.777384, -10.059095, -14.462010, -9.689913], dtype=np.float32))

        obs = vec_env.reset()
        extract_trajectory(obs, trajectories)
        # Print battery capacity
        print(f"Battery capacity for trajectory {expert_index}: {obs['battery_capacity'][0]}")
        
        # Print expert feature expectation
        print(f"Expert feature expectation for trajectory {expert_index}: {expert_data[expert_index]['feature_expectation']}")
        
        while True:
            action, _states = model.predict(obs, deterministic=True)

            obs, rewards, done, info = vec_env.step(action)

            # Print every action taken
            print(f"Action taken at timestep {obs['timestep'][0]}: {action[0]} with reward: {rewards[0]} with soc: {obs['soc'][0]} and soc target: {obs['soc_target'][0]}")
            # print feature expectation from info
            #print(f"Features at timestep {obs['timestep'][0]}: {info[0]['features']}")

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

        # Compute DTW distance between expert and agent SoC trajectories
        dtw_distance = compute_dtw(
            np.array(expert_data[expert_index]['soc_history'], dtype=np.float32),
            np.array(soc_history, dtype=np.float32)
        )

        plot_expert_trajectory(soc_history, expert_index, out_start_timestep, return_start_timestep, out_duration, return_duration, dtw_distance)

        # Plot accumulated reward over time
        # plt.figure()
        # plt.plot(range(len(accumulated_reward)), accumulated_reward, label='Accumulated Reward')
        # plt.xlabel('Timestep')
        # plt.ylabel('Accumulated Reward')
        # plt.title('Accumulated Reward over Time for Trajectory {}'.format(expert_index))
        # plt.legend()
        # plt.show()


# Evaluate on first 5 trajectories of the specified segment
    segment = "Male 50-59"
    expert_indexes = find_expert_indexes(segment)

    for idx in expert_indexes[:10]:
        print(f"Evaluating trajectory {idx} from segment {segment}")
        evaluate(idx)

        

    

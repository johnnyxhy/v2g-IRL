import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
import json

gym.register(
    id='V2GEnv-single',
    entry_point="irl.envs.V2GEnv:V2GEnv",
    max_episode_steps=96,
)

def make_env(initial_states=None):
    env = gym.make('V2GEnv-single')

    if initial_states is not None:
        env.set_initial_states(initial_states)

    return env


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
    with open("data/processed_trajectories.json", "r") as f:
        expert_data = json.load(f)

    expert_index = 2

    # Extract the first trajectory's initial state
    initial_states = expert_data[expert_index]['initial_values']

    # Load the trained model
    model = SAC.load("./models/sac_v2g_smoke_test")

    vec_env = DummyVecEnv([lambda: make_env(initial_states=initial_states)])

    obs = vec_env.reset()
    extract_trajectory(obs, trajectories)
    print(obs)
    
    while True:
        action, _states = model.predict(obs, deterministic=True)

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
            print("Lenght of soc history:", len(soc_history))

            extract_trajectory(terminal_obs, trajectories)
            break
        else:
            extract_trajectory(obs, trajectories)
    
   

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

    plt.title('V2G Environment Evaluation')
    plt.show()
    
    
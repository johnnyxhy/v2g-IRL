import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register(
    id='V2GEnv-single',
    entry_point="irl.envs.V2GEnv_Single:V2GEnv_SingleTimestep",
    max_episode_steps=96,
)

def make_env():
    env = gym.make('V2GEnv-single')

    return env

# Load the trained model
model = SAC.load("./models/sac_v2g_smoke_test_20260120_195950")

env = make_env()

vec_env = DummyVecEnv([make_env])

def extract_trajectory(obs, trajectory):
    trajectory['timestep'] = np.append(trajectory['timestep'], obs['timestep'])
    trajectory['soc'] = np.append(trajectory['soc'], obs['soc'][0])
    trajectory['soc_target'] = np.append(trajectory['soc_target'], obs['soc_target'][0])
    trajectory['energy_price'] = np.append(trajectory['energy_price'], obs['energy_price'][0])
    trajectory['time_to_next_journey'] = np.append(trajectory['time_to_next_journey'], obs['time_to_next_journey'])

if __name__ == "__main__":
    trajectories = {
        'timestep': np.array([], dtype=int),
        'soc': np.array([], dtype=np.float32),
        'soc_target': np.array([], dtype=np.float32),
        'energy_price': np.array([], dtype=np.float32),
        'time_to_next_journey': np.array([], dtype=int),
        'reward': np.array([0], dtype=np.float32),
    }

    accumulated_reward = [0.0]

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
            extract_trajectory(terminal_obs, trajectories)
            break
        else:
            extract_trajectory(obs, trajectories)
    
   

    # Plot the results
    fig, ax1 = plt.subplots(figsize=(12, 8))

# Left y-axis: SoC, target, price
    ax1.scatter(
        trajectories['timestep'],
        trajectories['soc'],
        label='State of Charge (SoC)'
    )
    ax1.scatter(
        trajectories['timestep'],
        trajectories['soc_target'],
        label='Target SoC'
    )
    ax1.plot(
        trajectories['timestep'],
        trajectories['energy_price'],
        label='Energy Price',
        linestyle=':'
    )

    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('SoC / Price')

    # Right y-axis: accumulated reward
    ax2 = ax1.twinx()
    ax2.plot(
        trajectories['timestep'],
        accumulated_reward,
        label='Accumulated Reward',
        linestyle='-.'
    )
    ax2.set_ylabel('Accumulated Reward')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    plt.title('V2G Environment Evaluation')
    plt.show()
    
    
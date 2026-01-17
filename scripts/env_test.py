import numpy as np
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

gym.register(
    id='V2GEnv-v0',
    entry_point="irl.envs.V2GEnv:V2GEnv",
    max_episode_steps=96,
)

def make_env():
    env = gym.make('V2GEnv-v0')

    return env

if __name__ == "__main__":
    env = make_env()

    # 1) Basic env validation (will catch obs/action space issues fast)
    check_env(env, warn=True)

    # 2) SB3 wants a VecEnv
    vec_env = DummyVecEnv([make_env])

    # 3) SAC with Dict obs => MultiInputPolicy
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        seed=0,
        device="cpu",
        learning_starts=100,   # small for smoke test
        buffer_size=10_000,
        batch_size=64,
        train_freq=1,
        gradient_steps=1,
    )

    # 4) Run a short training to exercise reset/step repeatedly
    model.learn(total_timesteps=2_000)

    # 5) Rollout a few steps deterministically
    obs = vec_env.reset()
    for _ in range(20):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if done.any():
            obs = vec_env.reset()

    print("SAC smoke test completed.")
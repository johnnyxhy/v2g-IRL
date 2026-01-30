import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO  
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- Register env once ---
gym.register(
    id="V2GEnv-simple",
    entry_point="irl.envs.V2GEnv_simple:V2GEnv",
    max_episode_steps=96,
)

def make_env(seed: int = 0, reward_weights: np.array = None):
    def _init():
        env = gym.make("V2GEnv-simple")
        if reward_weights is not None:
            env.set_initial_reward_weights(reward_weights)
        env = Monitor(env)
        # Seed is handled during reset
        return env
    return _init

def eval_policy_deterministic(model: PPO, env_fn, n_episodes: int = 5, base_seed: int = 10_000):
    returns = []
    for i in range(n_episodes):
        env = env_fn()
        obs, info = env.reset(seed=base_seed + i)
        done = False
        ep_ret = 0.0

        while not done:
            # model.predict works the same for PPO/SAC
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)

        returns.append(ep_ret)
        env.close()

    return float(np.mean(returns)), returns

if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)

    # 1) Basic env validation
    raw_env = gym.make("V2GEnv-simple")
    check_env(raw_env, warn=True)
    raw_env.close()

    # 2) VecEnv for SB3 training
    train_seed = 0
    vec_env = DummyVecEnv([make_env(seed=train_seed, reward_weights=np.array([-0.1, -10.0, -10.0], dtype=np.float32))])

    # 3) Create PPO model
    # Note: PPO doesn't use a Replay Buffer; it uses rollouts.
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        seed=train_seed,
        gamma=0.99,
        device="cuda",
        n_steps=2048,           # Number of steps to run per update
        batch_size=64,          # Minibatch size for the optimizer
        n_epochs=10,            # Number of epochs when optimizing the surrogate loss
        ent_coef=0.0,           # Entropy coefficient (increase for more exploration)
        learning_rate=3e-4,
    )

    print("SB3 model device:", model.device)

    # 5) Evaluate BEFORE training
    env_for_eval = lambda: gym.make("V2GEnv-simple")
    pre_mean, pre_all = eval_policy_deterministic(model, env_for_eval, n_episodes=5, base_seed=20_000)
    print("Pre-train eval mean return:", pre_mean)

    # 6) Training
    # total_timesteps should usually be a multiple of n_steps
    model.learn(total_timesteps=2048*5, log_interval=1)

    # 7) Evaluate AFTER training
    post_mean, post_all = eval_policy_deterministic(model, env_for_eval, n_episodes=5, base_seed=20_000)
    print("Post-train eval mean return:", post_mean)

    # 8) Save
    save_path = f"./models/ppo_v2g_simple"
    model.save(save_path)
    print("Saved model to:", save_path + ".zip")

    vec_env.close()
    print("PPO test completed.")
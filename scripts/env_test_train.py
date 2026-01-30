import os
import time
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# --- Register env once (safe to keep here) ---
gym.register(
    id="V2GEnv-single",
    entry_point="irl.envs.V2GEnv:V2GEnv",
    max_episode_steps=96,
)

def make_env(seed: int = 0):
    """
    Create a fresh env instance, wrapped with Monitor for episode stats.
    SB3 will handle per-episode seeds; this seed is for initial deterministic setup.
    """
    def _init():
        env = gym.make("V2GEnv-single")
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def eval_policy_deterministic(model: SAC, env_fn, n_episodes: int = 5, base_seed: int = 10_000):
    """
    Evaluate with a non-VecEnv env for clean terminal handling and predictable logging.
    Returns mean episode return.
    """
    returns = []
    for i in range(n_episodes):
        env = env_fn()
        obs, info = env.reset(seed=base_seed + i)
        done = False
        ep_ret = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(reward)

        returns.append(ep_ret)
        env.close()

    return float(np.mean(returns)), returns

if __name__ == "__main__":
    # 0) Make output directory
    os.makedirs("./models", exist_ok=True)

    # 1) Basic env validation (fast catch for obs/action mismatches)
    #    Use a raw env (not VecEnv) for check_env
    raw_env = gym.make("V2GEnv-single")
    check_env(raw_env, warn=True)
    raw_env.close()

    # 2) VecEnv for SB3 training
    #    Use different seeds per env if you ever scale n_envs > 1
    train_seed = 0
    vec_env = DummyVecEnv([make_env(seed=train_seed)])

    # 3) Create model (Dict obs => MultiInputPolicy)
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        seed=train_seed,
        gamma=0.99,
        device="cuda",
        learning_starts=1_000,     # more realistic than 100 for SAC stability
        buffer_size=50_000,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
    )

    # 4) Confirm device selection
    print("SB3 model device:", model.device)

    # 5) Evaluate BEFORE training on fixed seeds (deterministic)
    #    Use a clean single env instance (not VecEnv)
    env_for_eval = lambda: gym.make("V2GEnv-single")
    pre_mean, pre_all = eval_policy_deterministic(model, env_for_eval, n_episodes=5, base_seed=20_000)
    print("Pre-train eval mean return:", pre_mean)
    print("Pre-train eval returns:", pre_all)

    # 6) Short training smoke test (enough to exercise replay, updates, resets)
    model.learn(total_timesteps=10_000, log_interval=10)

    # 7) Evaluate AFTER training on the same fixed seeds
    post_mean, post_all = eval_policy_deterministic(model, env_for_eval, n_episodes=5, base_seed=20_000)
    print("Post-train eval mean return:", post_mean)
    print("Post-train eval returns:", post_all)

    # 8) Save with timestamp to avoid overwriting
    save_path = f"./models/sac_v2g_smoke_test"
    model.save(save_path)
    print("Saved model to:", save_path + ".zip")

    # 9) Clean up
    vec_env.close()
    print("SAC smoke test completed.")
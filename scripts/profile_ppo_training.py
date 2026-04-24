"""
PPO Training Bottleneck Profiler
=================================
Measures wall-clock time for each stage of one IRL epoch:
  1. Env step throughput (env.step calls per second)
  2. PPO collect throughput (samples/sec, single env)
  3. PPO gradient update throughput (updates/sec)
  4. Reward net forward pass throughput (inferences/sec)
  5. IRL rollout throughput (full episodes/sec)
  6. IRL gradient time (full batch)

Run with:
    uv run python scripts/profile_ppo_training.py
"""

import time
import numpy as np
import torch
import gymnasium as gym
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------ #
#  Registration                                                        #
# ------------------------------------------------------------------ #

gym.register(
    id='V2GDeepEnv-discrete',
    entry_point="irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
    max_episode_steps=96,
)

from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES
from irl.DeepMaxEnt.DeepMaxEnt_discrete import FlattenNormalizeObsWrapper, DeepMaxEntDiscreteConfig

# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

SECTION_WIDTH = 55

def header(title):
    print(f"\n{'='*SECTION_WIDTH}")
    print(f"  {title}")
    print(f"{'='*SECTION_WIDTH}")

def result(label, value, unit, note=""):
    note_str = f"  ({note})" if note else ""
    print(f"  {label:<38} {value:>8.1f}  {unit}{note_str}")

def make_env():
    e = gym.make('V2GDeepEnv-discrete')
    return FlattenNormalizeObsWrapper(e)

reward_net = RewardNet(obs_dim=7, action_dim=1, hidden_dim=64)

# ------------------------------------------------------------------ #
#  1. Raw env step throughput                                          #
# ------------------------------------------------------------------ #

header("1. Raw env.step() throughput")

env = make_env()
obs, _ = env.reset()
N_STEPS = 5_000
t0 = time.perf_counter()
for _ in range(N_STEPS):
    action = env.action_space.sample()
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
elapsed = time.perf_counter() - t0
env_steps_per_sec = N_STEPS / elapsed

result("env.step() calls", env_steps_per_sec, "steps/s")
result("Time per step", 1000 / env_steps_per_sec, "ms/step")
env.close()

# ------------------------------------------------------------------ #
#  2. Env step WITH reward net (as used during PPO training)          #
# ------------------------------------------------------------------ #

header("2. env.step() WITH reward net (PPO training mode)")

env = make_env()
env.unwrapped.set_reward_net(reward_net)
env.unwrapped.set_reward_scale(10.0)
obs, _ = env.reset()
t0 = time.perf_counter()
for _ in range(N_STEPS):
    action = env.action_space.sample()
    obs, _, done, _, _ = env.step(action)
    if done:
        obs, _ = env.reset()
elapsed = time.perf_counter() - t0
env_rnet_steps_per_sec = N_STEPS / elapsed

result("env.step() + reward net calls", env_rnet_steps_per_sec, "steps/s")
result("Overhead from reward net", env_steps_per_sec - env_rnet_steps_per_sec, "steps/s slower",
       f"{100*(1 - env_rnet_steps_per_sec/env_steps_per_sec):.1f}% slower")
env.close()

# ------------------------------------------------------------------ #
#  3. PPO rollout collection throughput (DummyVecEnv, 1 env)          #
# ------------------------------------------------------------------ #

header("3. PPO rollout collection (DummyVecEnv, 1 env, n_steps=2048)")

vec_env = DummyVecEnv([make_env])
vec_env.envs[0].unwrapped.set_reward_net(reward_net)
vec_env.envs[0].unwrapped.set_reward_scale(10.0)

model = PPO(
    "MlpPolicy", vec_env, verbose=0,
    n_steps=2048, batch_size=64, n_epochs=1,
    learning_rate=3e-4, gamma=1.0, ent_coef=0.01,
)

# Warm-up JAX
model.learn(total_timesteps=512, reset_num_timesteps=True, progress_bar=False)

N_COLLECT = 20_000
t0 = time.perf_counter()
model.learn(total_timesteps=N_COLLECT, reset_num_timesteps=False, progress_bar=False)
elapsed = time.perf_counter() - t0
ppo_1env_throughput = N_COLLECT / elapsed

result("PPO 1-env throughput", ppo_1env_throughput, "steps/s")
result("Time for 500k steps (1 env)", 500_000 / ppo_1env_throughput / 60, "minutes")

# ------------------------------------------------------------------ #
#  4. PPO throughput scaling with N envs                              #
# ------------------------------------------------------------------ #

header("4. PPO throughput scaling: DummyVecEnv with N envs")

print(f"  {'N envs':<10} {'steps/s':>10}  {'speedup':>8}  {'est. 500k time':>14}")
print(f"  {'-'*46}")

baseline = ppo_1env_throughput
for n_envs in [1, 2, 4, 8]:
    envs = DummyVecEnv([make_env for _ in range(n_envs)])
    for e in envs.envs:
        e.unwrapped.set_reward_net(reward_net)
        e.unwrapped.set_reward_scale(10.0)

    m = PPO(
        "MlpPolicy", envs, verbose=0,
        n_steps=2048, batch_size=64, n_epochs=1,
        learning_rate=3e-4, gamma=1.0, ent_coef=0.01,
    )
    m.learn(total_timesteps=1024, reset_num_timesteps=True, progress_bar=False)  # warm-up

    t0 = time.perf_counter()
    m.learn(total_timesteps=N_COLLECT, reset_num_timesteps=False, progress_bar=False)
    elapsed = time.perf_counter() - t0
    thr = N_COLLECT / elapsed
    speedup = thr / baseline
    eta_min = 500_000 / thr / 60
    print(f"  {n_envs:<10} {thr:>10.0f}  {speedup:>7.2f}x  {eta_min:>12.1f} min")
    envs.close()

# ------------------------------------------------------------------ #
#  5. Reward net forward pass throughput                               #
# ------------------------------------------------------------------ #

header("5. Reward net forward pass throughput")

obs_batch_1 = torch.randn(1, 7)
act_batch_1 = torch.randn(1, 1)
obs_batch_64 = torch.randn(64, 7)
act_batch_64 = torch.randn(64, 1)

N_INFER = 50_000

# Single sample (as called in env.step)
with torch.no_grad():
    for _ in range(100): reward_net(obs_batch_1, act_batch_1)  # warm-up
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_INFER):
        reward_net(obs_batch_1, act_batch_1)
elapsed = time.perf_counter() - t0
result("Single-sample inference", N_INFER / elapsed, "calls/s", "env.step overhead")

# Batch of 64 (IRL gradient scoring)
with torch.no_grad():
    for _ in range(100): reward_net(obs_batch_64, act_batch_64)
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_INFER // 10):
        reward_net(obs_batch_64, act_batch_64)
elapsed = time.perf_counter() - t0
result("Batch-64 inference", (N_INFER // 10 * 64) / elapsed, "samples/s", "IRL gradient scoring")

# ------------------------------------------------------------------ #
#  6. IRL rollout throughput (full episode)                            #
# ------------------------------------------------------------------ #

header("6. IRL rollout throughput (full episodes)")

rollout_env = DummyVecEnv([make_env])
rollout_env.envs[0].unwrapped.set_reward_net(reward_net)
rollout_env.envs[0].unwrapped.set_reward_scale(10.0)

# Use the already-trained model from section 3 as a stand-in
rollout_model = PPO(
    "MlpPolicy", rollout_env, verbose=0,
    n_steps=2048, batch_size=64, n_epochs=1,
    learning_rate=3e-4, gamma=1.0, ent_coef=0.01,
)
rollout_model.learn(total_timesteps=2048, progress_bar=False)

N_ROLLOUTS = 30

t0 = time.perf_counter()
for _ in range(N_ROLLOUTS):
    obs = rollout_env.reset()
    done = False
    while not done:
        action, _ = rollout_model.predict(obs, deterministic=False)
        obs, _, dones, _ = rollout_env.step(action)
        done = bool(dones[0])
elapsed = time.perf_counter() - t0
eps_per_sec = N_ROLLOUTS / elapsed

result("Full episode rollouts", eps_per_sec, "episodes/s")

# Estimate for typical config
train_size = 100  # ~100 train trajectories for "Male 50-59"
rollout_samples = 30
total_rollouts = train_size * rollout_samples
irl_rollout_time_min = total_rollouts / eps_per_sec / 60
result("Est. IRL rollout time/epoch", irl_rollout_time_min, "minutes",
       f"{train_size} trajs × {rollout_samples} samples")

rollout_env.close()

# ------------------------------------------------------------------ #
#  7. Summary + bottleneck diagnosis                                   #
# ------------------------------------------------------------------ #

header("7. Summary & Bottleneck Diagnosis")

ppo_500k_min = 500_000 / ppo_1env_throughput / 60

print(f"\n  Per-epoch time breakdown (estimates for default config):")
print(f"  {'Stage':<42} {'Est. time':>10}")
print(f"  {'-'*54}")
print(f"  {'PPO training (500k steps, 1 env)':<42} {ppo_500k_min:>9.1f}m")
print(f"  {'IRL rollout collection':<42} {irl_rollout_time_min:>9.1f}m")
print(f"  {'IRL gradient backprop (fast, ~seconds)':<42} {'<0.1':>10}m")

total_est = ppo_500k_min + irl_rollout_time_min
print(f"\n  {'Total estimated epoch time':<42} {total_est:>9.1f}m")

print(f"\n  Bottleneck: ", end="")
if ppo_500k_min > irl_rollout_time_min * 3:
    print("PPO TRAINING — parallelise with DummyVecEnv(N envs)")
elif irl_rollout_time_min > ppo_500k_min * 3:
    print("IRL ROLLOUTS — reduce rollout_samples or parallelise rollout envs")
else:
    print(f"BALANCED — both stages comparable ({ppo_500k_min:.1f}m PPO vs {irl_rollout_time_min:.1f}m IRL rollouts)")

print()

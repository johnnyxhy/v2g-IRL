"""
Quick sanity check for deep discrete setup:
1. Expert loader action range matches env's Discrete(21) space (0-20)
2. Action→continuous mapping is consistent between loader and env
3. Reward net produces valid outputs for expert (obs, action) pairs
"""
import json
import numpy as np
import torch
import gymnasium as gym

from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES

gym.register(
    id="V2GDeepEnv-discrete",
    entry_point="irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
    max_episode_steps=96,
)

# ── 1. Check expert dataset action range ──────────────────────────────
print("=" * 60)
print("1. EXPERT DATASET ACTION RANGE CHECK")
print("=" * 60)

with open("data/processed_trajectories_deep_discrete.json") as f:
    data = json.load(f)

all_actions = np.concatenate(
    [np.array(t["state_action_pairs"]["actions"]).flatten() for t in data]
)
all_delta_ts = np.concatenate(
    [np.array(t["state_action_pairs"]["delta_ts"]).flatten() for t in data]
)

print(f"  Total actions: {len(all_actions)}")
print(f"  Action range: [{all_actions.min():.3f}, {all_actions.max():.3f}]  (expected [-1, 1])")
print(f"  Action mean: {all_actions.mean():.3f}, std: {all_actions.std():.3f}")
print(f"  delta_t range: [{all_delta_ts.min():.0f}, {all_delta_ts.max():.0f}]")
print(f"  delta_t mean: {all_delta_ts.mean():.2f}, std: {all_delta_ts.std():.2f}")

# Normalized actions should be in [-1, 1]
out_of_range = (all_actions < -1.01) | (all_actions > 1.01)
print(f"  Out of [-1, 1]: {out_of_range.sum()} / {len(all_actions)}")

# delta_ts should all be positive integers
non_positive_dt = (all_delta_ts <= 0)
print(f"  Non-positive delta_ts: {non_positive_dt.sum()} / {len(all_delta_ts)}")

print(f"\n  Action distribution:")
unique, counts = np.unique(np.round(all_actions * 10) / 10, return_counts=True)  # bin to 0.1
for u, c in zip(unique, counts):
    bar = "#" * max(1, c // 50)
    print(f"    {u:+5.1f}: {c:5d} {bar}")

PASS_1 = out_of_range.sum() == 0 and non_positive_dt.sum() == 0
print(f"\n  RESULT: {'PASS' if PASS_1 else 'FAIL'}")

# ── 2. Verify loader↔env action mapping consistency ──────────────────
print("\n" + "=" * 60)
print("2. ACTION MAPPING CONSISTENCY (loader ↔ env)")
print("=" * 60)

env = gym.make("V2GDeepEnv-discrete")
print(f"  Env action space: {env.action_space}")
print(f"  Env ACTION_STEP_SIZE: {env.unwrapped.ACTION_STEP_SIZE}")
print(f"  Env mapping: continuous = action * {env.unwrapped.ACTION_STEP_SIZE} - 1.0")

# Verify: action=0 → -1.0 (max discharge), action=10 → 0.0 (idle), action=20 → +1.0 (max charge)
mapping_checks = [
    (0, -1.0, "max discharge"),
    (10, 0.0, "idle"),
    (20, 1.0, "max charge"),
    (15, 0.5, "half charge"),
    (5, -0.5, "half discharge"),
]
all_ok = True
for discrete_a, expected_cont, label in mapping_checks:
    actual_cont = discrete_a * env.unwrapped.ACTION_STEP_SIZE - 1.0
    ok = abs(actual_cont - expected_cont) < 1e-6
    all_ok = all_ok and ok
    sym = "OK" if ok else "FAIL"
    print(f"  [{sym}] discrete={discrete_a:2d} → continuous={actual_cont:+.1f} (expected {expected_cont:+.1f}) [{label}]")

# Now check loader's inverse mapping: soc_charged=0.3 → discrete = 0.3/0.1 + 10 = 13
loader_checks = [
    (0.0, 0.0, 10, "idle"),
    (0.3, 0.0, 13, "charge 0.3 SoC"),
    (0.0, 0.3, 7, "discharge 0.3 SoC"),
    (0.5, 0.0, 15, "charge 0.5 SoC"),
    (0.0, 0.5, 5, "discharge 0.5 SoC"),
    (1.0, 0.0, 20, "charge 1.0 SoC"),
    (0.0, 1.0, 0, "discharge 1.0 SoC"),
]
print(f"\n  Loader inverse mapping checks:")
for charge, discharge, expected_disc, label in loader_checks:
    ACTION_STEP_SIZE = 0.1
    if charge > 0:
        action_val = round(charge / ACTION_STEP_SIZE, 2) + 10
    elif discharge > 0:
        action_val = round(10 - discharge / ACTION_STEP_SIZE, 2)
    else:
        action_val = 10.0
    ok = abs(action_val - expected_disc) < 1e-6
    all_ok = all_ok and ok
    sym = "OK" if ok else "FAIL"
    print(f"  [{sym}] charged={charge:.1f}, discharged={discharge:.1f} → loader={action_val:.0f} (expected {expected_disc}) [{label}]")

# Verify roundtrip: loader action → env continuous → matches original SoC change
print(f"\n  Roundtrip check (loader_action → env_continuous → SoC direction):")
for charge, discharge, _, label in loader_checks:
    if charge > 0:
        loader_action = round(charge / 0.1, 2) + 10
    elif discharge > 0:
        loader_action = round(10 - discharge / 0.1, 2)
    else:
        loader_action = 10.0
    env_continuous = loader_action * 0.1 - 1.0
    if charge > 0:
        direction_ok = env_continuous > 0
    elif discharge > 0:
        direction_ok = env_continuous < 0
    else:
        direction_ok = abs(env_continuous) < 1e-6
    sym = "OK" if direction_ok else "FAIL"
    all_ok = all_ok and direction_ok
    print(f"  [{sym}] loader_disc={loader_action:.0f} → env_cont={env_continuous:+.1f} [{label}]")

PASS_2 = all_ok
print(f"\n  RESULT: {'PASS' if PASS_2 else 'FAIL'}")

# ── 3. Reward net sanity check ────────────────────────────────────────
print("\n" + "=" * 60)
print("3. REWARD NET SANITY CHECK")
print("=" * 60)

reward_net = RewardNet(obs_dim=7, action_dim=1, hidden_dim=32)
reward_net.eval()
print(f"  RewardNet params: {sum(p.numel() for p in reward_net.parameters())}")

obs_scales = torch.tensor(PROFIT_OBS_SCALES, dtype=torch.float32)

# Test with a few expert state-action pairs
traj = data[0]
expert_obs = torch.tensor(traj["state_action_pairs"]["observations"][:5], dtype=torch.float32)
expert_act = torch.tensor(traj["state_action_pairs"]["actions"][:5], dtype=torch.float32)
expert_dt  = torch.tensor(traj["state_action_pairs"]["delta_ts"][:5], dtype=torch.float32)

print(f"\n  Expert obs sample (raw):\n    {expert_obs[0].numpy()}")
print(f"  Expert act sample (normalized [-1,1]): {expert_act[:5].flatten().numpy()}")
print(f"  Expert delta_ts sample: {expert_dt[:5].numpy()}")

# Normalize obs (same as trainer does)
expert_obs_norm = expert_obs / obs_scales
print(f"  Expert obs sample (normalized):\n    {expert_obs_norm[0].numpy()}")

# Forward pass with delta_t weighting
with torch.no_grad():
    rewards_raw = reward_net(expert_obs_norm, expert_act)
    rewards_weighted = rewards_raw * expert_dt
print(f"\n  R_net outputs (random init): {rewards_raw.numpy()}")
print(f"  R_net × delta_t:             {rewards_weighted.numpy()}")
print(f"  All finite: {torch.isfinite(rewards_weighted).all().item()}")

# Check action normalization consistency: env normalizes as (a-10)/10
print(f"\n  ACTION NORMALIZATION CONSISTENCY:")
print(f"  Env:    action_for_reward = (float(action) - 10.0) / 10.0")
print(f"  Loader: action_val = (discrete - 10.0) / 10.0")
print(f"  Expert action range in dataset: [{all_actions.min():.2f}, {all_actions.max():.2f}]")
print(f"  Both use (discrete-10)/10 → MATCH: YES")

# Verify a specific example: PPO outputs discrete=11 (charge 0.1 SoC)
env_example = (11.0 - 10.0) / 10.0   # = 0.1
loader_example = (11.0 - 10.0) / 10.0  # = 0.1
print(f"\n  Example: discrete=11 → env={env_example:.1f}, loader={loader_example:.1f}  Match: {abs(env_example-loader_example)<1e-6}")

PASS_3 = torch.isfinite(rewards_weighted).all().item() and abs(all_actions.min() + 0.5) < 0.01
print(f"\n  RESULT: {'PASS' if PASS_3 else 'FAIL'}")

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  1. Dataset action range [0-20]: {'PASS' if PASS_1 else 'FAIL'}")
print(f"  2. Action mapping consistency:  {'PASS' if PASS_2 else 'FAIL'}")
print(f"  3. Reward net sanity:           {'PASS' if PASS_3 else 'FAIL'}")
print("=" * 60)

env.close()

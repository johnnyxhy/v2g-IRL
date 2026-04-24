import json, numpy as np

with open('data/processed_trajectories_deep_discrete.json') as f:
    data = json.load(f)

# Check a few trajectories
for i in range(3):
    traj = data[i]
    actions = np.array(traj['state_action_pairs']['actions'])
    obs = np.array(traj['state_action_pairs']['observations'])
    eid = traj['episodeID']
    seg = traj['segment']
    print(f"Episode {eid}, segment={seg}")
    print(f"  Actions shape: {actions.shape}, min={actions.min():.3f}, max={actions.max():.3f}")
    print(f"  Actions: {actions.flatten()[:20]}")
    print(f"  Obs shape: {obs.shape}")
    print(f"  Feature expectations: {np.array(traj['feature_expectation'])}")
    print()

# Global action stats
all_actions = np.concatenate([np.array(t['state_action_pairs']['actions']).flatten() for t in data])
print(f"Global action stats: min={all_actions.min():.3f}, max={all_actions.max():.3f}, mean={all_actions.mean():.3f}, std={all_actions.std():.3f}")
print(f"\nAction value distribution (rounded to 0.1):")
unique, counts = np.unique(np.round(all_actions, 1), return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u:7.1f}: {c}")

# What does the env produce for PPO?
# Discrete(21) -> actions are integers 0..20
# In env: action_for_reward = float(action)  -> 0.0 to 20.0
# In env: action = action * 0.1 - 1.0       -> -1.0 to 1.0
print("\n--- Action representation comparison ---")
print("Expert loader stores: charge_soc / ACTION_STEP_SIZE (e.g., 0.3 SoC -> 3.0)")
print("Env step receives: discrete 0-20, action_for_reward = float(0..20)")
print("Env converts to continuous: discrete * 0.1 - 1.0 -> [-1.0, 1.0]")

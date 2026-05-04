"""Probe reward component magnitudes including the action range penalty."""
import torch
import numpy as np
import json
import gymnasium as gym
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet

gym.register(id='V2GDeepEnv-profit', entry_point='irl.envs.V2GDeepEnv_profit:V2GDeepEnv', max_episode_steps=96)

reward_net = RewardNet(obs_dim=7, action_dim=1, hidden_dim=32)
reward_net.load_state_dict(torch.load(
    'models/Deep_profit_sum_0.1reg_50charge_1_continued/reward_net_epoch20.pt', weights_only=True))
reward_net.eval()
model = SAC.load('models/Deep_profit_sum_0.1reg_50charge_1_continued/sac_epoch20')

with open('data/processed_trajectories_deep_profit.json') as f:
    expert_data = json.load(f)
male_trajs = [t for t in expert_data if 'Male 50-59' in t['segment']]


def range_pen(a, coeff, mn=0.01, mx=0.3):
    if a == 0.0:
        return 0.0
    ab = abs(a)
    if ab < mn:
        return coeff * (1.0 - ab / mn)
    if ab > mx:
        return coeff * (ab / mx - 1.0)
    return 0.0


# ---- 1. Expert action distribution ----
print("=== EXPERT ACTION DISTRIBUTION ===")
expert_actions = []
for t in male_trajs:
    for a in t['state_action_pairs']['actions']:
        expert_actions.append(abs(a[0]))
ea = np.array(expert_actions)
print(f"N={len(ea)}, mean={ea.mean():.4f}, std={ea.std():.4f}, min={ea.min():.4f}, max={ea.max():.4f}")
print(f"  |a| == 0.0:          {(ea == 0.0).sum():4d} / {len(ea)} ({100*(ea == 0.0).mean():.1f}%)")
print(f"  0 < |a| < 0.01:      {((ea > 0) & (ea < 0.01)).sum():4d} / {len(ea)} ({100*((ea > 0) & (ea < 0.01)).mean():.1f}%)")
print(f"  0.01 <= |a| < 0.05:  {((ea >= 0.01) & (ea < 0.05)).sum():4d} / {len(ea)} ({100*((ea >= 0.01) & (ea < 0.05)).mean():.1f}%)")
print(f"  0.05 <= |a| <= 0.3:  {((ea >= 0.05) & (ea <= 0.3)).sum():4d} / {len(ea)} ({100*((ea >= 0.05) & (ea <= 0.3)).mean():.1f}%)")
print(f"  0.3 < |a| <= 0.5:   {((ea > 0.3) & (ea <= 0.5)).sum():4d} / {len(ea)} ({100*((ea > 0.3) & (ea <= 0.5)).mean():.1f}%)")
print(f"  |a| > 0.5:          {(ea > 0.5).sum():4d} / {len(ea)} ({100*(ea > 0.5).mean():.1f}%)")
print()

# ---- 2. Reward net output across action grid ----
print("=== REWARD NET OUTPUT vs ACTION (t=30, soc=0.5, target=0.3) ===")
test_obs = np.array([30/96, 0.5, 0.3, 0.30/0.47, 0.5, 20/96, 7.4/22], dtype=np.float32)
action_grid = np.linspace(-0.5, 0.5, 21)
with torch.no_grad():
    obs_t = torch.tensor(test_obs).unsqueeze(0).expand(len(action_grid), -1)
    act_t = torch.tensor(action_grid, dtype=torch.float32).unsqueeze(-1)
    rews = reward_net(obs_t, act_t).numpy()

header = f"{'a':>6s}  {'R_net':>7s}  {'a_pen':>7s}  {'rp@.01':>7s}  {'rp@.02':>7s}  {'tot@.02':>8s}"
print(f"  {header}")
for a, r in zip(action_grid, rews):
    ap = 0.1 * a**2
    rp1 = range_pen(a, 0.01)
    rp2 = range_pen(a, 0.02)
    tot = r - ap - rp2
    print(f"  {a:+.3f}  {r:+.4f}  {ap:.4f}  {rp1:.4f}  {rp2:.4f}  {tot:+.4f}")
print()

# ---- 3. Simulate episodes ----
print("=== SIMULATED EPISODE ANALYSIS (5 trajs) ===")
env = gym.make('V2GDeepEnv-profit')
env.unwrapped.set_reward_net(reward_net)
env.unwrapped.set_action_penalty_coeff(0.1)
vec_env = DummyVecEnv([lambda: env])

net_rs, a_pens, acts_abs = [], [], []
for ti in range(min(5, len(male_trajs))):
    traj = male_trajs[ti]
    vec_env.envs[0].unwrapped.set_initial_states(traj['initial_values'])
    obs = vec_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        e = vec_env.envs[0].unwrapped
        pof = e._flatten_obs_for_reward()
        av = float(np.round(action[0][0], 2))
        with torch.no_grad():
            ot = torch.tensor(pof).unsqueeze(0)
            at = torch.tensor([[av]])
            nr = reward_net(ot, at).item()
        net_rs.append(nr)
        a_pens.append(0.1 * av**2)
        acts_abs.append(abs(av))
        obs, _, dones, _ = vec_env.step(action)
        done = bool(dones[0])

net_rs = np.array(net_rs)
a_pens = np.array(a_pens)
acts_abs = np.array(acts_abs)

tiny = (acts_abs > 0) & (acts_abs < 0.01)
big = acts_abs > 0.3

print(f"\nAgent action dist: N={len(acts_abs)}")
print(f"  |a| == 0:     {(acts_abs == 0).sum()}")
print(f"  0<|a|<0.01:   {tiny.sum()}")
print(f"  0.01<=|a|<0.05: {((acts_abs >= 0.01) & (acts_abs < 0.05)).sum()}")
print(f"  0.05<=|a|<=0.3: {((acts_abs >= 0.05) & (acts_abs <= 0.3)).sum()}")
print(f"  |a|>0.3:      {big.sum()}")
print()

print(f"Reference magnitudes:")
print(f"  |R_net| mean: {np.abs(net_rs).mean():.4f}")
print(f"  a_pen mean:   {a_pens.mean():.4f}")
print()

# ---- 4. Range penalty sweep ----
print("=== RANGE PENALTY SWEEP (percentage-based) ===")
print(f"{'coeff':>6s}  {'mean_rp':>8s}  | {'rp@tiny':>8s}  {'|Rnet|@tiny':>11s}  {'ratio':>6s}  | {'rp@big':>8s}  {'|Rnet|@big':>11s}  {'ratio':>6s}")
for coeff in [0.005, 0.01, 0.02, 0.05, 0.1]:
    rps = np.array([range_pen(a, coeff) for a in acts_abs])
    rp_tiny = rps[tiny].mean() if tiny.any() else 0
    rp_big = rps[big].mean() if big.any() else 0
    nr_tiny = np.abs(net_rs[tiny]).mean() if tiny.any() else 1
    nr_big = np.abs(net_rs[big]).mean() if big.any() else 1
    r_tiny = rp_tiny / nr_tiny if nr_tiny > 0 else 0
    r_big = rp_big / nr_big if nr_big > 0 else 0
    print(f"{coeff:6.3f}  {rps.mean():8.4f}  | {rp_tiny:8.4f}  {nr_tiny:11.4f}  {r_tiny:6.2f}  | {rp_big:8.4f}  {nr_big:11.4f}  {r_big:6.2f}")

"""
probe_reward_discrete.py
========================
Analyses what the trained Deep MaxEnt discrete reward net has learned about
price-conditional charging behaviour.

Three questions answered:
  1. Reward sweep across actions at LOW vs HIGH price, holding SoC constant.
  2. Reward surface: charge action reward as a function of (SoC, price).
  3. Per-trajectory: compare expert reward on morning actions vs agent reward
     on the same (state, charge) pair — highlights if the net undervalues them.
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_DIR   = "models/DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male5059(old)"
EPOCH       = 20
DATA_PATH   = "data/processed_trajectories_deep_discrete_pricediff.json"
SEGMENT     = "Male 50-59"
HIDDEN_DIM  = 32
MEAN_PRICE  = 0.27    # ≈ mean of the profile
# ────────────────────────────────────────────────────────────────────────────

reward_net = RewardNet(obs_dim=7, action_dim=1, hidden_dim=HIDDEN_DIM)
reward_net.load_state_dict(torch.load(
    f"{MODEL_DIR}/reward_net_epoch{EPOCH}.pt", weights_only=True))
reward_net.eval()

with open(DATA_PATH) as f:
    expert_data = json.load(f)
trajs = [t for t in expert_data if t["segment"] == SEGMENT]
print(f"Loaded {len(trajs)} trajectories for segment '{SEGMENT}'")

# Discrete action space: 0-20, idle=10.  Normalise to [-1, 1] like the trainer.
ALL_ACTIONS   = np.arange(0, 21, dtype=np.float32)
NORM_ACTIONS  = (ALL_ACTIONS - 10.0) / 10.0   # −1 … +1

energy_price_profile = np.array([
    0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24, 0.26,
    0.27, 0.28, 0.30, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
    0.41, 0.42, 0.43, 0.44, 0.44, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.47,
    0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.42,
    0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32, 0.31, 0.30, 0.28,
    0.27, 0.26, 0.24, 0.23, 0.22, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
    0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07
])


def make_obs(timestep, soc, soc_target, price, battery_cap_idx=1,
             time_to_journey=20, charger_power=7.4):
    """Return a single normalised obs vector matching PROFIT_OBS_SCALES."""
    raw = np.array([timestep, soc, soc_target, price,
                    battery_cap_idx, time_to_journey, charger_power], dtype=np.float32)
    return raw / PROFIT_OBS_SCALES


def reward_for_actions(obs_vec, norm_actions):
    """Return R_net(obs, a) for each action in norm_actions (1-D array)."""
    with torch.no_grad():
        obs_t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0).expand(len(norm_actions), -1)
        act_t = torch.tensor(norm_actions, dtype=torch.float32).unsqueeze(-1)
        return reward_net(obs_t, act_t).numpy()


# ── 1. ACTION SWEEP: low vs mid vs high price, SoC fixed above target ────────
print("\n=== 1. ACTION SWEEP ACROSS PRICE CONDITIONS (soc=0.60, target=0.50) ===")
price_scenarios = {
    "Low  (t=2,  £0.07)": (2,  0.07),
    "Mid  (t=24, £0.27)": (24, 0.27),
    "High (t=47, £0.47)": (47, 0.47),
}
soc, soc_tgt = 0.60, 0.50

fig1, ax1 = plt.subplots(figsize=(9, 5))
for label, (t, p) in price_scenarios.items():
    obs = make_obs(t, soc, soc_tgt, p)
    rews = reward_for_actions(obs, NORM_ACTIONS)
    ax1.plot(ALL_ACTIONS - 10, rews, marker='o', markersize=3, label=label)
ax1.axvline(0, color='k', linestyle=':', linewidth=0.8, label='Idle (a=0)')
ax1.set_xlabel("Action (negative=discharge, positive=charge)")
ax1.set_ylabel("R_net(s, a)")
ax1.set_title(f"Reward vs Action at 3 Price Levels\n(soc={soc}, soc_target={soc_tgt})")
ax1.legend()
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/probe_action_sweep_price.png", dpi=150)
plt.show()

# ── 2. ACTION SWEEP: SoC above vs below target, at LOW price ─────────────────
print("\n=== 2. ACTION SWEEP: SoC ABOVE vs BELOW TARGET at LOW PRICE ===")
soc_scenarios = {
    "SoC=0.30 < target=0.50 (need to charge)": (0.30, 0.50),
    "SoC=0.50 = target=0.50 (at target)":      (0.50, 0.50),
    "SoC=0.70 > target=0.50 (above target)":   (0.70, 0.50),
}
low_price, low_t = 0.07, 2

fig2, ax2 = plt.subplots(figsize=(9, 5))
for label, (s, tgt) in soc_scenarios.items():
    obs = make_obs(low_t, s, tgt, low_price)
    rews = reward_for_actions(obs, NORM_ACTIONS)
    ax2.plot(ALL_ACTIONS - 10, rews, marker='o', markersize=3, label=label)
ax2.axvline(0, color='k', linestyle=':', linewidth=0.8, label='Idle')
ax2.set_xlabel("Action (negative=discharge, positive=charge)")
ax2.set_ylabel("R_net(s, a)")
ax2.set_title(f"Reward vs Action for Different SoC Levels\n(LOW price £{low_price}, timestep {low_t})")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/probe_soc_vs_action_lowprice.png", dpi=150)
plt.show()

# ── 3. REWARD SURFACE: R(charge action=+1 step) as a function of (SoC, price) ──
print("\n=== 3. REWARD SURFACE: charge(+1 step) vs (SoC, price) ===")
soc_grid   = np.linspace(0.1, 0.9, 33)
price_grid = energy_price_profile  # 96 real price values

# Small fixed charge: action index 11 (one step above idle = +0.1 normalised)
small_charge_norm = np.array([(11 - 10) / 10.0], dtype=np.float32)  # +0.1
idle_norm         = np.array([0.0],               dtype=np.float32)  # 0.0

surface_charge = np.zeros((len(soc_grid), len(price_grid)))
surface_idle   = np.zeros((len(soc_grid), len(price_grid)))

with torch.no_grad():
    for i, s in enumerate(soc_grid):
        obs_batch_charge = []
        obs_batch_idle   = []
        for j, (p, t) in enumerate(zip(price_grid, range(96))):
            obs_batch_charge.append(make_obs(t, s, 0.50, p))
            obs_batch_idle.append(make_obs(t, s, 0.50, p))

        obs_t = torch.tensor(np.array(obs_batch_charge), dtype=torch.float32)
        act_charge = torch.full((96, 1), small_charge_norm[0])
        act_idle   = torch.zeros((96, 1))
        surface_charge[i] = reward_net(obs_t, act_charge).numpy()
        surface_idle[i]   = reward_net(obs_t, act_idle).numpy()

# Advantage of charging over idle
advantage = surface_charge - surface_idle

fig3, axes = plt.subplots(1, 3, figsize=(15, 4))
im0 = axes[0].imshow(surface_charge, aspect='auto', origin='lower',
                     extent=[0, 95, soc_grid[0], soc_grid[-1]], cmap='RdYlGn')
axes[0].set_title("R_net(s, charge+1)")
axes[0].set_xlabel("Timestep (proxy for price)")
axes[0].set_ylabel("SoC")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(surface_idle, aspect='auto', origin='lower',
                     extent=[0, 95, soc_grid[0], soc_grid[-1]], cmap='RdYlGn')
axes[1].set_title("R_net(s, idle)")
axes[1].set_xlabel("Timestep (proxy for price)")
axes[1].set_ylabel("SoC")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(advantage, aspect='auto', origin='lower',
                     extent=[0, 95, soc_grid[0], soc_grid[-1]], cmap='RdYlGn')
axes[2].set_title("Charge advantage over idle\n(positive = net prefers charging)")
axes[2].set_xlabel("Timestep (proxy for price)")
axes[2].set_ylabel("SoC")
plt.colorbar(im2, ax=axes[2])

plt.suptitle(f"Reward Surface: small charge (+1 step) vs idle  |  epoch {EPOCH}")
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/probe_reward_surface.png", dpi=150)
plt.show()

# ── 4. PER-TRAJECTORY: expert morning actions (t<15) vs what the net scores ──
print("\n=== 4. EXPERT MORNING ACTIONS (t < 15, charge) — reward net scores ===")

morning_rewards_charge = []  # R_net for expert's morning charge actions
morning_rewards_idle   = []  # R_net for idle at the same state
morning_advantages     = []
morning_prices         = []
morning_socs           = []

for traj in trajs:
    obs_list = traj["state_action_pairs"]["observations"]
    act_list = traj["state_action_pairs"]["actions"]

    for obs_raw, act_raw in zip(obs_list, act_list):
        timestep = obs_raw[0]
        if timestep >= 15:
            continue
        action_idx = int(round(act_raw[0] * 10 + 10))  # denormalise back to 0-20
        if action_idx <= 10:   # only charge actions (>10) interest us
            continue

        price    = energy_price_profile[min(int(timestep), 95)]
        soc_val  = obs_raw[1]

        obs_norm = np.array(obs_raw, dtype=np.float32) / PROFIT_OBS_SCALES
        act_norm = np.array([act_raw[0]], dtype=np.float32)
        idle_act = np.array([0.0], dtype=np.float32)

        with torch.no_grad():
            r_charge = reward_net(
                torch.tensor(obs_norm).unsqueeze(0),
                torch.tensor(act_norm).unsqueeze(0)
            ).item()
            r_idle = reward_net(
                torch.tensor(obs_norm).unsqueeze(0),
                torch.tensor(idle_act).unsqueeze(0)
            ).item()

        morning_rewards_charge.append(r_charge)
        morning_rewards_idle.append(r_idle)
        morning_advantages.append(r_charge - r_idle)
        morning_prices.append(price)
        morning_socs.append(soc_val)

morning_rewards_charge = np.array(morning_rewards_charge)
morning_rewards_idle   = np.array(morning_rewards_idle)
morning_advantages     = np.array(morning_advantages)

print(f"  Morning charge actions found: {len(morning_advantages)}")
print(f"  R_net(charge) : mean={morning_rewards_charge.mean():.4f}, std={morning_rewards_charge.std():.4f}")
print(f"  R_net(idle)   : mean={morning_rewards_idle.mean():.4f},   std={morning_rewards_idle.std():.4f}")
print(f"  Advantage     : mean={morning_advantages.mean():.4f},   std={morning_advantages.std():.4f}")
frac_positive = (morning_advantages > 0).mean()
print(f"  Fraction where charge > idle: {frac_positive:.1%}  ← should be ~1.0 if net learned morning charging")

# Compare to HIGH-price actions (t 36-55)
print("\n=== 4b. EXPERT HIGH-PRICE CHARGE ACTIONS (35 < t < 56) — reward net scores ===")
high_rewards_charge = []
high_advantages     = []

for traj in trajs:
    obs_list = traj["state_action_pairs"]["observations"]
    act_list = traj["state_action_pairs"]["actions"]

    for obs_raw, act_raw in zip(obs_list, act_list):
        timestep = obs_raw[0]
        if not (35 < timestep < 56):
            continue
        action_idx = int(round(act_raw[0] * 10 + 10))
        if action_idx <= 10:
            continue

        obs_norm = np.array(obs_raw, dtype=np.float32) / PROFIT_OBS_SCALES
        act_norm = np.array([act_raw[0]], dtype=np.float32)
        idle_act = np.array([0.0], dtype=np.float32)

        with torch.no_grad():
            r_c = reward_net(torch.tensor(obs_norm).unsqueeze(0),
                             torch.tensor(act_norm).unsqueeze(0)).item()
            r_i = reward_net(torch.tensor(obs_norm).unsqueeze(0),
                             torch.tensor(idle_act).unsqueeze(0)).item()
        high_rewards_charge.append(r_c)
        high_advantages.append(r_c - r_i)

high_advantages = np.array(high_advantages)
print(f"  High-price charge actions found: {len(high_advantages)}")
if len(high_advantages):
    print(f"  Advantage  : mean={high_advantages.mean():.4f}  (vs morning: {morning_advantages.mean():.4f})")
    print(f"  → If morning advantage ≈ high-price advantage, net is IGNORING price timing.")
    print(f"  → If morning > high, net has PARTIALLY learned low-price preference.")

# ── 5. PLOT: advantage of charging as a function of price (holding SoC fixed) ─
print("\n=== 5. CHARGE ADVANTAGE vs PRICE (SoC sweep) ===")
prices_test = energy_price_profile[::4]  # every 4 timesteps
timesteps_test = list(range(0, 96, 4))
soc_cases = [0.30, 0.50, 0.70]
charge_action_norm = (11 - 10) / 10.0  # +1 step charge

fig5, ax5 = plt.subplots(figsize=(9, 5))
for s in soc_cases:
    advantages_vs_price = []
    with torch.no_grad():
        for t, p in zip(timesteps_test, prices_test):
            obs = torch.tensor(make_obs(t, s, 0.50, p), dtype=torch.float32).unsqueeze(0)
            r_c = reward_net(obs, torch.tensor([[charge_action_norm]])).item()
            r_i = reward_net(obs, torch.tensor([[0.0]])).item()
            advantages_vs_price.append(r_c - r_i)
    ax5.plot(prices_test, advantages_vs_price, marker='o', markersize=4, label=f"SoC={s}")

ax5.axhline(0, color='k', linestyle='--', linewidth=0.8)
ax5.axvline(MEAN_PRICE, color='gray', linestyle=':', linewidth=0.8, label=f"Mean price £{MEAN_PRICE}")
ax5.set_xlabel("Energy price (£/kWh)")
ax5.set_ylabel("R_net(charge) − R_net(idle)")
ax5.set_title("Charge advantage over idle vs price\n(positive = net prefers charging, soc_target=0.50)")
ax5.legend()
ax5.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/probe_charge_advantage_vs_price.png", dpi=150)
plt.show()

print("\nDone. Plots saved to model directory.")

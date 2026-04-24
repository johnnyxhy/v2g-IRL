"""
Analyze reward network evolution across epochs for the Deep MaxEnt Discrete experiment.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "DeepMaxEntIRL_discrete_profit_exp1"
N_EPOCHS = 20
HIDDEN_DIM = 32


def load_reward_net(epoch):
    net = RewardNet(obs_dim=7, action_dim=1, hidden_dim=HIDDEN_DIM)
    path = MODEL_DIR / f"reward_net_epoch{epoch}.pt"
    net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    net.eval()
    return net


def get_flat_params(net):
    return torch.cat([p.detach().flatten() for p in net.parameters()])


def get_layer_stats(net):
    stats = {}
    for name, p in net.named_parameters():
        d = p.detach()
        stats[name] = {"mean": d.mean().item(), "std": d.std().item(), "norm": d.norm().item()}
    return stats


def make_probe_inputs():
    scales = PROFIT_OBS_SCALES
    rng = np.random.RandomState(42)
    n = 200
    obs_raw = np.zeros((n, 7), dtype=np.float32)
    obs_raw[:, 0] = rng.uniform(0, 96, n)
    obs_raw[:, 1] = rng.uniform(0, 1, n)
    obs_raw[:, 2] = rng.uniform(0, 1, n)
    obs_raw[:, 3] = rng.uniform(0, 0.47, n)
    obs_raw[:, 4] = rng.choice([0, 1, 2], n)
    obs_raw[:, 5] = rng.uniform(0, 96, n)
    obs_raw[:, 6] = rng.choice([3, 7.4, 11, 22], n)
    obs_norm = obs_raw / scales
    # Actions are discrete 0-20, normalized to [-1,1] via (discrete-10)/10
    discrete_actions = rng.randint(0, 21, (n, 1)).astype(np.float32)
    actions_norm = (discrete_actions - 10.0) / 10.0

    scenarios = {
        "high_soc_charge":            (np.array([48, 0.9, 0.5, 0.1,  1, 48, 7.4]) / scales, np.array([0.8])),
        "low_soc_charge":             (np.array([48, 0.2, 0.5, 0.1,  1, 48, 7.4]) / scales, np.array([0.8])),
        "high_soc_discharge":         (np.array([48, 0.9, 0.5, 0.3,  1, 48, 7.4]) / scales, np.array([-0.8])),
        "low_soc_discharge":          (np.array([48, 0.2, 0.5, 0.3,  1, 48, 7.4]) / scales, np.array([-0.8])),
        "idle_high_soc":              (np.array([48, 0.8, 0.5, 0.2,  1, 48, 7.4]) / scales, np.array([0.0])),
        "idle_low_soc":               (np.array([48, 0.2, 0.5, 0.2,  1, 48, 7.4]) / scales, np.array([0.0])),
        "cheap_price_charge":         (np.array([48, 0.5, 0.7, 0.05, 1, 48, 7.4]) / scales, np.array([0.8])),
        "expensive_price_charge":     (np.array([48, 0.5, 0.7, 0.45, 1, 48, 7.4]) / scales, np.array([0.8])),
        "cheap_price_discharge":      (np.array([48, 0.5, 0.3, 0.05, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "expensive_price_discharge":  (np.array([48, 0.5, 0.3, 0.45, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "urgent_need_charge":         (np.array([48, 0.3, 0.8, 0.2,  1,  5, 7.4]) / scales, np.array([0.9])),
        "no_urgency_idle":            (np.array([48, 0.5, 0.5, 0.2,  1, 80, 7.4]) / scales, np.array([0.0])),
        "needs_soc_but_expensive":    (np.array([48, 0.3, 0.7, 0.45, 1, 10, 7.4]) / scales, np.array([0.8])),
        "has_soc_cheap_discharge":    (np.array([48, 0.7, 0.3, 0.05, 1, 48, 7.4]) / scales, np.array([-0.8])),
        # soc_target scenarios (soc fixed at 0.5, varying soc_target)
        "below_target_charge":        (np.array([48, 0.3, 0.8, 0.20, 1, 48, 7.4]) / scales, np.array([0.8])),
        "above_target_charge":        (np.array([48, 0.7, 0.3, 0.20, 1, 48, 7.4]) / scales, np.array([0.8])),
        "below_target_discharge":     (np.array([48, 0.3, 0.8, 0.20, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "above_target_discharge":     (np.array([48, 0.7, 0.3, 0.20, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "at_target_idle":             (np.array([48, 0.5, 0.5, 0.20, 1, 48, 7.4]) / scales, np.array([0.0])),
        "far_below_target_idle":      (np.array([48, 0.2, 0.8, 0.20, 1, 48, 7.4]) / scales, np.array([0.0])),
    }
    return (
        torch.tensor(obs_norm, dtype=torch.float32),
        torch.tensor(actions_norm, dtype=torch.float32),
        scenarios,
    )


def main():
    print(f"Analyzing reward networks from: {MODEL_DIR}\n")

    nets = {}
    flat_params = {}
    layer_stats_all = {}
    for e in range(1, N_EPOCHS + 1):
        nets[e] = load_reward_net(e)
        flat_params[e] = get_flat_params(nets[e])
        layer_stats_all[e] = get_layer_stats(nets[e])

    n_params = flat_params[1].numel()
    print(f"Network has {n_params} parameters\n")
    epochs = list(range(1, N_EPOCHS + 1))

    # ---- 1. Parameter drift ----
    drifts = [(flat_params[e] - flat_params[1]).norm().item() for e in epochs]
    print("=" * 60)
    print("1. PARAMETER DRIFT (L2 from epoch 1)")
    print("=" * 60)
    for e in [1, 5, 10, 20, 30, 40, 50]:
        if e <= N_EPOCHS:
            print(f"  Epoch {e:2d}: {drifts[e-1]:.5f}")

    # ---- 2. Cosine similarity ----
    cosines = [
        torch.nn.functional.cosine_similarity(flat_params[e].unsqueeze(0), flat_params[e+1].unsqueeze(0)).item()
        for e in range(1, N_EPOCHS)
    ]
    cos_1_last = torch.nn.functional.cosine_similarity(flat_params[1].unsqueeze(0), flat_params[N_EPOCHS].unsqueeze(0)).item()
    print(f"\n  Mean consecutive cosine: {np.mean(cosines):.5f}")
    print(f"  Epoch 1 vs {N_EPOCHS} cosine:    {cos_1_last:.5f}")

    # ---- 3. Probe rewards ----
    obs_probe, act_probe, scenarios = make_probe_inputs()
    rewards_per_epoch = {}
    for e in epochs:
        with torch.no_grad():
            rewards_per_epoch[e] = nets[e](obs_probe, act_probe).numpy()

    print("\n" + "=" * 60)
    print("2. REWARD OUTPUT ON 200 RANDOM PROBES")
    print("=" * 60)
    for e in [1, 5, 10, 20, 30, 40, 50]:
        if e <= N_EPOCHS:
            r = rewards_per_epoch[e]
            print(f"  Epoch {e:2d}: mean={r.mean():+.4f}, std={r.std():.4f}, min={r.min():+.4f}, max={r.max():+.4f}")

    print("\n  Spearman rank corr vs epoch 1:")
    for e in [10, 20, 30, 40, 50]:
        if e <= N_EPOCHS:
            corr, pval = spearmanr(rewards_per_epoch[1], rewards_per_epoch[e])
            print(f"    E1 vs E{e:2d}: ρ={corr:+.4f} (p={pval:.2e})")

    # ---- 4. Scenario rewards ----
    print("\n" + "=" * 60)
    print("3. REWARD ON TARGETED SCENARIOS (epoch 1 vs epoch 50)")
    print("=" * 60)
    scenario_evolution = {name: [] for name in scenarios}
    for e in epochs:
        for name, (obs, act) in scenarios.items():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor(act, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                scenario_evolution[name].append(nets[e](obs_t, act_t).item())

    r1  = {name: scenario_evolution[name][0]  for name in scenarios}
    r50 = {name: scenario_evolution[name][-1] for name in scenarios}
    for name in scenarios:
        print(f"  {name:35s}: E1={r1[name]:+.4f}  E{N_EPOCHS}={r50[name]:+.4f}  Δ={r50[name]-r1[name]:+.4f}")

    print(f"\n  Behavioral checks (Epoch {N_EPOCHS}):")
    r = r50
    checks = [
        ("Low SoC > high SoC for charging",            r["low_soc_charge"]           > r["high_soc_charge"],           r["low_soc_charge"],           r["high_soc_charge"]),
        ("High SoC > low SoC for discharging",         r["high_soc_discharge"]        > r["low_soc_discharge"],         r["high_soc_discharge"],        r["low_soc_discharge"]),
        ("Cheap > expensive for charging",             r["cheap_price_charge"]        > r["expensive_price_charge"],    r["cheap_price_charge"],        r["expensive_price_charge"]),
        ("Expensive > cheap for discharging",          r["expensive_price_discharge"] > r["cheap_price_discharge"],     r["expensive_price_discharge"], r["cheap_price_discharge"]),
        ("Urgent charge scores highly (|r|>0.1)",      abs(r["urgent_need_charge"])   > 0.1,                            r["urgent_need_charge"],        None),
        # soc_target checks
        ("Below target: charge > discharge",           r["below_target_charge"]       > r["below_target_discharge"],    r["below_target_charge"],       r["below_target_discharge"]),
        ("Above target: discharge > charge",           r["above_target_discharge"]    > r["above_target_charge"],       r["above_target_discharge"],    r["above_target_charge"]),
        ("Below target charge > above target charge",  r["below_target_charge"]       > r["above_target_charge"],       r["below_target_charge"],       r["above_target_charge"]),
        ("At target idle > far below target idle",     r["at_target_idle"]            > r["far_below_target_idle"],     r["at_target_idle"],            r["far_below_target_idle"]),
    ]
    for desc, passed, v1, v2 in checks:
        mark = "\u2713" if passed else "\u2717"
        if v2 is not None:
            print(f"    {mark} {desc}: {v1:+.4f} vs {v2:+.4f}")
        else:
            print(f"    {mark} {desc}: {v1:+.4f}")

    # ---- PLOTS ----
    out_dir = MODEL_DIR / "analysis"
    out_dir.mkdir(exist_ok=True)

    # Plot 1: Parameter drift
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, drifts, 'b-o', markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("L2 Distance from Epoch 1")
    ax.set_title("Parameter Drift Over Training"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "param_drift.png", dpi=150); plt.close()

    # Plot 2: Consecutive cosine similarity
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, N_EPOCHS), cosines, 'r-o', markersize=3)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cosine (t → t+1)")
    ax.set_title("Parameter Cosine Similarity Between Consecutive Epochs"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "cosine_similarity.png", dpi=150); plt.close()

    # Plot 3: Reward output distribution
    means = [rewards_per_epoch[e].mean() for e in epochs]
    stds  = [rewards_per_epoch[e].std()  for e in epochs]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(epochs, [m-s for m,s in zip(means,stds)], [m+s for m,s in zip(means,stds)], alpha=0.3)
    ax.plot(epochs, means, 'b-o', markersize=3, label="Mean ± Std")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
    ax.set_title("Reward Output Distribution on Fixed Probes"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "reward_distribution.png", dpi=150); plt.close()

    # Plot 4: Scenario reward evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    for name in ["low_soc_charge", "high_soc_charge", "cheap_price_charge", "expensive_price_charge"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=2, label=name)
    ax.set_title("Charging Scenarios"); ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name in ["high_soc_discharge", "low_soc_discharge", "cheap_price_discharge", "expensive_price_discharge"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=2, label=name)
    ax.set_title("Discharging Scenarios"); ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for name in ["idle_high_soc", "idle_low_soc", "urgent_need_charge", "no_urgency_idle"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=2, label=name)
    ax.set_title("Idle / Urgency Scenarios"); ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, [scenario_evolution["low_soc_charge"][i]  - scenario_evolution["high_soc_charge"][i]  for i in range(N_EPOCHS)], '-o', markersize=2, label="low_soc - high_soc (charge)")
    ax.plot(epochs, [scenario_evolution["high_soc_discharge"][i] - scenario_evolution["low_soc_discharge"][i] for i in range(N_EPOCHS)], '-o', markersize=2, label="high_soc - low_soc (discharge)")
    ax.plot(epochs, [scenario_evolution["cheap_price_charge"][i] - scenario_evolution["expensive_price_charge"][i] for i in range(N_EPOCHS)], '-o', markersize=2, label="cheap - expensive (charge)")
    ax.plot(epochs, [scenario_evolution["expensive_price_discharge"][i] - scenario_evolution["cheap_price_discharge"][i] for i in range(N_EPOCHS)], '-o', markersize=2, label="expensive - cheap (discharge)")
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_title("Reward Gaps (positive = sensible preference)"); ax.set_xlabel("Epoch"); ax.set_ylabel("Reward Gap")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Scenario-Based Reward Evolution", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "scenario_evolution.png", dpi=150); plt.close()

    # Plot 4b: soc_target scenario evolution
    soc_target_scenario_names = [
        "below_target_charge", "above_target_charge",
        "below_target_discharge", "above_target_discharge",
        "at_target_idle", "far_below_target_idle",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name in ["below_target_charge", "above_target_charge", "below_target_discharge", "above_target_discharge"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=3, label=name)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_title("SoC Target: Charge vs Discharge Scenarios")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, scenario_evolution["at_target_idle"],        '-o', markersize=3, label="at_target_idle")
    ax.plot(epochs, scenario_evolution["far_below_target_idle"], '-o', markersize=3, label="far_below_target_idle")
    gap = [scenario_evolution["below_target_charge"][i] - scenario_evolution["below_target_discharge"][i] for i in range(N_EPOCHS)]
    ax.plot(epochs, gap, '--s', markersize=3, label="below_target: charge−discharge gap")
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_title("SoC Target: Idle Behaviour & Preference Gap")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward / Gap")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("SoC Target Scenario Evolution", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "scenario_evolution_soc_target.png", dpi=150); plt.close()

    # Plot 5: SoC action value function heatmap at final epoch
    soc_values = np.linspace(0.1, 0.9, 17)
    action_values = np.linspace(-1, 1, 21)
    base_obs = np.array([48, 0.5, 0.5, 0.20, 1, 48, 7.4], dtype=np.float32)
    scales = PROFIT_OBS_SCALES
    reward_grid = np.zeros((len(soc_values), len(action_values)))
    net_final = nets[N_EPOCHS]
    for i, soc in enumerate(soc_values):
        for j, act in enumerate(action_values):
            obs_p = base_obs.copy(); obs_p[1] = soc
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor([[act]], dtype=torch.float32)
            with torch.no_grad():
                reward_grid[i, j] = net_final(obs_t, act_t).item()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(reward_grid, aspect='auto', origin='lower',
                   extent=[action_values[0], action_values[-1], soc_values[0], soc_values[-1]],
                   cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Reward')
    ax.set_xlabel("Action (negative=discharge, positive=charge)")
    ax.set_ylabel("State of Charge (SoC)")
    ax.set_title(f"R(SoC, action) Heatmap — Epoch {N_EPOCHS} (timestep=48, price=£0.20)")
    fig.tight_layout(); fig.savefig(out_dir / "reward_heatmap_soc_action.png", dpi=150); plt.close()

    # Plot 6: Price action value function heatmap at final epoch
    price_values = np.linspace(0.07, 0.47, 17)
    reward_grid_price = np.zeros((len(price_values), len(action_values)))
    for i, price in enumerate(price_values):
        for j, act in enumerate(action_values):
            obs_p = base_obs.copy(); obs_p[3] = price
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor([[act]], dtype=torch.float32)
            with torch.no_grad():
                reward_grid_price[i, j] = net_final(obs_t, act_t).item()

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(reward_grid_price, aspect='auto', origin='lower',
                   extent=[action_values[0], action_values[-1], price_values[0], price_values[-1]],
                   cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Reward')
    ax.set_xlabel("Action (negative=discharge, positive=charge)")
    ax.set_ylabel("Energy Price (£/kWh)")
    ax.set_title(f"R(price, action) Heatmap — Epoch {N_EPOCHS} (SoC=0.5, timestep=48)")
    fig.tight_layout(); fig.savefig(out_dir / "reward_heatmap_price_action.png", dpi=150); plt.close()

    # Plot 7: soc_target vs action heatmap at final epoch
    soc_target_values = np.linspace(0.05, 0.95, 19)
    # Fix soc at 0.5 so the deficit (soc_target - soc) varies from -0.45 to +0.45
    base_obs_t = np.array([48, 0.5, 0.5, 0.20, 1, 48, 7.4], dtype=np.float32)
    reward_grid_soc_target = np.zeros((len(soc_target_values), len(action_values)))
    for i, soc_t in enumerate(soc_target_values):
        for j, act in enumerate(action_values):
            obs_p = base_obs_t.copy(); obs_p[2] = soc_t
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor([[act]], dtype=torch.float32)
            with torch.no_grad():
                reward_grid_soc_target[i, j] = net_final(obs_t, act_t).item()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    im = ax.imshow(reward_grid_soc_target, aspect='auto', origin='lower',
                   extent=[action_values[0], action_values[-1], soc_target_values[0], soc_target_values[-1]],
                   cmap='RdYlGn')
    plt.colorbar(im, ax=ax, label='Reward')
    ax.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Action (negative=discharge, positive=charge)")
    ax.set_ylabel("SoC Target")
    ax.set_title(f"R(soc_target, action) — Epoch {N_EPOCHS}\n(SoC=0.5, timestep=48, price=£0.20)")

    # Annotate the soc=0.5 line (where deficit=0)
    ax.axhline(0.5, color='cyan', linestyle=':', linewidth=1.0)
    ax.text(action_values[-1] * 0.95, 0.5 + 0.02, 'SoC=target', color='cyan', fontsize=7, ha='right')

    # Side panel: reward vs action slices at low/mid/high soc_target
    ax = axes[1]
    for soc_t_val, label in [(0.2, 'soc_target=0.2 (surplus)'),
                              (0.5, 'soc_target=0.5 (at target)'),
                              (0.8, 'soc_target=0.8 (deficit)')]:
        idx = np.argmin(np.abs(soc_target_values - soc_t_val))
        ax.plot(action_values, reward_grid_soc_target[idx], '-o', markersize=2, label=label)
    ax.axvline(0, color='k', linestyle='--', alpha=0.4)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel("Action (negative=discharge, positive=charge)")
    ax.set_ylabel("Reward")
    ax.set_title(f"Reward vs Action Slices by SoC Target — Epoch {N_EPOCHS}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("SoC Target vs Action Reward Surface", fontsize=13)
    fig.tight_layout(); fig.savefig(out_dir / "reward_heatmap_soc_target_action.png", dpi=150); plt.close()

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    r_corr, _ = spearmanr(rewards_per_epoch[1], rewards_per_epoch[N_EPOCHS])
    print(f"  Total param drift (E1→E{N_EPOCHS}):          {drifts[-1]:.4f}")
    print(f"  Mean consecutive cosine:             {np.mean(cosines):.5f}")
    print(f"  E1 vs E{N_EPOCHS} cosine:                    {cos_1_last:.5f}")
    print(f"  E1 vs E{N_EPOCHS} reward rank corr (ρ):     {r_corr:+.4f}")
    print(f"  Reward mean  E1→E{N_EPOCHS}:  {rewards_per_epoch[1].mean():+.4f} → {rewards_per_epoch[N_EPOCHS].mean():+.4f}")
    print(f"  Reward std   E1→E{N_EPOCHS}:  {rewards_per_epoch[1].std():.4f} → {rewards_per_epoch[N_EPOCHS].std():.4f}")
    print(f"\n  Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()

"""
Analyze reward network evolution across 30 epochs to determine if meaningful learning occurred.

Checks:
1. Parameter drift (L2 distance from epoch 1 weights)
2. Per-layer weight statistics (mean, std, norm)
3. Cosine similarity between consecutive epochs
4. Reward output comparison on fixed inputs across epochs
5. Gradient of change over time
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "Deep_profit_sum_0.05reg_1e-3boundarypenalty_30_continued"
N_EPOCHS = 20
HIDDEN_DIM = 32  # from the training config


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
        stats[name] = {
            "mean": d.mean().item(),
            "std": d.std().item(),
            "norm": d.norm().item(),
            "min": d.min().item(),
            "max": d.max().item(),
            "abs_mean": d.abs().mean().item(),
        }
    return stats


def make_probe_inputs():
    """Create a fixed set of diverse (obs, action) inputs to evaluate reward across epochs."""
    scales = PROFIT_OBS_SCALES
    rng = np.random.RandomState(42)
    n = 200

    # Structured probes covering meaningful scenarios
    obs_raw = np.zeros((n, 7), dtype=np.float32)
    obs_raw[:, 0] = rng.uniform(0, 96, n)         # timestep
    obs_raw[:, 1] = rng.uniform(0, 1, n)           # soc
    obs_raw[:, 2] = rng.uniform(0, 1, n)           # soc_target
    obs_raw[:, 3] = rng.uniform(0, 0.47, n)        # energy_price
    obs_raw[:, 4] = rng.choice([0, 1, 2], n)       # battery_capacity index
    obs_raw[:, 5] = rng.uniform(0, 96, n)          # time_to_next_journey
    obs_raw[:, 6] = rng.choice([3, 7.4, 11, 22], n)  # current_charger_power

    obs_norm = obs_raw / scales
    actions = rng.uniform(-1, 1, (n, 1)).astype(np.float32)

    # Also create targeted scenarios
    scenarios = {
        "high_soc_charge": (np.array([48, 0.9, 0.5, 0.1, 1, 48, 7.4]) / scales, np.array([0.8])),
        "low_soc_charge": (np.array([48, 0.2, 0.5, 0.1, 1, 48, 7.4]) / scales, np.array([0.8])),
        "high_soc_discharge": (np.array([48, 0.9, 0.5, 0.3, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "low_soc_discharge": (np.array([48, 0.2, 0.5, 0.3, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "idle_high_soc": (np.array([48, 0.8, 0.5, 0.2, 1, 48, 7.4]) / scales, np.array([0.0])),
        "idle_low_soc": (np.array([48, 0.2, 0.5, 0.2, 1, 48, 7.4]) / scales, np.array([0.0])),
        "cheap_price_charge": (np.array([48, 0.5, 0.7, 0.05, 1, 48, 7.4]) / scales, np.array([0.8])),
        "expensive_price_charge": (np.array([48, 0.5, 0.7, 0.45, 1, 48, 7.4]) / scales, np.array([0.8])),
        "cheap_price_discharge": (np.array([48, 0.5, 0.3, 0.05, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "expensive_price_discharge": (np.array([48, 0.5, 0.3, 0.45, 1, 48, 7.4]) / scales, np.array([-0.8])),
        "urgent_need_charge": (np.array([48, 0.3, 0.8, 0.2, 1, 5, 7.4]) / scales, np.array([0.9])),
        "no_urgency_idle": (np.array([48, 0.5, 0.5, 0.2, 1, 80, 7.4]) / scales, np.array([0.0])),
    }

    return (
        torch.tensor(obs_norm, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.float32),
        scenarios,
    )


def main():
    print(f"Analyzing reward networks from: {MODEL_DIR}\n")

    # Load all networks
    nets = {}
    flat_params = {}
    layer_stats = {}
    for e in range(1, N_EPOCHS + 1):
        nets[e] = load_reward_net(e)
        flat_params[e] = get_flat_params(nets[e])
        layer_stats[e] = get_layer_stats(nets[e])

    n_params = flat_params[1].numel()
    print(f"Network has {n_params} parameters\n")

    # ---- 1. Parameter drift from epoch 1 ----
    print("=" * 60)
    print("1. PARAMETER DRIFT (L2 distance from epoch 1 weights)")
    print("=" * 60)
    drifts = []
    for e in range(1, N_EPOCHS + 1):
        d = (flat_params[e] - flat_params[1]).norm().item()
        drifts.append(d)
        if e % 5 == 0 or e == 1:
            print(f"  Epoch {e:2d}: L2 drift = {d:.6f}")
    print()

    # ---- 2. Consecutive cosine similarity ----
    print("=" * 60)
    print("2. CONSECUTIVE EPOCH COSINE SIMILARITY")
    print("=" * 60)
    cosines = []
    for e in range(1, N_EPOCHS):
        cos = torch.nn.functional.cosine_similarity(
            flat_params[e].unsqueeze(0), flat_params[e + 1].unsqueeze(0)
        ).item()
        cosines.append(cos)
        if e % 5 == 0 or e <= 3:
            print(f"  Epoch {e:2d} → {e+1:2d}: cosine = {cos:.6f}")
    print(f"  Mean cosine: {np.mean(cosines):.6f}")
    print(f"  Min cosine:  {np.min(cosines):.6f}")
    print()

    # Also compute epoch 1 vs epoch 30 cosine
    cos_1_30 = torch.nn.functional.cosine_similarity(
        flat_params[1].unsqueeze(0), flat_params[N_EPOCHS].unsqueeze(0)
    ).item()
    print(f"  Epoch 1 vs Epoch {N_EPOCHS} cosine: {cos_1_30:.6f}")
    print()

    # ---- 3. Per-layer weight statistics ----
    print("=" * 60)
    print("3. PER-LAYER WEIGHT STATISTICS (Epoch 1 vs 30)")
    print("=" * 60)
    for name in layer_stats[1]:
        s1 = layer_stats[1][name]
        s30 = layer_stats[N_EPOCHS][name]
        print(f"\n  {name}:")
        print(f"    Epoch  1: mean={s1['mean']:+.5f}, std={s1['std']:.5f}, norm={s1['norm']:.5f}")
        print(f"    Epoch 30: mean={s30['mean']:+.5f}, std={s30['std']:.5f}, norm={s30['norm']:.5f}")
        print(f"    Change:   Δmean={s30['mean']-s1['mean']:+.5f}, Δstd={s30['std']-s1['std']:+.5f}, Δnorm={s30['norm']-s1['norm']:+.5f}")
    print()

    # ---- 4. Reward outputs on probe inputs ----
    print("=" * 60)
    print("4. REWARD OUTPUT ON FIXED PROBE INPUTS")
    print("=" * 60)
    obs_probe, act_probe, scenarios = make_probe_inputs()

    rewards_per_epoch = {}
    for e in range(1, N_EPOCHS + 1):
        with torch.no_grad():
            rewards_per_epoch[e] = nets[e](obs_probe, act_probe).numpy()

    # Summary statistics of reward outputs
    print("\n  Reward output statistics on 200 random probes:")
    for e in [1] + list(range(5, N_EPOCHS + 1, 5)):
        if e > N_EPOCHS:
            continue
        r = rewards_per_epoch[e]
        print(f"    Epoch {e:2d}: mean={r.mean():+.4f}, std={r.std():.4f}, min={r.min():+.4f}, max={r.max():+.4f}")

    # Correlation of reward rankings
    print("\n  Spearman rank correlation of reward outputs vs epoch 1:")
    from scipy.stats import spearmanr
    for e in list(range(5, N_EPOCHS + 1, 5)):
        if e > N_EPOCHS:
            continue
        corr, pval = spearmanr(rewards_per_epoch[1], rewards_per_epoch[e])
        print(f"    Epoch  1 vs {e:2d}: ρ = {corr:+.4f} (p = {pval:.2e})")

    # ---- 5. Scenario-based analysis ----
    print("\n" + "=" * 60)
    print("5. REWARD ON TARGETED SCENARIOS (epoch 1 vs 30)")
    print("=" * 60)
    scenario_rewards_1 = {}
    scenario_rewards_30 = {}
    for name, (obs, act) in scenarios.items():
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        act_t = torch.tensor(act, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            r1 = nets[1](obs_t, act_t).item()
            r30 = nets[N_EPOCHS](obs_t, act_t).item()
        scenario_rewards_1[name] = r1
        scenario_rewards_30[name] = r30
        print(f"  {name:30s}:  E1={r1:+.4f}  E30={r30:+.4f}  Δ={r30-r1:+.4f}")

    # Check if learned preferences make sense
    print("\n  Key behavioral checks (Epoch 30):")
    r = scenario_rewards_30
    print(f"    Charge when low SoC vs high SoC:       {r['low_soc_charge']:+.4f} vs {r['high_soc_charge']:+.4f} → {'✓ prefers low-SoC charge' if r['low_soc_charge'] > r['high_soc_charge'] else '✗ prefers high-SoC charge'}")
    print(f"    Discharge when high SoC vs low SoC:    {r['high_soc_discharge']:+.4f} vs {r['low_soc_discharge']:+.4f} → {'✓ prefers high-SoC discharge' if r['high_soc_discharge'] > r['low_soc_discharge'] else '✗ prefers low-SoC discharge'}")
    print(f"    Charge at cheap vs expensive price:    {r['cheap_price_charge']:+.4f} vs {r['expensive_price_charge']:+.4f} → {'✓ prefers cheap charging' if r['cheap_price_charge'] > r['expensive_price_charge'] else '✗ prefers expensive charging'}")
    print(f"    Discharge at expensive vs cheap price: {r['expensive_price_discharge']:+.4f} vs {r['cheap_price_discharge']:+.4f} → {'✓ prefers expensive discharge' if r['expensive_price_discharge'] > r['cheap_price_discharge'] else '✗ prefers cheap discharge'}")
    print(f"    Urgent charge (low SoC, soon depart):  {r['urgent_need_charge']:+.4f} → {'high' if abs(r['urgent_need_charge']) > 0.5 else 'moderate' if abs(r['urgent_need_charge']) > 0.1 else 'low'} signal")

    # ---- 6. Per-epoch reward evolution on scenarios ----
    scenario_evolution = {name: [] for name in scenarios}
    for e in range(1, N_EPOCHS + 1):
        for name, (obs, act) in scenarios.items():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor(act, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                scenario_evolution[name].append(nets[e](obs_t, act_t).item())

    # ---- PLOTS ----
    out_dir = MODEL_DIR / "analysis"
    out_dir.mkdir(exist_ok=True)
    epochs = list(range(1, N_EPOCHS + 1))

    # Plot 1: Parameter drift
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, drifts, 'b-o', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Distance from Epoch 1")
    ax.set_title("Parameter Drift Over Training")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "param_drift.png", dpi=150)
    print(f"\nSaved: {out_dir / 'param_drift.png'}")

    # Plot 2: Consecutive cosine similarity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, N_EPOCHS), cosines, 'r-o', markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cosine Similarity (epoch t → t+1)")
    ax.set_title("Parameter Cosine Similarity Between Consecutive Epochs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cosine_similarity.png", dpi=150)
    print(f"Saved: {out_dir / 'cosine_similarity.png'}")

    # Plot 3: Layer norm evolution
    layer_names = list(layer_stats[1].keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, name in enumerate(layer_names):
        norms = [layer_stats[e][name]["norm"] for e in epochs]
        stds = [layer_stats[e][name]["std"] for e in epochs]
        ax = axes[i]
        ax.plot(epochs, norms, 'b-', label="norm")
        ax2 = ax.twinx()
        ax2.plot(epochs, stds, 'r--', label="std")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Norm", color='b')
        ax2.set_ylabel("Std", color='r')
        ax.grid(True, alpha=0.3)
    fig.suptitle("Per-Layer Weight Statistics Evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "layer_stats.png", dpi=150)
    print(f"Saved: {out_dir / 'layer_stats.png'}")

    # Plot 4: Reward output distribution over epochs
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [rewards_per_epoch[e].mean() for e in epochs]
    stds = [rewards_per_epoch[e].std() for e in epochs]
    ax.fill_between(epochs,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.3)
    ax.plot(epochs, means, 'b-o', markersize=3, label="Mean ± Std")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Output Distribution on Fixed Probes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_distribution.png", dpi=150)
    print(f"Saved: {out_dir / 'reward_distribution.png'}")

    # Plot 5: Scenario reward evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Charge scenarios
    ax = axes[0, 0]
    for name in ["low_soc_charge", "high_soc_charge", "cheap_price_charge", "expensive_price_charge"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=2, label=name)
    ax.set_title("Charging Scenarios")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Discharge scenarios
    ax = axes[0, 1]
    for name in ["high_soc_discharge", "low_soc_discharge", "cheap_price_discharge", "expensive_price_discharge"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=2, label=name)
    ax.set_title("Discharging Scenarios")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Idle + urgent
    ax = axes[1, 0]
    for name in ["idle_high_soc", "idle_low_soc", "urgent_need_charge", "no_urgency_idle"]:
        ax.plot(epochs, scenario_evolution[name], '-o', markersize=2, label=name)
    ax.set_title("Idle / Urgency Scenarios")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Charge vs discharge spread (reward gap for sensible behavior)
    ax = axes[1, 1]
    charge_gap = [scenario_evolution["low_soc_charge"][i] - scenario_evolution["high_soc_charge"][i] for i in range(N_EPOCHS)]
    discharge_gap = [scenario_evolution["high_soc_discharge"][i] - scenario_evolution["low_soc_discharge"][i] for i in range(N_EPOCHS)]
    price_charge_gap = [scenario_evolution["cheap_price_charge"][i] - scenario_evolution["expensive_price_charge"][i] for i in range(N_EPOCHS)]
    price_discharge_gap = [scenario_evolution["expensive_price_discharge"][i] - scenario_evolution["cheap_price_discharge"][i] for i in range(N_EPOCHS)]
    ax.plot(epochs, charge_gap, '-o', markersize=2, label="low_soc - high_soc (charge)")
    ax.plot(epochs, discharge_gap, '-o', markersize=2, label="high_soc - low_soc (discharge)")
    ax.plot(epochs, price_charge_gap, '-o', markersize=2, label="cheap - expensive (charge)")
    ax.plot(epochs, price_discharge_gap, '-o', markersize=2, label="expensive - cheap (discharge)")
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.set_title("Reward Gaps (positive = sensible preference)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Reward Gap"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle("Scenario-Based Reward Evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "scenario_evolution.png", dpi=150)
    print(f"Saved: {out_dir / 'scenario_evolution.png'}")

    # ---- Final verdict ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_drift = drifts[-1]
    mean_cosine = np.mean(cosines)
    r_corr, _ = spearmanr(rewards_per_epoch[1], rewards_per_epoch[N_EPOCHS])
    print(f"  Total parameter drift (E1→E30):   {total_drift:.4f}")
    print(f"  Mean consecutive cosine similarity: {mean_cosine:.6f}")
    print(f"  Epoch 1 vs 30 cosine similarity:    {cos_1_30:.6f}")
    print(f"  Epoch 1 vs 30 reward rank corr (ρ): {r_corr:+.4f}")
    print(f"  Reward mean shift (E1→E30):         {rewards_per_epoch[1].mean():+.4f} → {rewards_per_epoch[N_EPOCHS].mean():+.4f}")
    print(f"  Reward std shift (E1→E30):          {rewards_per_epoch[1].std():.4f} → {rewards_per_epoch[N_EPOCHS].std():.4f}")

    if total_drift < 0.01:
        verdict = "MINIMAL LEARNING — weights barely changed."
    elif total_drift < 0.1:
        verdict = "MODEST LEARNING — some parameter movement but limited."
    elif total_drift < 1.0:
        verdict = "MODERATE LEARNING — meaningful weight changes occurred."
    else:
        verdict = "SIGNIFICANT LEARNING — substantial parameter evolution."

    if abs(r_corr) < 0.3:
        verdict += " Reward rankings changed substantially (low rank corr)."
    elif abs(r_corr) > 0.9:
        verdict += " But reward rankings barely changed (high rank corr)."

    print(f"\n  VERDICT: {verdict}")

    # ---- 7. MAGNITUDE-PRICE INTERACTION ANALYSIS ----
    print("\n" + "=" * 60)
    print("7. MAGNITUDE × PRICE INTERACTION (Epoch 1 vs final)")
    print("=" * 60)
    print("  Testing if the reward net learned that price modulates magnitude,")
    print("  not willingness to act.\n")

    scales = PROFIT_OBS_SCALES
    magnitudes = np.linspace(0.05, 0.5, 10)
    prices_to_test = [0.07, 0.15, 0.25, 0.35, 0.47]

    base_obs = np.array([48, 0.5, 0.5, 0.20, 1, 48, 7.4], dtype=np.float32)

    net_first = nets[1]
    net_last = nets[N_EPOCHS]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Charge: R(a) at different prices
    ax = axes[0, 0]
    for price in prices_to_test:
        obs_p = base_obs.copy(); obs_p[3] = price
        obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
        rewards_mag = []
        for mag in magnitudes:
            act_t = torch.tensor([[mag]], dtype=torch.float32)
            with torch.no_grad():
                rewards_mag.append(net_last(obs_t, act_t).item())
        ax.plot(magnitudes, rewards_mag, '-o', markersize=3, label=f"£{price:.2f}")
    ax.set_title(f"Charge R(a) at different prices (Epoch {N_EPOCHS})")
    ax.set_xlabel("Charge magnitude"); ax.set_ylabel("Reward"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Discharge: R(a) at different prices
    ax = axes[0, 1]
    for price in prices_to_test:
        obs_p = base_obs.copy(); obs_p[3] = price
        obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
        rewards_mag = []
        for mag in magnitudes:
            act_t = torch.tensor([[-mag]], dtype=torch.float32)
            with torch.no_grad():
                rewards_mag.append(net_last(obs_t, act_t).item())
        ax.plot(magnitudes, rewards_mag, '-o', markersize=3, label=f"£{price:.2f}")
    ax.set_title(f"Discharge R(a) at different prices (Epoch {N_EPOCHS})")
    ax.set_xlabel("Discharge magnitude"); ax.set_ylabel("Reward"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Charge: compare E1 vs E_last at cheap vs expensive
    ax = axes[1, 0]
    for label, net, ls in [("E1", net_first, "--"), (f"E{N_EPOCHS}", net_last, "-")]:
        for price, color in [(0.07, "green"), (0.47, "red")]:
            obs_p = base_obs.copy(); obs_p[3] = price
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            rewards_mag = []
            for mag in magnitudes:
                act_t = torch.tensor([[mag]], dtype=torch.float32)
                with torch.no_grad():
                    rewards_mag.append(net(obs_t, act_t).item())
            ax.plot(magnitudes, rewards_mag, ls, color=color, markersize=3, label=f"{label} £{price:.2f}")
    ax.set_title("Charge: E1 vs E_last, cheap vs expensive")
    ax.set_xlabel("Charge magnitude"); ax.set_ylabel("Reward"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Discharge: compare E1 vs E_last at cheap vs expensive
    ax = axes[1, 1]
    for label, net, ls in [("E1", net_first, "--"), (f"E{N_EPOCHS}", net_last, "-")]:
        for price, color in [(0.07, "red"), (0.47, "green")]:
            obs_p = base_obs.copy(); obs_p[3] = price
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            rewards_mag = []
            for mag in magnitudes:
                act_t = torch.tensor([[-mag]], dtype=torch.float32)
                with torch.no_grad():
                    rewards_mag.append(net(obs_t, act_t).item())
            ax.plot(magnitudes, rewards_mag, ls, color=color, markersize=3, label=f"{label} £{price:.2f}")
    ax.set_title("Discharge: E1 vs E_last, cheap vs expensive")
    ax.set_xlabel("Discharge magnitude"); ax.set_ylabel("Reward"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle("Magnitude × Price Interaction — Does price modulate how much, not whether?", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "magnitude_price_interaction.png", dpi=150)
    print(f"\n  Saved: {out_dir / 'magnitude_price_interaction.png'}")

    # Numerical: slope of reward w.r.t. magnitude at different prices
    print("\n  Reward slope (dR/d|a|) at different prices (Epoch final):")
    for action_type, sign in [("Charge", 1), ("Discharge", -1)]:
        print(f"\n  {action_type}:")
        for price in prices_to_test:
            obs_p = base_obs.copy(); obs_p[3] = price
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            act_lo = torch.tensor([[sign * 0.1]], dtype=torch.float32)
            act_hi = torch.tensor([[sign * 0.5]], dtype=torch.float32)
            with torch.no_grad():
                r_lo = net_last(obs_t, act_lo).item()
                r_hi = net_last(obs_t, act_hi).item()
            slope = (r_hi - r_lo) / 0.4
            print(f"    Price £{price:.2f}: R(|a|=0.1)={r_lo:+.4f}, R(|a|=0.5)={r_hi:+.4f}, slope={slope:+.4f}")

    # Check: does the net have DIFFERENT slopes at different prices? (what we want)
    print("\n  Price-sensitivity of magnitude slope (desirable: varies by price):")
    for action_type, sign in [("Charge", 1), ("Discharge", -1)]:
        slopes = []
        for price in prices_to_test:
            obs_p = base_obs.copy(); obs_p[3] = price
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                r_lo = net_last(obs_t, torch.tensor([[sign * 0.1]])).item()
                r_hi = net_last(obs_t, torch.tensor([[sign * 0.5]])).item()
            slopes.append((r_hi - r_lo) / 0.4)
        slope_range = max(slopes) - min(slopes)
        print(f"    {action_type}: slope range across prices = {slope_range:.4f} ({'learned price-magnitude interaction' if slope_range > 0.01 else 'FLAT — no interaction learned'})")

    # ---- 8. SoC-MAGNITUDE INTERACTION ----
    print("\n" + "=" * 60)
    print("8. SoC × MAGNITUDE INTERACTION (does SoC modulate how much?)")
    print("=" * 60)

    soc_levels = [0.2, 0.4, 0.6, 0.8]
    for action_type, sign in [("Charge", 1), ("Discharge", -1)]:
        print(f"\n  {action_type}:")
        slopes = []
        for soc in soc_levels:
            obs_p = base_obs.copy(); obs_p[1] = soc
            obs_t = torch.tensor(obs_p / scales, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                r_lo = net_last(obs_t, torch.tensor([[sign * 0.1]])).item()
                r_hi = net_last(obs_t, torch.tensor([[sign * 0.5]])).item()
            slope = (r_hi - r_lo) / 0.4
            slopes.append(slope)
            print(f"    SoC={soc:.1f}: R(|a|=0.1)={r_lo:+.4f}, R(|a|=0.5)={r_hi:+.4f}, slope={slope:+.4f}")
        slope_range = max(slopes) - min(slopes)
        print(f"    Slope range across SoC = {slope_range:.4f} ({'learned SoC-magnitude interaction' if slope_range > 0.01 else 'FLAT — no interaction learned'})")


if __name__ == "__main__":
    main()

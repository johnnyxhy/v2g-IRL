"""
Compare reward networks from different population segments at a fixed epoch.

Evaluates each reward net on a suite of scenarios that vary:
  - SoC level (low / mid / high)
  - Energy price (cheap / expensive)
  - Action type (charge / discharge / idle)
  - SoC vs target (below / at / above)
  - Journey urgency (imminent / distant)

Usage: edit SEGMENTS and EPOCH at the top, then run the script.
Plots and a summary CSV are saved to OUTDIR.
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES

# ---------------------------------------------------------------------------
# Configuration – edit these
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

# Map display label -> model directory (relative to MODELS_ROOT or absolute)
SEGMENTS: dict[str, str] = {
    "Male 50-59":          "DeepMaxEntIRL_discrete_profit_exp2",
    "Female 50-59":        "DeepMaxEntIRL_discrete_profit_female_50-59_exp1",
}

EPOCH: int = 20          # which epoch's reward net to load for all segments
HIDDEN_DIM: int = 32     # must match the architecture used during training

OUTDIR = Path(__file__).resolve().parent.parent / "models" / "segment_comparison"
# ---------------------------------------------------------------------------

SCALES = PROFIT_OBS_SCALES  # [96, 1, 1, 0.47, 2, 96, 22]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_dir(entry: str) -> Path:
    p = Path(entry)
    return p if p.is_absolute() else MODELS_ROOT / p


def load_net(model_dir: Path, epoch: int) -> RewardNet:
    path = model_dir / f"reward_net_epoch{epoch}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No reward net found at {path}")
    net = RewardNet(obs_dim=len(SCALES), action_dim=1, hidden_dim=HIDDEN_DIM)
    net.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    net.eval()
    return net


def score(net: RewardNet, obs_raw: np.ndarray, action_norm: float) -> float:
    """Score a single (obs, action) pair. obs_raw is unnormalized."""
    obs_t = torch.tensor(obs_raw / SCALES, dtype=torch.float32).unsqueeze(0)
    act_t = torch.tensor([[action_norm]], dtype=torch.float32)
    with torch.no_grad():
        return net(obs_t, act_t).item()


# ---------------------------------------------------------------------------
# Scenario definitions
# obs columns: [timestep, soc, soc_target, energy_price, battery_capacity_idx,
#               time_to_next_journey, current_charger_power]
# action: continuous in [-1, 1], positive = charge, negative = discharge
# ---------------------------------------------------------------------------

# Base observation used as template; individual fields are overridden per scenario
_BASE = np.array([48.0, 0.5, 0.3, 0.27, 1.0, 20.0, 7.4], dtype=np.float32)
MEAN_PRICE = 0.27  # representative mean energy price £/kWh

SCENARIOS: dict[str, tuple[np.ndarray, float]] = {
    # # ── SoC level  ──────────────────────────────────────────────────────────
    # "low_soc_charge":          (_BASE * [1,0,1,1,1,1,1] + [0, 0.15, 0, 0, 0, 0, 0],   +0.8),
    # "mid_soc_charge":          (_BASE.copy(),                                             +0.8),
    # "high_soc_charge":         (_BASE * [1,0,1,1,1,1,1] + [0, 0.85, 0, 0, 0, 0, 0],   +0.8),
    # "low_soc_discharge":       (_BASE * [1,0,1,1,1,1,1] + [0, 0.15, 0, 0, 0, 0, 0],   -0.8),
    # "mid_soc_discharge":       (_BASE.copy(),                                             -0.8),
    # "high_soc_discharge":      (_BASE * [1,0,1,1,1,1,1] + [0, 0.85, 0, 0, 0, 0, 0],   -0.8),
    # "low_soc_idle":            (_BASE * [1,0,1,1,1,1,1] + [0, 0.15, 0, 0, 0, 0, 0],    0.0),
    # "mid_soc_idle":            (_BASE.copy(),                                              0.0),
    # "high_soc_idle":           (_BASE * [1,0,1,1,1,1,1] + [0, 0.85, 0, 0, 0, 0, 0],    0.0),

    # ── SoC vs target ────────────────────────────────────────────────────────
    "below_target_charge":     (np.array([48,0.3,0.4,0.27,1,48,7.4], dtype=np.float32), +0.1),
    "at_target_charge":        (np.array([48,0.4,0.4,0.27,1,48,7.4], dtype=np.float32), +0.1),
    "above_target_charge":     (np.array([48,0.5,0.4,0.27,1,48,7.4], dtype=np.float32), +0.1),
    "below_target_discharge":  (np.array([48,0.3,0.4,0.27,1,48,7.4], dtype=np.float32), -0.1),
    "at_target_discharge":     (np.array([48,0.4,0.4,0.27,1,48,7.4], dtype=np.float32), -0.1),
    "above_target_discharge":  (np.array([48,0.5,0.4,0.27,1,48,7.4], dtype=np.float32), -0.1),
    "at_target_idle":          (np.array([48,0.4,0.4,0.27,1,48,7.4], dtype=np.float32),  0.0),
    "below_target_idle":       (np.array([48,0.3,0.4,0.27,1,48,7.4], dtype=np.float32),  0.0),
    "above_target_idle":       (np.array([48,0.5,0.4,0.27,1,48,7.4], dtype=np.float32),  0.0),

    # ── Energy price  ────────────────────────────────────────────────────────
    "cheap_price_charge":      (_BASE * [1,1,1,0,1,1,1] + [0, 0, 0, 0.05, 0, 0, 0],   +0.1),
    "mean_price_charge":       (_BASE.copy(),                                             +0.1),
    "expensive_price_charge":  (_BASE * [1,1,1,0,1,1,1] + [0, 0, 0, 0.42, 0, 0, 0],   +0.1),
    "cheap_price_discharge":   (_BASE * [1,1,1,0,1,1,1] + [0, 0, 0, 0.05, 0, 0, 0],   -0.1),
    "mean_price_discharge":    (_BASE.copy(),                                             -0.1),
    "expensive_price_discharge":(_BASE * [1,1,1,0,1,1,1] + [0, 0, 0, 0.42, 0, 0, 0], -0.1),

    
    # ── Journey urgency  ─────────────────────────────────────────────────────
    "urgent_low_soc_charge":   (np.array([48,0.3,0.6,0.20,1, 4,7.4], dtype=np.float32),  +0.9),
    "distant_low_soc_charge":  (np.array([48,0.3,0.6,0.20,1,80,7.4], dtype=np.float32),  +0.9),
    "urgent_high_soc_idle":    (np.array([48,0.7,0.6,0.20,1, 4,7.4], dtype=np.float32),   0.0),
    "distant_high_soc_idle":   (np.array([48,0.7,0.6,0.20,1,80,7.4], dtype=np.float32),   0.0),

    # ── Trade-off scenarios  ─────────────────────────────────────────────────
    "needs_soc_but_expensive":   (np.array([48,0.3,0.7,0.42,1,10,7.4], dtype=np.float32), +0.8),
    "has_soc_cheap_discharge":   (np.array([48,0.7,0.3,0.05,1,48,7.4], dtype=np.float32), -0.8),
    "full_battery_expensive_dis":(np.array([48,0.9,0.5,0.42,1,48,7.4], dtype=np.float32), -0.8),
    "empty_battery_cheap_charge":(np.array([48,0.1,0.5,0.05,1,48,7.4], dtype=np.float32), +0.9),
}

# Group names for nicer plots
SCENARIO_GROUPS: dict[str, list[str]] = {
    "Price - Charge":        ["cheap_price_charge","mean_price_charge","expensive_price_charge"],
    "Price - Discharge":     ["cheap_price_discharge","mean_price_discharge","expensive_price_discharge"],
    "SoC vs Target - Charge":   ["below_target_charge","at_target_charge","above_target_charge"],
    "SoC vs Target - Discharge":["below_target_discharge","at_target_discharge","above_target_discharge"],
    "SoC vs Target - Idle":     ["below_target_idle","at_target_idle","above_target_idle"],
    "Journey Urgency":       ["urgent_low_soc_charge","distant_low_soc_charge",
                              "urgent_high_soc_idle","distant_high_soc_idle"],
    "Trade-offs":            ["needs_soc_but_expensive","has_soc_cheap_discharge",
                              "full_battery_expensive_dis","empty_battery_cheap_charge"],
}

# Pairwise differences to compare across segments
# Each entry: display label -> (scenario_a, scenario_b)  =>  reward(a) - reward(b)
DIFFERENCES: dict[str, tuple[str, str]] = {
    "cheap - expensive\n(charge)":          ("cheap_price_charge",      "expensive_price_charge"),
    "expensive - cheap\n(discharge)":        ("expensive_price_discharge","cheap_price_discharge"),
    "below - above target\n(charge)":        ("below_target_charge",     "above_target_charge"),
    "above - below target\n(discharge)":     ("above_target_discharge",  "below_target_discharge"),
}



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    segment_labels = list(SEGMENTS.keys())
    n_segs = len(segment_labels)

    # ------------------------------------------------------------------
    # 1. Load networks
    # ------------------------------------------------------------------
    nets: dict[str, RewardNet] = {}
    for label, entry in SEGMENTS.items():
        d = resolve_dir(entry)
        nets[label] = load_net(d, EPOCH)
        print(f"[OK] Loaded {label} from {d.name} epoch {EPOCH}")

    # ------------------------------------------------------------------
    # 2. Score all scenarios
    # ------------------------------------------------------------------
    scores: dict[str, dict[str, float]] = {}
    for label, net in nets.items():
        scores[label] = {}
        for name, (obs_raw, action) in SCENARIOS.items():
            scores[label][name] = score(net, obs_raw, action)

    # ------------------------------------------------------------------
    # 3. Print scenario table
    # ------------------------------------------------------------------
    col_w = max(len(l) for l in segment_labels) + 2
    name_w = max(len(n) for n in SCENARIOS) + 2

    print("\n" + "=" * (name_w + col_w * n_segs + 4))
    print(f"Scenario reward comparison — Epoch {EPOCH}")
    print("=" * (name_w + col_w * n_segs + 4))

    header = f"{'Scenario':<{name_w}}" + "".join(f"{l:>{col_w}}" for l in segment_labels)
    print(header)
    print("-" * len(header))

    prev_group = None
    for group_name, group_scenarios in SCENARIO_GROUPS.items():
        if prev_group is not None:
            print()
        print(f"  [{group_name}]")
        prev_group = group_name
        for sname in group_scenarios:
            row = f"  {sname:<{name_w - 2}}"
            for label in segment_labels:
                row += f"{scores[label][sname]:>{col_w}.4f}"
            print(row)

    # ------------------------------------------------------------------
    # 4. Save CSV summary
    # ------------------------------------------------------------------
    all_scenario_names = list(SCENARIOS.keys())
    rows = {label: [scores[label][n] for n in all_scenario_names] for label in segment_labels}
    df_all = pd.DataFrame(rows, index=all_scenario_names)
    df_all.index.name = "scenario"
    csv_path = OUTDIR / f"scenario_rewards_epoch{EPOCH}.csv"
    df_all.to_csv(csv_path)
    print(f"\nSaved scenario scores -> {csv_path}")

    # ==================================================================
    # PLOTS
    # ==================================================================

    # ------------------------------------------------------------------
    # Plot A: Grouped bar charts – one subplot per scenario group
    # X positions = scenarios, colours = segments
    # ------------------------------------------------------------------
    n_groups = len(SCENARIO_GROUPS)
    ncols = 2
    nrows = (n_groups + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 4))
    axes_flat = axes.flatten()

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    seg_colours = [prop_cycle[i % len(prop_cycle)] for i in range(n_segs)]

    for ax_idx, (group_name, group_scenarios) in enumerate(SCENARIO_GROUPS.items()):
        ax = axes_flat[ax_idx]
        n_scen = len(group_scenarios)
        x = np.arange(n_scen)
        width = 0.8 / n_segs
        offsets = np.linspace(-(n_segs - 1) / 2, (n_segs - 1) / 2, n_segs) * width

        for s_idx, label in enumerate(segment_labels):
            vals = [scores[label][sname] for sname in group_scenarios]
            ax.bar(x + offsets[s_idx], vals, width=width * 0.9,
                   label=label, color=seg_colours[s_idx], alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(group_scenarios, fontsize=8, rotation=20, ha="right")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title(group_name, fontsize=10)
        ax.set_ylabel("Reward")
        ax.set_ylim(bottom=0.15)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, axis="y", alpha=0.3)

    # hide unused subplots
    for ax_idx in range(n_groups, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(f"Scenario Reward Comparison Across Segments — Epoch {EPOCH}", fontsize=13)
    fig.tight_layout()
    p = OUTDIR / f"scenario_comparison_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot B: Difference comparisons bar chart
    # Each bar = reward(a) - reward(b) for a defined pair; colours = segments
    # ------------------------------------------------------------------
    diff_labels = list(DIFFERENCES.keys())
    n_diffs = len(diff_labels)
    x_d = np.arange(n_diffs)
    width_d = 0.8 / n_segs

    fig, ax = plt.subplots(figsize=(max(8, n_diffs * 2.5), 5))
    for s_idx, label in enumerate(segment_labels):
        offset = (s_idx - (n_segs - 1) / 2) * width_d
        vals = [
            scores[label][s_a] - scores[label][s_b]
            for s_a, s_b in DIFFERENCES.values()
        ]
        ax.bar(x_d + offset, vals, width=width_d * 0.9,
               label=label, color=seg_colours[s_idx], alpha=0.85)
        for xi, v in zip(x_d + offset, vals):
            ax.text(xi, v + 0.0005 if v >= 0 else v - 0.0015,
                    f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7)

    ax.set_xticks(x_d)
    ax.set_xticklabels(diff_labels, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_ylabel("Reward difference (a - b)")
    ax.set_title(f"Pairwise Reward Differences Across Segments — Epoch {EPOCH}")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = OUTDIR / f"difference_comparison_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot C: SoC x Action reward heatmaps (one per segment)
    # ------------------------------------------------------------------
    soc_vals = np.linspace(0.05, 0.95, 19)
    action_vals = np.linspace(-1.0, 1.0, 21)
    base_obs = np.array([48.0, 0.5, 0.5, 0.20, 1.0, 48.0, 7.4], dtype=np.float32)

    fig, axes = plt.subplots(1, n_segs, figsize=(6 * n_segs, 5), sharey=True)
    if n_segs == 1:
        axes = [axes]

    all_grids = []
    for label, net in nets.items():
        grid = np.zeros((len(soc_vals), len(action_vals)))
        for i, soc in enumerate(soc_vals):
            for j, act in enumerate(action_vals):
                obs_p = base_obs.copy()
                obs_p[1] = soc
                grid[i, j] = score(net, obs_p, act)
        all_grids.append(grid)

    vmin = min(g.min() for g in all_grids)
    vmax = max(g.max() for g in all_grids)

    for ax, label, grid in zip(axes, segment_labels, all_grids):
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       extent=[action_vals[0], action_vals[-1], soc_vals[0], soc_vals[-1]],
                       vmin=vmin, vmax=vmax, cmap="RdYlGn")
        ax.set_xlabel("Action (−1=discharge, +1=charge)")
        ax.set_ylabel("SoC")
        ax.set_title(label, fontsize=9)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        plt.colorbar(im, ax=ax, label="Reward")

    fig.suptitle(f"SoC × Action Reward Heatmap — Epoch {EPOCH}\n"
                 "(soc_target=0.5, price=0.20, time_to_journey=48)", fontsize=11)
    fig.tight_layout()
    p = OUTDIR / f"heatmap_soc_action_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot D: Price x Action reward heatmaps (SoC fixed at 0.5)
    # ------------------------------------------------------------------
    price_vals = np.linspace(0.01, 0.46, 20)

    fig, axes = plt.subplots(1, n_segs, figsize=(6 * n_segs, 5), sharey=True)
    if n_segs == 1:
        axes = [axes]

    all_grids_p = []
    for label, net in nets.items():
        grid = np.zeros((len(price_vals), len(action_vals)))
        for i, price in enumerate(price_vals):
            for j, act in enumerate(action_vals):
                obs_p = base_obs.copy()
                obs_p[3] = price
                grid[i, j] = score(net, obs_p, act)
        all_grids_p.append(grid)

    vmin_p = min(g.min() for g in all_grids_p)
    vmax_p = max(g.max() for g in all_grids_p)

    for ax, label, grid in zip(axes, segment_labels, all_grids_p):
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       extent=[action_vals[0], action_vals[-1], price_vals[0], price_vals[-1]],
                       vmin=vmin_p, vmax=vmax_p, cmap="RdYlGn")
        ax.set_xlabel("Action (−1=discharge, +1=charge)")
        ax.set_ylabel("Energy Price (£/kWh)")
        ax.set_title(label, fontsize=9)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(MEAN_PRICE, color="cyan", linewidth=0.8, linestyle="--", alpha=0.7,
                   label=f"mean price={MEAN_PRICE}")
        ax.legend(fontsize=7)
        plt.colorbar(im, ax=ax, label="Reward")

    fig.suptitle(f"Price × Action Reward Heatmap — Epoch {EPOCH}\n"
                 "(SoC=0.5, soc_target=0.5, time_to_journey=48)", fontsize=11)
    fig.tight_layout()
    p = OUTDIR / f"heatmap_price_action_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    print(f"\nAll outputs saved to {OUTDIR}")


if __name__ == "__main__":
    main()

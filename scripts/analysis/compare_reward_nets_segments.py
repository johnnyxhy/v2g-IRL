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
import shap
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, PROFIT_OBS_SCALES

# ---------------------------------------------------------------------------
# Configuration – edit these
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"

# Map display label -> model directory (relative to MODELS_ROOT or absolute)
SEGMENTS: dict[str, str] = {
    "Male 50-59":          "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male5059",
    "Male 40-49":          "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_pricediff_male4049",
}

EPOCH: int = 20          # which epoch's reward net to load for all segments
HIDDEN_DIM: int = 32     # must match the architecture used during training

OUTDIR = Path(__file__).resolve().parent.parent.parent / "models" / "segment_comparison"
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
# Advantage & sweep helpers
# ---------------------------------------------------------------------------

def compute_advantages(net: RewardNet) -> dict[str, float]:
    """
    A(s, a) = R(s, a) - R(s, idle) for canonical states.
    Sensitivity metrics are positive when the network has the correct
    directional preference.
    """
    CHARGE, DISCHARGE = 0.3, -0.3

    cheap_s  = np.array([48, 0.5, 0.5, 0.07, 1, 48, 7.4], dtype=np.float32)
    mid_s    = np.array([48, 0.5, 0.5, 0.27, 1, 48, 7.4], dtype=np.float32)
    exp_s    = np.array([48, 0.5, 0.5, 0.47, 1, 48, 7.4], dtype=np.float32)
    below_s  = np.array([48, 0.3, 0.7, 0.20, 1, 48, 7.4], dtype=np.float32)
    at_s     = np.array([48, 0.5, 0.5, 0.20, 1, 48, 7.4], dtype=np.float32)
    above_s  = np.array([48, 0.7, 0.3, 0.20, 1, 48, 7.4], dtype=np.float32)
    urgent_s = np.array([48, 0.3, 0.7, 0.20, 1,  5, 7.4], dtype=np.float32)

    a: dict[str, float] = {}
    for name, state in [("cheap", cheap_s), ("mid", mid_s), ("expensive", exp_s)]:
        a[f"charge_adv_{name}"]    = score(net, state, CHARGE)    - score(net, state, 0.0)
        a[f"discharge_adv_{name}"] = score(net, state, DISCHARGE) - score(net, state, 0.0)
    for name, state in [("below_target", below_s), ("at_target", at_s), ("above_target", above_s)]:
        a[f"charge_adv_{name}"]    = score(net, state, CHARGE)    - score(net, state, 0.0)
        a[f"discharge_adv_{name}"] = score(net, state, DISCHARGE) - score(net, state, 0.0)
    a["charge_adv_urgent"]     = score(net, urgent_s, CHARGE) - score(net, urgent_s, 0.0)
    a["charge_adv_non_urgent"] = score(net, at_s,     CHARGE) - score(net, at_s,     0.0)

    a["price_sensitivity_charge"]    = a["charge_adv_cheap"]           - a["charge_adv_expensive"]
    a["price_sensitivity_discharge"] = a["discharge_adv_expensive"]    - a["discharge_adv_cheap"]
    a["soc_need_sensitivity"]        = a["charge_adv_below_target"]    - a["charge_adv_above_target"]
    a["soc_surplus_sensitivity"]     = a["discharge_adv_above_target"] - a["discharge_adv_below_target"]
    a["urgency_sensitivity"]         = a["charge_adv_urgent"]          - a["charge_adv_non_urgent"]
    return a


def compute_price_sweep(net: RewardNet, charge_act: float = 0.3, discharge_act: float = -0.3):
    """Returns (prices, charge_advs, discharge_advs) sweeping price £0.07→£0.47."""
    prices = np.linspace(0.07, 0.47, 17)
    chg, dis = [], []
    for price in prices:
        state = np.array([48, 0.5, 0.5, price, 1, 48, 7.4], dtype=np.float32)
        chg.append(score(net, state, charge_act)    - score(net, state, 0.0))
        dis.append(score(net, state, discharge_act) - score(net, state, 0.0))
    return prices, np.array(chg), np.array(dis)


def compute_soc_deficit_sweep(net: RewardNet, charge_act: float = 0.3, discharge_act: float = -0.3):
    """Fix soc=0.5, sweep soc_target 0.1→0.9. Returns (deficits, charge_advs, discharge_advs)."""
    soc_targets = np.linspace(0.1, 0.9, 17)
    chg, dis = [], []
    for soc_t in soc_targets:
        state = np.array([48, 0.5, soc_t, 0.20, 1, 48, 7.4], dtype=np.float32)
        chg.append(score(net, state, charge_act)    - score(net, state, 0.0))
        dis.append(score(net, state, discharge_act) - score(net, state, 0.0))
    return soc_targets - 0.5, np.array(chg), np.array(dis)


OBS_FEATURE_NAMES = ["timestep", "soc", "soc_target", "price", "battery_cap", "time_to_dep", "charger_pwr"]


def compute_shap_sensitivity(
    net: RewardNet,
    n_background: int = 300,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Use SHAP GradientExplainer to estimate which obs features drive the
    *advantage* of each action over idle: A(s,a) = R(s,a) - R(s,idle).

    This is the behaviorally meaningful target: high price SHAP on the charge
    advantage means price changes whether charging is preferred over idling —
    i.e., the agent is genuinely price-timing its charging decisions.

    Returns a dict with keys "charge_adv" and "discharge_adv", each an array
    of shape (obs_dim,) containing mean |SHAP| per feature, normalised to sum
    to 1 for cross-segment comparability.
    """
    # Wrapper takes obs-only input and outputs advantage R(s,a) - R(s,idle)
    class _AdvantageWrapper(nn.Module):
        def __init__(self, rn: RewardNet, action_val: float):
            super().__init__()
            self.rn = rn
            self.action_val = action_val
        def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
            n = obs_norm.shape[0]
            act = torch.full((n, 1), self.action_val, dtype=obs_norm.dtype,
                             device=obs_norm.device)
            idle = torch.zeros(n, 1, dtype=obs_norm.dtype, device=obs_norm.device)
            return (self.rn(obs_norm, act) - self.rn(obs_norm, idle)).unsqueeze(-1)

    # Generate background states uniformly across the realistic operating range
    rng = np.random.default_rng(seed)
    bg_obs_norm = np.stack([
        rng.uniform(0.0, 1.0, n_background),                        # timestep / 96
        rng.uniform(0.1, 0.9, n_background),                        # soc
        rng.uniform(0.2, 0.8, n_background),                        # soc_target
        rng.uniform(0.07 / 0.47, 1.0, n_background),               # energy_price / 0.47
        rng.choice([0.0, 0.5, 1.0], n_background),                  # battery_cap / 2
        rng.uniform(0.0, 1.0, n_background),                        # time_to_dep / 96
        rng.choice([3/22, 7.4/22, 11/22, 1.0], n_background),      # charger_pwr / 22
    ], axis=1).astype(np.float32)  # (n_background, 7) — already normalized
    bg_tensor = torch.tensor(bg_obs_norm)

    results: dict[str, np.ndarray] = {}
    for adv_name, action_val in [("charge_adv", 0.3), ("discharge_adv", -0.3)]:
        wrapper = _AdvantageWrapper(net, action_val)
        wrapper.eval()

        explainer = shap.GradientExplainer(wrapper, bg_tensor)
        sv = explainer.shap_values(bg_tensor)  # list[(n, 7, 1)] or (n, 7, 1)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 0]  # (n_background, 7)

        mean_abs = np.abs(sv).mean(axis=0)  # (7,)
        total = mean_abs.sum()
        results[adv_name] = mean_abs / total if total > 0 else mean_abs

    return results


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
    "cheap_price_charge":      (_BASE * [1,0,0,0,1,1,1] + [0, 0.3, 0.5, 0.05, 0, 0, 0],   +0.1),
    "mean_price_charge":       (_BASE.copy(),                                             +0.1),
    "expensive_price_charge":  (_BASE * [1,0,0,0,1,1,1] + [0, 0.3, 0.5, 0.42, 0, 0, 0],   +0.1),
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

    # ------------------------------------------------------------------
    # Plot E: Advantage sweep overlays
    # ------------------------------------------------------------------
    sweep_price = {lbl: compute_price_sweep(net) for lbl, net in nets.items()}
    sweep_soc   = {lbl: compute_soc_deficit_sweep(net) for lbl, net in nets.items()}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for lbl, (prices, chg_advs, _) in sweep_price.items():
        ax.plot(prices, chg_advs, '-o', markersize=3, label=lbl)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel("Energy Price (\u00a3/kWh)"); ax.set_ylabel("Charge advantage over idle")
    ax.set_title("Charge Advantage vs Price\n(should slope \u2193: prefer cheap)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for lbl, (prices, _, dis_advs) in sweep_price.items():
        ax.plot(prices, dis_advs, '-o', markersize=3, label=lbl)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.set_xlabel("Energy Price (\u00a3/kWh)"); ax.set_ylabel("Discharge advantage over idle")
    ax.set_title("Discharge Advantage vs Price\n(should slope \u2191: prefer expensive)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for lbl, (deficits, chg_advs, dis_advs) in sweep_soc.items():
        ax.plot(deficits, chg_advs, '-o',  markersize=3, label=f"{lbl} charge")
        ax.plot(deficits, dis_advs, '--s', markersize=3, label=f"{lbl} discharge")
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.axvline(0, color='k', linestyle=':', alpha=0.4)
    ax.set_xlabel("SoC Deficit (soc_target \u2212 soc)"); ax.set_ylabel("Advantage over idle")
    ax.set_title("Action Advantage vs SoC Deficit")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f"Segment Comparison \u2014 Advantage Sweeps (Epoch {EPOCH})", fontsize=13)
    fig.tight_layout()
    p = OUTDIR / f"seg_comparison_sweeps_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot F: Behavioral sensitivity fingerprint
    # ------------------------------------------------------------------
    adv_per_seg = {lbl: compute_advantages(net) for lbl, net in nets.items()}

    SENS_KEYS = [
        "price_sensitivity_charge", "price_sensitivity_discharge",
        "soc_need_sensitivity", "soc_surplus_sensitivity", "urgency_sensitivity",
    ]
    SENS_LABELS = ["PriceSens\n(charge)", "PriceSens\n(discharge)",
                   "SoCNeed\nSens", "SoCPlus\nSens", "Urgency\nSens"]

    # Print sensitivity table
    col_w = 16
    lbl_w = max(len(l) for l in nets) + 2
    header = f"{'Segment':<{lbl_w}}" + "".join(f"{k[:col_w]:>{col_w}}" for k in SENS_KEYS)
    print("\n" + header + "\n" + "-" * len(header))
    for lbl, a in adv_per_seg.items():
        print(f"{lbl:<{lbl_w}}" + "".join(f"{a[k]:+{col_w}.4f}" for k in SENS_KEYS))

    x = np.arange(len(SENS_KEYS))
    width = 0.8 / n_segs
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (lbl, a) in enumerate(adv_per_seg.items()):
        offset = (i - n_segs / 2 + 0.5) * width
        ax.bar(x + offset, [a[k] for k in SENS_KEYS], width, label=lbl, color=seg_colours[i])
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(SENS_LABELS, fontsize=9)
    ax.set_ylabel("Advantage difference  (positive = correct preference direction)")
    ax.set_title(f"Behavioral Sensitivity Fingerprint by Segment (Epoch {EPOCH})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    p = OUTDIR / f"seg_comparison_fingerprint_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot G: Advantage heatmap (segments × canonical scenarios)
    # ------------------------------------------------------------------
    ADV_KEYS = [
        "charge_adv_cheap", "charge_adv_mid", "charge_adv_expensive",
        "discharge_adv_cheap", "discharge_adv_mid", "discharge_adv_expensive",
        "charge_adv_below_target", "charge_adv_at_target", "charge_adv_above_target",
        "discharge_adv_below_target", "discharge_adv_at_target", "discharge_adv_above_target",
        "charge_adv_urgent",
    ]
    SHORT = ["chg@cheap", "chg@mid", "chg@exp",
             "dis@cheap", "dis@mid", "dis@exp",
             "chg@blw",   "chg@at",  "chg@abv",
             "dis@blw",   "dis@at",  "dis@abv",
             "chg@urgent"]
    matrix = np.array([[adv_per_seg[lbl][k] for k in ADV_KEYS] for lbl in nets])
    vmax_h = max(np.abs(matrix).max(), 1e-6)
    fig, ax = plt.subplots(figsize=(14, max(3, n_segs * 0.9 + 2)))
    im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=-vmax_h, vmax=vmax_h)
    plt.colorbar(im, ax=ax, label='R(s,a) \u2212 R(s,idle)')
    ax.set_yticks(range(n_segs)); ax.set_yticklabels(segment_labels, fontsize=9)
    ax.set_xticks(range(len(ADV_KEYS)))
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=8)
    for i in range(n_segs):
        for j in range(len(ADV_KEYS)):
            color = 'white' if abs(matrix[i, j]) > vmax_h * 0.6 else 'black'
            ax.text(j, i, f"{matrix[i, j]:+.3f}", ha='center', va='center', fontsize=6, color=color)
    ax.set_title(f"Advantage Heatmap by Segment (Epoch {EPOCH})")
    fig.tight_layout()
    p = OUTDIR / f"seg_comparison_heatmap_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot H: SHAP on *advantage* A(s,a) = R(s,a) - R(s,idle) per segment
    # High price SHAP here = price changes whether action is preferred over idle
    # ------------------------------------------------------------------
    print("\nComputing SHAP advantage feature importance (this may take a moment)...")
    shap_per_seg = {lbl: compute_shap_sensitivity(net) for lbl, net in nets.items()}

    for adv_name, action_label in [("charge_adv", "Charge Advantage"), ("discharge_adv", "Discharge Advantage")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        n_feat = len(OBS_FEATURE_NAMES)
        x = np.arange(n_feat)
        width = 0.8 / n_segs
        for i, lbl in enumerate(segment_labels):
            offset = (i - n_segs / 2 + 0.5) * width
            vals = shap_per_seg[lbl][adv_name]
            bars = ax.bar(x + offset, vals, width, label=lbl, color=seg_colours[i], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(OBS_FEATURE_NAMES, fontsize=9)
        ax.set_ylabel("Mean |SHAP| on A(s,a) (normalised, sums to 1)")
        ax.set_title(f"SHAP: What drives {action_label} over Idle?\n"
                     f"(Epoch {EPOCH} — higher price bar = more price-timed behaviour)")
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        p = OUTDIR / f"shap_{adv_name}_epoch{EPOCH}.png"
        fig.savefig(p, dpi=150); plt.close(fig)
        print(f"Saved plot -> {p}")

    # Combined subplot: charge adv + discharge adv side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    n_feat = len(OBS_FEATURE_NAMES)
    x = np.arange(n_feat)
    width = 0.8 / n_segs
    for col_idx, (adv_name, action_label) in enumerate([("charge_adv", "Charge Advantage"), ("discharge_adv", "Discharge Advantage")]):
        ax = axes[col_idx]
        for i, lbl in enumerate(segment_labels):
            offset = (i - n_segs / 2 + 0.5) * width
            ax.bar(x + offset, shap_per_seg[lbl][adv_name], width, label=lbl,
                   color=seg_colours[i], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(OBS_FEATURE_NAMES, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Mean |SHAP| on A(s,a) (normalised)")
        ax.set_title(f"{action_label}")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"SHAP: Feature Importance for Action Advantage by Segment (Epoch {EPOCH})\n"
                 "Target = R(s,a) − R(s,idle) — price bar height = behavioural price-timing strength",
                 fontsize=11)
    fig.tight_layout()
    p = OUTDIR / f"shap_combined_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved plot -> {p}")

    print(f"\nAll outputs saved to {OUTDIR}")


if __name__ == "__main__":
    main()

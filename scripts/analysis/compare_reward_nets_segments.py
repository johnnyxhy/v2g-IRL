"""
Compare reward networks from different population segments at a fixed epoch.

Shows price-sweep and SoC-gap-sweep advantage plots, plus SHAP feature
importance plots for charge and discharge advantages.

Usage: edit SEGMENTS and EPOCH at the top, then run the script.
Plots are saved to OUTDIR.
"""

import sys
import shap
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from irl.DeepMaxEnt.DeepMaxEnt import RewardNet, OBS_SCALES

# ---------------------------------------------------------------------------
# Configuration – edit these
# ---------------------------------------------------------------------------
MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"

SEGMENTS: dict[str, str] = {
    "Normal":   "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_special_male5059",
    "High Bat":   "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_special_highbat_male5059",
    "Low Bat": "DeepMaxEnt/discrete/DeepMaxEntIRL_discrete_special_lowbat_male5059",
}

EPOCH: int = 20
HIDDEN_DIM: int = 32

OUTDIR = Path(__file__).resolve().parent.parent.parent / "models" / "Deep_pdp"
# ---------------------------------------------------------------------------

SCALES = OBS_SCALES  # [96, 1, 1, 0.47, 2, 96, 22]

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

MEAN_PRICE = float(energy_price_profile[24])  # 0.27 at timestep 24

SOC_GAP_SWEEP_LEVELS = (-0.2, 0.0, 0.2)
SOC_GAP_SWEEP_LABELS = ("soc_gap = −0.2 (below target)", "soc_gap = 0.0 (at target)", "soc_gap = +0.2 (above target)")


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
# Sweep helpers
# ---------------------------------------------------------------------------

def compute_price_sweep_decoupled(
    net: RewardNet,
    soc_gaps: tuple = SOC_GAP_SWEEP_LEVELS,
    charge_act: float = 0.3,
    discharge_act: float = -0.3,
) -> dict[float, tuple]:
    """Sweep price independently of timestep (ts=24 fixed).
    Returns dict: soc_gap -> (prices, charge_advs, discharge_advs).
    Advantage = R(s, a) - R(s, idle).
    """
    prices = np.linspace(0.07, 0.47, 30)
    result: dict[float, tuple] = {}
    for soc_gap in soc_gaps:
        chg, dis = [], []
        for price in prices:
            state = np.array([24, 0.5, float(soc_gap), price, 1, 20, 7.4], dtype=np.float32)
            chg.append(score(net, state, charge_act)    - score(net, state, 0.0))
            dis.append(score(net, state, discharge_act) - score(net, state, 0.0))
        result[float(soc_gap)] = (prices, np.array(chg), np.array(dis))
    return result


def compute_soc_gap_sweep(net: RewardNet, charge_act: float = 0.3, discharge_act: float = -0.3):
    """Fix soc=0.5, sweep soc_target 0.2→0.5. Returns (gaps, charge_advs, discharge_advs).
    gap = soc - soc_target: negative = below target, positive = above target."""
    soc_vals = np.linspace(0.2, 0.8, 20)
    chg, dis = [], []
    for soc in soc_vals:
        soc_gap = soc - 0.3
        state = np.array([24, 0.5, soc_gap, energy_price_profile[24], 1, 20, 7.4], dtype=np.float32)
        chg.append(score(net, state, charge_act)    - score(net, state, 0.0))
        dis.append(score(net, state, discharge_act) - score(net, state, 0.0))
    return soc_vals - 0.3, np.array(chg), np.array(dis)


def generate_background_states(n: int = 200, seed: int = 42) -> np.ndarray:
    """Sample n realistic unnormalized background states for PDP averaging."""
    rng = np.random.default_rng(seed)
    timestep = rng.integers(0, 96, n).astype(float)
    soc       = rng.uniform(0.2, 0.8, n)
    soc_t     = rng.uniform(0.2, 0.4, n)
    price     = energy_price_profile[timestep.astype(int)]
    bat_cap   = rng.choice([0.0, 1.0, 2.0], n)
    time_dep  = rng.uniform(0, 48, n)
    charger   = rng.choice([3.0, 7.4, 11.0, 22.0], n)
    return np.stack(
        [timestep, soc, soc - soc_t, price, bat_cap, time_dep, charger], axis=1
    ).astype(np.float32)


def compute_action_feature_heatmap(
    net: RewardNet,
    feat_idx: int,
    feat_vals: np.ndarray,
    action_vals: np.ndarray,
    background_states: np.ndarray,
    coupled: dict | None = None,
) -> np.ndarray:
    """2D PDP: rows = feat_vals (y-axis), cols = action_vals (x-axis).
    For each grid point, substitutes feat_val into all background states and
    averages the reward — a true Partial Dependence Plot.
    coupled: optional dict {feat_idx: array_of_values} for features that must
    change in tandem with feat_vals (e.g. timestep when sweeping price).
    Returns array of shape (len(feat_vals), len(action_vals)).
    """
    n_bg = len(background_states)
    grid = np.zeros((len(feat_vals), len(action_vals)), dtype=np.float32)
    for iy, fv in enumerate(feat_vals):
        states = background_states.copy()
        states[:, feat_idx] = fv
        if coupled:
            for cidx, cvals in coupled.items():
                states[:, cidx] = float(cvals[iy])
        obs_norm = torch.tensor(states / SCALES, dtype=torch.float32)
        for ix, av in enumerate(action_vals):
            act_t = torch.full((n_bg, 1), float(av), dtype=torch.float32)
            with torch.no_grad():
                grid[iy, ix] = net(obs_norm, act_t).mean().item()
    return grid


OBS_FEATURE_NAMES = ["timestep", "soc", "soc_gap", "price", "battery_cap", "time_to_dep", "charger_pwr"]


def compute_shap_per_decision(
    net: RewardNet,
    n_background: int = 300,
    seed: int = 42,
    charge_val: float = 0.3,
    discharge_val: float = -0.3,
) -> dict[str, np.ndarray]:
    """
    Compute SHAP values for R(s, a) with action fixed at charge / discharge /
    idle values. Explains which *state* features drive the reward for each
    decision type — action is held constant so it is not a SHAP feature.

    Returns a dict with keys "charge", "discharge", "idle", each an array
    of shape (obs_dim,) containing mean |SHAP| normalised to sum to 1.
    """
    class _FixedActionWrapper(nn.Module):
        def __init__(self, rn: RewardNet, action_val: float):
            super().__init__()
            self.rn = rn
            self.action_val = action_val
        def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
            n = obs_norm.shape[0]
            act = torch.full((n, 1), self.action_val,
                             dtype=obs_norm.dtype, device=obs_norm.device)
            return self.rn(obs_norm, act).unsqueeze(-1)

    rng = np.random.default_rng(seed)
    n = n_background
    _soc   = rng.uniform(0.2, 0.8, n)
    _soc_t = rng.uniform(0.2, 0.5, n)
    timestep = rng.integers(0, 96, n)
    # timestep = np.full(n, 48)
    price = energy_price_profile[timestep]
    # price = rng.uniform(0.07, 0.47, n)

    bg = np.stack([
        timestep / 96.0,
        _soc,
        _soc - _soc_t,
        price / 0.47,
        rng.choice([0.0, 0.5, 1.0], n),
        rng.uniform(0.0, 1.0, n),
        rng.choice([3/22, 7.4/22, 11/22, 1.0], n),
    ], axis=1).astype(np.float32)
    bg_tensor = torch.tensor(bg)

    results: dict[str, np.ndarray] = {}
    for key, action_val in [("charge", charge_val), ("discharge", discharge_val), ("idle", 0.0)]:
        wrapper = _FixedActionWrapper(net, action_val)
        wrapper.eval()
        explainer = shap.GradientExplainer(wrapper, bg_tensor)
        sv = explainer.shap_values(bg_tensor)
        if isinstance(sv, list):
            sv = sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 0]
        mean_abs = np.abs(sv).mean(axis=0)
        total = mean_abs.sum()
        results[key] = mean_abs / total if total > 0 else mean_abs

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    segment_labels = list(SEGMENTS.keys())
    n_segs = len(segment_labels)

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    seg_colours = [prop_cycle[i % len(prop_cycle)] for i in range(n_segs)]

    # ------------------------------------------------------------------
    # Load networks
    # ------------------------------------------------------------------
    nets: dict[str, RewardNet] = {}
    for label, entry in SEGMENTS.items():
        d = resolve_dir(entry)
        nets[label] = load_net(d, EPOCH)
        print(f"[OK] Loaded {label} from {d.name} epoch {EPOCH}")

    # ------------------------------------------------------------------
    # Plot: Decoupled price sweep (2 rows × 3 soc_gap levels)
    # ------------------------------------------------------------------
    sweep_price = {lbl: compute_price_sweep_decoupled(net) for lbl, net in nets.items()}

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey="row")
    for col, (soc_gap, soc_label) in enumerate(zip(SOC_GAP_SWEEP_LEVELS, SOC_GAP_SWEEP_LABELS)):
        ax_chg = axes[0, col]
        ax_dis = axes[1, col]
        for lbl, seg_colour in zip(segment_labels, seg_colours):
            prices, chg_advs, dis_advs = sweep_price[lbl][float(soc_gap)]
            slope_chg = float(np.polyfit(prices, chg_advs, 1)[0])
            slope_dis = float(np.polyfit(prices, dis_advs, 1)[0])
            ax_chg.plot(prices, chg_advs, '-', color=seg_colour,
                        label=f"{lbl}")
            ax_dis.plot(prices, dis_advs, '-', color=seg_colour,
                        label=f"{lbl}")
        for ax in (ax_chg, ax_dis):
            ax.axhline(0, color='k', linestyle='--', alpha=0.4)
            ax.axvline(MEAN_PRICE, color='grey', linestyle=':', alpha=0.6,
                       label=f"mean £{MEAN_PRICE:.2f}")
            ax.set_xlabel("Energy Price (£/kWh)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        ax_chg.set_title(f"Charge Advantage\n{soc_label}", fontsize=9)
        ax_dis.set_title(f"Discharge Advantage\n{soc_label}", fontsize=9)
        if col == 0:
            ax_chg.set_ylabel("Advantage over idle")
            ax_dis.set_ylabel("Advantage over idle")

    fig.suptitle(
        f"Decoupled Price Sweep — Segment Comparison (Epoch {EPOCH})\n"
        "ts=24 fixed; price swept independently [£0.07–£0.47];",
        fontsize=12,
    )
    fig.tight_layout()
    p = OUTDIR / f"sweep_price_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot C: SoC Gap x Action reward heatmaps (one per segment)
    # ------------------------------------------------------------------
    soc_gap_vals_c = np.linspace(-0.4, 0.4, 19)
    action_vals    = np.linspace(-1.0, 1.0, 21)

    print("Computing SoC Gap × Action PDP heatmaps...")
    background_states = generate_background_states(n=200, seed=42)

    fig, axes = plt.subplots(1, n_segs, figsize=(6 * n_segs, 5))
    if n_segs == 1:
        axes = [axes]

    all_grids = [
        compute_action_feature_heatmap(nets[lbl], 2, soc_gap_vals_c, action_vals, background_states)
        for lbl in segment_labels
    ]

    for ax, label, grid in zip(axes, segment_labels, all_grids):
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       extent=[action_vals[0], action_vals[-1], soc_gap_vals_c[0], soc_gap_vals_c[-1]],
                       vmin=grid.min(), vmax=grid.max(), cmap="RdYlGn")
        ax.set_xlabel("Action (−1=discharge, +1=charge)")
        ax.set_ylabel("SoC Gap (soc − soc_target)")
        ax.set_title(label, fontsize=9)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(0, color="white", linewidth=0.8, linestyle=":", alpha=0.7)
        plt.colorbar(im, ax=ax, label="Reward")

    fig.suptitle(f"SoC Gap × Action — 2D Partial Dependence Plot\n"
                 f"Averaged over {len(background_states)} background states", fontsize=11)
    fig.tight_layout()
    p = OUTDIR / f"heatmap_socgap_action_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # Plot D: Price x Action reward heatmaps (SoC fixed at 0.5)
    # ------------------------------------------------------------------
    # Sweep price monotonically 0.07→0.47; couple each price to the nearest
    # timestep in the ascending morning half (ts 0–47) so the mapping is 1-to-1.
    price_vals = np.linspace(0.07, 0.47, 20)
    _ts_asc = np.arange(0, 48)
    _prices_asc = energy_price_profile[_ts_asc]
    coupled_timesteps = np.array(
        [_ts_asc[np.argmin(np.abs(_prices_asc - p))] for p in price_vals]
    ).astype(float)

    print("Computing Price × Action PDP heatmaps...")
    fig, axes = plt.subplots(1, n_segs, figsize=(6 * n_segs, 5))
    if n_segs == 1:
        axes = [axes]

    all_grids_p = [
        compute_action_feature_heatmap(
            nets[lbl], 3, price_vals, action_vals, background_states,
            coupled={0: coupled_timesteps},
        )
        for lbl in segment_labels
    ]

    for ax, label, grid in zip(axes, segment_labels, all_grids_p):
        im = ax.imshow(grid, aspect="auto", origin="lower",
                       extent=[action_vals[0], action_vals[-1], price_vals[0], price_vals[-1]],
                       vmin=grid.min(), vmax=grid.max(), cmap="RdYlGn")
        ax.set_xlabel("Action (−1=discharge, +1=charge)")
        ax.set_ylabel("Energy Price (£/kWh)")
        ax.set_title(label, fontsize=9)
        ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axhline(MEAN_PRICE, color="cyan", linewidth=0.8, linestyle="--", alpha=0.7,
                   label=f"mean £{MEAN_PRICE:.2f} (ts=24)")
        ax.legend(fontsize=7)
        plt.colorbar(im, ax=ax, label="Reward")

    fig.suptitle(f"Price × Action — 2D Partial Dependence Plot\n"
                 f"Timestep coupled to price (morning half); averaged over {len(background_states)} background states", fontsize=11)
    fig.tight_layout()
    p = OUTDIR / f"heatmap_price_action_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    
    # ------------------------------------------------------------------
    # Plot: SoC gap sweep (charge adv | discharge adv)
    # ------------------------------------------------------------------
    sweep_soc = {lbl: compute_soc_gap_sweep(net) for lbl, net in nets.items()}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for lbl, (gaps, chg_advs, _) in sweep_soc.items():
        ax.plot(gaps, chg_advs, '-o', markersize=3, label=lbl)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.axvline(0, color='k', linestyle=':', alpha=0.4, label="soc = soc_target")
    ax.set_xlabel("SoC Gap (soc − soc_target)")
    ax.set_ylabel("Charge advantage over idle")
    ax.set_title("Charge Advantage vs SoC Gap\n(should slope ↓ left: prefer charging when below target)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for lbl, (gaps, _, dis_advs) in sweep_soc.items():
        ax.plot(gaps, dis_advs, '-o', markersize=3, label=lbl)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4)
    ax.axvline(0, color='k', linestyle=':', alpha=0.4, label="soc = soc_target")
    ax.set_xlabel("SoC Gap (soc − soc_target)")
    ax.set_ylabel("Discharge advantage over idle")
    ax.set_title("Discharge Advantage vs SoC Gap\n(should slope ↑ right: prefer discharging when above target)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"SoC Gap Sweep — Segment Comparison (Epoch {EPOCH})\n"
                 f"soc=0.5 fixed, price=£{MEAN_PRICE:.2f} [ts=24], time_to_next=20", fontsize=12)
    fig.tight_layout()
    p = OUTDIR / f"sweep_socgap_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    # ------------------------------------------------------------------
    # SHAP: per-decision (charge / discharge / idle)
    # ------------------------------------------------------------------
    print("\nComputing SHAP per decision type (charge / discharge / idle)...")
    shap_per_seg = {lbl: compute_shap_per_decision(net) for lbl, net in nets.items()}

    n_feat = len(OBS_FEATURE_NAMES)
    x = np.arange(n_feat)
    width = 0.8 / n_segs

    decision_titles = {
        "charge":    "Charge (action=+0.3)",
        "discharge": "Discharge (action=−0.3)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for col_idx, decision in enumerate(["charge", "discharge"]):
        ax = axes[col_idx]
        for i, lbl in enumerate(segment_labels):
            offset = (i - n_segs / 2 + 0.5) * width
            vals = shap_per_seg[lbl][decision]
            bars = ax.bar(x + offset, vals, width, label=lbl, color=seg_colours[i], alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(OBS_FEATURE_NAMES, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Mean |SHAP| on R(s,a) (normalised)")
        ax.set_title(decision_titles[decision], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"SHAP: State Feature Importance per Decision Type — Segment Comparison",
        fontsize=11,
    )
    fig.tight_layout()
    p = OUTDIR / f"shap_per_decision_epoch{EPOCH}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved plot -> {p}")

    print(f"\nAll outputs saved to {OUTDIR}")


if __name__ == "__main__":
    main()

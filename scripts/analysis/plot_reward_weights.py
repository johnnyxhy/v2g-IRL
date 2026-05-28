"""
plot_reward_weights.py — Plot reward weight evolution from a reward_weights_evolution.csv file.

Set CSV_PATH below, then run:
    python scripts/analysis/plot_reward_weights.py
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Edit this path ────────────────────────────────────────────────────────────
CSV_PATH = "models/MaxEnt/simple/MaxEntIRL_simple_male5059/reward_weights_evolution.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── Style ─────────────────────────────────────────────────────────────────────
GRID_COLOR  = "#E0E0E0"
SPINE_COLOR = "#000000"

# One color per feature (up to 9 features across all variants)
FEATURE_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # yellow-green
]

plt.rcParams.update({
    "font.size":          14,
    "axes.titlesize":     14,
    "axes.titleweight":   "regular",
    "axes.labelsize":     13,
    "axes.labelweight":   "regular",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.edgecolor":     SPINE_COLOR,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    12,
    "ytick.labelsize":    12,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.fontsize":    12,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   SPINE_COLOR,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})

FEATURE_LABELS = {
    "amount_charged":     "amount_charged",
    "amount_discharged":  "amount_discharged",
    "soc_below_target":   "soc_below_target",
    "soc_above_target":   "soc_above_target",
    "journey_failure":    "journey_failure",
    "charge_cost":        "charge_cost",
    "discharge_revenue":  "discharge_revenue",
    "charge_timing_quality":    "Charge_Timing_Quality",
    "discharge_timing_quality": "Discharge_Timing_Quality",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(csv_path: str) -> dict:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            try:
                data[k].append(float(v))
            except (ValueError, TypeError):
                data[k].append(None)
    return data


def plot_reward_weights(csv_path: str):
    data = load_csv(csv_path)
    epochs = data['epoch']
    features = [k for k in data.keys() if k != 'epoch']
    n = len(features)

    ncols = 2
    nrows = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.2))
    axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]
        label = FEATURE_LABELS.get(feature, feature.replace('_', ' ').title())
        color = FEATURE_COLORS[i % len(FEATURE_COLORS)]

        ax.plot(
            epochs, data[feature],
            color=color, linewidth=1.8,
            marker='o', markersize=4,
            markerfacecolor='white', markeredgewidth=1.4,
            zorder=2,
        )
        #ax.axhline(0, color=SPINE_COLOR, linewidth=0.6, linestyle='--', alpha=0.4, zorder=1)
        ax.set_title(label, pad=8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight')
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)

    # Hide any unused subplots
    for j in range(n, nrows * ncols):
        axes[j].set_visible(False)

    #fig.suptitle('Reward Weight Evolution', fontsize=13, y=1.01)
    plt.tight_layout()

    out = os.path.join(os.path.dirname(csv_path), 'reward_weights_evolution.png')
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == '__main__':
    if not os.path.isfile(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
    else:
        plot_reward_weights(CSV_PATH)

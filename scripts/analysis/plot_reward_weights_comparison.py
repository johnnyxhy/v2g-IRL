import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Plot style ─────────────────────────────────────────────────────────────────
GRID_COLOR  = "#E0E0E0"
SPINE_COLOR = "#000000"
FIG_SIZE    = (12, 5)

plt.rcParams.update({
    "font.size":          14,
    "axes.titlesize":     14,
    "axes.titleweight":   "regular",
    "axes.labelsize":     14,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.edgecolor":     SPINE_COLOR,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    12,
    "ytick.labelsize":    13,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.fontsize":    12,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   SPINE_COLOR,
    "figure.dpi":         150,
})
# ──────────────────────────────────────────────────────────────────────────────

FEAT_NAMES = [
    "amount\ncharged",
    "amount\ndischarged",
    "charge\nprice\nquality",
    "discharge\nprice\nquality",
    "soc\nbelow\ntarget",
    "soc\nabove\ntarget",
    "journey\nfailure",
]

SEGMENTS = {
    "Initial":       None,
    "Male 50–59":    "models/MaxEnt/discrete/MaxEntIRL_discrete_pricediff_male5059/final_reward_weights.txt",
    "Male 40–49":    "models/MaxEnt/discrete/MaxEntIRL_discrete_pricediff_male4049/final_reward_weights.txt",
    "Female 50–59":  "models/MaxEnt/discrete/MaxEntIRL_discrete_pricediff_female5059/final_reward_weights.txt",
}

INITIAL_WEIGHTS = np.array([0.5, 0.5, 0.5, 0.5, -1.0, -0.2, -1.0], dtype=np.float32)

COLORS = {
    "Initial":      "#AAAAAA",
    "Male 50–59":   "#1f77b4",
    "Male 40–49":   "#2ca02c",
    "Female 50–59": "#d62728",
}


def load_weights(path):
    weights = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                weights.append(float(line.split()[-1]))
    return np.array(weights, dtype=np.float32)


def plot_comparison(weights_dict):
    n_features = len(FEAT_NAMES)
    n_groups = len(weights_dict)
    bar_width = 0.18
    group_gap = 0.05
    total_bar_span = n_groups * bar_width + (n_groups - 1) * group_gap
    x = np.arange(n_features)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for i, (label, weights) in enumerate(weights_dict.items()):
        offset = (i - (n_groups - 1) / 2) * (bar_width + group_gap)
        bars = ax.bar(
            x + offset,
            weights,
            width=bar_width,
            color=COLORS[label],
            label=label,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        for bar, val in zip(bars, weights):
            y_pos = val + 0.02 if val >= 0 else val - 0.02
            va = "bottom" if val >= 0 else "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_pos,
                f"{val:.3f}",
                ha="center",
                va=va,
                fontsize=8,
                color=COLORS[label],
                zorder=4,
            )

    ax.axhline(0, color=SPINE_COLOR, linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(FEAT_NAMES)
    ax.set_ylabel("Reward Weight")
    ax.set_xlabel("Feature")
    ax.set_title("Linear MaxEnt IRL — Final Reward Weights by Population Segment")
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(
        "models/MaxEnt/discrete/reward_weights_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved: models/MaxEnt/discrete/reward_weights_comparison.png")
    plt.show()


if __name__ == "__main__":
    weights_dict = {}
    for label, path in SEGMENTS.items():
        if path is None:
            weights_dict[label] = INITIAL_WEIGHTS
        else:
            weights_dict[label] = load_weights(path)

    for label, w in weights_dict.items():
        feat_str = ", ".join(f"{n.replace(chr(10), ' ')}={v:.4f}" for n, v in zip(FEAT_NAMES, w))
        print(f"{label:15s}: {feat_str}")

    plot_comparison(weights_dict)

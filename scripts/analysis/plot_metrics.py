"""
plot_metrics.py — Plot training metrics from a metrics.csv file.

Set CSV_PATH below, then run:
    python scripts/analysis/plot_metrics.py
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Edit this path ────────────────────────────────────────────────────────────
CSV_PATH = "models/MaxEnt/discrete/MaxEntIRL_discrete_pricediff_male5059/metrics.csv"
# ─────────────────────────────────────────────────────────────────────────────

# ── Style ─────────────────────────────────────────────────────────────────────
TRAIN_COLOR = "#0000FF"
VAL_COLOR   = "#FF0000"
GRID_COLOR  = "#E0E0E0"
SPINE_COLOR = "#000000"

plt.rcParams.update({
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.titleweight":   "regular",
    "axes.labelsize":     11,
    "axes.labelweight":   "regular",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.edgecolor":     SPINE_COLOR,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   SPINE_COLOR,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
})
# ─────────────────────────────────────────────────────────────────────────────

FIG_SIZE = (5, 3.2)


def load_metrics(csv_path: str) -> dict:
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


def _make_ax(title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def _add_line(ax, x, y, label, color, zorder=2):
    ax.plot(x, y, color=color, linewidth=1.8, label=label,
            marker='o', markersize=4, markerfacecolor='white',
            markeredgewidth=1.4, zorder=zorder)


def _integer_xticks(ax, epochs):
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlim(epochs[0] - 0.3, epochs[-1] + 0.3)


def _save(fig, folder, name):
    out = os.path.join(folder, f'{name}.png')
    fig.savefig(out)
    print(f"Saved {out}")
    plt.close(fig)


def plot_metrics(csv_path: str):
    data = load_metrics(csv_path)
    epochs = data['epoch']
    folder = os.path.dirname(csv_path)
    path_lower = csv_path.replace('\\', '/').lower()

    # Detect variant
    has_reward_loss = 'reward_loss' in data and any(v is not None for v in data['reward_loss'])
    has_log_likelihood = 'log_likelihood' in data and any(v is not None for v in data['log_likelihood'])
    is_deepmaxent = ('deepmaxent' in path_lower) or has_reward_loss
    is_adversarial = 'disc_loss' in data and any(v is not None for v in data['disc_loss'])

    has_val = any(v is not None for v in data.get('val_dtw', []))

    # DTW
    fig, ax = _make_ax('DTW Distance', 'Epoch', 'DTW Distance')
    _add_line(ax, epochs, data['train_dtw'], 'Train', TRAIN_COLOR)
    if has_val:
        _add_line(ax, epochs, data['val_dtw'], 'Validation', VAL_COLOR)
        ax.legend()
    _integer_xticks(ax, epochs)
    plt.tight_layout()
    _save(fig, folder, 'dtw_distance')

    # MAE
    fig, ax = _make_ax('Mean Absolute Error', 'Epoch', 'MAE')
    _add_line(ax, epochs, data['train_mae'], 'Train', TRAIN_COLOR)
    if has_val:
        _add_line(ax, epochs, data['val_mae'], 'Validation', VAL_COLOR)
        ax.legend()
    _integer_xticks(ax, epochs)
    plt.tight_layout()
    _save(fig, folder, 'mae')

    # Feature L2
    l2_key = 'train_feat_l2' if 'train_feat_l2' in data else 'train_l2'
    val_l2_key = 'val_feat_l2' if 'val_feat_l2' in data else 'val_l2'
    if l2_key in data:
        fig, ax = _make_ax('Feature Expectation L2 Loss', 'Epoch', 'L2 Loss')
        _add_line(ax, epochs, data[l2_key], 'Train', TRAIN_COLOR)
        # if has_val and val_l2_key in data:
        #     _add_line(ax, epochs, data[val_l2_key], 'Validation', VAL_COLOR)
        #     ax.legend()
        _integer_xticks(ax, epochs)
        plt.tight_layout()
        _save(fig, folder, 'feature_l2')

    # Adversarial-specific metrics
    if is_adversarial:
        # Discriminator vs policy accuracy on one graph
        disc_acc_key = 'disc_acc' if 'disc_acc' in data else 'expert_acc'
        fig, ax = _make_ax('Discriminator Accuracy', 'Epoch', 'Accuracy')
        if disc_acc_key in data:
            _add_line(ax, epochs, data[disc_acc_key], 'Expert Accuracy', TRAIN_COLOR)
        if 'policy_acc' in data:
            _add_line(ax, epochs, data['policy_acc'], 'Policy Accuracy', VAL_COLOR)
        ax.legend()
        _integer_xticks(ax, epochs)
        plt.tight_layout()
        _save(fig, folder, 'accuracy')

        # Discriminator loss on separate graph
        fig, ax = _make_ax('Discriminator Loss', 'Epoch', 'Loss')
        _add_line(ax, epochs, data['disc_loss'], 'Discriminator Loss', TRAIN_COLOR)
        _integer_xticks(ax, epochs)
        plt.tight_layout()
        _save(fig, folder, 'disc_loss')

    # Reward Loss (DeepMaxEnt only)
    # If reward_loss is unavailable, use -log_likelihood as the mirrored proxy.
    if is_deepmaxent and (has_reward_loss or has_log_likelihood):
        fig, ax = _make_ax('Reward Loss', 'Epoch', 'Loss')
        if has_reward_loss:
            reward_loss_values = data['reward_loss']
            reward_label = 'Train'
        else:
            reward_loss_values = [(-v if v is not None else None) for v in data['log_likelihood']]
            reward_label = 'Train (-Log-Likelihood proxy)'
        _add_line(ax, epochs, reward_loss_values, reward_label, TRAIN_COLOR)
        _integer_xticks(ax, epochs)
        plt.tight_layout()
        _save(fig, folder, 'reward_loss')

    # Log-Likelihood (both DeepMaxEnt and MaxEnt)
    if has_log_likelihood:
        fig, ax = _make_ax('Expert Log-Likelihood', 'Epoch', r'$\log\, p(\tau_{\mathrm{expert}})$')
        _add_line(ax, epochs, data['log_likelihood'], 'Train', TRAIN_COLOR)
        _integer_xticks(ax, epochs)
        plt.tight_layout()
        _save(fig, folder, 'log_likelihood')

    # Learning Rate
    fig, ax = _make_ax('Reward Learning Rate Schedule', 'Epoch', 'Learning Rate')
    _add_line(ax, epochs, data['lr'], 'LR', TRAIN_COLOR)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    _integer_xticks(ax, epochs)
    plt.tight_layout()
    _save(fig, folder, 'learning_rate')


if __name__ == '__main__':
    if not os.path.isfile(CSV_PATH):
        print(f"File not found: {CSV_PATH}")
    else:
        plot_metrics(CSV_PATH)

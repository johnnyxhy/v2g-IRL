"""
plot_metrics.py — Plot training metrics from a metrics.csv file.

Usage:
    python scripts/analysis/plot_metrics.py <path/to/metrics.csv>
    python scripts/analysis/plot_metrics.py  # prompts for path
"""

import sys
import os
import csv
import matplotlib.pyplot as plt


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


def plot_metrics(csv_path: str):
    data = load_metrics(csv_path)
    epochs = data['epoch']
    folder = os.path.dirname(csv_path)

    has_val = any(v is not None for v in data.get('val_dtw', []))

    def _save(fig, name):
        out = os.path.join(folder, f'{name}.png')
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")
        plt.close(fig)

    # DTW
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data['train_dtw'], label='Train')
    if has_val:
        ax.plot(epochs, data['val_dtw'], label='Val')
        ax.legend()
    ax.set_title('DTW Distance')
    ax.set_xlabel('Epoch'); ax.set_ylabel('DTW'); ax.grid(True)
    plt.tight_layout(); _save(fig, 'dtw_distance')

    # MAE
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data['train_mae'], label='Train')
    if has_val:
        ax.plot(epochs, data['val_mae'], label='Val')
        ax.legend()
    ax.set_title('SoC MAE')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MAE'); ax.grid(True)
    plt.tight_layout(); _save(fig, 'mae')

    # Feature L2
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data['train_feat_l2'], label='Train')
    if has_val:
        ax.plot(epochs, data['val_feat_l2'], label='Val')
        ax.legend()
    ax.set_title('Feature L2 Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('L2'); ax.grid(True)
    plt.tight_layout(); _save(fig, 'feature_l2')

    # Reward Loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data['reward_loss'])
    ax.set_title('Reward Loss (neg log-likelihood)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True)
    plt.tight_layout(); _save(fig, 'reward_loss')

    # Log-Likelihood
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data['log_likelihood'])
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Converged (LL=0)')
    ax.set_title('Expert Log-Likelihood')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Avg log p(τ_expert)'); ax.legend(); ax.grid(True)
    plt.tight_layout(); _save(fig, 'log_likelihood')

    # Learning Rate
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, data['lr'])
    ax.set_title('Learning Rate')
    ax.set_xlabel('Epoch'); ax.set_ylabel('LR'); ax.grid(True)
    plt.tight_layout(); _save(fig, 'learning_rate')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Path to metrics.csv: ").strip().strip('"')

    if not os.path.isfile(path):
        print(f"File not found: {path}")
        sys.exit(1)

    plot_metrics(path)

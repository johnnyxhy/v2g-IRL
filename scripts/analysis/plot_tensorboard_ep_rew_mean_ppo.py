"""
Plot TensorBoard ep_rew_mean for PPO as an epoch-overlay figure.

PPO policy is reset every IRL epoch, so only an overlay plot is generated.

Usage:
    1) Edit RUN_CONFIG below.
    2) Run: python scripts/analysis/plot_tensorboard_ep_rew_mean_ppo.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


RUN_CONFIG = {
    "run_root": Path("old_models/MaxEntIRL_discrete_pricediff_male5059"),
    "tensorboard_subdir": Path("tensorboard"),
    "output_name": "ep_rew_mean_ppo_overlay.png",
    "tag": "ep_rew_mean",
    "title": "PPO Episode Reward Mean (IRL Epoch Overlay)",
    "smooth_window": 1,
    "dpi": 300,
}


def find_event_files(logdir: Path) -> list[Path]:
    return sorted(logdir.rglob("events.out.tfevents.*"))


def parse_epoch_from_path(event_file: Path) -> int | None:
    for part in event_file.parts:
        match = re.fullmatch(r"epoch_(\d+)_\d+", part)
        if match:
            return int(match.group(1))
    return None


def find_matching_tag(scalar_tags: list[str], requested_tag: str) -> str | None:
    if requested_tag in scalar_tags:
        return requested_tag

    suffix_matches = [tag for tag in scalar_tags if tag.endswith(f"/{requested_tag}")]
    if suffix_matches:
        return suffix_matches[0]

    contains_matches = [tag for tag in scalar_tags if requested_tag in tag]
    if contains_matches:
        return contains_matches[0]

    return None


def load_epoch_series(
    event_files: list[Path], requested_tag: str
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], str]:
    # Group by IRL epoch number parsed from TensorBoard folder names (epoch_X_Y).
    epoch_data: dict[int, dict[float, list[float]]] = {}
    matched_tag: str | None = None

    for event_file in event_files:
        epoch = parse_epoch_from_path(event_file)
        if epoch is None:
            continue

        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()

        scalar_tags = acc.Tags().get("scalars", [])
        tag_to_use = find_matching_tag(scalar_tags, requested_tag)
        if tag_to_use is None:
            continue

        if matched_tag is None:
            matched_tag = tag_to_use

        bucket = epoch_data.setdefault(epoch, {})
        for event in acc.Scalars(tag_to_use):
            step = float(event.step)
            bucket.setdefault(step, []).append(float(event.value))

    if not epoch_data:
        raise ValueError(
            f"No scalar data found for tag '{requested_tag}'. "
            "Check RUN_CONFIG['tag'] and verify event files contain that scalar."
        )

    series: list[tuple[int, np.ndarray, np.ndarray]] = []
    for epoch in sorted(epoch_data.keys()):
        step_to_values = epoch_data[epoch]
        steps = np.array(sorted(step_to_values.keys()), dtype=np.float64)
        values = np.array([np.mean(step_to_values[s]) for s in steps], dtype=np.float64)
        series.append((epoch, steps, values))

    return series, matched_tag or requested_tag


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def smooth_series(
    series: list[tuple[int, np.ndarray, np.ndarray]], smooth_window: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    smoothed: list[tuple[int, np.ndarray, np.ndarray]] = []
    for epoch, steps, values in series:
        smoothed_values = moving_average(values, smooth_window)
        if len(smoothed_values) == len(values):
            smoothed_steps = steps
        else:
            smoothed_steps = steps[smooth_window - 1 :]
        smoothed.append((epoch, smoothed_steps, smoothed_values))
    return smoothed


def build_epoch_palette(n_colors: int) -> list[tuple[float, float, float, float]]:
    palette: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])

    palette = palette[:n_colors]

    adjusted: list[tuple[float, float, float, float]] = []
    for r, g, b, _ in palette:
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if luminance > 0.68:
            scale = 0.68 / luminance
            r, g, b = r * scale, g * scale, b * scale
        adjusted.append((r, g, b, 1.0))
    return adjusted


def plot_ppo_overlay(
    logdir: Path,
    output: Path,
    tag: str,
    title: str,
    smooth_window: int,
    dpi: int,
) -> None:
    event_files = find_event_files(logdir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {logdir}")

    series, matched_tag = load_epoch_series(event_files, tag)
    series = smooth_series(series, smooth_window)

    epoch_ids = np.array([epoch for epoch, _, _ in series], dtype=np.int32)
    epoch_colors = build_epoch_palette(len(epoch_ids))
    cmap = colors.ListedColormap(epoch_colors)
    norm = colors.BoundaryNorm(np.arange(0.5, len(epoch_ids) + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    for color_idx, (_epoch, steps, values) in enumerate(series, start=1):
        local_steps = steps - steps.min()
        color = epoch_colors[color_idx - 1]
        ax.plot(local_steps, values, linewidth=1.5, color=color, alpha=0.95)

    ax.set_title(title)
    ax.set_xlabel("Local PPO Step Within IRL Epoch")
    ax.set_ylabel("Episode Reward Mean")
    ax.grid(True, alpha=0.35)

    scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(scalar_mappable, ax=ax, pad=0.02)
    cbar.set_label("IRL Epoch")

    if len(epoch_ids) <= 12:
        tick_positions = np.arange(1, len(epoch_ids) + 1)
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([str(int(epoch_ids[pos - 1])) for pos in tick_positions])
    else:
        tick_positions = np.unique(np.linspace(1, len(epoch_ids), num=8, dtype=np.int32))
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([str(int(epoch_ids[pos - 1])) for pos in tick_positions])

    ax.text(
        0.99,
        0.01,
        (
            f"Tag: {matched_tag}\n"
            f"Epochs: {int(epoch_ids.min())}-{int(epoch_ids.max())}\n"
            f"Smoothed: window={smooth_window}"
        ),
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.8,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def main() -> None:
    run_root = Path(RUN_CONFIG["run_root"])
    logdir = run_root / Path(RUN_CONFIG["tensorboard_subdir"])
    output = run_root / str(RUN_CONFIG["output_name"])

    plot_ppo_overlay(
        logdir=logdir,
        output=output,
        tag=str(RUN_CONFIG["tag"]),
        title=str(RUN_CONFIG["title"]),
        smooth_window=int(RUN_CONFIG["smooth_window"]),
        dpi=int(RUN_CONFIG["dpi"]),
    )
    print(f"Using logdir: {logdir}")
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()

"""
Plot TensorBoard scalar data for ep_rew_mean as a Matplotlib figure.

Usage:
    1) Edit RUN_CONFIG below.
    2) Run: python scripts/analysis/plot_tensorboard_ep_rew_mean.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# Edit these values to run the script.
RUN_CONFIG = {
    "run_root": Path("models/DeepMaxEnt/continuous/DeepMaxEntIRL_continuous_male5059_new"),
    "tensorboard_subdir": Path("tensorboard"),
    "output_name": "ep_rew_mean_report.png",
    "overlay_output_name": "ep_rew_mean_report_overlay.png",
    "tag": "ep_rew_mean",
    "title": "Deep MaxEnt IRL - ep_rew_mean by IRL Epoch",
    "smooth_window": 1,
    "dpi": 300,
}


def find_event_files(logdir: Path) -> list[Path]:
    return sorted(logdir.rglob("events.out.tfevents.*"))


def parse_epoch_folder_from_path(event_file: Path) -> tuple[str, int] | None:
    for part in event_file.parts:
        match = re.fullmatch(r"epoch_(\d+)_\d+", part)
        if match:
            return part, int(match.group(1))
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


def load_scalar_points(
    event_files: list[Path], requested_tag: str
) -> tuple[list[tuple[str, int, np.ndarray, np.ndarray]], str]:
    epoch_series: dict[str, dict[str, object]] = {}
    matched_tag: str | None = None

    for event_file in event_files:
        epoch_info = parse_epoch_folder_from_path(event_file)
        if epoch_info is None:
            continue
        folder_name, epoch_num = epoch_info

        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        scalar_tags = acc.Tags().get("scalars", [])

        tag_to_use = find_matching_tag(scalar_tags, requested_tag)
        if tag_to_use is None:
            continue

        if matched_tag is None:
            matched_tag = tag_to_use

        epoch_entry = epoch_series.setdefault(
            folder_name,
            {"epoch_num": epoch_num, "step_to_values": {}},
        )
        step_to_values = epoch_entry["step_to_values"]
        assert isinstance(step_to_values, dict)
        for event in acc.Scalars(tag_to_use):
            step = float(event.step)
            step_to_values.setdefault(step, []).append(float(event.value))

    if not epoch_series:
        raise ValueError(
            f"No scalar data found for tag '{requested_tag}'. "
            "Check --tag and verify event files contain that scalar."
        )

    series: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    sorted_items = sorted(
        epoch_series.items(),
        key=lambda item: (int(item[1]["epoch_num"]), item[0]),
    )
    for folder_name, entry in sorted_items:
        epoch_num = int(entry["epoch_num"])
        step_to_values = entry["step_to_values"]
        assert isinstance(step_to_values, dict)
        steps = np.array(sorted(step_to_values.keys()), dtype=np.float64)
        values = np.array([np.mean(step_to_values[s]) for s in steps], dtype=np.float64)
        series.append((folder_name, epoch_num, steps, values))

    return series, matched_tag or requested_tag


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="valid")


def smooth_series(
    series: list[tuple[str, int, np.ndarray, np.ndarray]], smooth_window: int
) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    smoothed: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for folder_name, epoch_num, steps, values in series:
        smoothed_values = moving_average(values, smooth_window)
        if len(smoothed_values) == len(values):
            smoothed_steps = steps
        else:
            smoothed_steps = steps[smooth_window - 1 :]
        smoothed.append((folder_name, epoch_num, smoothed_steps, smoothed_values))
    return smoothed


def build_epoch_palette(n_colors: int) -> list[tuple[float, float, float, float]]:
    """Build a discrete palette and darken overly light colors for readability."""
    palette: list[tuple[float, float, float, float]] = []
    for cmap_name in ("tab20", "tab20b", "tab20c"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])

    if n_colors > len(palette):
        extra = plt.get_cmap("Set1")
        palette.extend([extra(i) for i in range(extra.N)])

    palette = palette[:n_colors]

    adjusted: list[tuple[float, float, float, float]] = []
    for r, g, b, _ in palette:
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        if luminance > 0.68:
            scale = 0.68 / luminance
            r, g, b = r * scale, g * scale, b * scale
        adjusted.append((r, g, b, 1.0))

    return adjusted


def plot_ep_rew_mean(
    logdir: Path,
    output: Path,
    overlay_output: Path,
    tag: str,
    title: str,
    smooth_window: int,
    dpi: int,
) -> None:
    event_files = find_event_files(logdir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {logdir}")

    series, matched_tag = load_scalar_points(event_files, tag)
    series = smooth_series(series, smooth_window)

    epoch_ids = np.array([epoch_num for _, epoch_num, _, _ in series], dtype=np.int32)
    if len(epoch_ids) == 0:
        raise ValueError("No epoch data could be parsed from TensorBoard paths.")

    epoch_colors = build_epoch_palette(len(epoch_ids))
    cmap = colors.ListedColormap(epoch_colors)
    norm = colors.BoundaryNorm(np.arange(0.5, len(epoch_ids) + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    cumulative_offset = 0.0
    for color_idx, (_, _epoch_num, steps, values) in enumerate(series, start=1):
        local_steps = steps - steps.min()
        global_steps = local_steps + cumulative_offset
        color = epoch_colors[color_idx - 1]
        ax.plot(global_steps, values, linewidth=1.6, color=color, alpha=0.95)
        ax.scatter(global_steps, values, s=8, color=color, alpha=0.95)

        # Advance by the observed span from this TensorBoard epoch folder.
        cumulative_offset += float(local_steps.max()) if len(local_steps) > 0 else 0.0

    ax.set_title(title)
    ax.set_xlabel("SAC Step")
    ax.set_ylabel("Episode Reward Mean")
    ax.grid(True, alpha=0.35)

    scalar_mappable = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm,
    )
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

    # Overlay plot: all epochs aligned to local step within epoch.
    fig2, ax2 = plt.subplots(figsize=(8, 4.8))
    for color_idx, (_folder_name, _epoch_num, steps, values) in enumerate(series, start=1):
        local_steps = steps - steps.min()
        color = epoch_colors[color_idx - 1]
        ax2.plot(local_steps, values, linewidth=1.4, color=color, alpha=0.95)

    ax2.set_title(f"{title} (Epoch Overlay)")
    ax2.set_xlabel("Local SAC Step Within Epoch")
    ax2.set_ylabel("Episode Reward Mean")
    ax2.grid(True, alpha=0.35)

    scalar_mappable2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable2.set_array([])
    cbar2 = fig2.colorbar(scalar_mappable2, ax=ax2, pad=0.02)
    cbar2.set_label("IRL Epoch")
    if len(epoch_ids) <= 12:
        tick_positions2 = np.arange(1, len(epoch_ids) + 1)
        cbar2.set_ticks(tick_positions2)
        cbar2.set_ticklabels([str(int(epoch_ids[pos - 1])) for pos in tick_positions2])
    else:
        tick_positions2 = np.unique(np.linspace(1, len(epoch_ids), num=8, dtype=np.int32))
        cbar2.set_ticks(tick_positions2)
        cbar2.set_ticklabels([str(int(epoch_ids[pos - 1])) for pos in tick_positions2])

    ax2.text(
        0.99,
        0.01,
        (
            f"Tag: {matched_tag}\n"
            f"Smoothed: window={smooth_window}"
        ),
        ha="right",
        va="bottom",
        transform=ax2.transAxes,
        fontsize=9,
        alpha=0.8,
    )

    overlay_output.parent.mkdir(parents=True, exist_ok=True)
    fig2.tight_layout()
    fig2.savefig(overlay_output, dpi=dpi)
    plt.close(fig2)


def main() -> None:
    run_root = Path(RUN_CONFIG["run_root"])
    logdir = run_root / Path(RUN_CONFIG["tensorboard_subdir"])
    output = run_root / str(RUN_CONFIG["output_name"])
    overlay_output = run_root / str(RUN_CONFIG["overlay_output_name"])

    plot_ep_rew_mean(
        logdir=logdir,
        output=output,
        overlay_output=overlay_output,
        tag=str(RUN_CONFIG["tag"]),
        title=str(RUN_CONFIG["title"]),
        smooth_window=int(RUN_CONFIG["smooth_window"]),
        dpi=int(RUN_CONFIG["dpi"]),
    )
    print(f"Using logdir: {logdir}")
    print(f"Saved figure: {output}")
    print(f"Saved overlay figure: {overlay_output}")


if __name__ == "__main__":
    main()

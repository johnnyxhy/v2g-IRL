"""
Plot expert SoC trajectories from a processed trajectory JSON file.

Matches the figure style used in eval_deep_discrete.py:
  - Expert SoC (solid orange)
  - Out journey region (red shade)
  - Return journey region (blue shade)
  - Energy price profile (green)
"""

import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

JSON_PATH = "data/processed_trajectories_simple_probabilistic.json"
SEGMENT   = "Male 50-59"
N_FIGURES = 5       # number of trajectories to plot (None = all)
SHUFFLE   = False    # set True to pick random trajectories instead of first N

# ------------------------------------------------------------------ #
#  Energy price profile (shared with eval scripts)                    #
# ------------------------------------------------------------------ #

ENERGY_PRICE_PROFILE = np.array([
    0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24, 0.26,
    0.27, 0.28, 0.30, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
    0.41, 0.42, 0.43, 0.44, 0.44, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.47,
    0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.42,
    0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32, 0.31, 0.30, 0.28,
    0.27, 0.26, 0.24, 0.23, 0.22, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
    0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07,
])


def journey_duration(distance_miles: float, speed_mph: float) -> int:
    """Convert distance + speed to number of 15-min timesteps (ceiling)."""
    return int(np.ceil(distance_miles / speed_mph * 4))


def plot_expert(traj: dict, idx: int) -> None:
    iv = traj['initial_values']
    soc_history = traj['soc_history']
    T = len(soc_history)

    out_start  = iv['out_start_timestep']
    ret_start  = iv['return_start_timestep']
    out_dur    = journey_duration(iv['journey_distance'], iv['out_journey_speed'])
    ret_dur    = journey_duration(iv['journey_distance'], iv['return_journey_speed'])

    timesteps  = range(T)
    price      = ENERGY_PRICE_PROFILE[:T]

    plt.figure()
    plt.plot(timesteps, soc_history, label='Expert SoC', color='black', linestyle='--')
    plt.axvspan(out_start, out_start + out_dur,
                color='red',  alpha=0.3, label='Out Journey')
    plt.axvspan(ret_start, ret_start + ret_dur,
                color='blue', alpha=0.3, label='Return Journey')
    plt.plot(range(len(price)), price, label='Energy Price', color='green')
    plt.xlabel('Timestep')
    plt.ylabel('State of Charge (SoC)')
    plt.title(f'Expert Trajectory — {traj["segment"]}')
    plt.legend()
    plt.tight_layout()


if __name__ == "__main__":
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    trajectories = [t for t in data if SEGMENT in t['segment']]
    print(f"Found {len(trajectories)} trajectories for segment '{SEGMENT}'")

    if SHUFFLE:
        rng = np.random.default_rng(42)
        rng.shuffle(trajectories)

    to_plot = trajectories if N_FIGURES is None else trajectories[:N_FIGURES]

    for i, traj in enumerate(to_plot):
        print(f"  Plotting episode {traj['episodeID']} ({i+1}/{len(to_plot)})")
        plot_expert(traj, i)

    plt.show()

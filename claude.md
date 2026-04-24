# V2G-IRL — Maximum Entropy Inverse Reinforcement Learning for Vehicle-to-Grid

## Project Summary

This project applies **Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL)** to learn EV charging/discharging behaviour from expert demonstrations. Given a dataset of real-world EV usage trajectories, the system recovers a reward function (as a linear combination of hand-crafted features) that explains the observed behaviour, then trains a reinforcement learning agent (SAC) that reproduces that behaviour under the learned reward.

**Author:** Johnny Xiao Hong Yu (Imperial College London)

---

## High-Level Architecture

```
CSV Dataset
    │
    ▼
expert_loader_{variant}.py  ──►  processed_trajectories_{variant}.json
                                          │
                                          ▼
                              ExpertDataset (loads JSON, splits train/val)
                                          │
                                          ▼
                         train_MaxEnt_{variant}.py (entry point)
                                          │
                                          ▼
                       MaxEntIRLTrainer_Continuous (outer IRL loop)
                          ┌───────────────┼───────────────┐
                          ▼               ▼               ▼
                 SAC Policy          V2GEnv            Reward Weights w
                 (sbx/JAX)        (Gymnasium)         (gradient ascent)
                          │               │               │
                          └───────┬───────┘               │
                                  ▼                       │
                      Rollout feature expectations        │
                      φ_agent vs φ_expert                 │
                                  │                       │
                                  ▼                       │
                          ∇w = φ_expert − φ_agent ────────┘
                                  │
                                  ▼
                    Saved models + final_reward_weights.txt
```

---

## Environment Variants

There are three environment tiers, each building on the last:

| Variant | File | Action Space | Features | Key Addition |
|---------|------|-------------|----------|--------------|
| **Simple** | `V2GEnv_simple.py` | Discrete (charge / discharge / idle) | 4 | Baseline with SoC target tracking |
| **Continuous** | `V2GEnv_continuous.py` | Continuous `[-1, 1]` | 5 | Variable-timestep actions, continuous charge amount |
| **Profit** | `V2GEnv_profit.py` | Continuous `[-1, 1]` | 7 | Adds energy price awareness with charge cost & discharge revenue |

All environments share the same underlying scenario: **one EV completing a single return journey per day** (home → work → home) across **96 timesteps** (15-minute intervals). The agent controls charging/discharging while the car is parked.

---

## MDP Formulation (Continuous / Profit Variants)

### State Space

The observation is a `Dict` space:

| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `timestep` | int64 | 0–96 | Current 15-min interval |
| `soc` | float32 | 0–1 | Battery State of Charge |
| `soc_target` | float32 | 0–1 | SoC needed for next journey |
| `energy_price` | float32 | 0–0.47 | Current energy price in £/kWh *(profit variant only)* |
| `battery_capacity` | int64 | 0–2 | Index → {40, 60, 80} kWh |
| `time_to_next_journey` | int64 | 0–96 | Timesteps until next departure |
| `current_charger_power` | float32 | 0–22 | Charger power at current location (kW) |

### Action Space

- `Box(-1, 1)`: positive = charge, negative = discharge, 0 = idle
- Actions are rounded to 2–3 decimal places to avoid negligible transfers
- A single action can span **multiple 15-minute timesteps** (variable `delta_t`) — the env advances time in a while-loop until the requested energy transfer is complete or a stage boundary is reached

### Day Stages

| Stage | Location | Description |
|-------|----------|-------------|
| 0 | Home | Before work — can charge/discharge |
| 1 | Work | At work — can charge/discharge |
| 2 | Home | After work — can charge/discharge |

Journeys (driving) consume energy and are non-interactive — the agent cannot act during transit.

### Episode Initialisation

Each episode is configured with fixed parameters drawn from fitted distributions or provided explicitly via `set_initial_states()`:

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| Initial SoC | Normal(0.4944, 0.1481) | 0–1 |
| Battery capacity | Uniform | {40, 60, 80} kWh |
| Home charger | Uniform | {3, 7.4, 11} kW |
| Work charger | Uniform | {7.4, 11, 22} kW |
| Journey distance | Exponential(λ=10.13) + 10 | 10–80 miles |
| Journey speeds | Gamma(8.55, 3.47) | 10–75 mph |
| Out start timestep | Gamma(11.48, 2.61) | 0–80 |
| Return start timestep | Gamma(24.66, 1.89) + 20 | out_end–90 |

### Energy Model

- Efficiency: 3 miles/kWh (`kwh_per_mile = 1/3`)
- Max energy per timestep: `charger_power × 0.25h / battery_capacity_kWh`
- Journey energy: `distance × kwh_per_mile / battery_capacity_kWh` (as SoC fraction)
- Boundary penalty: 0.01 £/kWh for charging above 0.8 SoC or discharging below 0.2 SoC

---

## Feature Vector φ

The reward is linear in features: $R(s, a) = \mathbf{w} \cdot \boldsymbol{\phi}(s, a)$

### Continuous Variant (5 features)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | `amount_charged` | SoC increase this action (≥ 0) |
| 1 | `amount_discharged` | SoC decrease this action (≥ 0) |
| 2 | `battery_needed_target` | $\max(0,\ \text{soc\_target} - \text{soc})^2 \times \Delta t$ |
| 3 | `battery_exceeded_target` | $\max(0,\ \text{soc} - \text{soc\_target})^2 \times \Delta t$ |
| 4 | `journey_failure` | 1 if SoC insufficient for journey, else 0 |

### Profit Variant (7 features)

Adds two price-timing interaction features:

| Index | Feature | Description |
|-------|---------|-------------|
| 5 | `charge_timing_quality` | `amount_charged × (mean_price - current_price) / mean_price` — positive when charging below average price |
| 6 | `discharge_timing_quality` | `amount_discharged × (current_price - mean_price) / mean_price` — positive when discharging above average price |

### Feature Expectations

Feature expectations are computed as the **mean** of per-step feature values across an episode:

$$\bar{\phi} = \frac{1}{T} \sum_{t=1}^{T} \phi(s_t, a_t)$$

These are compared between expert and agent to drive the IRL gradient.

---

## MaxEnt IRL Algorithm

### Objective

Find reward weights $\mathbf{w}$ such that the agent's expected feature counts match the expert's:

$$\mathbb{E}_{\pi_\mathbf{w}}[\boldsymbol{\phi}] = \mathbb{E}_{\text{expert}}[\boldsymbol{\phi}]$$

Under the Maximum Entropy framework, the policy distribution over trajectories is:

$$p(\tau) \propto \exp\left(R(\tau)\right) = \exp\left(\mathbf{w} \cdot \boldsymbol{\Phi}(\tau)\right)$$

### Gradient Update

Per epoch, the gradient for each expert trajectory $i$ is:

$$\nabla_\mathbf{w} = \frac{1}{N} \sum_{i=1}^{N} \left[\boldsymbol{\phi}_{\text{expert}}^{(i)} - \sum_j \text{softmax}(R(\tau_j)) \cdot \boldsymbol{\phi}(\tau_j)\right]$$

Where the softmax weighting uses **LogSumExp** for numerical stability:

$$\text{weights}_j \propto \exp(R(\tau_j) - \max_k R(\tau_k))$$

Weights are updated via **gradient ascent**:

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \cdot \nabla_\mathbf{w}$$

### Training Loop (Per Epoch)

1. **Set reward weights** in the training environment
2. **Train SAC policy** for `policy_train_steps_per_iter` steps under current reward
3. **Reset SAC entropy coefficient** to `log(0.1)` each epoch (auto-tuning restarts fresh)
4. **For each expert trajectory** in the training set:
   - Run `rollout_samples` stochastic rollouts from the expert's initial state
   - Compute trajectory rewards and feature expectations for each rollout
   - Weight rollouts by softmax of trajectory rewards (MaxEnt weighting)
   - Accumulate gradient: $\phi_\text{expert} - \phi_\text{agent\_weighted}$
5. **Clip gradient** (max norm = 5.0)
6. **Update weights** with linearly decaying learning rate
7. **(Optional) Validate** with deterministic rollouts on held-out trajectories

---

## Policy: SAC (Soft Actor-Critic)

The forward RL policy is **SAC** from the `sbx` library (JAX-accelerated Stable-Baselines3):

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (`sbx.SAC`) |
| Policy network | `MultiInputPolicy` (handles Dict obs) |
| Device | CUDA (JAX) |
| Learning rate | 3e-4 |
| Gamma | 0.99 |
| Batch size | 64 |
| Buffer size | 100,000 |
| Learning starts | 1,000 |
| Entropy coefficient | Auto-tuned (reset per IRL epoch) |
| Replay buffer | `VariableDtReplayBuffer` (custom) |

### Variable-Timestep Replay Buffer

Since a single action can span multiple environment timesteps (`delta_t > 1`), the standard fixed-discount Bellman update is incorrect. The custom `VariableDtReplayBuffer` stores per-transition `delta_t` and computes $\gamma^{\Delta t}$ as the per-transition discount factor, ensuring correct temporal credit assignment.

---

## Training Configuration

### `MaxEntConfig` Defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 10 | Outer IRL loop iterations |
| `reward_lr` | 0.01 | Initial reward weight learning rate |
| `reward_lr_end` | 0.0 | Final LR (linear decay) |
| `rollout_samples` | 20 | Agent rollouts per expert trajectory |
| `policy_train_steps_per_iter` | 5,000 | SAC steps per IRL epoch |
| `policy_train_lr` | 3e-4 | SAC learning rate |
| `policy_gamma` | 0.99 | Discount factor |
| `policy_batch_size` | 64 | SAC batch size |
| `train_ratio` | 0.8 | Train/validation split ratio |
| `segment` | None | Demographic filter (e.g. `"Male 50-59"`) |
| `grad_clip_norm` | 5.0 | Max gradient norm for clipping |
| `validation` | False | Enable validation loop |

### Current Experiment Configs

**Continuous v7 (latest):** Initial weights `[0.7, 0.7, -1, -1, -1]`, `reward_lr=1→0.01`, 5 epochs, 100k SAC steps/epoch, segment `"Male 50-59"`, validation enabled.

**Profit v3 (latest):** Initial weights `[0.5, 0.5, -1, -1, -1, 0.5, 0.5]`, `reward_lr=1→0.1`, 10 epochs, segment `"Male 50-59"`. Uses price-timing interaction features (`charge_timing_quality`, `discharge_timing_quality`) instead of raw cost/revenue.

---

## Expert Data Pipeline

### Raw Data

CSV files in `data/` contain simulated EV usage records with columns for timestep, location, battery state, charging/discharging amounts, journey parameters, energy prices, and demographic segments.

### Preprocessing (`expert_loader_{variant}.py`)

1. Assign `EpisodeID` per trajectory (each starts at timestep 0)
2. Map locations to integers (home=0, work=1, driving_out=2, driving_return=3, towed=4)
3. Compute journey start timesteps and durations
4. Compute time-to-next-journey for each timestep
5. Convert energy levels to SoC fractions
6. Compute per-step features (charged/discharged amounts, target tracking, journey failure)
7. Group consecutive timesteps into actions (action boundary = where previous `Action_Duration_Timesteps == 0`)
8. Compute mean feature expectations per episode
9. Output JSON with `episodeID`, `segment`, `feature_expectation`, `features`, `soc_history`, `initial_values`

### `ExpertDataset` Class

- Loads JSON trajectories into `ExpertTrajectory` dataclasses
- `split_dataset(train_ratio, segment)`: optionally filters by demographic segment, then splits into train/val sets

---

## Evaluation

Evaluation scripts (`scripts/eval_continuous.py`, `scripts/eval_simple.py`) load a trained SAC model and:

1. Set expert initial states in the environment
2. Set the learned reward weights
3. Run deterministic rollouts
4. Compare agent SoC trajectory against expert using **Dynamic Time Warping (DTW)** distance
5. Plot side-by-side SoC curves with journey regions highlighted

---

## Monitoring Metrics

| Metric | Description |
|--------|-------------|
| **L2 Loss** | $\|\phi_\text{expert} - \phi_\text{agent}\|_2$ averaged over training set |
| **DTW Distance** | Dynamic Time Warping between expert and agent SoC trajectories |
| **Reward Weight Evolution** | Per-epoch weight values plotted per feature |
| **Accumulated Profit** | Total £ earned/spent over an episode (profit variant) |

---

## Key Design Decisions

1. **Variable-timestep actions**: Rather than one action per 15-min slot, an action runs until the requested energy transfer is complete. This is more realistic and requires the custom `VariableDtReplayBuffer`.

2. **LogSumExp weighting**: Rollouts are weighted by softmax of their trajectory reward (not uniformly averaged), implementing the MaxEnt importance weighting.

3. **Entropy coefficient reset**: SAC's auto-tuned entropy is reset each IRL epoch to prevent the policy from becoming overly deterministic as reward weights shift.

4. **Gradient clipping**: Max norm of 5.0 prevents reward weight oscillation from noisy gradient estimates.

5. **Linear LR decay**: The reward learning rate decays linearly from `reward_lr` to `reward_lr_end` across epochs for stable convergence.

6. **Boundary penalties**: A soft penalty (0.01 £/kWh) for SoC outside [0.2, 0.8] encourages battery-healthy behaviour without hard constraints.

7. **Demographic segmentation**: The dataset can be filtered by segment (e.g. `"Male 50-59"`) to learn segment-specific reward functions.

---

## File Reference

| Path | Description |
|------|-------------|
| `src/irl/envs/V2GEnv_simple.py` | Simple environment (discrete actions, 4 features) |
| `src/irl/envs/V2GEnv_continuous.py` | Continuous environment (5 features) |
| `src/irl/envs/V2GEnv_profit.py` | Profit-aware environment (7 features, includes energy price) |
| `src/irl/MaxEnt/MaxEnt_continuous.py` | MaxEnt IRL trainer (works with both continuous and profit envs) |
| `src/irl/MaxEnt/MaxEnt_simple.py` | MaxEnt IRL trainer for simple (discrete) variant |
| `src/irl/dataset/expert_dataset.py` | `ExpertDataset` and `ExpertTrajectory` classes |
| `src/irl/dataset/expert_loader_continuous.py` | CSV→JSON preprocessor (continuous) |
| `src/irl/dataset/expert_loader_profit.py` | CSV→JSON preprocessor (profit) |
| `src/irl/dataset/expert_loader_simple.py` | CSV→JSON preprocessor (simple) |
| `src/irl/utils/tools.py` | `AdamOptimizer` (unused), `compute_dtw` |
| `src/irl/utils/variable_dt_buffer.py` | Custom replay buffer for variable-timestep discounting |
| `scripts/train_MaxEnt_continuous.py` | Training entry point (continuous) |
| `scripts/train_MaxEnt_profit.py` | Training entry point (profit) |
| `scripts/train_MaxEnt_simple.py` | Training entry point (simple) |
| `scripts/eval_continuous.py` | Evaluation and plotting |
| `scripts/eval_simple.py` | Evaluation for simple variant |
| `scripts/data_analysis.py` | Dataset exploration and distribution fitting |
| `data/EVDataset*.csv` | Raw expert demonstration datasets |
| `data/processed_trajectories*.json` | Preprocessed trajectory JSONs |
| `models/MaxEntIRL_*/` | Saved model checkpoints, plots, reward weights |

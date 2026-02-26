# V2G-IRL Project Context

## Project Overview
Inverse Reinforcement Learning (IRL) applied to Vehicle-to-Grid (V2G) energy management.
The goal is to recover reward weights from expert EV charging/discharging behaviour using
Maximum Entropy IRL, then use those weights to train an agent that mimics expert behaviour.

---

## Project Structure

```
v2g-IRL/
├── src/
│   └── irl/
│       ├── envs/
│       │   └── V2GEnv_continuous.py       # Gymnasium MDP environment
│       ├── MaxEnt/
│       │   └── MaxEnt_continuous.py       # MaxEnt IRL training loop
│       ├── dataset/
│       │   └── expert_dataset_continuous.py  # Expert trajectory loading/splitting
│       └── utils/
│           └── tools.py                   # AdamOptimizer, compute_dtw, etc.
└── models/
    └── MaxEntIRL_continuous/              # Saved models, plots, weights
```

---

## MDP Environment (`V2GEnv_continuous.py`)

### Scenario
- A single EV completes one return journey per day (home → work → home)
- The day is divided into 96 timesteps (15-minute intervals)
- The agent controls charging/discharging while the car is parked

### Day Stages
| Stage | Value | Description |
|-------|-------|-------------|
| Before work | 0 | Car parked at home, can charge/discharge |
| At work | 1 | Car parked at work, can charge/discharge |
| After work | 2 | Car back at home, can charge/discharge |

### Observation Space (`spaces.Dict`)
| Key | Type | Range | Description |
|-----|------|-------|-------------|
| `timestep` | int64 | 0–96 | Current 15-min interval |
| `soc` | float32 | 0.0–1.0 | Current battery State of Charge |
| `soc_target` | float32 | 0.0–1.0 | SoC needed for next journey |
| `battery_capacity` | int64 | 0–2 | Index → {40, 60, 80} kWh |
| `time_to_next_journey` | int64 | 0–96 | Timesteps until next departure |
| `current_charger_power` | float32 | 0–22 kW | Power available at current location |

### Action Space
- `spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)`
- `a > 0`: charge | `a < 0`: discharge | `a = 0`: idle
- Actions rounded to 2 decimal places
- Penalty applied for charging above 0.8 SoC or discharging below 0.2 SoC

### Fixed States Per Episode (randomised or provided)
| Variable | Distribution | Range |
|----------|-------------|-------|
| `soc` (initial) | Normal(0.4944, 0.1481) | 0.0–1.0 |
| `battery_capacity` | Uniform choice | {0, 1, 2} |
| `home_charge_power` | Uniform choice | {0, 1, 2} → {3, 7.4, 11} kW |
| `work_charge_power` | Uniform choice | {0, 1, 2} → {7.4, 11, 22} kW |
| `journey_distance` | Exponential(10.1331) + 10 | 10–80 miles |
| `out_journey_speed` | Gamma(8.5518, 3.4713) | 10–75 mph |
| `return_journey_speed` | Gamma(8.5518, 3.4713) | 10–75 mph |
| `out_start_timestep` | Gamma(11.4837, 2.61428) | 0–80 |
| `return_start_timestep` | Gamma(24.6641, 1.8946) + 20 | out_end–90 |

### Energy Model
- Assumed efficiency: 3 miles/kWh → `kwh_per_mile = 1/3`
- `energy_for_out = (journey_distance × kwh_per_mile) / battery_capacity_kWh`
- `energy_for_return = energy_for_out` (same distance)
- Max energy transfer per timestep: `(charger_power × 0.25) / battery_capacity_kWh`

### Feature Vector φ (6 elements)
| Index | Name | Formula |
|-------|------|---------|
| 0 | `amount_charged` | SoC increase this action (≥ 0) |
| 1 | `amount_discharged` | SoC decrease this action (≥ 0) |
| 2 | `battery_needed_target` | `max(0, soc_target - soc)²` |
| 3 | `battery_exceeded_target` | `max(0, soc - soc_target)²` |
| 4 | `journey_failure` | 1 if SoC insufficient for journey |
| 5 | `charge_action_taken` | 1 if any charge/discharge occurred |

### Reward Function
```
R = w · φ(s, a) − (10.0 if bad_discharge else 0.0)
```
- Linear in features: reward weights are learned via IRL
- Default weights: `[0.1, 0.1, -1.0, -1.0, -10.0, 1.0]`
- `bad_discharge`: charging above 0.8 SoC or discharging below 0.2 SoC

### Episode Termination
- `terminated = True` when `timestep >= 95` in stage 2 (after work)
- `truncated = False` (no truncation implemented)

---

## MaxEnt IRL (`MaxEnt_continuous.py`)

### Algorithm: Maximum Entropy IRL
Recovers reward weights `w` such that the agent's feature expectations match the expert's:

```
E_π[φ] = E_expert[φ]
```

Gradient update per epoch:
```
∇w = (1/N) Σ_i [ φ_expert_i − Σ_j softmax(R(τ_j)) · φ(τ_j) ]
w  ← w + lr × ∇w
```

LogSumExp weighting (numerically stable MaxEnt):
```python
weights ∝ exp(R(τ) − max(R))
```

### Training Loop
1. Set reward weights in training env
2. Train SAC policy for `policy_train_steps_per_iter` steps
3. For each expert trajectory:
   - Perform `rollout_samples` rollouts from expert initial state
   - Weight rollouts by softmax of trajectory reward
   - Compute gradient as difference in feature expectations
4. Update `w` via gradient ascent

### Configuration (`MaxEntConfig`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `device` | `'cuda'` | Training device |
| `n_epochs` | 10 | IRL outer loop iterations |
| `reward_lr` | 0.01 | Reward weight learning rate |
| `rollout_samples` | 20 | Agent rollouts per expert trajectory |
| `policy_train_steps_per_iter` | 5,000 | SAC steps per IRL epoch |
| `policy_train_lr` | 3e-4 | SAC learning rate |
| `policy_gamma` | 0.99 | Discount factor |
| `policy_batch_size` | 64 | SAC batch size |
| `train_ratio` | 0.8 | Train/val split |
| `segment` | None | Optional dataset segment filter |
| `folder_name` | `'MaxEntIRL_continuous'` | Output directory |
| `validation` | False | Enable validation loop |

### SAC Policy Setup
- Library: `sbx` (JAX-based SB3 equivalent)
- Policy: `MultiInputPolicy` (handles `Dict` observation space)
- Training env: `DummyVecEnv` + `VecNormalize` (reward normalisation only)
- Rollout env: Separate unnormalized `DummyVecEnv`
- Replay buffer: 100,000 transitions
- Entropy coefficient: 0.1 (fixed)

### Monitoring & Outputs
| Output | Description |
|--------|-------------|
| `train_l2_loss` | L2 norm between expert and agent feature expectations |
| `train_dtw_distance` | DTW distance between expert and agent SoC trajectories |
| `val_l2_loss` | Validation L2 loss (if enabled) |
| `reward_weights_history` | Weight vector per epoch |
| `final_reward_weights.txt` | Saved learned weights |
| `maxent_irl_epoch{n}.zip` | SAC model checkpoint per epoch |

---

## Key Conventions
- Timesteps are 15-minute intervals; a full day = 96 timesteps (0–95)
- SoC is always normalised to [0.0, 1.0]
- Feature expectations are **summed** (not averaged) over an episode
- Journey energy deduction happens at stage transition, not during travel
- The `soc_history` list length may vary per episode due to journey skipping
- `set_initial_states(None)` triggers random scenario generation in `reset()`
- `set_initial_states(dict)` pins the episode to a specific expert scenario

---

## Known Incomplete Areas (as of 2026-02-18)
- Charging/discharging SoC update logic inside `step()` while-loop is incomplete
- Journey failure feature update (`features[4]`) not yet assigned in `step()`
- SoC history update during journey transitions not yet implemented
- Adam optimizer for reward weights is commented out (using plain gradient ascent)
- `energy_price` observation is commented out (price profile exists but unused in obs)
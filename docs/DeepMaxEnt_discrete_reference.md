# Deep MaxEnt IRL — Discrete Variant: Technical Reference

**Files covered:**
- `src/irl/DeepMaxEnt/DeepMaxEnt.py` — `RewardNet`, shared constants
- `src/irl/DeepMaxEnt/DeepMaxEnt_discrete.py` — config, trainer, data loader
- `src/irl/envs/V2GDeepEnv_discrete.py` — Gymnasium environment
- `src/irl/dataset/expert_loader_deep_discrete.py` — CSV → JSON preprocessor
- `scripts/train_DeepMaxEnt_discrete.py` — training entry point

---

## 1. Overview

This variant applies **Deep Maximum Entropy Inverse Reinforcement Learning** to recover a *neural* reward function from expert EV charging demonstrations. It differs from the linear MaxEnt variants in one critical way:

| Aspect | Linear MaxEnt | Deep MaxEnt Discrete |
|--------|--------------|----------------------|
| Reward function | $R = \mathbf{w} \cdot \boldsymbol{\phi}(s,a)$ | $R = R_\theta(s, a)$ (neural net) |
| Reward parameters | Weight vector **w** | Network weights **θ** |
| Inner-loop policy | SAC (off-policy, continuous) | PPO (on-policy, discrete) |
| Action space | Continuous `[-1, 1]` | Discrete 0–20 |
| Obs input to policy | Dict (MultiInputPolicy) | Flat normalized vector (MlpPolicy) |

Both wrap the same underlying physical scenario: one EV completing a home→work→home return journey across 96 timesteps.

---

## 2. Architecture Diagram

```
EVDataset_discrete.csv
        │
        ▼
expert_loader_deep_discrete.py
        │  Extracts per-action (obs, action, delta_t) pairs + feature expectations
        ▼
processed_trajectories_deep_discrete.json
        │
        ▼
load_deep_discrete_expert_data()  ─── train/val split ─── DeepDiscreteExpertTrajectory[]
        │
        ▼
DeepMaxEntDiscreteTrainer.train()
    │
    ├─── [Outer IRL loop, each epoch]
    │       │
    │       ├── 1. PPO policy training
    │       │       env: V2GDeepEnv_discrete (wrapped: flatten+normalize → MlpPolicy)
    │       │       reward: R_θ(s_normalized, a_normalized) × delta_t × reward_scale
    │       │
    │       ├── 2. IRL gradient update (for each expert trajectory)
    │       │       - Score expert traj: Σ R_θ(s_e, a_e) × delta_t
    │       │       - Collect K stochastic rollouts from same initial state
    │       │       - Score rollouts: {r_k = Σ R_θ(s_k, a_k) × delta_t}
    │       │       - Softmax importance weights: w_k = softmax({r_k})
    │       │       - Loss: -(R_expert - Σ w_k × r_k) / N
    │       │       - Backprop into θ
    │       │
    │       └── 3. AdamW step + CosineAnnealing LR scheduler
    │
    └─── Saved: reward_net_epochN.pt, ppo_epochN.zip, metrics.csv, plots
```

---

## 3. Expert Data Pipeline

### 3.1 Input

`data/EVDataset_discrete.csv` — raw EV simulation records with one row per 15-minute timestep.

### 3.2 Preprocessing (`expert_loader_deep_discrete.py`)

Key steps:

1. **Episode segmentation** — each trajectory starts at `Timestep == 0`.
2. **Journey timestamps** — `out_start_timestep`, `return_start_timestep`, durations.
3. **Time-to-next-journey** — computed per timestep based on stage.
4. **SoC conversion** — `SoC = Battery_Energy_Level_kWh / Battery_Capacity_kWh`.
5. **Action grouping** — consecutive timesteps with `Action_Duration_Timesteps > 0` belong to the same action. The action boundary is where the previous row has `Action_Duration_Timesteps == 0` (i.e. the action just completed). This mirrors the env's variable-timestep execution.
6. **Action encoding** — the expert's net charge/discharge per action is converted to a discrete index 0–20 (10 = idle), then **immediately normalized to `[-1, 1]`** as `(discrete - 10) / 10`. This matches `action_for_reward` in the env step.
7. **delta_t** stored per action — the number of 15-min timesteps the action consumed.
8. **Feature expectations** — 7 features computed per action for monitoring/comparison (not used in the neural reward gradient).

### 3.3 Output JSON Schema

```json
{
  "episodeID": 42,
  "segment": "Male 50-59",
  "initial_values": { "soc": 0.5, "battery_capacity": 1, ... },
  "soc_history": [0.5, 0.51, ...],           // 96 timestep SoC values
  "feature_expectation": [f0, f1, ..., f6],  // 7 monitoring features (summed)
  "state_action_pairs": {
    "observations": [[t, soc, soc_t, price, cap, t2j, charger_kW], ...],  // unnormalized
    "actions": [[a_normalized], ...],   // already in [-1, 1]
    "delta_ts": [dt1, dt2, ...]         // int, timesteps per action
  }
}
```

---

## 4. Environment: `V2GDeepEnv` (V2GDeepEnv_discrete.py)

Identical physics to `V2GEnv_profit`, but with:

### 4.1 Action Space

```python
spaces.Discrete(21)   # 0 = max discharge, 10 = idle, 20 = max charge
```

Internally scaled: `action_continuous = action * 0.1 - 1.0` → `[-1.0, 1.0]`.

For the **reward network**, the action is separately normalized as:
```python
action_for_reward = (float(action) - 10.0) / 10.0  # matches expert encoding
```

### 4.2 Observation Space

Dict with 7 keys (same as profit variant):

| Key | Range | Norm scale |
|-----|-------|-----------|
| `timestep` | 0–96 | 96.0 |
| `soc` | 0–1 | 1.0 |
| `soc_target` | 0–1 | 1.0 |
| `energy_price` | 0–0.47 | 0.47 |
| `battery_capacity` | 0–2 | 2.0 |
| `time_to_next_journey` | 0–96 | 96.0 |
| `current_charger_power` | 0–22 | 22.0 |

### 4.3 Reward Computation

```python
reward = R_θ(obs_normalized, action_for_reward) × delta_t × reward_scale
       - boundary_penalty_per_kwh × boundary_violation × battery_cap × delta_t
       - action_penalty_coeff × action_for_reward² × delta_t
```

**Scaling by `delta_t`** is critical: it ensures PPO's cumulative episode reward is proportional to the time-integral of `R(s,a)`, not the number of discrete action events. Without it, PPO would be incentivized to take many tiny actions rather than large expert-like charge/discharge steps.

### 4.4 Variable-Timestep Execution

A single action can span multiple 15-min slots. The env advances timestep-by-timestep within a `while` loop until the requested SoC transfer is completed or a stage deadline is reached. The `delta_t` returned in `info` records how many slots were consumed.

### 4.5 Obs Wrapper for PPO

`FlattenNormalizeObsWrapper` (in `DeepMaxEnt_discrete.py`) converts the Dict obs to a normalized flat `(7,)` float32 vector, required because SBX's PPO only supports `MlpPolicy` (no `MultiInputPolicy`). The normalization uses `PROFIT_OBS_SCALES` — the same scales used by the reward network.

---

## 5. Neural Reward Network (`RewardNet`)

Defined in `src/irl/DeepMaxEnt/DeepMaxEnt.py`.

### Architecture

```
Input: [obs_normalized (7,), action_normalized (1,)]  →  (8,)
Linear(8 → hidden_dim)  →  ReLU
Linear(hidden_dim → hidden_dim)  →  ReLU
Linear(hidden_dim → 1)           →  scalar (unbounded)
```

- **Unbounded output** — provides PPO with a strong learning signal; no `tanh` clamping.
- **Default `hidden_dim = 64`** (configurable; experiments use 32).
- **L2 regularization** via `AdamW(weight_decay=reward_l2_reg)` prevents reward divergence.

### Normalization Contract

The reward net **always** receives:
- `obs`: divided by `PROFIT_OBS_SCALES = [96, 1, 1, 0.47, 2, 96, 22]`
- `action`: in `[-1, 1]`

This contract is enforced at three call sites:
1. `V2GDeepEnv_discrete.step()` — live env reward
2. `DeepMaxEntDiscreteTrainer` during IRL gradient — expert and rollout scoring
3. The obs wrapper — PPO policy training

---

## 6. Training Algorithm (`DeepMaxEntDiscreteTrainer`)

### 6.1 Outer Loop (per epoch)

```
for epoch in range(n_epochs):
    1. [Optional] Reset PPO model (reset_ppo_each_epoch)
    2. Set reward_net in envs
    3. PPO.learn(policy_train_steps_per_iter)
    4. IRL gradient update (see §6.2)
    5. AdamW.step() + scheduler.step()
    6. [Optional] Validation rollouts
    7. Save PPO + reward_net checkpoints, append metrics.csv
```

### 6.2 IRL Gradient Update (per expert trajectory)

For each trajectory $i$ in the training set:

**Step 1 — Score the expert:**
$$R_\text{expert}^{(i)} = \sum_t R_\theta(s_t^e, a_t^e) \cdot \Delta t_t$$

where $(s_t^e, a_t^e)$ are normalized obs/action pairs and $\Delta t_t$ is the action duration.

**Step 2 — Collect K stochastic rollouts** from the expert's initial state using the current PPO policy:
$$\{\tau_k\}_{k=1}^{K}, \quad r_k = \sum_t R_\theta(s_t^k, a_t^k) \cdot \Delta t_t$$

Note: rollout actions are raw discrete integers from `model.predict()`, then normalized `(a - 10) / 10` before scoring.

**Step 3 — Importance weights (MaxEnt):**
$$w_k = \text{softmax}(\{r_k\}) = \frac{\exp(r_k)}{\sum_j \exp(r_j)}$$

**Step 4 — Weighted agent reward:**
$$R_\text{agent}^{(i)} = \sum_k w_k \cdot r_k$$

**Step 5 — Loss (negative log-likelihood):**
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left(R_\text{expert}^{(i)} - R_\text{agent}^{(i)}\right)$$

At convergence, $R_\text{expert} \approx R_\text{agent}$, so $\mathcal{L} \approx 0$.

**Step 6 — Gradient clip + AdamW update:**
```python
torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm=reward_grad_clip)
reward_optimizer.step()
reward_scheduler.step()
```

### 6.3 Log-likelihood Monitoring

A separate diagnostic quantity is computed:
$$\log p(\tau_e) \approx R_\text{expert} - \log\!\left(\frac{1}{K}\sum_k \exp(r_k)\right)$$

This is a Monte Carlo estimate of $R(\tau_e) - \log Z$, the log-likelihood under the MaxEnt distribution. It approaches 0 at convergence and is tracked in `metrics.csv`.

---

## 7. Optimizers and Schedulers

| Component | Type | Configuration |
|-----------|------|--------------|
| Reward net optimizer | `AdamW` | `lr=reward_lr`, `weight_decay=reward_l2_reg` |
| Reward LR scheduler | `CosineAnnealingLR` | `T_max=n_epochs`, `eta_min=reward_lr_end` |
| PPO policy | SBX `PPO("MlpPolicy")` | `lr=policy_train_lr`, `gamma=policy_gamma`, `ent_coef=policy_ent_coef` |

The **cosine annealing** schedule replaces the linear decay used in the linear MaxEnt variants, providing a smooth warm decay with a low-LR tail.

`reset_ppo_each_epoch=True` discards the PPO model and optimizer state at the start of each epoch. This avoids Adam momentum accumulated under the old reward landscape interfering with optimization under the updated reward. Set to `False` to allow the PPO policy to "continue learning" across epochs (used in some experiments).

---

## 8. Configuration (`DeepMaxEntDiscreteConfig`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 10 | Outer IRL iterations |
| `reward_lr` | 1e-3 | Initial reward net LR (AdamW) |
| `reward_lr_end` | 1e-4 | Final LR (cosine annealing) |
| `rollout_samples` | 10 | Agent rollouts per expert trajectory per epoch |
| `policy_train_steps_per_iter` | 100,000 | PPO env steps per IRL epoch |
| `policy_train_lr` | 3e-4 | PPO actor/critic LR |
| `policy_gamma` | 1.0 | PPO discount factor |
| `policy_batch_size` | 64 | PPO minibatch size |
| `policy_n_steps` | 2048 | PPO rollout buffer length |
| `policy_n_epochs` | 10 | PPO gradient steps per rollout |
| `policy_ent_coef` | 0.01 | PPO entropy bonus coefficient |
| `reward_hidden_dim` | 64 | Reward net hidden layer width |
| `reward_obs_dim` | 7 | Reward net input obs dimension |
| `reward_action_dim` | 1 | Reward net input action dimension |
| `reward_grad_clip` | 5.0 | Max gradient norm for reward net |
| `reward_l2_reg` | 0.01 | AdamW weight decay |
| `reward_scale` | 1.0 | Multiplier on reward before PPO sees it (does not affect IRL gradient) |
| `action_penalty_coeff` | 0.0 | Coefficient for $-\lambda a^2$ action magnitude penalty |
| `reset_ppo_each_epoch` | True | Recreate PPO model each epoch |
| `pretrained_reward_net_path` | None | Warm-start from `.pt` checkpoint |
| `train_ratio` | 0.8 | Train/val split |
| `segment` | None | Demographic filter (e.g. `"Male 50-59"`) |
| `validation` | False | Run deterministic val rollouts each epoch |
| `folder_name` | "DeepMaxEntIRL_discrete" | Output directory under `models/` |

---

## 9. Monitoring Features (7 features)

These are computed by both the env and the expert loader **for monitoring only** — they are not used as reward inputs, unlike the linear MaxEnt variants.

| Index | Feature | Formula |
|-------|---------|---------|
| 0 | `amount_charged` | SoC increase this action |
| 1 | `amount_discharged` | SoC decrease this action |
| 2 | `charge_cost_penalty` | $10 \cdot \text{charged}^2 \cdot (\bar{p}_a / p_\max)$ |
| 3 | `discharge_cost_penalty` | $10 \cdot \text{discharged}^2 \cdot ((p_\max - \bar{p}_a) / p_\max)$ |
| 4 | `battery_needed_target` | $\max(0, \text{soc\_target} - \text{soc})^2 \cdot \Delta t$ |
| 5 | `battery_exceeded_target` | $\max(0, \text{soc} - \text{soc\_target})^2 \cdot \Delta t$ |
| 6 | `journey_failure` | 1 if SoC insufficient for journey |

Feature L2 distance $\|\phi_\text{expert} - \phi_\text{agent}\|_2$ is logged each epoch to track imitation quality independently of reward loss.

---

## 10. Key Design Decisions

### 10.1 Discrete Action Space with Neural Reward

Previous linear MaxEnt variants used a continuous action space with a linear reward. This variant switches to:
- **Discrete actions** (21 levels, step 0.1 SoC units): enables PPO (on-policy); cleaner gradient signal for each action choice.
- **Neural reward**: drops the hand-crafted feature engineering assumption — the net learns what features matter implicitly.

### 10.2 Action Normalization Consistency

Expert actions and rollout actions are always normalized identically before entering the reward net:
- Expert loader stores `(discrete - 10) / 10`
- Env computes `action_for_reward = (float(action) - 10.0) / 10.0`
- Trainer normalizes rollout actions: `(act_t - 10.0) / 10.0`

Breaking this consistency would cause mismatched scoring between expert and agent.

### 10.3 `delta_t` Weighting

All reward sums — expert scoring, rollout scoring, and env reward to PPO — are weighted by `delta_t`. This makes the IRL loss consistent with the PPO objective: both use the time-integral of rewards rather than the mean over discrete action events.

### 10.4 On-Policy PPO (no replay buffer needed)

Because the action space is discrete and PPO is on-policy, there is no need for the `VariableDtReplayBuffer` used in the SAC variants. PPO collects a fresh `n_steps=2048` rollout buffer each update, which naturally avoids stale transitions under the old reward function.

### 10.5 AdamW + L2 Regularization

`AdamW` with `weight_decay=reward_l2_reg` regularizes the reward network to prevent it from assigning arbitrarily large magnitudes to out-of-distribution (s, a) pairs. Without this, the reward net can diverge in regions the rollout policy never visits.

### 10.6 Gradient Clipping on Reward Net

`clip_grad_norm_(max_norm=5.0)` on the reward net prevents instability when the expert and agent rollout rewards are far apart in early training.

### 10.7 Warm-Start Support

`pretrained_reward_net_path` allows loading a network from a prior experiment, useful when extending training or fine-tuning on a different demographic segment without training from scratch.

---

## 11. Saved Outputs

All outputs go to `./models/{folder_name}/`:

| File | Contents |
|------|---------|
| `reward_net_epochN.pt` | `state_dict` of reward net after epoch N |
| `ppo_epochN.zip` | SBX PPO model checkpoint |
| `metrics.csv` | Per-epoch: DTW, feat L2, reward loss, log-likelihood, LR |
| `monitor.csv` | SBX Monitor: per-episode reward during PPO training |
| `dtw_distance.png` | Train (+ val) DTW over epochs |
| `feature_l2.png` | Train (+ val) feature L2 over epochs |
| `reward_loss.png` | Reward loss (neg log-likelihood) over epochs |
| `log_likelihood.png` | Expert log-likelihood over epochs (target: → 0) |
| `about.md` | Free-text description field from `cfg.description` |
| `tensorboard/` | PPO TensorBoard logs per epoch |

# v2g-IRL

# Installation

To install this repository `uv` is highly recommended. Install it via following the instructions on the [uv website](https://docs.astral.sh/uv/getting-started/installation/) or running the following command with `pip` installed:

```bash
pip install uv
```

After installing `uv`, install the repository by running the following command in the project root:

```bash
uv sync
```

CUDA GPU is highly recommended as CUDA-enabled PyTorch is installed by default. For CPU-only PyTorch, adjust `pyproject.toml` accordingly.

---

# Project Structure

```
data/               Raw CSV datasets and processed trajectory JSON files
demo/               Interactive Streamlit demo application
models/             Saved model checkpoints (created during training)
scripts/
    load/           Data loading scripts
    train/          Training scripts for all IRL methods
    evaluate/       Evaluation scripts
    analysis/       Plotting and analysis scripts
src/irl/
    envs/           Gymnasium environment definitions
    dataset/        Expert data loaders and trajectory processors
    MaxEnt/         Linear Maximum Entropy IRL implementations
    DeepMaxEnt/     Deep Maximum Entropy IRL implementations
    Adversarial/    Adversarial IRL (AIRL) implementations
    utils/          Shared utilities
```

---

# Loading Data

Before training, expert trajectories must be extracted from the raw CSV datasets and saved as JSON files. Edit the `args` dict in `scripts/load/load_data.py` to select the loader and files, then run:

```bash
uv run scripts/load/load_data.py
```

Available `expert_loader` values and their corresponding datasets:

| `expert_loader`              | Input CSV                              | Output JSON                                              |
|-----------------------------|----------------------------------------|----------------------------------------------------------|
| `linear_simple`             | `EVDataset_simple.csv`                 | `processed_trajectories_simple.json`                     |
| `linear_discrete`           | `EVDataset_discrete.csv`               | `processed_trajectories_discrete_pricediff.json`         |
| `linear_continuous`         | `EVDataset_continuous.csv`             | `processed_trajectories_continuous.json`                 |
| `deep_discrete`             | `EVDataset_discrete_special.csv`       | `processed_trajectories_deep_discrete_special.json`      |
| `deep_continuous`           | `EVDataset_continuous.csv`             | `processed_trajectories_continuous.json`                 |
| `airl_discrete`             | `EVDataset_discrete.csv`               | `processed_trajectories_airl_discrete_pricediff.json`    |
| `airl_continuous`           | `EVDataset_continuous.csv`             | `processed_trajectories_airl_continuous.json`            |

Pre-processed JSON files are already included in `data/` for all standard configurations.

---

# Usage

## Linear Maximum Entropy IRL

**Related files:**
- `src/irl/envs/V2GEnv_simple.py` — Simple V2G environment
- `src/irl/envs/V2GEnv_discrete.py` — Discrete V2G environment
- `src/irl/envs/V2GEnv_continuous.py` — Continuous V2G environment
- `src/irl/dataset/expert_loader_simple.py` — Simple expert data loader
- `src/irl/dataset/expert_loader_discrete.py` — Discrete expert data loader
- `src/irl/dataset/expert_loader_continuous.py` — Continuous expert data loader
- `src/irl/MaxEnt/MaxEnt_simple.py` — Simple MaxEnt IRL
- `src/irl/MaxEnt/MaxEnt_discrete.py` — Discrete MaxEnt IRL
- `src/irl/MaxEnt/MaxEnt_continuous.py` — Continuous MaxEnt IRL

**Train:**
```bash
uv run scripts/train/train_MaxEnt_simple.py
uv run scripts/train/train_MaxEnt_discrete.py
uv run scripts/train/train_MaxEnt_continuous.py
```

**Evaluate:**
```bash
uv run scripts/evaluate/eval_simple.py
uv run scripts/evaluate/eval_discrete.py
uv run scripts/evaluate/eval_continuous.py
```

---

## Deep Maximum Entropy IRL

**Related files:**
- `src/irl/envs/V2GDeepEnv_discrete.py` — Deep discrete V2G environment
- `src/irl/envs/V2GDeepEnv_continuous.py` — Deep continuous V2G environment
- `src/irl/dataset/expert_loader_deep_discrete.py` — Deep discrete expert data loader
- `src/irl/dataset/expert_loader_deep_continuous.py` — Deep continuous expert data loader
- `src/irl/DeepMaxEnt/DeepMaxEnt_discrete.py` — Deep MaxEnt IRL (discrete)
- `src/irl/DeepMaxEnt/DeepMaxEnt_continuous.py` — Deep MaxEnt IRL (continuous)

**Train:**
```bash
uv run scripts/train/train_DeepMaxEnt_discrete.py
uv run scripts/train/train_DeepMaxEnt_continuous.py
```

**Evaluate:**
```bash
uv run scripts/evaluate/eval_deep_discrete.py
uv run scripts/evaluate/eval_deep_continuous.py
```

---

## Adversarial IRL

**Related files:**
- `src/irl/envs/V2GDeepEnv_discrete.py` — Discrete V2G environment (shared with DeepMaxEnt)
- `src/irl/envs/V2GDeepEnv_continuous.py` — Continuous V2G environment (shared with DeepMaxEnt)
- `src/irl/dataset/expert_loader_airl_discrete.py` — AIRL discrete expert data loader
- `src/irl/dataset/expert_loader_airl_continuous.py` — AIRL continuous expert data loader
- `src/irl/Adversarial/Adversarial_discrete.py` — AIRL trainer (discrete)
- `src/irl/Adversarial/Adversarial_continuous.py` — AIRL trainer (continuous)
- `src/irl/Adversarial/Adversarial.py` — Shared base classes and reward/shaping networks

**Train:**
```bash
uv run scripts/train/train_Adversarial_discrete.py
uv run scripts/train/train_Adversarial_continuous.py
```

**Evaluate:**
```bash
uv run scripts/evaluate/eval_adversarial_discrete.py
uv run scripts/evaluate/eval_adversarial_continuous.py
```

---

# Demo

The interactive demo runs as a Streamlit web application. Launch it from the project root:

```bash
uv run streamlit run demo/app.py
```

The demo allows you to visualise and compare learned policies across different IRL methods and EV driver segments interactively.


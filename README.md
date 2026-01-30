# v2g-IRL

# Installation

To install this repository `uv` is highly recommended. Install it via following the instructions on the [uv website](https://docs.astral.sh/uv/getting-started/installation/) or running the following command with `pip` installed:

```bash
pip install uv
```

After installing `uv`, you can install this repository by running the following command in your terminal:

```bash
uv sync
```
CUDA GPU is highly recommended as cuda enabled PyTorch is installed. For CPU only PyTorch, adjust `pyproject.toml` accordingly.

# Usage

## Maximum Entropy IRL

The training script can be run using 
```bash
uv run .\scripts\MaxEntIRL_simple.py 
```

The related files are as follows:
- `src/irl/env/V2GEnv_simple.py`: The simple V2G environment definition.
- `src/irl/dataset/expert_dataset_simple.py`: The expert dataset loader for the simple V2G environment.
- `src/irl/dataset/expert_loader_simple.py`: The expert data extraction script for the simple V2G environment.
- `src/irl/MaxEnt/MaxEnt_simple.py`: The Maximum Entropy IRL implementation for the simple V2G environment.


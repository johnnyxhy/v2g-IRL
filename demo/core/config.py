"""
Central configuration for the V2G-IRL demo.

Each scenario defines:
  - env_id / entry_point: gymnasium registration details
  - canonical_json: the expert-trajectory JSON used for the
    "Expert Comparison" episode picker
  - methods: dict keyed by short name (maxent / deepmaxent / airl)
      - label          : display name
      - available      : bool – False ⟹ greyed-out in the UI
      - model_folder   : path relative to project root's models/
      - algo           : 'PPO' | 'SAC'
      - model_prefix   : filename stem before the epoch number
      - last_epoch     : last trained epoch (used as default)
      - json_path      : expert JSON for this specific method
      - wrapper        : 'flatten_simple' | 'flatten_discrete' |
                         'flatten_deepmaxent_discrete' | 'flatten_airl_discrete' |
                         'flatten_airl_continuous' | None
      - reward_net     : bool – whether to load a reward .pt file
      - shaping_net    : bool – whether to load a shaping .pt file
      - reward_hidden  : hidden dim for reward net (default 64)
      - shaping_hidden : hidden dim for shaping net
"""

import os

# ── Project root (two levels up from this file) ────────────────────────────
ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

SCENARIOS = {
    "Simple (Probabilistic)": {
        "env_id": "V2GEnv-simple",
        "entry_point": "irl.envs.V2GEnv_simple:V2GEnv",
        "canonical_json": os.path.join(ROOT, "data", "processed_trajectories_simple_probabilistic.json"),
        "description": "Discrete charge / discharge / idle actions with probabilistic expert.",
        "methods": {
            "maxent": {
                "label": "MaxEnt (Linear)",
                "available": True,
                "model_folder": os.path.join(ROOT, "models", "MaxEnt", "simple", "MaxEntIRL_simple_prob_male5059"),
                "algo": "PPO",
                "model_prefix": "ppo_epoch",
                "last_epoch": 20,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_simple_probabilistic.json"),
                "wrapper": "flatten_simple",
                "reward_net": False,
                "shaping_net": False,
            },
            "deepmaxent": {
                "label": "Deep MaxEnt",
                "available": False,
                "model_folder": None,
                "algo": None,
                "model_prefix": None,
                "last_epoch": None,
                "json_path": None,
                "wrapper": None,
                "reward_net": False,
                "shaping_net": False,
            },
            "airl": {
                "label": "AIRL",
                "available": False,
                "model_folder": None,
                "algo": None,
                "model_prefix": None,
                "last_epoch": None,
                "json_path": None,
                "wrapper": None,
                "reward_net": False,
                "shaping_net": False,
            },
        },
    },

    "Continuous (Profit)": {
        "env_id_maxent": "V2GEnv-continuous",
        "entry_point_maxent": "irl.envs.V2GEnv_continuous:V2GEnv",
        "env_id_deep": "V2GDeepEnv-continuous",
        "entry_point_deep": "irl.envs.V2GDeepEnv_continuous:V2GDeepEnv",
        "canonical_json": os.path.join(ROOT, "data", "processed_trajectories_profit.json"),
        "description": "Continuous charging power with profit-maximising reward.",
        "methods": {
            "maxent": {
                "label": "MaxEnt (Linear)",
                "available": True,
                "model_folder": os.path.join(ROOT, "models", "MaxEnt", "continuous", "MaxEntIRL_continuous_male5059"),
                "algo": "SAC",
                "model_prefix": "maxent_irl_epoch",
                "last_epoch": 30,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_profit.json"),
                "env_id": "V2GEnv-continuous",
                "entry_point": "irl.envs.V2GEnv_continuous:V2GEnv",
                "wrapper": None,
                "reward_net": False,
                "shaping_net": False,
            },
            "deepmaxent": {
                "label": "Deep MaxEnt",
                "available": True,
                "model_folder": os.path.join(ROOT, "models", "DeepMaxEnt", "continuous", "DeepMaxEntIRL_continuous_male5059"),
                "algo": "SAC",
                "model_prefix": "sac_epoch",
                "last_epoch": 30,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_deep_profit.json"),
                "env_id": "V2GDeepEnv-continuous",
                "entry_point": "irl.envs.V2GDeepEnv_continuous:V2GDeepEnv",
                "wrapper": None,
                "reward_net": True,
                "reward_hidden": 32,
                "shaping_net": False,
            },
            "airl": {
                "label": "AIRL",
                "available": True,
                "model_folder": os.path.join(ROOT, "models", "Adversarial", "continuous", "AIRL_continuous_male5059"),
                "algo": "SAC",
                "model_prefix": "sac_epoch",
                "last_epoch": 30,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_airl_continuous.json"),
                "env_id": "V2GDeepEnv-continuous",
                "entry_point": "irl.envs.V2GDeepEnv_continuous:V2GDeepEnv",
                "wrapper": "flatten_airl_continuous",
                "reward_net": True,
                "reward_hidden": 32,
                "shaping_net": True,
                "shaping_hidden": 32,
            },
        },
    },

    "Discrete (Profit)": {
        "env_id_maxent": "V2GEnv-discrete",
        "entry_point_maxent": "irl.envs.V2GEnv_discrete:V2GEnv",
        "env_id_deep": "V2GDeepEnv-discrete",
        "entry_point_deep": "irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
        "canonical_json": os.path.join(ROOT, "data", "processed_trajectories_discrete_pricediff.json"),
        "description": "Discrete charging levels with price-differential and profit features.",
        # Segments with trained models for this scenario
        "available_segments": ["Male 50-59", "Male 40-49", "Female 50-59"],
        "methods": {
            "maxent": {
                "label": "MaxEnt (Linear)",
                "available": True,
                "algo": "PPO",
                "model_prefix": "ppo_epoch",
                "last_epoch": 20,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_discrete_pricediff.json"),
                "env_id": "V2GEnv-discrete",
                "entry_point": "irl.envs.V2GEnv_discrete:V2GEnv",
                "wrapper": "flatten_discrete",
                "reward_net": False,
                "shaping_net": False,
                # Per-segment model folders
                "segments": {
                    "Male 50-59":   os.path.join(ROOT, "models", "MaxEnt", "discrete", "MaxEntIRL_discrete_pricediff_male5059"),
                    "Male 40-49":   os.path.join(ROOT, "models", "MaxEnt", "discrete", "MaxEntIRL_discrete_pricediff_male4049"),
                    "Female 50-59": os.path.join(ROOT, "models", "MaxEnt", "discrete", "MaxEntIRL_discrete_pricediff_female5059"),
                },
            },
            "deepmaxent": {
                "label": "Deep MaxEnt",
                "available": True,
                "algo": "PPO",
                "model_prefix": "ppo_epoch",
                "last_epoch": 20,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_deep_discrete_gap_pricediff.json"),
                "env_id": "V2GDeepEnv-discrete",
                "entry_point": "irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
                "wrapper": "flatten_deepmaxent_discrete",
                "reward_net": True,
                "reward_hidden": 32,
                "shaping_net": False,
                "segments": {
                    "Male 50-59":   os.path.join(ROOT, "models", "DeepMaxEnt", "discrete", "DeepMaxEntIRL_discrete_male5059"),
                    "Male 40-49":   os.path.join(ROOT, "models", "DeepMaxEnt", "discrete", "DeepMaxEntIRL_discrete_male4049"),
                    "Female 50-59": os.path.join(ROOT, "models", "DeepMaxEnt", "discrete", "DeepMaxEntIRL_discrete_female5059"),
                },
            },
            "airl": {
                "label": "AIRL",
                "available": True,
                "algo": "PPO",
                "model_prefix": "ppo_epoch",
                "last_epoch": 20,
                "json_path": os.path.join(ROOT, "data", "processed_trajectories_airl_discrete_pricediff.json"),
                "env_id": "V2GDeepEnv-discrete",
                "entry_point": "irl.envs.V2GDeepEnv_discrete:V2GDeepEnv",
                "wrapper": "flatten_airl_discrete",
                "reward_net": True,
                "reward_hidden": 32,
                "shaping_net": True,
                "shaping_hidden": 32,
                "segments": {
                    "Male 50-59":   os.path.join(ROOT, "models", "Adversarial", "discrete", "Adversarial_discrete_male5059"),
                    "Male 40-49":   os.path.join(ROOT, "models", "Adversarial", "discrete", "Adversarial_discrete_male4049"),
                    "Female 50-59": os.path.join(ROOT, "models", "Adversarial", "discrete", "Adversarial_discrete_female5059_new"),
                },
            },
        },
    },
}


SEGMENT = "Male 50-59"
DISCRETE_SEGMENTS = ["Male 50-59", "Male 40-49", "Female 50-59"]

# Colour palette per method
METHOD_COLOURS = {
    "maxent":     "#2563EB",   # blue
    "deepmaxent": "#D97706",   # amber
    "airl":       "#DC2626",   # red
    "expert":     "#111827",   # near-black
}

# Human-readable charger power options
BATTERY_OPTIONS   = {0: "40 kWh", 1: "60 kWh", 2: "80 kWh"}
HOME_CHARGER_OPTIONS = {0: "3.0 kW", 1: "7.4 kW", 2: "11.0 kW"}
WORK_CHARGER_OPTIONS = {0: "7.4 kW", 1: "11.0 kW", 2: "22.0 kW"}

ENERGY_PRICE_PROFILE = [
    0.07, 0.07, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.10, 0.10, 0.11, 0.12,
    0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.21, 0.22, 0.23, 0.24, 0.26,
    0.27, 0.28, 0.30, 0.31, 0.32, 0.33, 0.35, 0.36, 0.37, 0.38, 0.39, 0.40,
    0.41, 0.42, 0.43, 0.44, 0.44, 0.45, 0.45, 0.46, 0.46, 0.47, 0.47, 0.47,
    0.47, 0.47, 0.47, 0.47, 0.46, 0.46, 0.45, 0.45, 0.44, 0.44, 0.43, 0.42,
    0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.33, 0.32, 0.31, 0.30, 0.28,
    0.27, 0.26, 0.24, 0.23, 0.22, 0.21, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14,
    0.13, 0.12, 0.11, 0.10, 0.10, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07,
]

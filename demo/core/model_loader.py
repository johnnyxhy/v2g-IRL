"""
Model loader for the V2G-IRL demo.

Handles gym environment registration and PPO/SAC loading for all
scenario × method combinations.  Results are cached with
``@st.cache_resource`` so models load only once per Streamlit session.

Public API
----------
get_model(scenario_name, method_name) -> (model, env_factory, method_cfg)
    Returns the loaded RL model, a callable that creates a fresh wrapped
    VecEnv, and the method config dict.
"""

import os
import sys
import gymnasium as gym
import torch
import streamlit as st
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import SCENARIOS, ROOT

# ── Ensure the project src/ is on sys.path ────────────────────────────────
_SRC = os.path.join(ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ── Gym registration ───────────────────────────────────────────────────────
_REGISTERED: set[str] = set()


def _register_env(env_id: str, entry_point: str, max_episode_steps: int = 96) -> None:
    if env_id in _REGISTERED:
        return
    try:
        gym.register(id=env_id, entry_point=entry_point, max_episode_steps=max_episode_steps)
    except gym.error.Error:
        pass  # already registered by another import
    _REGISTERED.add(env_id)


def _get_env_id_and_entry(scenario_name: str, method_name: str, method_cfg: dict) -> tuple[str, str]:
    """Resolve env_id / entry_point for a given scenario × method."""
    scen = SCENARIOS[scenario_name]
    # Methods that have an explicit env_id override (Continuous/Discrete dual-env)
    if "env_id" in method_cfg:
        return method_cfg["env_id"], method_cfg["entry_point"]
    # Simple scenario has a single env_id at the scenario level
    return scen["env_id"], scen["entry_point"]


# ── Wrapper factories ──────────────────────────────────────────────────────

def _make_wrapper(wrapper_key: str | None, env_id: str):
    """Return a function that wraps a raw gym env with the right normaliser."""
    if wrapper_key is None:
        def factory():
            return gym.make(env_id)
        return factory

    if wrapper_key == "flatten_simple":
        from irl.MaxEnt.MaxEnt_simple import FlattenNormalizeObsWrapper, OBS_SCALES
        def factory():
            return FlattenNormalizeObsWrapper(gym.make(env_id), OBS_SCALES)
        return factory

    if wrapper_key == "flatten_discrete":
        from irl.MaxEnt.MaxEnt_discrete import FlattenNormalizeObsWrapper, OBS_SCALES
        def factory():
            return FlattenNormalizeObsWrapper(gym.make(env_id), OBS_SCALES)
        return factory

    if wrapper_key == "flatten_deepmaxent_discrete":
        from irl.DeepMaxEnt.DeepMaxEnt_discrete import FlattenNormalizeObsWrapper
        def factory():
            return FlattenNormalizeObsWrapper(gym.make(env_id))
        return factory

    if wrapper_key == "flatten_airl_discrete":
        from irl.Adversarial.Adversarial_discrete import FlattenNormalizeObsWrapper
        def factory():
            return FlattenNormalizeObsWrapper(gym.make(env_id))
        return factory

    if wrapper_key == "flatten_airl_continuous":
        from irl.Adversarial.Adversarial_continuous import FlattenNormalizeObsWrapper
        def factory():
            return FlattenNormalizeObsWrapper(gym.make(env_id))
        return factory

    raise ValueError(f"Unknown wrapper key: {wrapper_key!r}")


# ── Network loaders ────────────────────────────────────────────────────────

def _load_reward_net(folder: str, epoch: int, obs_dim: int = 7,
                     action_dim: int = 1, hidden_dim: int = 64):
    from irl.DeepMaxEnt.DeepMaxEnt import RewardNet
    net = RewardNet(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    path = os.path.join(folder, f"reward_net_epoch{epoch}.pt")
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


def _load_shaping_net(folder: str, epoch: int, obs_dim: int = 7, hidden_dim: int = 32):
    from irl.Adversarial.Adversarial import ShapingNet
    net = ShapingNet(obs_dim=obs_dim, hidden_dim=hidden_dim)
    path = os.path.join(folder, f"shaping_net_epoch{epoch}.pt")
    net.load_state_dict(torch.load(path, weights_only=True))
    net.eval()
    return net


# ── Main cached loader ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_model(scenario_name: str, method_name: str, segment: str | None = None):
    """Load and cache the RL model + env factory for a scenario × method × segment triple.

    Returns
    -------
    model          : loaded SBX PPO or SAC instance
    env_factory    : callable() → wrapped gym.Env (single instance)
    method_cfg     : the raw config dict
    reward_net     : RewardNet | None
    shaping_net    : ShapingNet | None
    """
    method_cfg = SCENARIOS[scenario_name]["methods"][method_name]
    if not method_cfg["available"]:
        raise RuntimeError(f"{method_name} is not available for {scenario_name}")

    env_id, entry_point = _get_env_id_and_entry(scenario_name, method_name, method_cfg)
    _register_env(env_id, entry_point)

    env_factory = _make_wrapper(method_cfg["wrapper"], env_id)

    # Resolve model folder — supports per-segment overrides
    if "segments" in method_cfg and segment is not None:
        folder = method_cfg["segments"][segment]
    else:
        folder = method_cfg["model_folder"]

    epoch = method_cfg["last_epoch"]
    model_path = os.path.join(folder, f"{method_cfg['model_prefix']}{epoch}")

    algo = method_cfg["algo"]
    if algo == "PPO":
        from sbx import PPO
        model = PPO.load(model_path)
    elif algo == "SAC":
        from sbx import SAC
        model = SAC.load(model_path)
    else:
        raise ValueError(f"Unknown algo: {algo!r}")

    reward_net = None
    shaping_net = None

    if method_cfg.get("reward_net"):
        reward_net = _load_reward_net(
            folder, epoch,
            hidden_dim=method_cfg.get("reward_hidden", 64),
        )

    if method_cfg.get("shaping_net"):
        shaping_hidden = method_cfg.get("shaping_hidden", 32)
        shaping_net = _load_shaping_net(folder, epoch, hidden_dim=shaping_hidden)

    return model, env_factory, method_cfg, reward_net, shaping_net

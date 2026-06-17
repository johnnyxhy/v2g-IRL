"""
Expert trajectory loader for the V2G-IRL demo.

Provides:
  - load_expert_trajectories(json_path, segment) → list[dict]
  - get_canonical_episodes(scenario_cfg, segment) → list[dict]
"""

import json
import functools
from .config import SEGMENT


@functools.lru_cache(maxsize=16)
def _load_json(json_path: str) -> list:
    with open(json_path, "r") as f:
        return json.load(f)


def load_expert_trajectories(json_path: str, segment: str = SEGMENT) -> list[dict]:
    """Return all episodes from *json_path* filtered to *segment*.

    Each returned dict has the keys present in the raw JSON:
      episodeID, segment, soc_history, initial_values, feature_expectation, features
    """
    data = _load_json(json_path)
    return [ep for ep in data if ep.get("segment") == segment]


def get_canonical_episodes(scenario_cfg: dict, segment: str = SEGMENT) -> list[dict]:
    """Return the episodes from the canonical expert JSON for the given scenario."""
    return load_expert_trajectories(scenario_cfg["canonical_json"], segment)


def format_episode_label(ep: dict) -> str:
    """Human-readable episode label for the selectbox."""
    iv = ep.get("initial_values", {})
    soc_pct = round(iv.get("soc", 0) * 100)
    out_ts = iv.get("out_start_timestep", 0)
    ret_ts = iv.get("return_start_timestep", 0)
    out_hhmm = _ts_to_hhmm(out_ts)
    ret_hhmm = _ts_to_hhmm(ret_ts)
    return f"Episode {ep['episodeID']}  |  SoC {soc_pct}%  |  Dep {out_hhmm}  Ret {ret_hhmm}"


def _ts_to_hhmm(ts: int) -> str:
    """Convert a 15-min timestep index (0–95) to HH:MM string."""
    total_minutes = int(ts) * 15
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"

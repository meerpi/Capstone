#!/usr/bin/env python3
"""Reproducible showcase run selector across classical heuristics and RL policies.

Scans deterministic seed spaces to extract extreme, best-looking, and failure-mode
demonstration episodes for visual rendering.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch

import baseline_classical
from deepset_v2.env import make_deepset_env_v2
from deepset_v2.model import DeepSetAgent
import env_config
import ppo
import ppo_lstm
from scripts.diagnose_hb_crashes import categorize_crash
from train_optimal_overtaker import OptimalAgent

RESERVED_SEED_RANGES = [
    (2000, 4999),
    (9000, 9999),
]

POLICY_TABLE: dict[str, dict[str, Any]] = {
    "classical_idm_legacy_fa": {
        "agent_type": "classical",
        "controller": "idm",
        "checkpoint_path": None,
        "profile": "legacy_50",
        "tier": "full_adas",
        "default_density": 1.0,
        "default_vehicles": 10,
        "default_duration": 40,
        "target_speeds": None,
        "observation_source": "full_adas",
    },
    "classical_idm_legacy_fo": {
        "agent_type": "classical",
        "controller": "idm",
        "checkpoint_path": None,
        "profile": "legacy_50",
        "tier": "front_only",
        "default_density": 1.0,
        "default_vehicles": 10,
        "default_duration": 40,
        "target_speeds": None,
        "observation_source": "front_only",
    },
    "classical_mobil_legacy_fa": {
        "agent_type": "classical",
        "controller": "mobil",
        "checkpoint_path": None,
        "profile": "legacy_50",
        "tier": "full_adas",
        "default_density": 1.0,
        "default_vehicles": 10,
        "default_duration": 40,
        "target_speeds": None,
        "observation_source": "full_adas",
    },
    "classical_mobil_legacy_fo": {
        "agent_type": "classical",
        "controller": "mobil",
        "checkpoint_path": None,
        "profile": "legacy_50",
        "tier": "front_only",
        "default_density": 1.0,
        "default_vehicles": 10,
        "default_duration": 40,
        "target_speeds": None,
        "observation_source": "front_only",
    },
    "classical_idm_current_fa": {
        "agent_type": "classical",
        "controller": "idm",
        "checkpoint_path": None,
        "profile": "current_75",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": None,
        "observation_source": "full_adas",
    },
    "classical_idm_current_fo": {
        "agent_type": "classical",
        "controller": "idm",
        "checkpoint_path": None,
        "profile": "current_75",
        "tier": "front_only",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": None,
        "observation_source": "front_only",
    },
    "classical_mobil_current_fa": {
        "agent_type": "classical",
        "controller": "mobil",
        "checkpoint_path": None,
        "profile": "current_75",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": None,
        "observation_source": "full_adas",
    },
    "classical_mobil_current_fo": {
        "agent_type": "classical",
        "controller": "mobil",
        "checkpoint_path": None,
        "profile": "current_75",
        "tier": "front_only",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": None,
        "observation_source": "front_only",
    },
    "ppo_ff_legacy_fo": {
        "agent_type": "ppo_ff",
        "controller": "ppo",
        "checkpoint_path": "models/ppo_highway_front_only_seed101.pt",
        "profile": "legacy_50",
        "tier": "front_only",
        "default_density": 1.0,
        "default_vehicles": 10,
        "default_duration": 40,
        "target_speeds": [20.0, 25.0, 30.0],
        "observation_source": "front_only",
    },
    "ppo_lstm_legacy_fo": {
        "agent_type": "ppo_lstm",
        "controller": "ppo_lstm",
        "checkpoint_path": "models/ppo_lstm_highway_front_only_seed101.pt",
        "profile": "legacy_50",
        "tier": "front_only",
        "default_density": 1.0,
        "default_vehicles": 10,
        "default_duration": 40,
        "target_speeds": [20.0, 25.0, 30.0],
        "observation_source": "front_only",
    },
    "ppo_ff_current_fa": {
        "agent_type": "ppo_ff",
        "controller": "ppo",
        "checkpoint_path": "models/ppo_highway_full_adas_seed101.pt",
        "profile": "current_75",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [20.0, 25.0, 30.0],
        "observation_source": "full_adas",
    },
    "overtaker_baseline": {
        "agent_type": "optimal_overtaker",
        "controller": "optimal_agent",
        "checkpoint_path": "models/ppo_optimal_overtaker_4lane_best.pt",
        "profile": "optimal_4lane_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [20.0, 25.0, 30.0],
        "observation_source": "simulator state (not sensor-limited)",
    },
    "overtaker_ext_seed42_best": {
        "agent_type": "optimal_overtaker",
        "controller": "optimal_agent",
        "checkpoint_path": "scratch/optimal_overtaker_extended_speeds/ppo_optimal_overtaker_4lane_best.pt",
        "profile": "optimal_4lane_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "observation_source": "simulator state (not sensor-limited)",
    },
    "overtaker_ext_seed42_final": {
        "agent_type": "optimal_overtaker",
        "controller": "optimal_agent",
        "checkpoint_path": "scratch/optimal_overtaker_extended_speeds/final.pt",
        "profile": "optimal_4lane_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "observation_source": "simulator state (not sensor-limited)",
    },
    "overtaker_ext_seed137_best": {
        "agent_type": "optimal_overtaker",
        "controller": "optimal_agent",
        "checkpoint_path": "scratch/optimal_overtaker_extended_speeds_seed137/ppo_optimal_overtaker_4lane_best.pt",
        "profile": "optimal_4lane_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "observation_source": "simulator state (not sensor-limited)",
    },
    "overtaker_ext_seed137_final": {
        "agent_type": "optimal_overtaker",
        "controller": "optimal_agent",
        "checkpoint_path": "scratch/optimal_overtaker_extended_speeds_seed137/final.pt",
        "profile": "optimal_4lane_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "observation_source": "simulator state (not sensor-limited)",
    },
    "ppo_lagrangian_sep_norm": {
        "agent_type": "optimal_overtaker",
        "controller": "optimal_agent",
        "checkpoint_path": "scratch/ppo_lagrangian_1m_sep_norm/ppo_optimal_overtaker_4lane_best.pt",
        "profile": "optimal_4lane_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [20.0, 25.0, 30.0],
        "observation_source": "simulator state (not sensor-limited)",
    },
    "deepset_v2_2m_baseline": {
        "agent_type": "deepset_v2",
        "controller": "deepset_agent",
        "checkpoint_path": "scratch/deepset_v2_highway_2m/deepset_v2_highway_best.pt",
        "profile": "deepset_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [20.0, 25.0, 30.0],
        "observation_source": "set observation (kinematics)",
    },
    "deepset_v2_retrain_seed42": {
        "agent_type": "deepset_v2",
        "controller": "deepset_agent",
        "checkpoint_path": "scratch/deepset_v2_highway_retrain_seed42/deepset_v2_highway_best.pt",
        "profile": "deepset_current",
        "tier": "full_adas",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "observation_source": "set observation (kinematics)",
    },
}

DEFAULT_SCAN_POLICIES = [
    "overtaker_baseline",
    "overtaker_ext_seed42_best",
    "overtaker_ext_seed137_best",
    "classical_mobil_current_fa",
    "classical_mobil_current_fo",
    "ppo_lagrangian_sep_norm",
    "deepset_v2_retrain_seed42",
    "ppo_ff_current_fa",
]


def validate_seed_range(seed_start: int, num_seeds: int) -> None:
    """Validate that requested seed range does not overlap reserved ranges."""
    seed_end = seed_start + num_seeds - 1
    for r_start, r_end in RESERVED_SEED_RANGES:
        if not (seed_end < r_start or seed_start > r_end):
            raise ValueError(
                f"Requested seed range [{seed_start}, {seed_end}] overlaps reserved range "
                f"[{r_start}, {r_end}]. Scan seeds 5000 and above (excluding 9000-9999)."
            )


def get_git_commit() -> str:
    """Retrieve HEAD commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_package_versions() -> dict[str, str]:
    """Retrieve package runtime versions."""
    import highway_env

    return {
        "highway_env": getattr(highway_env, "__version__", "unknown"),
        "gymnasium": getattr(gym, "__version__", "unknown"),
        "torch": getattr(torch, "__version__", "unknown"),
        "numpy": getattr(np, "__version__", "unknown"),
    }


def make_policy_env(
    policy_name: str,
    seed: int,
    density_override: float | None = None,
    vehicles_override: int | None = None,
    duration_override: int | None = None,
) -> tuple[gym.Env, dict[str, Any]]:
    """Create environment matching the policy's manifest profile and target ladder."""
    p_info = POLICY_TABLE[policy_name]
    density = density_override if density_override is not None else p_info["default_density"]
    v_count = vehicles_override if vehicles_override is not None else p_info["default_vehicles"]
    duration = duration_override if duration_override is not None else p_info["default_duration"]
    agent_type = p_info["agent_type"]
    tier = p_info["tier"]
    target_speeds = p_info["target_speeds"]

    env_params = {
        "vehicles_density": float(density),
        "vehicles_count": int(v_count),
        "duration": int(duration),
    }

    if agent_type == "classical":
        env = env_config.make_env(
            scenario="highway",
            tier=tier,
            seed=seed,
            duration=duration,
            vehicles_density=density,
            vehicles_count=v_count,
            tactical_overtaking=True,
        )
    elif agent_type in ("ppo_ff", "ppo_lstm"):
        extra_kwargs: dict[str, Any] = {}
        if p_info["profile"] == "legacy_50":
            extra_kwargs["observation"] = {"vehicles_count": 10}
            extra_kwargs["vehicles_count"] = v_count
            extra_kwargs["vehicles_density"] = density
            extra_kwargs["duration"] = duration
        elif p_info["profile"] == "current_75":
            extra_kwargs["observation"] = {"vehicles_count": 15}
            extra_kwargs["vehicles_count"] = v_count
            extra_kwargs["vehicles_density"] = density
            extra_kwargs["duration"] = duration

        env = env_config.make_env(
            scenario="highway",
            tier=tier,
            seed=seed,
            tactical_overtaking=True,
            **extra_kwargs,
        )
    elif agent_type == "optimal_overtaker":
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=density,
            vehicles_count=v_count,
            seed=seed,
            frame_stack_k=3,
            duration=duration,
            target_speeds=target_speeds,
        )
    elif agent_type == "deepset_v2":
        env = make_deepset_env_v2(
            tier=tier,
            scenario="highway",
            seed=seed,
            target_speeds=target_speeds,
        )
        env.unwrapped.config["vehicles_density"] = density
        env.unwrapped.config["vehicles_count"] = v_count
        env.unwrapped.config["duration"] = duration
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    return env, env_params


def load_agent(policy_name: str, env: gym.Env, device: str = "cpu") -> Any:
    """Load neural network agent for learned policies or return None for classical."""
    p_info = POLICY_TABLE[policy_name]
    agent_type = p_info["agent_type"]
    ckpt_path = p_info["checkpoint_path"]

    if agent_type == "classical":
        return None

    dev = torch.device(device)
    if agent_type == "ppo_ff":
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        agent = ppo.Agent(obs_dim=obs_dim, action_dim=action_dim).to(dev)
        raw_ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state_dict = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent

    elif agent_type == "ppo_lstm":
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        agent = ppo_lstm.RecurrentAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(dev)
        raw_ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state_dict = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent

    elif agent_type == "optimal_overtaker":
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim).to(dev)
        raw_ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state_dict = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
        agent.load_state_dict(state_dict, strict=False)
        agent.eval()
        return agent

    elif agent_type == "deepset_v2":
        agent = DeepSetAgent().to(dev)
        raw_ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state_dict = raw_ckpt["model_state_dict"] if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt else raw_ckpt
        agent.load_state_dict(state_dict)
        agent.eval()
        return agent

    raise ValueError(f"Unknown agent type: {agent_type}")


def rollout_episode(
    policy_name: str,
    seed: int,
    density_override: float | None = None,
    vehicles_override: int | None = None,
    duration_override: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Execute a single deterministic episode and extract all metrics and diagnostics."""
    env, env_params = make_policy_env(
        policy_name,
        seed,
        density_override=density_override,
        vehicles_override=vehicles_override,
        duration_override=duration_override,
    )
    agent = load_agent(policy_name, env, device=device)
    p_info = POLICY_TABLE[policy_name]
    agent_type = p_info["agent_type"]
    controller = p_info["controller"]

    floor_speed = float(p_info["target_speeds"][0]) if p_info["target_speeds"] else 20.0

    obs, info = env.reset(seed=seed)
    dev = torch.device(device)

    actor_state = None
    if agent_type == "ppo_lstm":
        actor_state, _ = agent.get_initial_states(batch_size=1, device=dev)

    speeds: list[float] = []
    lane_changes = 0
    overtakes = 0
    step_count = 0
    crashed = False
    speed_at_impact = None
    crash_category = None

    history: collections.deque[dict[str, Any]] = collections.deque(maxlen=10)
    ego = env.unwrapped.vehicle
    prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

    while True:
        ego = env.unwrapped.vehicle
        cur_lane = int(ego.lane_index[2]) if hasattr(ego, "lane_index") else 0
        if cur_lane != prev_lane:
            lane_changes += 1
            prev_lane = cur_lane

        target_lane = getattr(ego, "target_lane_index", None)
        target_idx = target_lane[2] if target_lane is not None else cur_lane
        lane_change_in_progress = (target_idx != cur_lane)

        # Decide action
        if agent_type == "classical":
            params = baseline_classical.DEFAULT_IDM_PARAMS
            if controller == "mobil":
                action_idx = baseline_classical.compute_idm_mobil_discrete_action(env, obs, params)
            else:
                action_idx = baseline_classical.compute_idm_discrete_action(env, obs, params)
        elif agent_type == "ppo_lstm":
            action_dim = env.action_space.n
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            obs_t = torch.tensor(obs.reshape(1, -1), dtype=torch.float32, device=dev)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0)
            with torch.no_grad():
                hidden = agent.actor.feature_net(obs_t)
                h_step, actor_state = agent.actor.lstm(hidden.unsqueeze(0), actor_state)
                logits = agent.actor.actor_head(h_step.squeeze(0))
                logits = torch.where(mask_t, logits, torch.tensor(-1e8, device=dev, dtype=logits.dtype))
                action_idx = int(torch.argmax(logits, dim=-1).item())
        else:
            action_dim = env.action_space.n
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            obs_t = torch.tensor(obs, dtype=torch.float32, device=dev).unsqueeze(0)
            mask_t = torch.tensor(mask, dtype=torch.bool, device=dev).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs_t, deterministic=True, action_mask=mask_t)
                action_idx = int(action.item())

        surrounding = []
        for v in env.unwrapped.road.vehicles:
            if v is not ego:
                surrounding.append({
                    "id": id(v),
                    "rel_x": float(v.position[0] - ego.position[0]),
                    "rel_y": float(v.position[1] - ego.position[1]),
                    "speed_ms": float(v.speed),
                    "speed_kmh": float(v.speed * 3.6),
                    "lane": int(v.lane_index[2]) if hasattr(v, "lane_index") else None,
                    "crashed": bool(getattr(v, "crashed", False)),
                })

        step_snap = {
            "step": step_count,
            "ego_speed_ms": float(ego.speed),
            "ego_speed_kmh": float(ego.speed * 3.6),
            "ego_lane": cur_lane,
            "action": action_idx,
            "lane_change_in_progress": lane_change_in_progress,
            "surrounding": surrounding,
        }
        history.append(step_snap)

        obs, reward, terminated, truncated, info = env.step(action_idx)
        step_count += 1

        cur_speed_kmh = float(info.get("speed", getattr(env.unwrapped.vehicle, "speed", 0.0))) * 3.6
        speeds.append(cur_speed_kmh)
        overtakes += int(info.get("overtake_count", 0))

        if getattr(ego, "crashed", False) or info.get("crashed", False) or (terminated and not truncated):
            crashed = True
            speed_at_impact = float(ego.speed * 3.6)

        if terminated or truncated:
            break

    if crashed:
        collided_veh_id = None
        closest_dist = float("inf")
        collided_veh_data = None
        final_surrounding = history[-1]["surrounding"] if history else []
        for sv in final_surrounding:
            dist = math.hypot(sv["rel_x"], sv["rel_y"])
            if sv["crashed"]:
                collided_veh_id = sv["id"]
                collided_veh_data = copy.deepcopy(sv)
                break
            elif dist < closest_dist:
                closest_dist = dist
                collided_veh_id = sv["id"]
                collided_veh_data = copy.deepcopy(sv)

        if collided_veh_data is not None and collided_veh_id is not None:
            traj = []
            for s in list(history)[-5:]:
                m = [v for v in s["surrounding"] if v["id"] == collided_veh_id]
                if m:
                    traj.append(m[0])
            collided_veh_data["history"] = traj

        cat, _ = categorize_crash(list(history), collided_veh_data, floor_speed=floor_speed)
        crash_category = cat

    env.close()

    mean_speed_kmh = float(np.mean(speeds)) if speeds else 0.0
    max_speed_kmh = float(np.max(speeds)) if speeds else 0.0

    return {
        "policy": policy_name,
        "seed": seed,
        "profile": p_info["profile"],
        "parameters": env_params,
        "crashed": crashed,
        "crash_category": crash_category,
        "steps": step_count,
        "mean_speed_kmh": round(mean_speed_kmh, 2),
        "max_speed_kmh": round(max_speed_kmh, 2),
        "speed_at_impact": round(speed_at_impact, 2) if speed_at_impact is not None else None,
        "overtakes": overtakes,
        "lane_changes": lane_changes,
    }


def make_replay_command(policy_name: str, seed: int, parameters: dict[str, Any]) -> str:
    """Generate exact visual reproduction command line for a run."""
    p_info = POLICY_TABLE[policy_name]
    agent_type = p_info["agent_type"]
    ckpt = p_info["checkpoint_path"]
    tier = p_info["tier"]

    density_arg = f" --density {parameters['vehicles_density']}" if parameters["vehicles_density"] != p_info["default_density"] else ""

    if agent_type == "classical":
        ctrl = p_info["controller"]
        return f"python record_visual_driving.py --agent classical_{ctrl} --tier {tier} --seed {seed}{density_arg}"
    elif agent_type == "ppo_ff":
        return f"python record_visual_driving.py --agent ppo --tier {tier} --seed {seed}{density_arg}"
    elif agent_type == "ppo_lstm":
        return f"python record_visual_driving.py --agent ppo_lstm --tier {tier} --seed {seed}{density_arg}"
    elif agent_type == "optimal_overtaker":
        return f"python -c \"import render_top10; render_top10.render_optimal_episode(seed={seed}, rank=1, model_path='{ckpt}')\""
    elif agent_type == "deepset_v2":
        return f"python scripts/diagnose_crashes.py --model-path {ckpt} --seed-start {seed} --episodes 1 --max-gifs 1"
    return f"# replay for {policy_name} seed {seed}"


def select_categories(
    pool_records: list[dict[str, Any]],
    per_category: int = 10,
    per_policy: int = 2,
    num_seeds_scanned: int = 1000,
    seed_start: int = 5000,
) -> dict[str, Any]:
    """Apply extreme ranking rules across scanned pools to construct manifest."""
    categories: dict[str, Any] = {}

    # Compute per-policy median lane changes
    policy_lcs: dict[str, list[int]] = collections.defaultdict(list)
    policy_runs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in pool_records:
        policy_lcs[r["policy"]].append(r["lane_changes"])
        policy_runs[r["policy"]].append(r)

    policy_median_lc: dict[str, float] = {
        pol: float(np.median(lcs)) if lcs else 0.0 for pol, lcs in policy_lcs.items()
    }

    max_scanned_density = max((r["parameters"]["vehicles_density"] for r in pool_records), default=1.4)

    def attach_manifest_fields(run: dict[str, Any], cat_name: str, rank_idx: int) -> dict[str, Any]:
        item = copy.deepcopy(run)
        item["pool_seeds"] = num_seeds_scanned
        item["seed_range"] = [seed_start, seed_start + num_seeds_scanned - 1]

        p_def = POLICY_TABLE[run["policy"]]
        non_default_parts = []
        filename_parts = []
        if run["parameters"]["vehicles_density"] != p_def["default_density"]:
            non_default_parts.append(f"density={run['parameters']['vehicles_density']}")
            filename_parts.append(f"dens{run['parameters']['vehicles_density']}")
        if run["parameters"]["vehicles_count"] != p_def["default_vehicles"]:
            non_default_parts.append(f"vehicles={run['parameters']['vehicles_count']}")
            filename_parts.append(f"veh{run['parameters']['vehicles_count']}")
        if run["parameters"]["duration"] != p_def["default_duration"]:
            non_default_parts.append(f"duration={run['parameters']['duration']}s")
            filename_parts.append(f"dur{run['parameters']['duration']}")

        param_str = f" ({', '.join(non_default_parts)})" if non_default_parts else ""
        tag_str = "_" + "_".join(filename_parts) if filename_parts else ""

        item["caption"] = f"Selected from {num_seeds_scanned} seeds{param_str}"
        item["file_name"] = f"{cat_name}_rank{rank_idx}_{run['policy']}_seed{run['seed']}{tag_str}.gif"
        item["replay_command"] = make_replay_command(run["policy"], run["seed"], run["parameters"])
        return item

    # 1. clean_best_overall
    rule_1 = "no crash, full episode, lane changes <= policy median; sort by overtakes desc, mean speed desc"
    cand_1 = [
        r for r in pool_records
        if not r["crashed"] and r["steps"] >= (r["parameters"]["duration"] * 5) and r["lane_changes"] <= policy_median_lc[r["policy"]]
    ]
    cand_1.sort(key=lambda r: (-r["overtakes"], -r["mean_speed_kmh"]))
    top_1 = [attach_manifest_fields(r, "clean_best_overall", i + 1) for i, r in enumerate(cand_1[:per_category])]
    categories["clean_best_overall"] = {
        "ranking_rule": rule_1,
        "eligible_count": len(cand_1),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_1) >= per_category else f"Only {len(cand_1)} runs met zero-crash, full-duration, low-chatter criteria",
        "entries": top_1,
    }

    # 2. clean_fastest
    rule_2 = "no crash, full episode, mean speed descending"
    cand_2 = [
        r for r in pool_records
        if not r["crashed"] and r["steps"] >= (r["parameters"]["duration"] * 5)
    ]
    cand_2.sort(key=lambda r: (-r["mean_speed_kmh"],))
    top_2 = [attach_manifest_fields(r, "clean_fastest", i + 1) for i, r in enumerate(cand_2[:per_category])]
    categories["clean_fastest"] = {
        "ranking_rule": rule_2,
        "eligible_count": len(cand_2),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_2) >= per_category else f"Only {len(cand_2)} runs survived full horizon",
        "entries": top_2,
    }

    # 3. clean_most_overtakes
    rule_3 = "no crash, overtakes descending"
    cand_3 = [r for r in pool_records if not r["crashed"]]
    cand_3.sort(key=lambda r: (-r["overtakes"], -r["mean_speed_kmh"]))
    top_3 = [attach_manifest_fields(r, "clean_most_overtakes", i + 1) for i, r in enumerate(cand_3[:per_category])]
    categories["clean_most_overtakes"] = {
        "ranking_rule": rule_3,
        "eligible_count": len(cand_3),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_3) >= per_category else f"Only {len(cand_3)} surviving runs available",
        "entries": top_3,
    }

    # 4. clean_smoothest
    rule_4 = "no crash, at least one overtake, lane changes ascending, overtakes descending"
    cand_4 = [r for r in pool_records if not r["crashed"] and r["overtakes"] >= 1]
    cand_4.sort(key=lambda r: (r["lane_changes"], -r["overtakes"], -r["mean_speed_kmh"]))
    top_4 = [attach_manifest_fields(r, "clean_smoothest", i + 1) for i, r in enumerate(cand_4[:per_category])]
    categories["clean_smoothest"] = {
        "ranking_rule": rule_4,
        "eligible_count": len(cand_4),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_4) >= per_category else f"Only {len(cand_4)} runs had >= 1 overtake without crashing",
        "entries": top_4,
    }

    # 5. clean_dense_traffic
    rule_5 = f"no crash, full episode, vehicles_density == {max_scanned_density}, overtakes descending"
    cand_5 = [
        r for r in pool_records
        if not r["crashed"] and r["steps"] >= (r["parameters"]["duration"] * 5) and r["parameters"]["vehicles_density"] == max_scanned_density
    ]
    cand_5.sort(key=lambda r: (-r["overtakes"], -r["mean_speed_kmh"]))
    top_5 = [attach_manifest_fields(r, "clean_dense_traffic", i + 1) for i, r in enumerate(cand_5[:per_category])]
    categories["clean_dense_traffic"] = {
        "ranking_rule": rule_5,
        "eligible_count": len(cand_5),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_5) >= per_category else f"Only {len(cand_5)} runs survived at maximum density {max_scanned_density}",
        "entries": top_5,
    }

    # 6. crash_fastest
    rule_6 = "crash, speed at impact descending"
    cand_6 = [r for r in pool_records if r["crashed"] and r["speed_at_impact"] is not None]
    cand_6.sort(key=lambda r: (-r["speed_at_impact"],))
    top_6 = [attach_manifest_fields(r, "crash_fastest", i + 1) for i, r in enumerate(cand_6[:per_category])]
    categories["crash_fastest"] = {
        "ranking_rule": rule_6,
        "eligible_count": len(cand_6),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_6) >= per_category else f"Only {len(cand_6)} crashes recorded",
        "entries": top_6,
    }

    # 7. crash_floor_speed
    rule_7 = "crash classified as (a) Floor-speed rear-end, speed at impact descending"
    cand_7 = [r for r in pool_records if r["crashed"] and r["crash_category"] == "(a) Floor-speed rear-end" and r["speed_at_impact"] is not None]
    cand_7.sort(key=lambda r: (-r["speed_at_impact"],))
    top_7 = [attach_manifest_fields(r, "crash_floor_speed", i + 1) for i, r in enumerate(cand_7[:per_category])]
    categories["crash_floor_speed"] = {
        "ranking_rule": rule_7,
        "eligible_count": len(cand_7),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_7) >= per_category else f"Only {len(cand_7)} floor-speed rear-end crashes found",
        "entries": top_7,
    }

    # 8. crash_lane_change
    rule_8 = "crash classified as (b) Lane-change collision, speed at impact descending"
    cand_8 = [r for r in pool_records if r["crashed"] and r["crash_category"] == "(b) Lane-change collision" and r["speed_at_impact"] is not None]
    cand_8.sort(key=lambda r: (-r["speed_at_impact"],))
    top_8 = [attach_manifest_fields(r, "crash_lane_change", i + 1) for i, r in enumerate(cand_8[:per_category])]
    categories["crash_lane_change"] = {
        "ranking_rule": rule_8,
        "eligible_count": len(cand_8),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_8) >= per_category else f"Only {len(cand_8)} lane-change collisions found",
        "entries": top_8,
    }

    # 9. crash_blind_spot
    rule_9 = "crash classified as (c) Blind-spot/other, speed at impact descending"
    cand_9 = [r for r in pool_records if r["crashed"] and r["crash_category"] == "(c) Blind-spot/other" and r["speed_at_impact"] is not None]
    cand_9.sort(key=lambda r: (-r["speed_at_impact"],))
    top_9 = [attach_manifest_fields(r, "crash_blind_spot", i + 1) for i, r in enumerate(cand_9[:per_category])]
    categories["crash_blind_spot"] = {
        "ranking_rule": rule_9,
        "eligible_count": len(cand_9),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_9) >= per_category else f"Only {len(cand_9)} blind-spot/other crashes found",
        "entries": top_9,
    }

    # 10. most_lane_changes
    rule_10 = "any outcome, lane changes descending (chatter example)"
    cand_10 = list(pool_records)
    cand_10.sort(key=lambda r: (-r["lane_changes"], -r["mean_speed_kmh"]))
    top_10 = [attach_manifest_fields(r, "most_lane_changes", i + 1) for i, r in enumerate(cand_10[:per_category])]
    categories["most_lane_changes"] = {
        "ranking_rule": rule_10,
        "eligible_count": len(cand_10),
        "target_k": per_category,
        "reason_if_fewer": None if len(cand_10) >= per_category else f"Only {len(cand_10)} runs available",
        "entries": top_10,
    }

    # 11. paired_floor_fix
    rule_11 = "same seed & profile, baseline overtaker crashed (floor-speed rear-end) while extended-ladder seed 42 best survived; rank pairs by crashed speed at impact"
    base_runs = {r["seed"]: r for r in pool_records if r["policy"] == "overtaker_baseline"}
    ext42_runs = {r["seed"]: r for r in pool_records if r["policy"] == "overtaker_ext_seed42_best" and r["parameters"]["vehicles_density"] == 1.4}
    paired_floor_candidates = []
    for s_seed, r_base in base_runs.items():
        if r_base["crashed"] and r_base["crash_category"] == "(a) Floor-speed rear-end":
            r_ext = ext42_runs.get(s_seed)
            if r_ext and not r_ext["crashed"]:
                paired_floor_candidates.append((r_base["speed_at_impact"] or 0.0, r_base, r_ext))
    paired_floor_candidates.sort(key=lambda item: -item[0])
    top_floor_pairs = []
    for i, (_, r_base, r_ext) in enumerate(paired_floor_candidates[:per_category]):
        crashed_entry = attach_manifest_fields(r_base, "paired_floor_fix_crashed", i + 1)
        surviving_entry = attach_manifest_fields(r_ext, "paired_floor_fix_surviving", i + 1)
        top_floor_pairs.append({
            "rank": i + 1,
            "seed": r_base["seed"],
            "policy": r_base["policy"],
            "pool_seeds": num_seeds_scanned,
            "seed_range": [seed_start, seed_start + num_seeds_scanned - 1],
            "parameters": r_base["parameters"],
            "rank_metric": r_base["speed_at_impact"],
            "crashed_run": crashed_entry,
            "surviving_run": surviving_entry,
            "caption": f"Selected from {num_seeds_scanned} seeds",
            "file_name": f"paired_floor_fix_rank{i+1}_seed{r_base['seed']}.gif",
            "replay_command": crashed_entry["replay_command"],
        })
    categories["paired_floor_fix"] = {
        "ranking_rule": rule_11,
        "eligible_count": len(paired_floor_candidates),
        "target_k": per_category,
        "reason_if_fewer": None if len(paired_floor_candidates) >= per_category else f"Only {len(paired_floor_candidates)} pairs matched floor-speed crash vs survival",
        "entries": top_floor_pairs,
    }

    # 12. paired_sensor_tier
    rule_12 = "same seed, MOBIL front_only crashed while MOBIL full_adas survived; rank pairs by crashed speed at impact"
    fo_runs = {r["seed"]: r for r in pool_records if r["policy"] in ("classical_mobil_current_fo", "classical_mobil_legacy_fo")}
    fa_runs = {r["seed"]: r for r in pool_records if r["policy"] in ("classical_mobil_current_fa", "classical_mobil_legacy_fa")}
    paired_tier_candidates = []
    for s_seed, r_fo in fo_runs.items():
        if r_fo["crashed"]:
            r_fa = fa_runs.get(s_seed)
            if r_fa and not r_fa["crashed"]:
                paired_tier_candidates.append((r_fo["speed_at_impact"] or 0.0, r_fo, r_fa))
    paired_tier_candidates.sort(key=lambda item: -item[0])
    top_tier_pairs = []
    for i, (_, r_fo, r_fa) in enumerate(paired_tier_candidates[:per_category]):
        crashed_entry = attach_manifest_fields(r_fo, "paired_sensor_tier_crashed", i + 1)
        surviving_entry = attach_manifest_fields(r_fa, "paired_sensor_tier_surviving", i + 1)
        top_tier_pairs.append({
            "rank": i + 1,
            "seed": r_fo["seed"],
            "policy": r_fo["policy"],
            "pool_seeds": num_seeds_scanned,
            "seed_range": [seed_start, seed_start + num_seeds_scanned - 1],
            "parameters": r_fo["parameters"],
            "rank_metric": r_fo["speed_at_impact"],
            "crashed_run": crashed_entry,
            "surviving_run": surviving_entry,
            "caption": f"Selected from {num_seeds_scanned} seeds",
            "file_name": f"paired_sensor_tier_rank{i+1}_seed{r_fo['seed']}.gif",
            "replay_command": crashed_entry["replay_command"],
        })
    categories["paired_sensor_tier"] = {
        "ranking_rule": rule_12,
        "eligible_count": len(paired_tier_candidates),
        "target_k": per_category,
        "reason_if_fewer": None if len(paired_tier_candidates) >= per_category else f"Only {len(paired_tier_candidates)} pairs matched front_only crash vs full_adas survival",
        "entries": top_tier_pairs,
    }

    # 13. paired_density
    rule_13 = "same policy & seed, survived at default density and crashed at highest density (or vice versa); rank by crashed speed at impact"
    paired_density_candidates = []
    # Index by (policy, seed)
    runs_by_pol_seed: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for r in pool_records:
        runs_by_pol_seed[(r["policy"], r["seed"])].append(r)

    for (pol, s_seed), runs in runs_by_pol_seed.items():
        if len(runs) >= 2:
            runs_sorted = sorted(runs, key=lambda x: x["parameters"]["vehicles_density"])
            low_dens_run = runs_sorted[0]
            high_dens_run = runs_sorted[-1]
            if low_dens_run["parameters"]["vehicles_density"] != high_dens_run["parameters"]["vehicles_density"]:
                if (not low_dens_run["crashed"] and high_dens_run["crashed"]):
                    paired_density_candidates.append((high_dens_run["speed_at_impact"] or 0.0, high_dens_run, low_dens_run))
                elif (low_dens_run["crashed"] and not high_dens_run["crashed"]):
                    paired_density_candidates.append((low_dens_run["speed_at_impact"] or 0.0, low_dens_run, high_dens_run))

    paired_density_candidates.sort(key=lambda item: -item[0])
    top_density_pairs = []
    for i, (_, r_crash, r_surv) in enumerate(paired_density_candidates[:per_category]):
        crashed_entry = attach_manifest_fields(r_crash, "paired_density_crashed", i + 1)
        surviving_entry = attach_manifest_fields(r_surv, "paired_density_surviving", i + 1)
        top_density_pairs.append({
            "rank": i + 1,
            "seed": r_crash["seed"],
            "policy": r_crash["policy"],
            "pool_seeds": num_seeds_scanned,
            "seed_range": [seed_start, seed_start + num_seeds_scanned - 1],
            "parameters": r_crash["parameters"],
            "rank_metric": r_crash["speed_at_impact"],
            "crashed_run": crashed_entry,
            "surviving_run": surviving_entry,
            "caption": f"Selected from {num_seeds_scanned} seeds",
            "file_name": f"paired_density_rank{i+1}_{r_crash['policy']}_seed{r_crash['seed']}.gif",
            "replay_command": crashed_entry["replay_command"],
        })
    categories["paired_density"] = {
        "ranking_rule": rule_13,
        "eligible_count": len(paired_density_candidates),
        "target_k": per_category,
        "reason_if_fewer": None if len(paired_density_candidates) >= per_category else f"Only {len(paired_density_candidates)} pairs exhibited density sensitivity",
        "entries": top_density_pairs,
    }

    # Per policy categories
    # 14. policy_best_clean
    pol_best_clean: dict[str, list[dict[str, Any]]] = {}
    flat_clean_entries: list[dict[str, Any]] = []
    clean_eligible_total = 0
    clean_short_pols = []
    for pol, runs in policy_runs.items():
        p_cands = [
            r for r in runs
            if not r["crashed"] and r["steps"] >= (r["parameters"]["duration"] * 5) and r["lane_changes"] <= policy_median_lc[pol]
        ]
        clean_eligible_total += len(p_cands)
        p_cands.sort(key=lambda r: (-r["overtakes"], -r["mean_speed_kmh"]))
        selected_p = [attach_manifest_fields(r, f"policy_best_clean_{pol}", i + 1) for i, r in enumerate(p_cands[:per_policy])]
        pol_best_clean[pol] = selected_p
        flat_clean_entries.extend(selected_p)
        if len(selected_p) < per_policy:
            clean_short_pols.append(pol)

    target_k_clean = per_policy * len(policy_runs)
    categories["policy_best_clean"] = {
        "ranking_rule": "clean_best_overall applied within each policy pool",
        "eligible_count": clean_eligible_total,
        "target_k": target_k_clean,
        "per_policy_k": per_policy,
        "reason_if_fewer": f"Policies with fewer than {per_policy} clean runs: {', '.join(clean_short_pols)}" if clean_short_pols else None,
        "entries": flat_clean_entries,
        "by_policy": pol_best_clean,
    }

    # 15. policy_best_crash
    pol_best_crash: dict[str, list[dict[str, Any]]] = {}
    flat_crash_entries: list[dict[str, Any]] = []
    crash_eligible_total = 0
    no_crash_pols = []
    for pol, runs in policy_runs.items():
        p_cands = [r for r in runs if r["crashed"] and r["speed_at_impact"] is not None]
        crash_eligible_total += len(p_cands)
        p_cands.sort(key=lambda r: (-r["speed_at_impact"],))
        selected_p = [attach_manifest_fields(r, f"policy_best_crash_{pol}", 1) for r in p_cands[:1]]
        pol_best_crash[pol] = selected_p
        flat_crash_entries.extend(selected_p)
        if not selected_p:
            no_crash_pols.append(pol)

    target_k_crash = len(policy_runs)
    categories["policy_best_crash"] = {
        "ranking_rule": "crash_fastest applied within each policy pool (1 entry per policy)",
        "eligible_count": crash_eligible_total,
        "target_k": target_k_crash,
        "per_policy_k": 1,
        "reason_if_fewer": f"Policies with 0 crashes in pool: {', '.join(no_crash_pols)}" if no_crash_pols else None,
        "entries": flat_crash_entries,
        "by_policy": pol_best_crash,
    }

    return categories


def _scan_worker_task(args_tuple: tuple[str, int, float | None, int | None, int | None]) -> dict[str, Any] | None:
    p_name, s, d, v, dur = args_tuple
    try:
        return rollout_episode(p_name, s, density_override=d, vehicles_override=v, duration_override=dur)
    except Exception as e:
        print(f"Error in {p_name} seed {s}: {e}", file=sys.stderr)
        return None


def execute_scan(
    policies: list[str],
    seed_start: int = 5000,
    num_seeds: int = 1000,
    density_overrides: list[float] | None = None,
    vehicles_override: int | None = None,
    duration_override: int | None = None,
    workers: int = 8,
) -> list[dict[str, Any]]:
    """Scan deterministic seed spaces using parallel worker processes."""
    tasks = []
    for pol in policies:
        densities = density_overrides if density_overrides else [POLICY_TABLE[pol]["default_density"]]
        for d in densities:
            for s_idx in range(num_seeds):
                s = seed_start + s_idx
                tasks.append((pol, s, d, vehicles_override, duration_override))

    results: list[dict[str, Any]] = []
    print(f"Scanning {len(tasks)} total episodes across {len(policies)} policies using {workers} workers...", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for i, res in enumerate(executor.map(_scan_worker_task, tasks, chunksize=5)):
            if res is not None:
                results.append(res)
            if (i + 1) % 500 == 0 or (i + 1) == len(tasks):
                print(f"Completed {i + 1}/{len(tasks)} episodes...", flush=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible showcase run selector across RL policies.")
    parser.add_argument("--policies", type=str, nargs="+", default=None,
                        help="Policies to scan (keys in POLICY_TABLE). Default: core policy set.")
    parser.add_argument("--seed-start", type=int, default=5000,
                        help="Starting random seed (default: 5000).")
    parser.add_argument("--num-seeds", type=int, default=1000,
                        help="Number of deterministic seeds to scan (default: 1000).")
    parser.add_argument("--vehicles-density", type=float, nargs="+", default=None,
                        help="One or more traffic densities to evaluate.")
    parser.add_argument("--vehicles-count", type=int, default=None,
                        help="Optional override for vehicles count.")
    parser.add_argument("--per-category", type=int, default=10,
                        help="Maximum runs per global category (default: 10).")
    parser.add_argument("--per-policy", type=int, default=2,
                        help="Maximum runs per individual policy category (default: 2).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="Number of parallel worker processes.")
    parser.add_argument("--output", type=str, default="gifs/manifest.json",
                        help="Output path for showcase manifest (default: gifs/manifest.json).")

    args = parser.parse_args()

    validate_seed_range(args.seed_start, args.num_seeds)

    selected_policies = args.policies if args.policies else DEFAULT_SCAN_POLICIES
    for pol in selected_policies:
        if pol not in POLICY_TABLE:
            raise ValueError(f"Unknown policy '{pol}'. Available: {list(POLICY_TABLE.keys())}")

    # Primary scan
    records = execute_scan(
        policies=selected_policies,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        density_overrides=args.vehicles_density,
        vehicles_override=args.vehicles_count,
        workers=args.workers,
    )

    # If density override wasn't passed, run high-density on overtaker_ext_seed42_best for paired_density & clean_dense_traffic
    if args.vehicles_density is None and "overtaker_ext_seed42_best" in selected_policies:
        print("Executing auxiliary high-density scan (density=1.8) for overtaker_ext_seed42_best...", flush=True)
        high_dens_records = execute_scan(
            policies=["overtaker_ext_seed42_best"],
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            density_overrides=[1.8],
            vehicles_override=args.vehicles_count,
            workers=args.workers,
        )
        records.extend(high_dens_records)

    # Compute policy pool statistics
    policy_stats: dict[str, dict[str, Any]] = {}
    policy_runs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        policy_runs[r["policy"]].append(r)

    for pol, runs in policy_runs.items():
        total = len(runs)
        crashes = sum(1 for r in runs if r["crashed"])
        rate = (crashes / total) * 100.0 if total > 0 else 0.0
        speeds = [r["mean_speed_kmh"] for r in runs]
        lcs = [r["lane_changes"] for r in runs]
        policy_stats[pol] = {
            "pool_episodes": total,
            "pool_seeds": args.num_seeds,
            "crashes": crashes,
            "crash_rate_pct": round(rate, 2),
            "mean_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "median_lane_changes": float(np.median(lcs)) if lcs else 0.0,
        }

    categories = select_categories(
        pool_records=records,
        per_category=args.per_category,
        per_policy=args.per_policy,
        num_seeds_scanned=args.num_seeds,
        seed_start=args.seed_start,
    )

    manifest = {
        "git_commit": get_git_commit(),
        "package_versions": get_package_versions(),
        "scan_configuration": {
            "seed_start": args.seed_start,
            "num_seeds": args.num_seeds,
            "seed_range": [args.seed_start, args.seed_start + args.num_seeds - 1],
            "policies": selected_policies,
            "total_episodes_scanned": len(records),
        },
        "policy_statistics": policy_stats,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"\nManifest successfully written to: {args.output}", flush=True)


if __name__ == "__main__":
    main()

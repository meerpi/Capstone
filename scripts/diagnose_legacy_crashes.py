#!/usr/bin/env python3
"""Detailed crash failure-mode breakdown for legacy Optimal Overtaker & PPO-Lagrangian models.

Applies the exact same crash categorization logic as diagnose_hb_crashes.py:
- (a) Floor-speed rear-end: ego in-lane, not actively changing lanes, closing on lead vehicle
      traveling below floor speed (or closing at floor speed <= 20.5 m/s).
- (b) Lane-change collision: crash occurred during or just after an active lane-change maneuver.
- (c) Blind-spot/other: adjacent vehicle cut-in, rear-ended from behind, or high-speed rear-end.
"""

import argparse
import collections
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch

from env_config import make_optimal_env
from train_optimal_overtaker import OptimalAgent

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def categorize_crash(
    history: list[dict[str, Any]],
    collided_veh_info: dict[str, Any] | None,
    floor_speed: float = 20.0,
) -> tuple[str, str]:
    """Categorize crash into (a) floor-speed rear-end, (b) lane-change, or (c) blind-spot/other."""
    if not history:
        return "(c) Blind-spot/other", "No step history available"

    final_step = history[-1]
    ego_speed = final_step["ego_speed_ms"]
    ego_lane = final_step["ego_lane"]

    # Check (b) Lane-change collision:
    recent_actions = [s["action"] for s in history[-5:]]
    lane_change_action_recent = any(a in (0, 2) for a in recent_actions)
    lane_change_active = any(s["lane_change_in_progress"] for s in history[-3:])

    if lane_change_action_recent or lane_change_active:
        explanation = (
            f"Crash occurred during/after lane change. "
            f"Actions in last 5 steps: {[ACTION_NAMES.get(a, a) for a in recent_actions]}. "
            f"Ego lane: {final_step['ego_lane']}, speed: {ego_speed:.2f} m/s."
        )
        return "(b) Lane-change collision", explanation

    # If no lane change was in progress, check lead vehicle collision
    if collided_veh_info is not None:
        rel_x = collided_veh_info.get("rel_x", 0.0)
        rel_y = collided_veh_info.get("rel_y", 0.0)
        other_speed = collided_veh_info.get("speed_ms", 0.0)
        veh_history = collided_veh_info.get("history", [])

        # Check if the colliding vehicle cut into ego lane from adjacent (blind spot / cut-in)
        if len(veh_history) >= 2:
            initial_rel_y = veh_history[0].get("rel_y", rel_y)
            if abs(initial_rel_y) > 2.0 and abs(rel_y) <= 2.0:
                explanation = (
                    f"Adjacent vehicle cut into ego lane (lateral rel_y shifted from "
                    f"{initial_rel_y:+.2f}m to {rel_y:+.2f}m). Ego speed: {ego_speed:.2f} m/s."
                )
                return "(c) Blind-spot/other", explanation

        # Check lead vehicle in same lane
        is_lead_in_lane = (rel_x > -2.0) and (abs(rel_y) < 2.5)

        if is_lead_in_lane:
            if other_speed < floor_speed or (ego_speed <= (floor_speed + 0.5) and other_speed <= ego_speed):
                explanation = (
                    f"In-lane rear-end at floor speed: lead vehicle speed was {other_speed:.2f} m/s "
                    f"(< {floor_speed:.1f} m/s floor), rel_x={rel_x:+.2f}m. Ego at {ego_speed:.2f} m/s cannot brake below {floor_speed:.1f} m/s."
                )
                return "(a) Floor-speed rear-end", explanation
            else:
                explanation = (
                    f"High-speed rear-end: lead vehicle was at {other_speed:.2f} m/s (>= {floor_speed:.1f} m/s), "
                    f"ego speed was {ego_speed:.2f} m/s. Failed to brake or steer in time."
                )
                return "(c) Blind-spot/other", explanation

        # Collided from behind
        if rel_x < -2.0 and abs(rel_y) < 2.5:
            explanation = (
                f"Ego was rear-ended from behind by vehicle at rel_x={rel_x:+.2f}m, "
                f"speed={other_speed:.2f} m/s (ego speed={ego_speed:.2f} m/s)."
            )
            return "(c) Blind-spot/other", explanation

    return "(c) Blind-spot/other", f"Complex in-lane contact. Ego speed: {ego_speed:.2f} m/s, lane: {ego_lane}."


def evaluate_and_diagnose(
    model_path: str,
    n_episodes: int = 100,
    seed_start: int = 2000,
    output_json: str = "eval_out/crash_breakdown_legacy.json",
    floor_speed: float = 20.0,
    target_speeds: list[float] | None = None,
) -> dict[str, Any]:
    """Run 100-episode evaluation of legacy agent and extract crash breakdown."""
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    if target_speeds is None and isinstance(checkpoint, dict) and "args" in checkpoint:
        ckpt_args = checkpoint["args"]
        if isinstance(ckpt_args, dict) and "target_speeds" in ckpt_args and ckpt_args["target_speeds"]:
            target_speeds = [float(x) for x in ckpt_args["target_speeds"]]
            print(f"Auto-detected target_speeds from checkpoint args: {target_speeds}")

    if isinstance(checkpoint, dict) and "obs_dim" in checkpoint:
        obs_dim = checkpoint["obs_dim"]
    elif isinstance(state_dict, dict) and "actor.0.weight" in state_dict:
        obs_dim = state_dict["actor.0.weight"].shape[1]
    else:
        obs_dim = 81

    action_dim = 5
    has_cost_critic = any(k.startswith("cost_critic.") for k in state_dict.keys())

    agent = OptimalAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        use_cost_critic=has_cost_critic,
    ).to(device)
    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    include_continuous_features = (obs_dim in (30, 90))

    env = make_optimal_env(
        lanes_count=4,
        vehicles_density=1.4,
        vehicles_count=14,
        seed=seed_start,
        frame_stack_k=3,
        include_continuous_features=include_continuous_features,
        collision_penalty=-50.0,
        overtake_bonus=1.0,
        target_speeds=target_speeds,
    )

    crashes: list[dict[str, Any]] = []
    category_counts: dict[str, int] = collections.defaultdict(int)

    all_speeds_ms: list[float] = []
    all_lane_changes: list[int] = []
    all_returns: list[float] = []

    print(f"Evaluating {n_episodes} episodes on seeds {seed_start}..{seed_start + n_episodes - 1}...")

    for ep_idx in range(n_episodes):
        seed = seed_start + ep_idx
        obs, info = env.reset(seed=seed)
        step = 0
        ep_crashed = False
        ep_return = 0.0
        ep_speeds: list[float] = []
        lane_changes = 0

        history: collections.deque[dict[str, Any]] = collections.deque(maxlen=10)

        ego = env.unwrapped.vehicle
        prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

        while True:
            ego = env.unwrapped.vehicle
            ego_speed_ms = float(ego.speed)
            ego_speed_kmh = ego_speed_ms * 3.6
            cur_lane = int(ego.lane_index[2]) if hasattr(ego, "lane_index") else 0

            if cur_lane != prev_lane:
                lane_changes += 1
                prev_lane = cur_lane

            target_lane = getattr(ego, "target_lane_index", None)
            target_idx = target_lane[2] if target_lane is not None else cur_lane
            lane_change_in_progress = (target_idx != cur_lane)

            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_t, deterministic=True, action_mask=mask_t
                )
                action_idx = int(action.item())

            # Log snapshot of surrounding traffic
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
                "step": step,
                "ego_speed_ms": ego_speed_ms,
                "ego_speed_kmh": ego_speed_kmh,
                "ego_lane": cur_lane,
                "ego_pos_x": float(ego.position[0]),
                "ego_pos_y": float(ego.position[1]),
                "target_lane": target_idx,
                "action": action_idx,
                "action_name": ACTION_NAMES.get(action_idx, str(action_idx)),
                "lane_change_in_progress": lane_change_in_progress,
                "surrounding": surrounding,
            }
            history.append(step_snap)
            ep_speeds.append(ego_speed_ms)

            obs, reward, terminated, truncated, info = env.step(action_idx)
            ep_return += float(reward)
            step += 1

            if getattr(ego, "crashed", False) or info.get("crashed", False):
                ep_crashed = True

            if terminated or truncated:
                break

        all_speeds_ms.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        all_lane_changes.append(lane_changes)
        all_returns.append(ep_return)

        if ep_crashed:
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

            effective_floor = float(target_speeds[0]) if target_speeds else floor_speed
            category, explanation = categorize_crash(list(history), collided_veh_data, floor_speed=effective_floor)
            category_counts[category] += 1

            pre_5_steps = []
            for s in list(history)[-5:]:
                snap = {
                    "step": s["step"],
                    "ego_speed_ms": s["ego_speed_ms"],
                    "ego_speed_kmh": s["ego_speed_kmh"],
                    "ego_lane": s["ego_lane"],
                    "action": s["action_name"],
                    "lane_change_in_progress": s["lane_change_in_progress"],
                }
                if collided_veh_id is not None:
                    target_v = [v for v in s["surrounding"] if v["id"] == collided_veh_id]
                    if target_v:
                        tv = target_v[0]
                        snap["collided_vehicle"] = {
                            "rel_x": tv["rel_x"],
                            "rel_y": tv["rel_y"],
                            "speed_ms": tv["speed_ms"],
                            "speed_kmh": tv["speed_kmh"],
                            "lane": tv["lane"],
                        }
                pre_5_steps.append(snap)

            crash_record = {
                "seed": seed,
                "impact_step": step,
                "category": category,
                "explanation": explanation,
                "final_ego_speed_ms": history[-1]["ego_speed_ms"] if history else 0.0,
                "final_ego_lane": history[-1]["ego_lane"] if history else 0,
                "collided_vehicle": {
                    "rel_x": collided_veh_data.get("rel_x", 0.0) if collided_veh_data else 0.0,
                    "rel_y": collided_veh_data.get("rel_y", 0.0) if collided_veh_data else 0.0,
                    "speed_ms": collided_veh_data.get("speed_ms", 0.0) if collided_veh_data else 0.0,
                    "speed_kmh": collided_veh_data.get("speed_kmh", 0.0) if collided_veh_data else 0.0,
                    "lane": collided_veh_data.get("lane", None) if collided_veh_data else None,
                } if collided_veh_data else None,
                "pre_impact_5_steps": pre_5_steps,
            }
            crashes.append(crash_record)
            print(f"  [Crash #{len(crashes):02d}] Seed {seed:4d} @ Step {step:3d}: {category} | {explanation}")

    env.close()

    total_crashes = len(crashes)
    summary = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + n_episodes - 1,
        "total_crashes": total_crashes,
        "crash_rate_pct": (total_crashes / n_episodes) * 100.0,
        "mean_speed_ms": float(np.mean(all_speeds_ms)),
        "mean_speed_kmh": float(np.mean(all_speeds_ms)) * 3.6,
        "mean_lane_changes": float(np.mean(all_lane_changes)),
        "mean_return": float(np.mean(all_returns)),
        "category_breakdown": {
            cat: {
                "count": count,
                "percentage": (count / total_crashes) * 100.0 if total_crashes > 0 else 0.0,
            }
            for cat, count in sorted(category_counts.items())
        },
        "crashes": crashes,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("CRASH-CAUSE BREAKDOWN SUMMARY (LEGACY PIPELINE)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes} (seeds {seed_start}..{seed_start + n_episodes - 1})")
    print(f"Total Crashes: {total_crashes}/{n_episodes} ({summary['crash_rate_pct']:.1f}%)")
    print(f"Mean Speed: {summary['mean_speed_ms']:.2f} m/s ({summary['mean_speed_kmh']:.1f} km/h)")
    print(f"Mean Lane Changes: {summary['mean_lane_changes']:.2f}")
    print(f"Mean Return: {summary['mean_return']:.2f}")
    print("-" * 80)
    for cat, data in summary["category_breakdown"].items():
        print(f"  {cat:<30}: {data['count']:2d} crashes ({data['percentage']:5.1f}%)")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--output-json", type=str, default="eval_out/crash_breakdown_legacy.json")
    parser.add_argument("--floor-speed", type=float, default=20.0)
    parser.add_argument(
        "--target-speeds",
        type=float,
        nargs="+",
        default=None,
        help="Target speeds for DiscreteMetaAction (e.g. 10 15 20 25 30)",
    )
    args = parser.parse_args()

    evaluate_and_diagnose(
        model_path=args.model_path,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        output_json=args.output_json,
        floor_speed=args.floor_speed,
        target_speeds=args.target_speeds,
    )

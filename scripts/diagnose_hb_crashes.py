#!/usr/bin/env python3
"""Detailed crash failure-mode breakdown for DeepSet v2 on highway-fast-v0.

Evaluates a trained DeepSet v2 checkpoint on 100 seeds to test floor-speed crash causes:
For every crashing episode, logs:
- Step of impact
- Ego speed (m/s and km/h), lane, lateral position, action history
- Colliding vehicle speed, relative position (rel_x, rel_y), lane over the 5 steps before impact
- Lane-change status during the 5 steps before impact

Categorizes each crash into:
  (a) Floor-speed rear-end: ego was in-lane, not actively lane-changing, closing on a lead vehicle below the 20 m/s floor
  (b) Lane-change collision: crash occurred during or just after an active lane-change maneuver
  (c) Blind-spot/other: anything not covered above (e.g. cut-in by adjacent vehicle, rear-ended from behind, high-speed rear-end)
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

from deepset_v2.env import make_deepset_env_v2
from deepset_v2.model import DeepSetAgent

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
    # Did ego execute a lane-change action (0=LANE_LEFT, 2=LANE_RIGHT) in the last 5 steps,
    # or was target_lane != current_lane, or was ego laterally shifting?
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
            # If it started in adjacent lane (|rel_y| > 2.0) and moved into ego lane (|rel_y| < 1.5)
            if abs(initial_rel_y) > 2.0 and abs(rel_y) <= 2.0:
                explanation = (
                    f"Adjacent vehicle cut into ego lane (lateral rel_y shifted from "
                    f"{initial_rel_y:+.2f}m to {rel_y:+.2f}m). Ego speed: {ego_speed:.2f} m/s."
                )
                return "(c) Blind-spot/other", explanation

        # Check lead vehicle in same lane: other vehicle was ahead (rel_x > -2.0) and in same lane (|rel_y| < 2.5)
        is_lead_in_lane = (rel_x > -2.0) and (abs(rel_y) < 2.5)

        if is_lead_in_lane:
            # Check (a) Floor-speed rear-end:
            # Ego is in-lane, not changing lanes, and colliding with a lead vehicle traveling below floor speed (or closing at floor speed)
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

        # Collided from behind (ego was rear-ended)
        if rel_x < -2.0 and abs(rel_y) < 2.5:
            explanation = (
                f"Ego was rear-ended from behind by vehicle at rel_x={rel_x:+.2f}m, "
                f"speed={other_speed:.2f} m/s (ego speed={ego_speed:.2f} m/s)."
            )
            return "(c) Blind-spot/other", explanation

    # Fallback
    return "(c) Blind-spot/other", f"Complex in-lane contact. Ego speed: {ego_speed:.2f} m/s, lane: {ego_lane}."


def evaluate_and_diagnose(
    model_path: str,
    n_episodes: int = 100,
    seed_start: int = 2000,
    output_json: str = "eval_out/crash_diagnosis.json",
    target_speeds: list[float] | None = None,
) -> dict[str, Any]:
    """Run 100-episode evaluation and extract 5-step pre-impact telemetry for every crash."""
    device = torch.device("cpu")
    print(f"Loading checkpoint from: {model_path}")
    agent = DeepSetAgent().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    agent.load_state_dict(state_dict)
    agent.eval()

    env = make_deepset_env_v2(
        tier="full_adas",
        scenario="highway",
        seed=seed_start,
        target_speeds=target_speeds,
    )

    action_dim = int(env.action_space.n)
    crashes: list[dict[str, Any]] = []
    category_counts: dict[str, int] = collections.defaultdict(int)

    all_speeds_ms: list[float] = []
    all_lane_changes: list[int] = []
    all_distinct_lane_changes: list[int] = []
    all_rapid_reversals: list[int] = []
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
        rapid_reversals = 0
        lc_events: list[tuple[int, int, int]] = []

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
                if lc_events:
                    prev_lc_step, prev_lc_from, _ = lc_events[-1]
                    if cur_lane == prev_lc_from and (step - prev_lc_step) <= 5:
                        rapid_reversals += 1
                lc_events.append((step, prev_lane, cur_lane))
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
        all_rapid_reversals.append(rapid_reversals)
        all_distinct_lane_changes.append(max(0, lane_changes - rapid_reversals))
        all_returns.append(ep_return)

        if ep_crashed:
            # Identify the vehicle collided with
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

            # Build trajectory of collided vehicle across 5 steps before impact
            if collided_veh_data is not None and collided_veh_id is not None:
                traj = []
                for s in list(history)[-5:]:
                    m = [v for v in s["surrounding"] if v["id"] == collided_veh_id]
                    if m:
                        traj.append(m[0])
                collided_veh_data["history"] = traj

            floor_speed = float(target_speeds[0]) if target_speeds else 20.0
            category, explanation = categorize_crash(list(history), collided_veh_data, floor_speed=floor_speed)
            category_counts[category] += 1

            # Build 5-step pre-impact telemetry
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
    from parity.metrics import clopper_pearson
    ci_low, ci_high = clopper_pearson(total_crashes, n_episodes)

    summary = {
        "model_path": model_path,
        "n_episodes": n_episodes,
        "seed_start": seed_start,
        "seed_end": seed_start + n_episodes - 1,
        "total_crashes": total_crashes,
        "crash_rate_pct": (total_crashes / n_episodes) * 100.0,
        "crash_ci_low_pct": ci_low * 100.0,
        "crash_ci_high_pct": ci_high * 100.0,
        "mean_speed_ms": float(np.mean(all_speeds_ms)),
        "mean_speed_kmh": float(np.mean(all_speeds_ms)) * 3.6,
        "mean_lane_changes": float(np.mean(all_lane_changes)),
        "mean_distinct_lane_changes": float(np.mean(all_distinct_lane_changes)),
        "mean_rapid_reversals": float(np.mean(all_rapid_reversals)),
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
    print("CRASH-CAUSE BREAKDOWN SUMMARY")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes} (seeds {seed_start}..{seed_start + n_episodes - 1})")
    print(f"Total Crashes: {total_crashes}/{n_episodes} ({summary['crash_rate_pct']:.1f}%, 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%])")
    print(f"Mean Speed: {summary['mean_speed_ms']:.2f} m/s ({summary['mean_speed_kmh']:.1f} km/h)")
    print(f"Mean Lane Changes (raw crossings): {summary['mean_lane_changes']:.2f}")
    print(f"Mean Distinct LC Maneuvers (deduped): {summary['mean_distinct_lane_changes']:.2f}")
    print(f"Mean Rapid Reversals (<=5 steps): {summary['mean_rapid_reversals']:.2f}")
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
    parser.add_argument("--output-json", type=str, default="eval_out/crash_breakdown.json")
    parser.add_argument("--target-speeds", type=float, nargs="+", default=None,
                        help="Target speeds for DiscreteMetaAction (e.g. 10 15 20 25 30)")
    args = parser.parse_args()

    evaluate_and_diagnose(
        model_path=args.model_path,
        n_episodes=args.episodes,
        seed_start=args.seed_start,
        output_json=args.output_json,
        target_speeds=args.target_speeds,
    )

#!/usr/bin/env python3
"""Crash failure-mode diagnostic script for DeepSet v2.

Evaluates on the 100 benchmark seeds (2000-2099).
For every crashing episode:
- Logs seed, step, ego speed, lane, action, and lane-change status.
- Logs other vehicle's relative position (x, y), speed, and lane for the 5 steps leading to impact.
- Categorizes crash type: rear-end, side-swipe during lane change, blind-spot cut-in, cut-off, other.
- Renders HUD-annotated GIFs for representative crashes.
- Outputs complete structured diagnostic data to JSON.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import os
import sys
from typing import Any

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw

from deepset_v2.env import make_deepset_env_v2
from deepset_v2.model import DeepSetAgent

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def overlay_hud(
    frame: np.ndarray,
    step: int,
    speed_kmh: float,
    action: int,
    current_lane: int,
    crashed: bool,
    crash_type: str = "",
    seed: int = 0,
) -> Image.Image:
    """Overlay telemetry HUD onto rendered simulation frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    banner_height = 30
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(20, 24, 32))

    status_text = f"CRASH: {crash_type}" if crashed else "RUNNING"
    status_color = (255, 60, 60) if crashed else (50, 230, 110)

    telemetry_str = (
        f"Seed: {seed} | Step: {step:3d} | "
        f"Speed: {speed_kmh:4.1f} km/h | Lane: {current_lane} | "
        f"Act: {ACTION_NAMES.get(action, str(action))}"
    )

    draw.text((10, 8), telemetry_str, fill=(240, 245, 250))
    draw.text((img.width - 240, 8), status_text, fill=status_color)
    return img


def categorize_crash(
    history: list[dict[str, Any]],
    collided_veh_info: dict[str, Any] | None,
) -> tuple[str, str]:
    """Categorize crash into failure mode using pre-impact kinematics.

    Categories:
    - rear-end: ego hit rear of a lead vehicle in its lane
    - side-swipe during lane change: ego was executing a lane change and collided
    - blind-spot cut-in: neighbor vehicle moved into ego lane from adjacent
    - cut-off by vehicle ahead: vehicle ahead decelerated abruptly or cut in
    - other: miscellaneous contacts
    """
    if not history:
        return "other", "No step history available"

    final_step = history[-1]
    ego_speed = final_step["ego_speed_kmh"]
    lane_change_active = any(s["lane_change_in_progress"] for s in history[-3:])

    if lane_change_active:
        explanation = (
            f"Ego initiated lane change (lane {history[-3]['ego_lane']} -> {final_step['ego_lane']}) "
            f"and sideswiped adjacent traffic."
        )
        return "side-swipe during lane change", explanation

    if collided_veh_info is not None:
        rel_x = collided_veh_info.get("rel_x", 0.0)
        rel_y = collided_veh_info.get("rel_y", 0.0)
        other_speed = collided_veh_info.get("speed_kmh", 0.0)
        other_lane = collided_veh_info.get("lane", None)
        ego_lane = final_step["ego_lane"]

        # Check lead vehicle rear-end:
        # Other vehicle was ahead (rel_x > -2.0) and in same lane (|rel_y| < 2.5), ego traveling faster
        if rel_x > -2.0 and abs(rel_y) < 2.5 and ego_speed > other_speed:
            explanation = (
                f"Ego traveling at {ego_speed:.1f} km/h rear-ended lead vehicle (speed {other_speed:.1f} km/h, "
                f"rel_x={rel_x:+.1f}m, same lane {ego_lane}) without braking or changing lanes."
            )
            return "rear-end", explanation

        # Check cut-in: other vehicle was in adjacent lane earlier and moved across
        other_history = collided_veh_info.get("history", [])
        if len(other_history) >= 2:
            prev_rel_y = other_history[0].get("rel_y", rel_y)
            if abs(prev_rel_y) > 2.0 and abs(rel_y) <= 2.0:
                explanation = (
                    f"Surrounding vehicle cut into ego lane (lateral rel_y changed from {prev_rel_y:+.1f}m to {rel_y:+.1f}m)."
                )
                return "blind-spot cut-in", explanation

        # Other vehicle ahead but slower/braking
        if rel_x > 0.0 and abs(rel_y) < 3.0:
            explanation = (
                f"Ego closed distance on vehicle ahead (rel_x={rel_x:+.1f}m, delta_v={ego_speed - other_speed:+.1f} km/h) in lane {ego_lane}."
            )
            return "rear-end", explanation

    # Default to rear-end if ego was traveling fast in lane with no lane change
    if ego_speed > 60.0 and not lane_change_active:
        return "rear-end", f"Ego traveling at high speed ({ego_speed:.1f} km/h) failed to brake for traffic ahead."

    return "other", "Complex multi-vehicle interaction."


def run_diagnostics(
    model_path: str,
    episodes: int = 100,
    seed_start: int = 2000,
    output_dir: str = "eval_out/crash_diagnostics",
    render_gifs: bool = True,
    max_gifs: int = 5,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    gif_dir = os.path.join(output_dir, "gifs")
    if render_gifs:
        os.makedirs(gif_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} onto {device}...")
    agent = DeepSetAgent().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    agent.load_state_dict(state_dict)
    agent.eval()

    env = make_deepset_env_v2(
        tier="full_adas",
        scenario="highway",
        seed=seed_start,
        render_mode="rgb_array",
    )
    env.unwrapped.config["offscreen_rendering"] = True

    action_dim = int(env.action_space.n)
    crashes: list[dict[str, Any]] = []
    category_counts: dict[str, int] = collections.defaultdict(int)
    gifs_saved = 0

    print(f"Evaluating {episodes} seeds ({seed_start}-{seed_start+episodes-1})...")

    for ep in range(episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        step = 0
        crashed = False
        frames: list[np.ndarray] = []
        hud_frames: list[Image.Image] = []

        history_buffer: collections.deque[dict[str, Any]] = collections.deque(maxlen=10)
        previous_lane = (
            env.unwrapped.vehicle.lane_index[2]
            if hasattr(env.unwrapped, "vehicle") and hasattr(env.unwrapped.vehicle, "lane_index")
            else 0
        )

        while True:
            ego = env.unwrapped.vehicle
            ego_speed_kmh = float(ego.speed * 3.6)
            current_lane = int(ego.lane_index[2]) if hasattr(ego, "lane_index") else 0

            # Check if lane change is underway
            target_lane = getattr(ego, "target_lane_index", None)
            target_idx = target_lane[2] if target_lane is not None else current_lane
            lane_change_in_progress = (target_idx != current_lane) or (current_lane != previous_lane)

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_tensor, deterministic=True, action_mask=mask_t
                )
                action_idx = int(action.item())

            # Store step state in buffer
            # Gather other vehicles kinematics
            surrounding_info: list[dict[str, Any]] = []
            for v in env.unwrapped.road.vehicles:
                if v is not ego:
                    surrounding_info.append({
                        "id": id(v),
                        "rel_x": float(v.position[0] - ego.position[0]),
                        "rel_y": float(v.position[1] - ego.position[1]),
                        "speed_kmh": float(v.speed * 3.6),
                        "lane": int(v.lane_index[2]) if hasattr(v, "lane_index") else None,
                        "crashed": bool(getattr(v, "crashed", False)),
                    })

            step_record = {
                "step": step,
                "ego_speed_kmh": ego_speed_kmh,
                "ego_lane": current_lane,
                "action": action_idx,
                "action_name": ACTION_NAMES.get(action_idx, str(action_idx)),
                "lane_change_in_progress": lane_change_in_progress,
                "surrounding": surrounding_info,
            }
            history_buffer.append(step_record)

            if render_gifs and gifs_saved < max_gifs:
                raw_frame = env.render()
                if raw_frame is not None:
                    frames.append(raw_frame)
                    hud_frame = overlay_hud(
                        raw_frame,
                        step=step,
                        speed_kmh=ego_speed_kmh,
                        action=action_idx,
                        current_lane=current_lane,
                        crashed=False,
                        seed=seed,
                    )
                    hud_frames.append(hud_frame)

            obs, reward, terminated, truncated, info = env.step(action_idx)
            step += 1
            previous_lane = current_lane

            if info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False):
                crashed = True

            if terminated or truncated:
                break

        if crashed:
            # Identify collided vehicle
            collided_veh_id = None
            closest_dist = float("inf")
            collided_veh_data: dict[str, Any] | None = None

            # Look at final surrounding info
            final_surrounding = history_buffer[-1]["surrounding"] if history_buffer else []
            for sv in final_surrounding:
                dist = math.hypot(sv["rel_x"], sv["rel_y"])
                if sv["crashed"] or dist < closest_dist:
                    closest_dist = dist
                    collided_veh_id = sv["id"]
                    collided_veh_data = copy.deepcopy(sv)

            # Build vehicle trajectory across the 5 steps before crash
            if collided_veh_data is not None and collided_veh_id is not None:
                veh_traj = []
                for s in list(history_buffer)[-5:]:
                    matching = [v for v in s["surrounding"] if v["id"] == collided_veh_id]
                    if matching:
                        veh_traj.append(matching[0])
                collided_veh_data["history"] = veh_traj

            crash_category, explanation = categorize_crash(list(history_buffer), collided_veh_data)
            category_counts[crash_category] += 1

            # Extract 5 steps prior to impact
            pre_impact_steps = []
            for s in list(history_buffer)[-5:]:
                s_copy = copy.deepcopy(s)
                # Filter surrounding to just the collided vehicle if available
                if collided_veh_id is not None:
                    s_copy["target_vehicle"] = [v for v in s_copy["surrounding"] if v["id"] == collided_veh_id]
                del s_copy["surrounding"]
                pre_impact_steps.append(s_copy)

            crash_entry = {
                "seed": seed,
                "crash_step": step,
                "crash_category": crash_category,
                "explanation": explanation,
                "final_ego_speed_kmh": history_buffer[-1]["ego_speed_kmh"] if history_buffer else 0.0,
                "final_ego_lane": history_buffer[-1]["ego_lane"] if history_buffer else 0,
                "lane_change_in_progress": history_buffer[-1]["lane_change_in_progress"] if history_buffer else False,
                "collided_vehicle": collided_veh_data,
                "pre_impact_5_steps": pre_impact_steps,
            }
            crashes.append(crash_entry)
            print(f"  [Crash #{len(crashes)}] Seed {seed} @ Step {step}: {crash_category} -- {explanation}")

            # Save GIF if under limit
            if render_gifs and gifs_saved < max_gifs and hud_frames:
                gif_filename = f"crash_seed{seed}_{crash_category.replace(' ', '_')}.gif"
                gif_path = os.path.join(gif_dir, gif_filename)
                # Keep last 40 frames (~8 seconds) leading to crash
                clip_frames = hud_frames[-45:]
                if clip_frames:
                    clip_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=clip_frames[1:],
                        duration=120,
                        loop=0,
                        optimize=True,
                    )
                    crash_entry["gif_path"] = gif_path
                    gifs_saved += 1
                    print(f"    -> Saved GIF to {gif_path}")

    env.close()

    report: dict[str, Any] = {
        "model_path": model_path,
        "total_episodes": episodes,
        "total_crashes": len(crashes),
        "crash_rate_pct": (len(crashes) / episodes) * 100.0,
        "category_counts": dict(category_counts),
        "crashes": crashes,
    }

    report_path = os.path.join(output_dir, "crash_diagnosis_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Complete diagnostic report written to {report_path}")

    print("\n" + "=" * 65)
    print("CRASH FAILURE-MODE BREAKDOWN SUMMARY")
    print("=" * 65)
    print(f"Total Evaluated: {episodes} episodes | Crashes: {len(crashes)} ({len(crashes)/episodes*100:.1f}%)")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(crashes)) * 100.0 if crashes else 0.0
        print(f"  - {cat:<32}: {count:>2} crashes ({pct:>5.1f}%)")
    print("=" * 65)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="scratch/seed_replicates/highway_1m_seed42/deepset_v2_highway_best.pt",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="eval_out/crash_diagnostics")
    parser.add_argument("--max-gifs", type=int, default=5)
    args = parser.parse_args()

    run_diagnostics(
        model_path=args.model_path,
        episodes=args.episodes,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        render_gifs=True,
        max_gifs=args.max_gifs,
    )

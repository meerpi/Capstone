#!/usr/bin/env python3
"""Verify lane-change dynamics for the Optimal Overtaker Final checkpoint.

Renders 5 episodes as GIF/MP4 with HUD overlay.
Analyzes the step-by-step lane-change sequence for rapid back-and-forth oscillations,
boundary crossing intervals, and traffic-responsive tactical repositioning.
"""

import os
import sys
import shutil
import json
import math
from typing import Any
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import env_config
from train_optimal_overtaker import OptimalAgent

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
    target_lane: int,
    overtakes: int,
    lead_dist: float | None,
    lead_speed_kmh: float | None,
    crashed: bool,
    max_steps: int = 500,
) -> Image.Image:
    """Overlay telemetry HUD onto rendered simulation frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    banner_height = 30
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(16, 20, 28))

    status_text = "CRASH" if crashed else "SURVIVING"
    status_color = (255, 60, 60) if crashed else (50, 230, 110)

    lead_str = f"Lead: {lead_dist:.1f}m ({lead_speed_kmh:.0f}km/h)" if lead_dist is not None else "Lead: Clear"

    telemetry_str = (
        f"Step: {step:3d}/{max_steps} | "
        f"Spd: {speed_kmh:4.1f} km/h | Lane: {current_lane} (tgt:{target_lane}) | "
        f"Act: {ACTION_NAMES.get(action, str(action)):<10} | {lead_str} | OT: {overtakes}"
    )

    draw.text((10, 8), telemetry_str, fill=(240, 245, 250))
    draw.text((img.width - 90, 8), status_text, fill=status_color)
    return img


def get_rendered_frame(env) -> np.ndarray:
    """Safely render frame from environment."""
    frame = env.render()
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "viewer") and unwrapped.viewer is not None:
        if not unwrapped.viewer.enabled:
            unwrapped.viewer.enabled = True
            frame = env.render()
    return frame


def analyze_and_record_episodes(
    model_path: str = "scratch/optimal_overtaker_extended_speeds/final.pt",
    seeds: list[int] = [2000, 2001, 2002, 2003, 2004],
    max_steps: int = 500,
    output_dir: str = "visualizations/lane_change_verification",
    artifact_dir: str | None = "/home/meerpi/.gemini/antigravity-ide/brain/14ddad04-ca6f-4784-91b9-fd67d1dc5404",
    target_speeds: list[float] = [10.0, 15.0, 20.0, 25.0, 30.0],
    fps: int = 15,
) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    obs_dim = checkpoint.get("obs_dim", 81)
    action_dim = checkpoint.get("action_dim", 5)

    agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    episode_analyses = []

    for seed in seeds:
        duration_sec = int(np.ceil(max_steps / 5)) + 10
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            seed=seed,
            frame_stack_k=3,
            duration=duration_sec,
            render_mode="rgb_array",
            target_speeds=target_speeds,
        )
        env.unwrapped.config["offscreen_rendering"] = True

        obs, info = env.reset(seed=seed)
        frames: list[Image.Image] = []
        step_count = 0
        total_overtakes = 0
        crashed = False

        ego = env.unwrapped.vehicle
        prev_lane = int(ego.lane_index[2]) if hasattr(ego, "lane_index") else 0

        # Detailed step records
        steps_log = []
        lane_changes = []

        while step_count < max_steps:
            raw_frame = get_rendered_frame(env)
            ego = env.unwrapped.vehicle
            speed_mps = float(ego.speed)
            speed_kmh = speed_mps * 3.6
            cur_lane = int(ego.lane_index[2]) if hasattr(ego, "lane_index") else 0

            target_lane_obj = getattr(ego, "target_lane_index", None)
            target_lane_idx = int(target_lane_obj[2]) if target_lane_obj is not None else cur_lane

            total_overtakes += int(info.get("overtake_count", 0))

            # Find lead vehicle in current lane
            lead_dist = None
            lead_speed_kmh = None
            for v in env.unwrapped.road.vehicles:
                if v is not ego:
                    v_lane = int(v.lane_index[2]) if hasattr(v, "lane_index") else None
                    rel_x = float(v.position[0] - ego.position[0])
                    rel_y = float(v.position[1] - ego.position[1])
                    if v_lane == cur_lane and rel_x > 0:
                        if lead_dist is None or rel_x < lead_dist:
                            lead_dist = rel_x
                            lead_speed_kmh = float(v.speed * 3.6)

            # Check if lane change occurred at this step
            if cur_lane != prev_lane:
                lc_event = {
                    "step": step_count,
                    "from_lane": prev_lane,
                    "to_lane": cur_lane,
                    "speed_kmh": speed_kmh,
                    "lead_dist_before": steps_log[-1]["lead_dist"] if steps_log else None,
                    "lead_speed_before": steps_log[-1]["lead_speed_kmh"] if steps_log else None,
                    "action": steps_log[-1]["action_name"] if steps_log else None,
                }
                lane_changes.append(lc_event)
                prev_lane = cur_lane

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_tensor, deterministic=True, action_mask=mask_t
                )
                action_idx = int(action.item())

            frame_hud = overlay_hud(
                raw_frame,
                step=step_count,
                speed_kmh=speed_kmh,
                action=action_idx,
                current_lane=cur_lane,
                target_lane=target_lane_idx,
                overtakes=total_overtakes,
                lead_dist=lead_dist,
                lead_speed_kmh=lead_speed_kmh,
                crashed=crashed,
                max_steps=max_steps,
            )
            frames.append(frame_hud)

            steps_log.append({
                "step": step_count,
                "lane": cur_lane,
                "target_lane": target_lane_idx,
                "action": action_idx,
                "action_name": ACTION_NAMES.get(action_idx, str(action_idx)),
                "speed_kmh": speed_kmh,
                "lead_dist": lead_dist,
                "lead_speed_kmh": lead_speed_kmh,
            })

            obs, reward, term, trunc, info = env.step(action_idx)
            step_count += 1

            if info.get("crashed", False) or getattr(ego, "crashed", False):
                crashed = True

            if term or trunc:
                final_raw = get_rendered_frame(env)
                frames.append(
                    overlay_hud(
                        final_raw,
                        step=step_count,
                        speed_kmh=speed_kmh,
                        action=action_idx,
                        current_lane=cur_lane,
                        target_lane=target_lane_idx,
                        overtakes=total_overtakes,
                        lead_dist=lead_dist,
                        lead_speed_kmh=lead_speed_kmh,
                        crashed=crashed,
                        max_steps=max_steps,
                    )
                )
                break

        env.close()

        # Save GIF & MP4
        gif_filename = f"optimal_final_seed{seed}.gif"
        mp4_filename = f"optimal_final_seed{seed}.mp4"
        output_gif = os.path.join(output_dir, gif_filename)
        output_mp4 = os.path.join(output_dir, mp4_filename)

        frame_duration = int(1000 / fps)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True,
        )

        try:
            import imageio
            with imageio.get_writer(output_mp4, fps=fps, codec="libx264", quality=8) as writer:
                for f in frames:
                    writer.append_data(np.array(f))
        except Exception as e:
            print(f"Error saving mp4: {e}")

        # Copy to artifact directory
        if artifact_dir:
            shutil.copy2(output_gif, os.path.join(artifact_dir, gif_filename))
            if os.path.exists(output_mp4):
                shutil.copy2(output_mp4, os.path.join(artifact_dir, mp4_filename))

        # Analyze lane change sequences
        # 1. Intervals between consecutive lane changes
        intervals = []
        for i in range(1, len(lane_changes)):
            delta = lane_changes[i]["step"] - lane_changes[i - 1]["step"]
            intervals.append(delta)

        # 2. Oscillations / rapid reversals: L1 -> L2 -> L1
        oscillations_3 = 0  # within <= 3 steps
        oscillations_5 = 0  # within <= 5 steps
        oscillations_10 = 0 # within <= 10 steps
        rapid_events = []

        for i in range(1, len(lane_changes)):
            prev_lc = lane_changes[i - 1]
            curr_lc = lane_changes[i]
            # Did it cross back to the previous lane?
            if curr_lc["to_lane"] == prev_lc["from_lane"]:
                delta_t = curr_lc["step"] - prev_lc["step"]
                if delta_t <= 3:
                    oscillations_3 += 1
                if delta_t <= 5:
                    oscillations_5 += 1
                if delta_t <= 10:
                    oscillations_10 += 1
                    rapid_events.append({
                        "step": curr_lc["step"],
                        "delta_steps": delta_t,
                        "lane_cycle": f"{prev_lc['from_lane']} -> {prev_lc['to_lane']} -> {curr_lc['to_lane']}",
                    })

        # 3. Purposefulness: what fraction of lane changes occurred when a lead vehicle was present within 45m?
        purposeful_count = 0
        for lc in lane_changes:
            ld = lc.get("lead_dist_before")
            if ld is not None and ld < 45.0:
                purposeful_count += 1

        mean_int = float(np.mean(intervals)) if intervals else None
        min_int = min(intervals) if intervals else None

        ep_analysis = {
            "seed": seed,
            "total_steps": step_count,
            "crashed": crashed,
            "total_overtakes": total_overtakes,
            "total_lane_changes": len(lane_changes),
            "lane_changes_per_step": len(lane_changes) / max(step_count, 1),
            "min_interval_steps": min_int,
            "mean_interval_steps": mean_int,
            "median_interval_steps": float(np.median(intervals)) if intervals else None,
            "oscillations_within_3_steps": oscillations_3,
            "oscillations_within_5_steps": oscillations_5,
            "oscillations_within_10_steps": oscillations_10,
            "rapid_events": rapid_events,
            "purposeful_overtake_repositions": purposeful_count,
            "purposeful_ratio": purposeful_count / len(lane_changes) if lane_changes else 1.0,
            "lane_changes": lane_changes,
            "gif_path": output_gif,
            "mp4_path": output_mp4,
        }
        episode_analyses.append(ep_analysis)
        mean_str = f"{mean_int:.1f}" if mean_int is not None else "N/A"
        print(f"Seed {seed:4d}: {len(lane_changes):2d} lane changes, min_interval={min_int}, "
              f"mean_interval={mean_str} steps, "
              f"oscillations<=5: {oscillations_5}, purposeful={purposeful_count}/{len(lane_changes)}")

    summary_file = os.path.join(output_dir, "lane_change_dynamics_analysis.json")
    with open(summary_file, "w") as f:
        json.dump(episode_analyses, f, indent=2)

    return {
        "episodes": episode_analyses,
        "summary_file": summary_file,
    }


if __name__ == "__main__":
    analyze_and_record_episodes()

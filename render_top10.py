"""Batch render top 10 optimal overtaking demonstration seeds to GIF and MP4.

Renders full 500-step (100-second) episodes with real-time telemetry HUD:
- Step count / Total steps
- Real-time ego vehicle speed (km/h)
- Mean NPC traffic flow speed (km/h)
- Relative speed (ego - flow) (km/h)
- Current lane index
- Discrete action executed
- Cumulative strictly adjacent overtakes
- Lane changes count
- Cumulative % of steps at the floor setpoint (20 m/s / 72 km/h)
- Survival / collision status
"""

import os
import shutil
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
import subprocess
import env_config
from train_optimal_overtaker import OptimalAgent
from render_utils import (
    apply_anti_alias_stripes,
    get_rendered_frame,
    compute_mean_npc_speed,
    frame_duration_ms,
    FAST_FPS,
)

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
    ego_speed_kmh: float,
    flow_speed_kmh: float,
    rel_speed_kmh: float,
    floor_pct: float,
    action: int,
    current_lane: int,
    overtakes: int,
    lane_changes: int,
    crashed: bool,
    max_steps: int = 500,
    seed: int = 0,
    rank: int = 0,
) -> Image.Image:
    """Overlay rich telemetry HUD onto rendered simulation frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    banner_height = 28
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(16, 20, 28))

    status_text = "CRASHED" if crashed else "SURVIVING"
    status_color = (255, 60, 60) if crashed else (50, 230, 110)

    act_str = ACTION_NAMES.get(action, str(action))
    telemetry_str = (
        f"TOP #{rank} (S{seed}) | "
        f"{step:3d}/{max_steps} | "
        f"Ego:{ego_speed_kmh:4.1f} Flow:{flow_speed_kmh:4.1f} Rel:{rel_speed_kmh:+4.1f}kph | "
        f"L{current_lane} {act_str:<9} | "
        f"OT:{overtakes:2d} LC:{lane_changes:2d} Flr:{floor_pct:2.0f}%"
    )

    draw.text((8, 7), telemetry_str, fill=(240, 245, 250))
    draw.text((img.width - 82, 7), status_text, fill=status_color)
    return img


def render_seed_episode(
    agent: OptimalAgent,
    seed: int,
    rank: int,
    output_dir: str,
    artifact_dir: str | None = None,
    max_steps: int = 500,
    fps: int = FAST_FPS,
    realtime: bool = False,
    device: str = "cpu",
) -> dict:
    apply_anti_alias_stripes()

    os.makedirs(output_dir, exist_ok=True)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)

    gif_filename = f"ppo_top_{rank}_seed_{seed}.gif"
    mp4_filename = f"ppo_top_{rank}_seed_{seed}.mp4"

    local_gif = os.path.join(output_dir, gif_filename)
    local_mp4 = os.path.join(output_dir, mp4_filename)

    env = env_config.make_optimal_env(
        lanes_count=4,
        vehicles_density=1.4,
        vehicles_count=14,
        seed=seed,
        frame_stack_k=3,
        duration=110,
        render_mode="rgb_array",
    )
    env.unwrapped.config["offscreen_rendering"] = True

    obs, info = env.reset(seed=seed)
    frames: list[Image.Image] = []
    actions: list[int] = []
    speeds: list[float] = []
    flow_speeds: list[float] = []
    rel_speeds: list[float] = []

    step_count = 0
    total_overtakes = 0
    lane_changes = 0
    floor_steps = 0
    prev_lane = env.unwrapped.vehicle.lane_index[2]
    crashed = False

    while step_count < max_steps:
        raw_frame = get_rendered_frame(env)
        speed_mps = float(info.get("speed", env.unwrapped.vehicle.speed))
        speed_kmh = speed_mps * 3.6
        speeds.append(speed_kmh)

        flow_mps = compute_mean_npc_speed(env)
        flow_kmh = flow_mps * 3.6
        flow_speeds.append(flow_kmh)

        rel_kmh = speed_kmh - flow_kmh
        rel_speeds.append(rel_kmh)

        # Track floor setpoint steps (20 m/s target speed)
        target_spd = getattr(env.unwrapped.vehicle, "target_speed", 25.0)
        if target_spd <= 20.1:
            floor_steps += 1
        floor_pct = (floor_steps / max(1, step_count + 1)) * 100.0

        current_lane = env.unwrapped.vehicle.lane_index[2]

        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask = info.get("action_mask", np.ones(5, dtype=bool))
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                obs_t, deterministic=True, action_mask=mask_t
            )
            act = int(action.item())
        actions.append(act)

        hud_frame = overlay_hud(
            raw_frame,
            step=step_count,
            ego_speed_kmh=speed_kmh,
            flow_speed_kmh=flow_kmh,
            rel_speed_kmh=rel_kmh,
            floor_pct=floor_pct,
            action=act,
            current_lane=current_lane,
            overtakes=total_overtakes,
            lane_changes=lane_changes,
            crashed=crashed,
            max_steps=max_steps,
            seed=seed,
            rank=rank,
        )
        frames.append(hud_frame)

        obs, reward, term, trunc, info = env.step(act)
        step_count += 1
        total_overtakes += int(info.get("overtake_count", 0))

        if env.unwrapped.vehicle.lane_index[2] != prev_lane:
            lane_changes += 1
            prev_lane = env.unwrapped.vehicle.lane_index[2]

        if info.get("crashed", False):
            crashed = True

        if term or trunc:
            final_raw = get_rendered_frame(env)
            hud_frame = overlay_hud(
                final_raw,
                step=step_count,
                ego_speed_kmh=speed_kmh,
                flow_speed_kmh=flow_kmh,
                rel_speed_kmh=rel_kmh,
                floor_pct=floor_pct,
                action=act,
                current_lane=current_lane,
                overtakes=total_overtakes,
                lane_changes=lane_changes,
                crashed=crashed,
                max_steps=max_steps,
                seed=seed,
                rank=rank,
            )
            frames.append(hud_frame)
            break

    env.close()

    # Determine frame duration and playback rate
    eff_fps = 5 if realtime else fps
    duration_ms = frame_duration_ms(realtime=realtime)

    # Save GIF
    frames[0].save(
        local_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    # Save MP4 via ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", local_gif,
        "-r", str(eff_fps),
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        "-movflags", "+faststart", local_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Copy to artifact dir if provided
    if artifact_dir and os.path.exists(artifact_dir):
        shutil.copy2(local_gif, os.path.join(artifact_dir, gif_filename))
        shutil.copy2(local_mp4, os.path.join(artifact_dir, mp4_filename))

    summary = {
        "rank": rank,
        "seed": seed,
        "steps": step_count,
        "overtakes": total_overtakes,
        "lane_changes": lane_changes,
        "faster_cnt": actions.count(3),
        "idle_cnt": actions.count(1),
        "slower_cnt": actions.count(4),
        "mean_speed_kmh": round(float(np.mean(speeds)), 1) if speeds else 0.0,
        "max_speed_kmh": round(float(np.max(speeds)), 1) if speeds else 0.0,
        "mean_flow_speed_kmh": round(float(np.mean(flow_speeds)), 1) if flow_speeds else 0.0,
        "mean_rel_speed_kmh": round(float(np.mean(rel_speeds)), 1) if rel_speeds else 0.0,
        "pct_steps_at_floor_setpoint": round((floor_steps / max(1, step_count)) * 100.0, 1),
        "playback_mode": "realtime" if realtime else "fast",
        "frame_duration_ms": duration_ms,
        "playback_speed": "1.0x (200ms/frame, 5fps)" if realtime else f"{1000/(duration_ms*5):.1f}x ({duration_ms}ms/frame, {eff_fps}fps)",
        "crashed": crashed,
        "gif_file": gif_filename,
        "mp4_file": mp4_filename,
    }
    print(
        f"Completed Rank #{rank} (Seed {seed}): OT={total_overtakes}, LC={lane_changes}, "
        f"EgoSpd={summary['mean_speed_kmh']} km/h, FlowSpd={summary['mean_flow_speed_kmh']} km/h, "
        f"RelSpd={summary['mean_rel_speed_kmh']:+.1f} km/h, Floor={summary['pct_steps_at_floor_setpoint']}%, "
        f"Crashed={crashed}"
    )
    return summary

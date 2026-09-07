"""Batch render top 10 optimal overtaking demonstration seeds to GIF and MP4.

Renders full 500-step (100-second) episodes with real-time telemetry HUD:
- Step count / Total steps
- Real-time ego vehicle speed (km/h)
- Current lane index
- Discrete action executed
- Cumulative strictly adjacent overtakes
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
        f"TOP #{rank} (Seed {seed}) | "
        f"Step: {step:3d}/{max_steps} | "
        f"{speed_kmh:4.1f} km/h | Lane {current_lane} | "
        f"Act: {act_str:<10} | "
        f"Overtakes: {overtakes:2d} | LC: {lane_changes:2d}"
    )

    draw.text((10, 7), telemetry_str, fill=(240, 245, 250))
    draw.text((img.width - 85, 7), status_text, fill=status_color)
    return img


def render_seed_episode(
    agent: OptimalAgent,
    seed: int,
    rank: int,
    output_dir: str,
    artifact_dir: str,
    max_steps: int = 500,
    fps: int = 15,
    device: str = "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    gif_filename = f"ppo_top_{rank}_seed_{seed}.gif"
    mp4_filename = f"ppo_top_{rank}_seed_{seed}.mp4"

    local_gif = os.path.join(output_dir, gif_filename)
    local_mp4 = os.path.join(output_dir, mp4_filename)
    art_gif = os.path.join(artifact_dir, gif_filename)
    art_mp4 = os.path.join(artifact_dir, mp4_filename)

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

    step_count = 0
    total_overtakes = 0
    lane_changes = 0
    prev_lane = env.unwrapped.vehicle.lane_index[2]
    crashed = False

    while step_count < max_steps:
        raw_frame = env.render()
        speed_mps = float(info.get("speed", env.unwrapped.vehicle.speed))
        speed_kmh = speed_mps * 3.6
        speeds.append(speed_kmh)
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
            speed_kmh=speed_kmh,
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
            final_raw = env.render()
            hud_frame = overlay_hud(
                final_raw,
                step=step_count,
                speed_kmh=speed_kmh,
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

    # Save GIF
    frame_duration = int(1000 / fps)
    frames[0].save(
        local_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )

    # Save MP4 via ffmpeg
    cmd = [
        "ffmpeg", "-y", "-i", local_gif,
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        "-movflags", "+faststart", local_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Copy to artifact dir
    shutil.copy2(local_gif, art_gif)
    shutil.copy2(local_mp4, art_mp4)

    summary = {
        "rank": rank,
        "seed": seed,
        "steps": step_count,
        "overtakes": total_overtakes,
        "lane_changes": lane_changes,
        "faster_cnt": actions.count(3),
        "idle_cnt": actions.count(1),
        "slower_cnt": actions.count(4),
        "mean_speed_kmh": round(float(np.mean(speeds)), 1),
        "max_speed_kmh": round(float(np.max(speeds)), 1),
        "crashed": crashed,
        "gif_file": gif_filename,
        "mp4_file": mp4_filename,
    }
    print(f"Completed Rank #{rank} (Seed {seed}): OT={total_overtakes}, LC={lane_changes}, FASTER={actions.count(3)}, MeanSpd={summary['mean_speed_kmh']} km/h, Crashed={crashed}")
    return summary

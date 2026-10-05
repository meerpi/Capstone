"""Visual Representation Generator for Sensor-Budgeted Highway Driving.

Generates high-definition animated GIF and MP4 recordings of trained agents (PPO-LSTM, Classical MOBIL)
executing highway driving episodes with telemetry overlays (speed, action, lane, sensor tier).
Fixes headless offscreen rendering to prevent black screens caused by SDL dummy driver disabling graphics.
"""

import os
import subprocess
import uuid
import concurrent.futures
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import env_config
from ppo_lstm import RecurrentAgent
from render_utils import apply_anti_alias_stripes, get_rendered_frame

ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def draw_telemetry(
    frame: np.ndarray,
    step: int,
    action: int,
    speed_kmh: float,
    current_lane: int,
    tier: str,
    overtakes: int,
    crashed: bool,
    max_steps: int = 500,
) -> Image.Image:
    """Overlay telemetry HUD onto rendered simulation frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    banner_height = 24
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(25, 25, 30))

    status_text = "CRASH" if crashed else "ACTIVE"
    status_color = (255, 70, 70) if crashed else (80, 240, 100)

    telemetry_str = (
        f"Step: {step:3d}/{max_steps} | Tier: {tier.upper()} | "
        f"Speed: {speed_kmh:4.1f} km/h | Lane: {current_lane} | "
        f"Act: {ACTION_NAMES.get(action, str(action)):<10} | "
        f"Overtakes: {overtakes}"
    )

    draw.text((8, 5), telemetry_str, fill=(240, 240, 240))
    draw.text((img.width - 65, 5), status_text, fill=status_color)
    return img


def save_video_artifacts(frames: list[Image.Image], output_gif: str, output_mp4: str | None, fps: int = 20) -> None:
    """Save captured frames to optimized GIF and universally playable H.264 MP4."""
    if not frames:
        print("Warning: No frames to save.")
        return

    # 1. Save animated GIF
    if output_gif:
        os.makedirs(os.path.dirname(os.path.abspath(output_gif)), exist_ok=True)
        frame_duration = int(1000 / fps)
        frames[0].save(
            output_gif,
            save_all=True,
            append_images=frames[1:],
            duration=frame_duration,
            loop=0,
            optimize=True,
        )
        print(f"Saved animated GIF: {output_gif} ({len(frames)} frames at {fps} FPS)")

    # 2. Save MP4 with 2x nearest-neighbor scaling for high-definition clarity
    if output_mp4:
        os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
        temp_dir = f"/tmp/render_frames_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        os.makedirs(temp_dir, exist_ok=True)
        temp_pattern = os.path.join(temp_dir, "frame_%04d.png")

        for i, f in enumerate(frames):
            f.save(temp_pattern % i)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", temp_pattern,
            "-vf", "scale=1200:300:flags=neighbor",
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.0",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-movflags", "+faststart",
            output_mp4,
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"Saved MP4 video: {output_mp4} ({len(frames)} frames at {fps} FPS)")

        # Clean up temporary frames
        for i in range(len(frames)):
            try:
                os.remove(temp_pattern % i)
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def record_episode(
    model_path: str,
    tier: str = "front_only",
    seed: int = 2015,
    output_gif: str = "visualizations/ppo_lstm_front_only_seed2015.gif",
    output_mp4: str | None = "visualizations/ppo_lstm_front_only_seed2015.mp4",
    device_name: str = "cpu",
    fps: int = 20,
    max_steps: int = 500,
) -> dict:
    apply_anti_alias_stripes()
    device = torch.device(device_name)
    state_dict = torch.load(model_path, map_location=device)
    obs_dim_ckpt = state_dict["actor.feature_net.0.weight"].shape[1]
    veh_count = obs_dim_ckpt // 5

    env_kwargs = {"scenario": "highway", "tier": tier, "seed": seed, "render_mode": "rgb_array", "tactical_overtaking": True}
    if veh_count != 15:
        env_kwargs["observation"] = {"vehicles_count": veh_count}

    env = env_config.make_env(**env_kwargs)
    env.unwrapped.config["offscreen_rendering"] = True

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agent = RecurrentAgent(obs_dim, action_dim, hidden_dim=128).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    obs_np, info = env.reset(seed=seed)
    actor_state, _ = agent.get_initial_states(batch_size=1, device=device)

    frames: list[Image.Image] = []
    lane_changes = 0
    overtakes = 0
    step_count = 0
    unwrapped = env.unwrapped
    last_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]
    done = False
    crashed = False

    while not done and step_count < max_steps:
        raw_frame = get_rendered_frame(env)

        obs_tensor = torch.tensor(obs_np.reshape(1, obs_dim), dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            hidden = agent.actor.feature_net(obs_tensor)
            h_step, actor_state = agent.actor.lstm(hidden.unsqueeze(0), actor_state)
            logits = agent.actor.actor_head(h_step.squeeze(0))
            logits = torch.where(mask_tensor, logits, torch.tensor(-1e8, device=device, dtype=logits.dtype))
            action = int(torch.argmax(logits, dim=-1).item())

        obs_np, reward, term, trunc, info = env.step(action)
        step_count += 1
        done = term or trunc
        crashed = info.get("crashed", False)

        veh = unwrapped.vehicle
        speed_kmh = float(getattr(veh, "speed", 0.0)) * 3.6
        current_lane = getattr(veh, "lane_index", [0, 0, 0])[2]
        if current_lane != last_lane:
            lane_changes += 1
            last_lane = current_lane

        overtakes += info.get("overtake_count", 0)

        if raw_frame is not None:
            annotated_img = draw_telemetry(
                raw_frame,
                step=step_count,
                action=action,
                speed_kmh=speed_kmh,
                current_lane=current_lane,
                tier=f"PPO-LSTM-{tier}",
                overtakes=overtakes,
                crashed=crashed,
                max_steps=max_steps,
            )
            frames.append(annotated_img)

    env.close()
    save_video_artifacts(frames, output_gif, output_mp4, fps=fps)

    return {
        "steps": step_count,
        "lane_changes": lane_changes,
        "overtakes": overtakes,
        "crashed": crashed,
        "frames": len(frames),
    }


def record_ppo_episode(
    model_path: str,
    tier: str = "front_only",
    seed: int = 2015,
    output_gif: str = "visualizations/ppo_front_only_seed2015.gif",
    output_mp4: str | None = "visualizations/ppo_front_only_seed2015.mp4",
    device_name: str = "cpu",
    fps: int = 20,
    max_steps: int = 500,
) -> dict:
    """Record an episode of trained Feedforward PPO and export to GIF and MP4."""
    apply_anti_alias_stripes()
    import ppo
    device = torch.device(device_name)
    env = env_config.make_env(
        scenario="highway",
        tier=tier,
        seed=seed,
        render_mode="rgb_array",
        tactical_overtaking=True,
    )
    env.unwrapped.config["offscreen_rendering"] = True

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n

    agent = ppo.Agent(obs_dim, action_dim).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs_np, info = env.reset(seed=seed)

    frames: list[Image.Image] = []
    lane_changes = 0
    overtakes = 0
    step_count = 0
    unwrapped = env.unwrapped
    last_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]
    done = False
    crashed = False

    while not done and step_count < max_steps:
        raw_frame = get_rendered_frame(env)

        obs_tensor = torch.tensor(obs_np.reshape(1, obs_dim), dtype=torch.float32, device=device)
        mask_tensor = torch.tensor(info["action_mask"], dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = agent.actor(obs_tensor)
            logits = torch.where(mask_tensor, logits, torch.tensor(-1e8, device=device, dtype=logits.dtype))
            action = int(torch.argmax(logits, dim=-1).item())

        obs_np, reward, term, trunc, info = env.step(action)
        step_count += 1
        done = term or trunc
        crashed = info.get("crashed", False)

        veh = unwrapped.vehicle
        speed_kmh = float(getattr(veh, "speed", 0.0)) * 3.6
        current_lane = getattr(veh, "lane_index", [0, 0, 0])[2]
        if current_lane != last_lane:
            lane_changes += 1
            last_lane = current_lane

        overtakes += info.get("overtake_count", 0)

        if raw_frame is not None:
            annotated_img = draw_telemetry(
                raw_frame,
                step=step_count,
                action=action,
                speed_kmh=speed_kmh,
                current_lane=current_lane,
                tier=f"PPO-{tier}",
                overtakes=overtakes,
                crashed=crashed,
                max_steps=max_steps,
            )
            frames.append(annotated_img)

    env.close()
    save_video_artifacts(frames, output_gif, output_mp4, fps=fps)

    return {
        "steps": step_count,
        "lane_changes": lane_changes,
        "overtakes": overtakes,
        "crashed": crashed,
        "frames": len(frames),
    }


def record_classical_episode(
    tier: str = "front_only",
    seed: int = 2002,
    use_mobil: bool = True,
    output_gif: str = "visualizations/classical_mobil_front_only_seed2002.gif",
    output_mp4: str | None = "visualizations/classical_mobil_front_only_seed2002.mp4",
    fps: int = 20,
    max_steps: int = 500,
) -> dict:
    """Record a Classical IDM or IDM+MOBIL episode and export to GIF/MP4."""
    apply_anti_alias_stripes()
    import baseline_classical
    params = baseline_classical.DEFAULT_IDM_PARAMS

    env = env_config.make_env(
        scenario="highway",
        tier=tier,
        seed=seed,
        render_mode="rgb_array",
        tactical_overtaking=True,
    )
    env.unwrapped.config["offscreen_rendering"] = True

    obs_np, info = env.reset(seed=seed)
    frames: list[Image.Image] = []
    lane_changes = 0
    overtakes = 0
    step_count = 0
    unwrapped = env.unwrapped
    last_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]
    done = False
    crashed = False

    agent_name = "MOBIL" if use_mobil else "IDM"

    while not done and step_count < max_steps:
        raw_frame = get_rendered_frame(env)
        if use_mobil:
            action = baseline_classical.compute_idm_mobil_discrete_action(env, obs_np, params)
        else:
            action = baseline_classical.compute_idm_discrete_action(env, obs_np, params)
        obs_np, reward, term, trunc, info = env.step(action)
        step_count += 1
        done = term or trunc
        crashed = info.get("crashed", False)

        veh = unwrapped.vehicle
        speed_kmh = float(getattr(veh, "speed", 0.0)) * 3.6
        current_lane = getattr(veh, "lane_index", [0, 0, 0])[2]
        if current_lane != last_lane:
            lane_changes += 1
            last_lane = current_lane

        overtakes += info.get("overtake_count", 0)

        if raw_frame is not None:
            annotated_img = draw_telemetry(
                raw_frame,
                step=step_count,
                action=action,
                speed_kmh=speed_kmh,
                current_lane=current_lane,
                tier=f"{agent_name}-{tier}",
                overtakes=overtakes,
                crashed=crashed,
                max_steps=max_steps,
            )
            frames.append(annotated_img)

    env.close()
    save_video_artifacts(frames, output_gif, output_mp4, fps=fps)

    return {
        "steps": step_count,
        "lane_changes": lane_changes,
        "overtakes": overtakes,
        "crashed": crashed,
        "frames": len(frames),
    }


def run_recording_job(job: dict) -> dict:
    """Execute a single episode recording job, suitable for concurrent execution."""
    import time
    start_t = time.time()
    agent_type = job["agent"]
    tier = job["tier"]
    seed = job["seed"]
    max_steps = job.get("max_steps", 500)
    fps = job.get("fps", 20)
    output_gif = job["gif"]
    output_mp4 = job["mp4"]
    device_name = job.get("device_name", "cpu")

    print(f"[START] Rendering {agent_type} ({tier}, seed {seed}) at {fps} FPS -> {output_gif}")

    if agent_type in ("classical_mobil", "classical_idm"):
        res = record_classical_episode(
            tier=tier,
            seed=seed,
            use_mobil=(agent_type == "classical_mobil"),
            output_gif=output_gif,
            output_mp4=output_mp4,
            fps=fps,
            max_steps=max_steps,
        )
    elif agent_type == "ppo":
        model_path = f"models/ppo_highway_{tier}_seed101.pt"
        res = record_ppo_episode(
            model_path=model_path,
            tier=tier,
            seed=seed,
            output_gif=output_gif,
            output_mp4=output_mp4,
            device_name=device_name,
            fps=fps,
            max_steps=max_steps,
        )
    elif agent_type == "ppo_lstm":
        model_path = f"models/ppo_lstm_highway_{tier}_seed101.pt"
        res = record_episode(
            model_path=model_path,
            tier=tier,
            seed=seed,
            output_gif=output_gif,
            output_mp4=output_mp4,
            device_name=device_name,
            fps=fps,
            max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    elapsed = time.time() - start_t
    res["agent"] = agent_type
    res["tier"] = tier
    res["seed"] = seed
    res["fps"] = fps
    res["elapsed_sec"] = round(elapsed, 2)
    print(f"[DONE] {agent_type} ({tier}, seed {seed}) in {elapsed:.1f}s | steps={res['steps']}, overtakes={res['overtakes']}, crashed={res['crashed']}")
    return res


if __name__ == "__main__":
    import argparse
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Multi-core visual driving renderer with telemetry overlays.")
    parser.add_argument("--agent", type=str, default="ppo_lstm", choices=["ppo", "ppo_lstm", "classical_mobil", "classical_idm"])
    parser.add_argument("--tier", type=str, default="front_only")
    parser.add_argument("--seed", type=int, default=2015)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=20, help="Playback FPS (default: 20 for fast, dynamic playback)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--all", action="store_true", help="Render all standard benchmark episodes in parallel across CPU cores")
    parser.add_argument("--gif", type=str, default=None)
    parser.add_argument("--mp4", type=str, default=None)
    args = parser.parse_args()

    if args.all:
        benchmark_jobs = [
            {"agent": "ppo", "tier": "full_adas", "seed": 2015, "gif": "visualizations/ppo_full_adas_seed2015.gif", "mp4": "visualizations/ppo_full_adas_seed2015.mp4"},
            {"agent": "ppo", "tier": "front_only", "seed": 2015, "gif": "visualizations/ppo_front_only_seed2015.gif", "mp4": "visualizations/ppo_front_only_seed2015.mp4"},
            {"agent": "ppo", "tier": "full_adas", "seed": 2002, "gif": "visualizations/ppo_full_adas_seed2002.gif", "mp4": "visualizations/ppo_full_adas_seed2002.mp4"},
            {"agent": "ppo", "tier": "front_only", "seed": 2002, "gif": "visualizations/ppo_front_only_seed2002.gif", "mp4": "visualizations/ppo_front_only_seed2002.mp4"},
            {"agent": "ppo_lstm", "tier": "full_adas", "seed": 2015, "gif": "visualizations/ppo_lstm_full_adas_seed2015.gif", "mp4": "visualizations/ppo_lstm_full_adas_seed2015.mp4"},
            {"agent": "ppo_lstm", "tier": "front_only", "seed": 2015, "gif": "visualizations/ppo_lstm_front_only_seed2015.gif", "mp4": "visualizations/ppo_lstm_front_only_seed2015.mp4"},
            {"agent": "classical_mobil", "tier": "full_adas", "seed": 2002, "gif": "visualizations/classical_mobil_full_adas_seed2002.gif", "mp4": "visualizations/classical_mobil_full_adas_seed2002.mp4"},
            {"agent": "classical_mobil", "tier": "front_only", "seed": 2002, "gif": "visualizations/classical_mobil_front_only_seed2002.gif", "mp4": "visualizations/classical_mobil_front_only_seed2002.mp4"},
            {"agent": "classical_idm", "tier": "full_adas", "seed": 2002, "gif": "visualizations/classical_idm_full_adas_seed2002.gif", "mp4": "visualizations/classical_idm_full_adas_seed2002.mp4"},
            {"agent": "classical_idm", "tier": "front_only", "seed": 2002, "gif": "visualizations/classical_idm_front_only_seed2002.gif", "mp4": "visualizations/classical_idm_front_only_seed2002.mp4"},
        ]
        for job in benchmark_jobs:
            job["fps"] = args.fps
            job["max_steps"] = args.max_steps

        print(f"Launching {len(benchmark_jobs)} benchmark rendering jobs across {args.workers} CPU cores (fps={args.fps})...")
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_recording_job, job): job for job in benchmark_jobs}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as exc:
                    job_info = futures[fut]
                    print(f"Job {job_info['agent']} ({job_info['tier']}) generated an exception: {exc}")

        print(f"\nAll {len(results)}/{len(benchmark_jobs)} benchmark rendering jobs completed successfully.")
    else:
        prefix = args.agent
        default_gif = f"visualizations/{prefix}_{args.tier}_seed{args.seed}.gif"
        default_mp4 = f"visualizations/{prefix}_{args.tier}_seed{args.seed}.mp4"
        single_job = {
            "agent": args.agent,
            "tier": args.tier,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "fps": args.fps,
            "gif": args.gif or default_gif,
            "mp4": args.mp4 or default_mp4,
        }
        res = run_recording_job(single_job)
        print(f"Result: {res}")

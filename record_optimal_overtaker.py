"""Record visual driving of the trained Optimal Tactical Overtaker PPO model.

Captures simulation frames with telemetry overlay (speed, lane, action, overtakes, crash status).
Exports to GIF and MP4.
"""

import os
import shutil
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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
    crashed: bool,
    max_steps: int = 500,
) -> Image.Image:
    """Overlay telemetry HUD onto rendered simulation frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    banner_height = 26
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(20, 24, 32))

    status_text = "CRASH" if crashed else "SURVIVING"
    status_color = (255, 60, 60) if crashed else (50, 230, 110)

    telemetry_str = (
        f"Step: {step:3d}/{max_steps} | "
        f"Speed: {speed_kmh:4.1f} km/h | Lane: {current_lane} | "
        f"Act: {ACTION_NAMES.get(action, str(action)):<10} | "
        f"Overtakes: {overtakes}"
    )

    draw.text((10, 6), telemetry_str, fill=(240, 245, 250))
    draw.text((img.width - 85, 6), status_text, fill=status_color)
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


def record_optimal_episode(
    model_path: str = "models/ppo_optimal_overtaker_4lane_best.pt",
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    seed: int = 2002,
    max_steps: int = 500,
    output_gif: str = "visualizations/ppo_optimal_overtaker_4lane.gif",
    output_mp4: str = "visualizations/ppo_optimal_overtaker_4lane.mp4",
    artifact_dir: str | None = None,
    fps: int = 15,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(os.path.dirname(output_gif), exist_ok=True)

    duration_sec = int(np.ceil(max_steps / 5)) + 10
    env = env_config.make_optimal_env(
        lanes_count=lanes_count,
        vehicles_density=vehicles_density,
        vehicles_count=vehicles_count,
        seed=seed,
        frame_stack_k=3,
        duration=duration_sec,
        render_mode="rgb_array",
    )
    env.unwrapped.config["offscreen_rendering"] = True

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()

    obs, info = env.reset(seed=seed)
    frames: list[Image.Image] = []
    step_count = 0
    total_overtakes = 0
    crashed = False

    while step_count < max_steps:
        raw_frame = get_rendered_frame(env)
        speed_mps = float(info.get("speed", 0.0))
        speed_kmh = speed_mps * 3.6
        current_lane = (
            env.unwrapped.vehicle.lane_index[2]
            if hasattr(env.unwrapped, "vehicle")
            else 0
        )
        total_overtakes += int(info.get("overtake_count", 0))

        obs_tensor = torch.tensor(
            obs, dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
        mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                obs_tensor, deterministic=True, action_mask=mask_t
            )
            action_idx = int(action.item())

        frame_with_hud = overlay_hud(
            raw_frame,
            step=step_count,
            speed_kmh=speed_kmh,
            action=action_idx,
            current_lane=current_lane,
            overtakes=total_overtakes,
            crashed=crashed,
            max_steps=max_steps,
        )
        frames.append(frame_with_hud)

        obs, reward, term, trunc, info = env.step(action_idx)
        step_count += 1

        if info.get("crashed", False):
            crashed = True

        if term or trunc:
            # Capture final frame
            final_raw = get_rendered_frame(env)
            frames.append(
                overlay_hud(
                    final_raw,
                    step=step_count,
                    speed_kmh=speed_kmh,
                    action=action_idx,
                    current_lane=current_lane,
                    overtakes=total_overtakes,
                    crashed=crashed,
                    max_steps=max_steps,
                )
            )
            break

    env.close()

    # Save GIF
    frame_duration = int(1000 / fps)
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=True,
    )
    print(f"Saved GIF: {output_gif} ({len(frames)} frames)")

    # Save MP4 via imageio or cv2
    try:
        import imageio
        with imageio.get_writer(output_mp4, fps=fps, codec="libx264", quality=8) as writer:
            for f in frames:
                writer.append_data(np.array(f))
        print(f"Saved MP4: {output_mp4}")
    except Exception as e:
        print(f"Could not save MP4 via imageio: {e}")

    # Copy to artifact directory if provided
    if artifact_dir and os.path.exists(artifact_dir):
        art_gif = os.path.join(artifact_dir, os.path.basename(output_gif))
        shutil.copy2(output_gif, art_gif)
        print(f"Copied to artifact dir: {art_gif}")
        if os.path.exists(output_mp4):
            art_mp4 = os.path.join(artifact_dir, os.path.basename(output_mp4))
            shutil.copy2(output_mp4, art_mp4)
            print(f"Copied to artifact dir: {art_mp4}")

    return {
        "steps": step_count,
        "overtakes": total_overtakes,
        "crashed": crashed,
        "speed_kmh": speed_kmh,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/ppo_optimal_overtaker_4lane_best.pt")
    parser.add_argument("--seed", type=int, default=2002)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--artifact-dir", type=str, default="/home/meerpi/.gemini/antigravity-ide/brain/fd4ad70f-1abc-49f5-8b2f-9bfd02929cd9")
    args = parser.parse_args()

    res = record_optimal_episode(
        model_path=args.model,
        seed=args.seed,
        max_steps=args.steps,
        artifact_dir=args.artifact_dir,
    )
    print(f"Result: {res}")

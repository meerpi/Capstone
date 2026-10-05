"""Automated runner for Top 10 Overtaking Visualizations with Anti-Aliasing and Extended Telemetry."""

import argparse
import os
import sys
import json
import torch
import env_config
from train_optimal_overtaker import OptimalAgent
from render_top10 import render_seed_episode

DEFAULT_MODEL_PATH = "models/ppo_optimal_overtaker_4lane_best.pt"
DEFAULT_OUTPUT_DIR = "visualizations/top10_v2"
DEFAULT_SEEDS = [3094, 3115, 3010, 3000, 3019, 3092, 3078, 3028, 2010, 3049]


def main(
    selected_seeds: list[int] | None = None,
    model_path: str = DEFAULT_MODEL_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    artifact_dir: str | None = None,
    realtime: bool = False,
):
    seeds = selected_seeds or DEFAULT_SEEDS
    print(f"Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = OptimalAgent(obs_dim=81, action_dim=5).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        agent.load_state_dict(ckpt["model_state_dict"])
    else:
        agent.load_state_dict(ckpt)
    agent.eval()

    os.makedirs(output_dir, exist_ok=True)
    results = []
    for rank, seed in enumerate(seeds, 1):
        print(f"\n[{rank}/{len(seeds)}] Rendering Rank #{rank} (Seed {seed})...")
        res = render_seed_episode(
            agent=agent,
            seed=seed,
            rank=rank,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            max_steps=500,
            realtime=realtime,
            device=device,
        )
        results.append(res)

    # Save manifest
    manifest_path = os.path.join(output_dir, "top10_episodes_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote manifest to {manifest_path}")

    if artifact_dir and os.path.exists(artifact_dir):
        art_manifest = os.path.join(artifact_dir, "top10_episodes_manifest.json")
        with open(art_manifest, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote artifact manifest to {art_manifest}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render top 10 overtaking episodes.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save GIFs and MP4s (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=None,
        help="Optional external artifact directory to copy videos to.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Model checkpoint path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Render at 1.0x real-time speed (200ms/frame, 5fps) instead of 15fps.",
    )
    parser.add_argument(
        "seeds",
        nargs="*",
        type=int,
        help="Specific seeds to render (defaults to top 10 benchmark seeds).",
    )
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else None
    main(
        selected_seeds=seeds,
        model_path=args.model_path,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
        realtime=args.realtime,
    )

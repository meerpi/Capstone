"""Automated runner for Top 10 Overtaking Visualizations."""

import os
import sys
import json
import torch
import env_config
from train_optimal_overtaker import OptimalAgent
from render_top10 import render_seed_episode

MODEL_PATH = "models/ppo_optimal_overtaker_4lane_best.pt"
OUTPUT_DIR = "visualizations/top10"
ARTIFACT_DIR = "/home/meerpi/.gemini/antigravity-ide/brain/fd4ad70f-1abc-49f5-8b2f-9bfd02929cd9"


def main(selected_seeds: list[int]):
    print(f"Loading model from {MODEL_PATH}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = OptimalAgent(obs_dim=81, action_dim=5).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()

    results = []
    for rank, seed in enumerate(selected_seeds, 1):
        print(f"\n[{rank}/10] Rendering Rank #{rank} (Seed {seed})...")
        res = render_seed_episode(
            agent=agent,
            seed=seed,
            rank=rank,
            output_dir=OUTPUT_DIR,
            artifact_dir=ARTIFACT_DIR,
            max_steps=500,
            fps=15,
            device=device,
        )
        results.append(res)

    results_file = os.path.join(ARTIFACT_DIR, "top10_episodes_manifest.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote manifest to {results_file}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seeds = [int(s) for s in sys.argv[1:]]
    else:
        # Definitive Top 10 non-idle, high-overtaking, tactical avoidance seeds
        seeds = [3094, 3115, 3010, 3000, 3019, 3092, 3078, 3028, 2010, 3049]
    main(seeds)

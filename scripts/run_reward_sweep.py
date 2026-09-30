"""Empirical Reward Sweep across Collision Penalty and Overtake Bonus pairs.

Runs a short local training sweep across the four target configurations:
1. (-50.0, 1.0) [current defaults]
2. (-50.0, 2.5)
3. (-50.0, 5.0) [harmonized with Round-1 pipeline]
4. (-30.0, 5.0) [fully harmonized with Round-1]

Uses lexicographic model selection for all runs and evaluates each selected model
across deterministic evaluation seeds (20 episodes, seeds 2000-2019).
Reports empirical crash rate, mean speed, overtakes, lane changes, and duration
alongside theoretical p* values.
"""

import json
import os
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env_config
from train_optimal_overtaker import evaluate, train, parse_args
import argparse


SWEEP_CONFIGS = [
    {
        "name": "cp-50_ot1.0",
        "label": "(-50.0, 1.0) [Current]",
        "collision_penalty": -50.0,
        "overtake_bonus": 1.0,
        "p_star": 1.0 / (50.0 + 1.0),
    },
    {
        "name": "cp-50_ot2.5",
        "label": "(-50.0, 2.5)",
        "collision_penalty": -50.0,
        "overtake_bonus": 2.5,
        "p_star": 2.5 / (50.0 + 2.5),
    },
    {
        "name": "cp-50_ot5.0",
        "label": "(-50.0, 5.0) [Harmonized Bonus]",
        "collision_penalty": -50.0,
        "overtake_bonus": 5.0,
        "p_star": 5.0 / (50.0 + 5.0),
    },
    {
        "name": "cp-30_ot5.0",
        "label": "(-30.0, 5.0) [Fully Harmonized]",
        "collision_penalty": -30.0,
        "overtake_bonus": 5.0,
        "p_star": 5.0 / (30.0 + 5.0),
    },
]


def run_sweep(
    total_timesteps: int = 20480,
    num_envs: int = 4,
    num_steps: int = 128,
    eval_episodes: int = 20,
    seed: int = 42,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """Execute training sweep across all 4 configurations."""
    os.makedirs("scratch/sweep_results", exist_ok=True)
    results = []

    print("=" * 80)
    print("EMPIRICAL REWARD SWEEP: (Collision Penalty, Overtake Bonus)")
    print(f"Total timesteps per run: {total_timesteps} ({total_timesteps // (num_envs * num_steps)} iterations)")
    print(f"Envs: {num_envs} | Steps/Env: {num_steps} | Seed: {seed} | Device: {device}")
    print(f"Eval episodes: {eval_episodes} (seeds 2000..{2000 + eval_episodes - 1})")
    print("=" * 80, flush=True)

    for idx, cfg in enumerate(SWEEP_CONFIGS, start=1):
        print(f"\n[{idx}/4] >>> Running Config: {cfg['label']} <<<")
        print(f"  Collision Penalty: {cfg['collision_penalty']}, Overtake Bonus: {cfg['overtake_bonus']}")
        print(f"  Theoretical p* = {cfg['p_star'] * 100:.2f}% (break-even crash probability per pass)")

        ckpt_dir = os.path.join("scratch/sweep_results", cfg["name"])
        os.makedirs(ckpt_dir, exist_ok=True)
        final_model_path = os.path.join(ckpt_dir, "final.pt")
        best_model_path = os.path.join(ckpt_dir, "ppo_optimal_overtaker_4lane_best.pt")

        args = argparse.Namespace(
            lanes=4,
            density=1.4,
            vehicles=14,
            frame_stack=3,
            seed=seed,
            total_timesteps=total_timesteps,
            learning_rate=2.5e-4,
            lr_end=1e-5,
            num_envs=num_envs,
            num_steps=num_steps,
            gamma=0.995,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            clip_coef=0.2,
            ent_coef=0.02,
            ent_end=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            cuda=(device == "cuda"),
            bootstrap_truncation=False,
            checkpoint_dir=ckpt_dir,
            save_frequency=50000,
            resume=None,
            output_path=final_model_path,
            collision_penalty=cfg["collision_penalty"],
            overtake_bonus=cfg["overtake_bonus"],
            eval_only=False,
            model_path=None,
            eval_episodes=eval_episodes,
            eval_seed_start=2000,
        )

        start_time = time.time()
        train(args)
        train_duration = time.time() - start_time

        # Check whether lexicographic selection saved a best model
        eval_path = best_model_path if os.path.exists(best_model_path) else final_model_path
        used_best = os.path.exists(best_model_path)
        print(f"  Evaluating candidate ({'lexicographic best' if used_best else 'final'}): {eval_path}")

        eval_res = evaluate(
            model_path=eval_path,
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            episodes=eval_episodes,
            seed_start=2000,
            frame_stack_k=3,
            device=device,
            collision_penalty=cfg["collision_penalty"],
            overtake_bonus=cfg["overtake_bonus"],
        )

        run_summary = {
            "name": cfg["name"],
            "label": cfg["label"],
            "collision_penalty": cfg["collision_penalty"],
            "overtake_bonus": cfg["overtake_bonus"],
            "p_star_percent": round(cfg["p_star"] * 100.0, 2),
            "train_duration_sec": round(train_duration, 1),
            "eval_crashes": eval_res["crashes"],
            "eval_crash_rate": eval_res["crash_rate"],
            "eval_mean_speed_kmh": round(eval_res["mean_speed_kmh"], 2),
            "eval_mean_overtakes": round(eval_res["mean_overtakes"], 2),
            "eval_mean_lane_changes": round(eval_res["mean_lane_changes"], 2),
            "eval_mean_duration": round(eval_res["mean_duration"], 1),
            "eval_model_used": "best_lexicographic" if used_best else "final",
        }
        results.append(run_summary)

    # Save results to json
    results_path = "scratch/sweep_results/empirical_sweep_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Sweep complete! Results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"{'Configuration':<32} | {'p* (Theor.)':<11} | {'Crash Rate':<12} | {'Speed (km/h)':<13} | {'Overtakes':<10} | {'Lane Chg':<9}")
    print("-" * 95)
    for r in results:
        crash_str = f"{r['eval_crashes']}/{eval_episodes} ({r['eval_crash_rate']:4.1f}%)"
        print(f"{r['label']:<32} | {r['p_star_percent']:>9.2f}% | {crash_str:<12} | {r['eval_mean_speed_kmh']:>11.1f} | {r['eval_mean_overtakes']:>9.2f} | {r['eval_mean_lane_changes']:>8.2f}")
    print("=" * 95)

    return results


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_sweep(device=device)

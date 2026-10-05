"""Benchmark runner comparing Classical (IDM+MOBIL) and PPO agents.

Runs evaluation on benchmark test seeds across sensor tiers:
- full_adas (360-degree observation)
- front_only (forward cone radar, blind spots behind)

RL agents are trained with:
- Tailgating / Headway penalty (penalizes passive tailgating)
- Overtake bonus (+1.0 per vehicle overtaken)
- Invalid action masking (prevents boundary exploits)
"""

import argparse
import copy
import json
import os
import subprocess
import time
from typing import Any

import gymnasium
import highway_env
import numpy
import torch

import baseline_classical
import env_config
import ppo

try:
    from parity.metrics import clopper_pearson
except ImportError:
    clopper_pearson = None


CLASSICAL_PROFILES: dict[str, dict[str, Any]] = {
    "legacy": {
        "observation": {"vehicles_count": 10},
        "vehicles_count": 10,
        "vehicles_density": 1.0,
        "duration": 40,
        "step_cap": 200,
        "policy_frequency": 5,
    },
    "current": {
        "observation": {"vehicles_count": 15},
        "vehicles_count": 14,
        "vehicles_density": 1.4,
        "duration": 100,
        "step_cap": 500,
        "policy_frequency": 5,
    },
}


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def get_package_versions() -> dict[str, str]:
    """Retrieve runtime versions of core environment packages."""
    return {
        "highway_env": getattr(highway_env, "__version__", "unknown"),
        "gymnasium": getattr(gymnasium, "__version__", "unknown"),
        "torch": getattr(torch, "__version__", "unknown"),
        "numpy": getattr(numpy, "__version__", "unknown"),
    }


def evaluate_classical_baselines(
    profile_name: str,
    episodes: int = 100,
    seed_start: int = 2000,
    save_dir: str = "eval_out",
) -> dict[str, Any]:
    """Evaluate IDM-only and IDM+MOBIL across full_adas and front_only tiers."""
    if profile_name not in CLASSICAL_PROFILES:
        raise ValueError(f"Unknown classical profile: {profile_name}")

    profile_cfg = CLASSICAL_PROFILES[profile_name]
    profile_overrides = {
        k: v
        for k, v in profile_cfg.items()
        if k in ("observation", "vehicles_count", "vehicles_density", "duration")
    }

    real_make_env = env_config.make_env

    def patched_make_env(*args: Any, **kwargs: Any) -> Any:
        merged = copy.deepcopy(profile_overrides)
        merged.update(kwargs)
        return real_make_env(*args, **merged)

    env_config.make_env = patched_make_env

    results_per_row: dict[str, dict[str, Any]] = {}
    try:
        for tier in ["full_adas", "front_only"]:
            for controller in ["idm_only", "idm_mobil"]:
                raw = baseline_classical.evaluate_controller(
                    scenario="highway",
                    tier=tier,
                    controller_name=controller,
                    episodes=episodes,
                    seed_start=seed_start,
                )
                key = f"{tier}_{controller}"
                crashes = int(raw["crashes"])
                n = int(raw["episodes"])
                rate = (crashes / n * 100.0) if n > 0 else 0.0

                ci_low: float | None = None
                ci_high: float | None = None
                if clopper_pearson is not None and n > 0:
                    ci_low, ci_high = clopper_pearson(crashes, n)

                policy_label = (
                    "IDM Only (Lane-Keeping)"
                    if controller == "idm_only"
                    else "IDM + MOBIL (Overtaking)"
                )

                results_per_row[key] = {
                    "policy": policy_label,
                    "tier": tier,
                    "controller": controller,
                    "profile": profile_name,
                    "episodes": n,
                    "crashes": crashes,
                    "crash_fraction": f"{crashes}/{n}",
                    "crash_rate": rate,
                    "ci_95": [ci_low, ci_high] if ci_low is not None else None,
                    "mean_duration": float(raw["mean_duration"]),
                    "std_duration": float(raw["std_duration"]),
                    "mean_speed_kmh": float(raw["mean_speed_kmh"]),
                    "mean_lane_changes": float(raw["mean_lane_changes"]),
                }
    finally:
        env_config.make_env = real_make_env

    output_doc = {
        "profile": profile_name,
        "profile_parameters": profile_cfg,
        "git_commit": get_git_commit(),
        "package_versions": get_package_versions(),
        "seeds": {
            "start": seed_start,
            "end": seed_start + episodes - 1,
            "count": episodes,
        },
        "results": results_per_row,
    }

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"classical_baselines_{profile_name}.json")
    with open(out_path, "w") as f:
        json.dump(output_doc, f, indent=2)
    print(f"Saved classical baselines to {out_path}", flush=True)
    return output_doc


def load_classical_baseline_results(
    profile_name: str = "current",
    episodes: int = 100,
    seed_start: int = 2000,
    save_dir: str = "eval_out",
) -> dict[str, dict[str, Any]]:
    """Load or evaluate classical benchmark results."""
    path = os.path.join(save_dir, f"classical_baselines_{profile_name}.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if "results" in data:
                return data["results"]
        except Exception:
            pass
    doc = evaluate_classical_baselines(
        profile_name=profile_name,
        episodes=episodes,
        seed_start=seed_start,
        save_dir=save_dir,
    )
    return doc["results"]


def run_ppo_experiments(
    tiers: list[str],
    total_timesteps: int = 800000,
    seed: int = 101,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate PPO across specified sensor tiers."""
    ppo_results: dict[str, dict[str, Any]] = {}

    for tier in tiers:
        print("=" * 80, flush=True)
        print(f"RUNNING PPO: tier={tier} | steps={total_timesteps:,}", flush=True)
        print("=" * 80, flush=True)

        args = argparse.Namespace(
            scenario="highway",
            tier=tier,
            seed=seed,
            total_timesteps=total_timesteps,
            learning_rate=3e-4,
            num_envs=8,
            num_steps=512,
            anneal_lr=True,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=4,
            update_epochs=4,
            norm_adv=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            cuda=True,
            async_envs=True,
        )

        t0 = time.time()
        model_path = ppo.train_ppo(args)
        train_time = time.time() - t0

        print(f"Evaluating PPO {tier} on 30 test seeds...", flush=True)
        eval_res = ppo.evaluate_ppo_policy(
            model_path=model_path,
            scenario="highway",
            tier=tier,
            episodes=30,
            seed_start=2000,
            deterministic=True,
        )
        eval_res["policy"] = "PPO (MLP)"
        eval_res["train_duration_sec"] = train_time
        ppo_results[f"ppo_{tier}"] = eval_res

    return ppo_results


def format_benchmark_table(
    classical: dict[str, dict[str, dict[str, Any]]],
    ppo_res: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Format and return the final comparative Markdown table."""
    lines: list[str] = []
    lines.append(
        "| Profile | Sensor Tier | Policy / Controller | Crashes | Rate | "
        "Duration (steps) | Mean Speed | Lane Changes / ep |"
    )
    lines.append(
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    )

    for profile_name in ["legacy", "current"]:
        prof_data = classical.get(profile_name, {})
        for tier in ["full_adas", "front_only"]:
            idm = prof_data.get(f"{tier}_idm_only")
            if idm:
                lines.append(
                    f"| **{profile_name}** | **{tier}** | {idm['policy']} | {idm['crash_fraction']} | "
                    f"{idm['crash_rate']:.1f}% | {idm['mean_duration']:.1f} ± "
                    f"{idm['std_duration']:.1f} | {idm['mean_speed_kmh']:.1f} km/h | "
                    f"{idm['mean_lane_changes']:.2f} |"
                )
            mobil = prof_data.get(f"{tier}_idm_mobil")
            if mobil:
                lines.append(
                    f"| | | {mobil['policy']} | {mobil['crash_fraction']} | "
                    f"{mobil['crash_rate']:.1f}% | {mobil['mean_duration']:.1f} ± "
                    f"{mobil['std_duration']:.1f} | {mobil['mean_speed_kmh']:.1f} km/h | "
                    f"{mobil['mean_lane_changes']:.2f} |"
                )

    if ppo_res:
        for tier in ["full_adas", "front_only"]:
            ppo_entry = ppo_res.get(f"ppo_{tier}")
            if ppo_entry:
                lines.append(
                    f"| *eval* | **{tier}** | **{ppo_entry['policy']}** | "
                    f"{ppo_entry['crash_fraction']} | "
                    f"{ppo_entry['crash_rate']:.1f}% | "
                    f"{ppo_entry['mean_duration']:.1f} ± "
                    f"{ppo_entry['std_duration']:.1f} | "
                    f"{ppo_entry['mean_speed_kmh']:.1f} km/h | "
                    f"{ppo_entry['mean_lane_changes']:.2f} |"
                )

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line configuration arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark runner comparing Classical (IDM+MOBIL) and PPO."
    )
    parser.add_argument(
        "--classical-only",
        action="store_true",
        default=False,
        help="Run classical baselines only (skip PPO training).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per condition for classical evaluation (default: 100).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2000,
        help="Starting random seed for evaluation (default: 2000).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute complete benchmark sweep."""
    args = parse_args()
    tiers = ["full_adas", "front_only"]

    print("Running Classical Baselines under baseline and current speed profiles...")
    legacy_doc = evaluate_classical_baselines(
        "legacy", episodes=args.episodes, seed_start=args.seed_start
    )
    current_doc = evaluate_classical_baselines(
        "current", episodes=args.episodes, seed_start=args.seed_start
    )

    classical_all = {
        "legacy": legacy_doc["results"],
        "current": current_doc["results"],
    }

    ppo_res: dict[str, dict[str, Any]] = {}
    if not args.classical_only:
        ppo_res = run_ppo_experiments(tiers=tiers, total_timesteps=800000, seed=101)

    all_results = {
        "classical_legacy": legacy_doc,
        "classical_current": current_doc,
        "ppo": ppo_res,
    }

    with open("ultimate_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    table = format_benchmark_table(classical_all, ppo_res)
    print("\n" + "=" * 80)
    print(
        f"COMPARISON BENCHMARK (N={args.episodes} Seeds: {args.seed_start}-{args.seed_start + args.episodes - 1})"
    )
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()


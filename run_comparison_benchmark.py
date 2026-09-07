"""Comprehensive benchmark runner comparing Classical (IDM+MOBIL), PPO, and Rainbow DQN.

Runs evaluation on the 30 benchmark test seeds (2000-2029) across sensor tiers:
- full_adas (360-degree observation)
- front_only (forward cone radar, blind spots behind)

Both RL agents are trained with:
- Mechanism A: Tailgating / Headway penalty (penalizes passive tailgating)
- Mechanism B: Overtake bonus (+1.0 per vehicle overtaken)
- Mechanism C: Invalid action masking (prevents boundary exploits)
"""

import argparse
import json
import os
import time
from typing import Any

import ppo
import rainbow_highway


def load_classical_baseline_results() -> dict[str, dict[str, Any]]:
    """Return verified Stage 2 classical benchmark results across 30 seeds."""
    return {
        "full_adas_idm_only": {
            "policy": "IDM Only (Lane-Keeping)",
            "tier": "full_adas",
            "crashes": 0,
            "crash_fraction": "0/30",
            "crash_rate": 0.0,
            "mean_duration": 200.0,
            "std_duration": 0.0,
            "mean_speed_kmh": 81.4,
            "mean_lane_changes": 0.0,
        },
        "full_adas_idm_mobil": {
            "policy": "IDM + MOBIL (Overtaking)",
            "tier": "full_adas",
            "crashes": 2,
            "crash_fraction": "2/30",
            "crash_rate": 6.7,
            "mean_duration": 193.6,
            "std_duration": 32.5,
            "mean_speed_kmh": 83.4,
            "mean_lane_changes": 5.80,
        },
        "front_only_idm_only": {
            "policy": "IDM Only (Lane-Keeping)",
            "tier": "front_only",
            "crashes": 0,
            "crash_fraction": "0/30",
            "crash_rate": 0.0,
            "mean_duration": 200.0,
            "std_duration": 0.0,
            "mean_speed_kmh": 81.4,
            "mean_lane_changes": 0.0,
        },
        "front_only_idm_mobil": {
            "policy": "IDM + MOBIL (Overtaking)",
            "tier": "front_only",
            "crashes": 15,
            "crash_fraction": "15/30",
            "crash_rate": 50.0,
            "mean_duration": 147.1,
            "std_duration": 64.3,
            "mean_speed_kmh": 84.2,
            "mean_lane_changes": 4.40,
        },
    }


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


def run_rainbow_experiments(
    tiers: list[str],
    total_timesteps: int = 100000,
    seed: int = 101,
) -> dict[str, dict[str, Any]]:
    """Train and evaluate Rainbow DQN across specified sensor tiers."""
    rainbow_results: dict[str, dict[str, Any]] = {}

    for tier in tiers:
        print("=" * 80, flush=True)
        print(
            f"RUNNING RAINBOW DQN: tier={tier} | steps={total_timesteps:,}",
            flush=True,
        )
        print("=" * 80, flush=True)

        args = argparse.Namespace(
            scenario="highway",
            tier=tier,
            seed=seed,
            total_timesteps=total_timesteps,
            learning_rate=1e-4,
            buffer_size=100000,
            gamma=0.99,
            batch_size=64,
            learning_starts=5000,
            train_frequency=4,
            target_network_frequency=1000,
            n_step=3,
            prioritized_replay_alpha=0.6,
            prioritized_replay_beta=0.4,
            n_atoms=51,
            v_min=-20.0,
            v_max=20.0,
            cuda=True,
        )

        t0 = time.time()
        model_path = rainbow_highway.train_rainbow(args)
        train_time = time.time() - t0

        print(f"Evaluating Rainbow DQN {tier} on 30 test seeds...", flush=True)
        eval_res = rainbow_highway.evaluate_rainbow_policy(
            model_path=model_path,
            scenario="highway",
            tier=tier,
            episodes=30,
            seed_start=2000,
        )
        eval_res["policy"] = "Rainbow DQN"
        eval_res["train_duration_sec"] = train_time
        rainbow_results[f"rainbow_{tier}"] = eval_res

    return rainbow_results


def format_benchmark_table(
    classical: dict[str, dict[str, Any]],
    ppo_res: dict[str, dict[str, Any]],
    rainbow_res: dict[str, dict[str, Any]],
) -> str:
    """Format and return the final 3-way comparative Markdown table."""
    lines: list[str] = []
    lines.append(
        "| Sensor Tier | Policy / Controller | Crashes | Rate | "
        "Duration (steps) | Mean Speed | Lane Changes / ep |"
    )
    lines.append(
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
    )

    tiers = ["full_adas", "front_only"]
    for tier in tiers:
        # IDM Only
        idm = classical[f"{tier}_idm_only"]
        lines.append(
            f"| **{tier}** | {idm['policy']} | {idm['crash_fraction']} | "
            f"{idm['crash_rate']:.1f}% | {idm['mean_duration']:.1f} ± "
            f"{idm['std_duration']:.1f} | {idm['mean_speed_kmh']:.1f} km/h | "
            f"{idm['mean_lane_changes']:.2f} |"
        )
        # IDM + MOBIL
        mobil = classical[f"{tier}_idm_mobil"]
        lines.append(
            f"| | {mobil['policy']} | {mobil['crash_fraction']} | "
            f"{mobil['crash_rate']:.1f}% | {mobil['mean_duration']:.1f} ± "
            f"{mobil['std_duration']:.1f} | {mobil['mean_speed_kmh']:.1f} km/h | "
            f"{mobil['mean_lane_changes']:.2f} |"
        )
        # PPO
        ppo_entry = ppo_res.get(f"ppo_{tier}")
        if ppo_entry:
            lines.append(
                f"| | **{ppo_entry['policy']}** | "
                f"{ppo_entry['crash_fraction']} | "
                f"{ppo_entry['crash_rate']:.1f}% | "
                f"{ppo_entry['mean_duration']:.1f} ± "
                f"{ppo_entry['std_duration']:.1f} | "
                f"{ppo_entry['mean_speed_kmh']:.1f} km/h | "
                f"{ppo_entry['mean_lane_changes']:.2f} |"
            )
        # Rainbow DQN
        rb_entry = rainbow_res.get(f"rainbow_{tier}")
        if rb_entry:
            lines.append(
                f"| | **{rb_entry['policy']}** | "
                f"{rb_entry['crash_fraction']} | "
                f"{rb_entry['crash_rate']:.1f}% | "
                f"{rb_entry['mean_duration']:.1f} ± "
                f"{rb_entry['std_duration']:.1f} | "
                f"{rb_entry['mean_speed_kmh']:.1f} km/h | "
                f"{rb_entry['mean_lane_changes']:.2f} |"
            )

    return "\n".join(lines)


def main() -> None:
    """Execute complete 3-way benchmark sweep."""
    tiers = ["full_adas", "front_only"]

    print("Starting 3-Way Benchmark Suite: Classical vs PPO vs Rainbow DQN...")
    classical_res = load_classical_baseline_results()

    # Step 1: Run PPO with Mechanisms A, B, C
    ppo_res = run_ppo_experiments(tiers=tiers, total_timesteps=800000, seed=101)

    # Step 2: Run Rainbow DQN with Mechanisms A, B, C
    rainbow_res = run_rainbow_experiments(
        tiers=tiers, total_timesteps=100000, seed=101
    )

    # Step 3: Combine and save results
    all_results = {
        "classical": classical_res,
        "ppo": ppo_res,
        "rainbow": rainbow_res,
    }

    with open("ultimate_comparison_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    table = format_benchmark_table(classical_res, ppo_res, rainbow_res)
    print("\n" + "=" * 80)
    print("ULTIMATE 3-WAY COMPARISON (N=30 Benchmark Seeds: 2000-2029)")
    print("=" * 80)
    print(table)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

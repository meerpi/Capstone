"""Orchestration script for SOTA parity ladder across R0 to R7.

Executes:
(a) Fixed baselines: random, always-IDLE, always-SLOWER, always-FASTER, always-LANE_RIGHT, IDM+MOBIL
(b) SB3 PPO: 200k steps x 3 seeds (seeds 0, 1, 2)
(c) Our PPO: 200k steps x 3 seeds (seeds 0, 1, 2)
All evaluated on seeds 6000-6099.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, ".")
from parity.env_factory import make_ladder_env
from parity.evaluator import (
    evaluate_policy,
    RandomPolicy,
    ConstantPolicy,
    IDMMobilPolicy,
)
from parity.train_sb3 import train_and_eval_sb3
from parity.train_our_ppo import train_and_eval_our_ppo


def run_baselines_for_rung(rung: str) -> list[dict[str, Any]]:
    print(f"\n==========================================")
    print(f"Running Baselines for {rung} (seeds 6000-6099)")
    print(f"==========================================")
    results = []

    eval_env_fn = lambda: make_ladder_env(rung)

    # 1. Random
    print(f"[Baseline] Random policy...")
    res = evaluate_policy(eval_env_fn, RandomPolicy(5), seed_start=6000, n_episodes=100)
    res.update({"rung": rung, "method": "random", "seed": "none"})
    results.append(res)
    print(f"  Random: Crash {res['ci_str']}, Speed {res['mean_speed_ms']:.2f} m/s, Return {res['mean_return']:.2f}")

    # 2. always-IDLE
    print(f"[Baseline] always-IDLE...")
    res = evaluate_policy(eval_env_fn, ConstantPolicy(1), seed_start=6000, n_episodes=100)
    res.update({"rung": rung, "method": "always_IDLE", "seed": "none"})
    results.append(res)
    print(f"  always-IDLE: Crash {res['ci_str']}, Speed {res['mean_speed_ms']:.2f} m/s, Return {res['mean_return']:.2f}")

    # 3. always-SLOWER
    print(f"[Baseline] always-SLOWER...")
    res = evaluate_policy(eval_env_fn, ConstantPolicy(4), seed_start=6000, n_episodes=100)
    res.update({"rung": rung, "method": "always_SLOWER", "seed": "none"})
    results.append(res)
    print(f"  always-SLOWER: Crash {res['ci_str']}, Speed {res['mean_speed_ms']:.2f} m/s, Return {res['mean_return']:.2f}")

    # 4. always-FASTER
    print(f"[Baseline] always-FASTER...")
    res = evaluate_policy(eval_env_fn, ConstantPolicy(3), seed_start=6000, n_episodes=100)
    res.update({"rung": rung, "method": "always_FASTER", "seed": "none"})
    results.append(res)
    print(f"  always-FASTER: Crash {res['ci_str']}, Speed {res['mean_speed_ms']:.2f} m/s, Return {res['mean_return']:.2f}")

    # 5. always-LANE_RIGHT
    print(f"[Baseline] always-LANE_RIGHT...")
    res = evaluate_policy(eval_env_fn, ConstantPolicy(2), seed_start=6000, n_episodes=100)
    res.update({"rung": rung, "method": "always_LANE_RIGHT", "seed": "none"})
    results.append(res)
    print(f"  always-LANE_RIGHT: Crash {res['ci_str']}, Speed {res['mean_speed_ms']:.2f} m/s, Return {res['mean_return']:.2f}")

    # 6. IDM+MOBIL
    print(f"[Baseline] IDM+MOBIL...")
    res = evaluate_policy(eval_env_fn, IDMMobilPolicy(), seed_start=6000, n_episodes=100, is_env_aware=True)
    res.update({"rung": rung, "method": "IDM_MOBIL", "seed": "none"})
    results.append(res)
    print(f"  IDM+MOBIL: Crash {res['ci_str']}, Speed {res['mean_speed_ms']:.2f} m/s, Return {res['mean_return']:.2f}")

    return results


def run_rung(rung: str, steps: int = 200000, gamma: float = 0.8, run_sb3: bool = True, run_our: bool = True) -> list[dict[str, Any]]:
    rung_results = []

    # 1. Run baselines
    baselines = run_baselines_for_rung(rung)
    rung_results.extend(baselines)

    seeds = [0, 1, 2]

    # For R0: also run 20k steps official Farama script
    if rung.upper() == "R0":
        print("\n[R0 Special] Running Official 20k steps Farama SB3 baseline...")
        res_20k = train_and_eval_sb3("R0", seed=0, total_timesteps=20000, gamma=0.8)
        res_20k["method"] = "SB3_Official_20k"
        rung_results.append(res_20k)

    # 2. Run SB3 PPO 200k x 3 seeds
    if run_sb3:
        for seed in seeds:
            print(f"\n--- Training SB3 PPO on {rung} (seed {seed}, {steps} steps, gamma {gamma}) ---")
            res = train_and_eval_sb3(rung, seed=seed, total_timesteps=steps, gamma=gamma)
            rung_results.append(res)

    # 3. Run Our PPO 200k x 3 seeds
    if run_our:
        for seed in seeds:
            print(f"\n--- Training Our PPO on {rung} (seed {seed}, {steps} steps, gamma {gamma}) ---")
            res = train_and_eval_our_ppo(rung, seed=seed, total_timesteps=steps, gamma=gamma)
            rung_results.append(res)

    # Save rung results to disk
    out_file = Path("parity") / f"results_{rung.lower()}.json"
    with open(out_file, "w") as f:
        json.dump(rung_results, f, indent=2)
    print(f"\nSaved rung results to {out_file}")

    return rung_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", type=str, required=True, help="Rung name (R0, R1, R2, R3, R4, R5_956, R5_995, R6, R7_gt, R7_deepset)")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--skip-sb3", action="store_true")
    parser.add_argument("--skip-our", action="store_true")
    args = parser.parse_args()

    run_rung(args.rung, steps=args.steps, gamma=args.gamma, run_sb3=not args.skip_sb3, run_our=not args.skip_our)

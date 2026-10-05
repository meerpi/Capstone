"""Sequential rung executor for baseline parity ladder.

Runs:
1. Baselines on seeds 6000-6099
2. SB3 Official 20k steps (if R0)
3. SB3 PPO seeds 0, 1, 2
4. Our PPO seeds 0, 1, 2
5. Saves combined results to parity/results_{rung}.json
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


def evaluate_baselines(rung: str) -> list[dict[str, Any]]:
    print(f"\n==========================================")
    print(f"Evaluating Baselines for {rung} (seeds 6000-6099)")
    print(f"==========================================")
    results = []
    eval_env_fn = lambda: make_ladder_env(rung)

    baselines = [
        ("random", RandomPolicy(5), False),
        ("always_IDLE", ConstantPolicy(1), False),
        ("always_SLOWER", ConstantPolicy(4), False),
        ("always_FASTER", ConstantPolicy(3), False),
        ("always_LANE_RIGHT", ConstantPolicy(2), False),
        ("IDM_MOBIL", IDMMobilPolicy(), True),
    ]

    for name, policy, is_aware in baselines:
        t0 = time.time()
        res = evaluate_policy(eval_env_fn, policy, seed_start=6000, n_episodes=100, is_env_aware=is_aware)
        dt = time.time() - t0
        res.update({"rung": rung, "method": name, "seed": "none", "eval_time_s": round(dt, 2)})
        results.append(res)
        print(f"  [{name:17s}] Crash: {res['ci_str']:28s} | Speed: {res['mean_speed_ms']:.2f} m/s | Return: {res['mean_return']:.2f} | Dom: {res['dominant_action']} ({res['dominant_action_frac']:.1%}) ({dt:.1f}s)")

    return results


def run_single_sb3(rung: str, seed: int, steps: int, gamma: float) -> dict[str, Any]:
    t0 = time.time()
    res = train_and_eval_sb3(rung, seed=seed, total_timesteps=steps, gamma=gamma)
    res["runtime_s"] = round(time.time() - t0, 2)
    return res


def run_single_our(rung: str, seed: int, steps: int, gamma: float) -> dict[str, Any]:
    t0 = time.time()
    res = train_and_eval_our_ppo(rung, seed=seed, total_timesteps=steps, gamma=gamma)
    res["runtime_s"] = round(time.time() - t0, 2)
    return res


def run_rung(
    rung: str,
    steps: int = 200000,
    gamma: float = 0.8,
    include_20k: bool = False,
    run_baselines: bool = True,
    run_sb3: bool = True,
    run_our: bool = True,
    seeds: list[int] = [0, 1, 2],
):
    rung_clean = rung.upper().replace("-", "_")
    print(f"\n#################################################################")
    print(f"STARTING LADDER RUNG: {rung_clean} (steps={steps}, gamma={gamma})")
    print(f"#################################################################")
    start_total = time.time()
    all_results = []

    # 1. Baselines
    if run_baselines:
        baseline_results = evaluate_baselines(rung_clean)
        all_results.extend(baseline_results)

    # If R0, run 20k Farama official baseline
    if include_20k or rung_clean == "R0":
        print("\n--- Training SB3 Official Farama 20k steps (seed 0) ---")
        t0 = time.time()
        res_20k = train_and_eval_sb3(rung_clean, seed=0, total_timesteps=20000, gamma=gamma)
        res_20k["method"] = "SB3_Official_20k"
        res_20k["runtime_s"] = round(time.time() - t0, 2)
        all_results.append(res_20k)

    # 2. SB3 PPO seeds
    if run_sb3:
        for seed in seeds:
            print(f"\n--- Training SB3 PPO on {rung_clean} (seed {seed}, {steps} steps) ---")
            res = run_single_sb3(rung_clean, seed, steps, gamma)
            all_results.append(res)
            print(f"[Finished SB3 seed {seed}] Crash: {res['ci_str']} | Speed: {res['mean_speed_ms']:.2f} m/s | Return: {res['mean_return']:.2f} | Dom: {res['dominant_action']} ({res['dominant_action_frac']:.1%}) ({res['runtime_s']:.1f}s)")

    # 3. Our PPO seeds
    if run_our:
        for seed in seeds:
            print(f"\n--- Training Our PPO on {rung_clean} (seed {seed}, {steps} steps) ---")
            res = run_single_our(rung_clean, seed, steps, gamma)
            all_results.append(res)
            print(f"[Finished Our PPO seed {seed}] Crash: {res['ci_str']} | Speed: {res['mean_speed_ms']:.2f} m/s | Return: {res['mean_return']:.2f} | Dom: {res['dominant_action']} ({res['dominant_action_frac']:.1%}) ({res['runtime_s']:.1f}s)")

    total_time = round(time.time() - start_total, 2)
    print(f"\n[Rung {rung_clean} Completed in {total_time}s ({total_time/60:.2f} min)]")

    # Save to json (append to existing if present)
    out_file = Path("parity") / f"results_{rung_clean.lower()}.json"
    existing = []
    if out_file.exists():
        try:
            with open(out_file, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    combined = existing + all_results
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved results to {out_file}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", type=str, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--include-20k", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-sb3", action="store_true")
    parser.add_argument("--skip-our", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else [0, 1, 2]

    run_rung(
        args.rung,
        steps=args.steps,
        gamma=args.gamma,
        include_20k=args.include_20k,
        run_baselines=not args.skip_baselines,
        run_sb3=not args.skip_sb3,
        run_our=not args.skip_our,
        seeds=seeds,
    )

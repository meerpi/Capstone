"""Script to sweep a grid of modest PID-Lagrangian gains (Kp, Ki) at 20,480 steps."""

import os
import sys
import json
import argparse
import subprocess
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_gain_point(kp: float, ki: float, timesteps: int = 20480, base_dir: str = "scratch/gain_sweep") -> dict[str, Any]:
    ckpt_dir = os.path.join(base_dir, f"kp_{kp}_ki_{ki}".replace(".", "p"))
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, "final.pt")

    cmd = [
        "uv", "run", "python", "train_optimal_overtaker.py",
        "--lagrangian",
        "--total-timesteps", str(timesteps),
        "--num-envs", "8",
        "--num-steps", "512",
        "--cost-limit", "0.05",
        "--lagrangian-kp", str(kp),
        "--lagrangian-ki", str(ki),
        "--checkpoint-dir", ckpt_dir,
        "--output-path", out_path,
        "--eval-episodes", "5",
        "--cuda",
    ]

    print(f"\n=======================================================", flush=True)
    print(f"--- Running Gain Point: Kp={kp}, Ki={ki} ---", flush=True)
    print(f"Directory: {ckpt_dir}", flush=True)
    print(f"=======================================================", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"Error running point: {result.stderr}", flush=True)
        return {"kp": kp, "ki": ki, "error": result.stderr}

    # Load history
    history_file = os.path.join(ckpt_dir, "lagrange_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            history = json.load(f)
    else:
        history = []

    lambdas = [h["lambda"] for h in history]
    ar_mags = [h["mag_reward"] for h in history]
    ac_mags = [h["mag_cost"] for h in history]
    ratios = [h["cost_to_reward_ratio"] for h in history]

    # Evaluate smoothness: variance of first differences in lambda
    if len(lambdas) > 1:
        diffs = [lambdas[i+1] - lambdas[i] for i in range(len(lambdas)-1)]
        is_oscillating = any(diffs[i] * diffs[i+1] < -1e-6 for i in range(len(diffs)-1))
        flatlined = all(abs(d) < 1e-6 for d in diffs) and (lambdas[-1] < 1e-4)
    else:
        is_oscillating = False
        flatlined = False

    return {
        "kp": kp,
        "ki": ki,
        "lambdas": lambdas,
        "final_lambda": lambdas[-1] if lambdas else 0.0,
        "mean_mag_ar": float(sum(ar_mags) / len(ar_mags)) if ar_mags else 0.0,
        "mean_mag_ac": float(sum(ac_mags) / len(ac_mags)) if ac_mags else 0.0,
        "mean_ratio": float(sum(ratios) / len(ratios)) if ratios else 0.0,
        "is_oscillating": is_oscillating,
        "flatlined": flatlined,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep PID gains.")
    parser.add_argument("--timesteps", type=int, default=20480)
    args = parser.parse_args()

    grid = [
        (0.1, 0.001),
        (0.1, 0.01),
        (0.5, 0.001),
        (0.5, 0.01),
        (1.0, 0.001),
        (1.0, 0.01),
    ]

    all_results = []
    for kp, ki in grid:
        res = run_gain_point(kp, ki, timesteps=args.timesteps)
        all_results.append(res)

    print("\n" + "=" * 90)
    print("=== PID-LAGRANGIAN GAIN SWEEP RESULTS (20,480 Timesteps) ===")
    print("=" * 90)
    header = f"{'Kp':<6} | {'Ki':<6} | {'Final λ':<9} | {'Mean |A_R|':<10} | {'Mean |λA_C|':<11} | {'|λA_C|/|A_R|':<12} | {'Trajectory (λ_1 → λ_5)':<25} | {'Behavior':<12}"
    print(header)
    print("-" * len(header))

    for r in all_results:
        kp = r["kp"]
        ki = r["ki"]
        fin_lam = f"{r['final_lambda']:.4f}"
        ar = f"{r['mean_mag_ar']:.3f}"
        ac = f"{r['mean_mag_ac']:.3f}"
        ratio = f"{r['mean_ratio']*100:.1f}%"
        traj_str = " → ".join([f"{l:.3f}" for l in r["lambdas"][:5]])
        if r["is_oscillating"]:
            beh = "Oscillating"
        elif r["flatlined"]:
            beh = "Flatlined"
        else:
            beh = "Smooth"
        print(f"{kp:<6.2f} | {ki:<6.3f} | {fin_lam:<9} | {ar:<10} | {ac:<11} | {ratio:<12} | {traj_str:<25} | {beh:<12}")
    print("=" * 90)

    # Save summary
    with open("scratch/gain_sweep/sweep_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

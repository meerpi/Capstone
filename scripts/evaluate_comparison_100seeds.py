import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from train_optimal_overtaker import evaluate


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute exact Clopper-Pearson binomial confidence interval [p_lower, p_upper].

    Uses binary search over the exact binomial tail sums with math.comb.
    Zero external dependencies (no scipy required).
    """
    if n == 0:
        return 0.0, 1.0

    alpha = 1.0 - confidence

    # Lower bound: P(Bin(n, p) >= k) = alpha / 2
    if k == 0:
        p_lower = 0.0
    else:
        target = alpha / 2.0
        low, high = 0.0, float(k) / n
        for _ in range(80):
            mid = (low + high) / 2.0
            # P(X >= k) = sum_{j=k}^n comb(n, j) * p^j * (1-p)^(n-j)
            tail_prob = sum(math.comb(n, j) * (mid ** j) * ((1.0 - mid) ** (n - j)) for j in range(k, n + 1))
            if tail_prob < target:
                low = mid
            else:
                high = mid
        p_lower = (low + high) / 2.0

    # Upper bound: P(Bin(n, p) <= k) = alpha / 2
    if k == n:
        p_upper = 1.0
    else:
        target = alpha / 2.0
        low, high = float(k) / n, 1.0
        for _ in range(80):
            mid = (low + high) / 2.0
            # P(X <= k) = sum_{j=0}^k comb(n, j) * p^j * (1-p)^(n-j)
            tail_prob = sum(math.comb(n, j) * (mid ** j) * ((1.0 - mid) ** (n - j)) for j in range(k + 1))
            if tail_prob < target:
                high = mid
            else:
                low = mid
        p_upper = (low + high) / 2.0

    return p_lower, p_upper


def evaluate_model(
    model_path: str,
    episodes: int = 100,
    seed_start: int = 2000,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate model on fixed deterministic seeds and return metrics with CI."""
    print(f"\nEvaluating: {model_path} on {episodes} seeds (start={seed_start})...", flush=True)
    results = evaluate(
        model_path=model_path,
        lanes_count=4,
        vehicles_density=1.4,
        vehicles_count=14,
        episodes=episodes,
        seed_start=seed_start,
        frame_stack_k=3,
        device=device,
        deterministic=True,
    )

    crashes = results["crashes"]
    total = results["episodes"]
    crash_rate = crashes / total
    p_lower, p_upper = clopper_pearson_ci(crashes, total, confidence=0.95)

    return {
        "crashes": crashes,
        "total": total,
        "crash_rate": crash_rate,
        "ci_lower": p_lower,
        "ci_upper": p_upper,
        "mean_speed_kmh": results["mean_speed_kmh"],
        "mean_overtakes": results["mean_overtakes"],
        "mean_lane_changes": results["mean_lane_changes"],
        "mean_duration": results["mean_duration"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="100-seed side-by-side model comparison.")
    parser.add_argument("--baseline", type=str, default="models/ppo_optimal_overtaker_4lane_best.pt")
    parser.add_argument("--failed-run", type=str, default=None, help="Path to failed unnormalized calibration run")
    parser.add_argument("--lagrangian", type=str, required=True, help="Path to new PPO-Lagrangian model")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    models = [("Baseline PPO (-50.0, 1.0)", args.baseline)]
    if args.failed_run and os.path.exists(args.failed_run):
        models.append(("Failed Unnorm (82% Crash)", args.failed_run))
    models.append(("PPO-Lagrangian (Sep Norm)", args.lagrangian))

    all_metrics = []
    for name, path in models:
        metrics = evaluate_model(path, episodes=args.episodes, seed_start=args.seed_start, device=args.device)
        all_metrics.append((name, metrics))

    col_w = 26
    print("\n" + "=" * (32 + col_w * len(models)))
    print(f"=== 100-SEED FAIR COMPARISON (Seeds {args.seed_start} - {args.seed_start + args.episodes - 1}) ===")
    print("=" * (32 + col_w * len(models)))
    
    header = f"{'Metric':<32}" + "".join(f" | {name:<{col_w}}" for name, _ in all_metrics)
    print(header)
    print("-" * len(header))

    def row(label, val_fn):
        cols = [f"{val_fn(m):<{col_w}}" for _, m in all_metrics]
        print(f"{label:<32}" + "".join(f" | {c}" for c in cols))

    row("Crash Rate (95% Clopper-Pearson)", lambda m: f"{m['crash_rate']*100:.1f}% [{m['ci_lower']*100:.1f}%, {m['ci_upper']*100:.1f}%]")
    row("Crashes / Episodes", lambda m: f"{m['crashes']}/{m['total']}")
    row("Mean Ego Speed", lambda m: f"{m['mean_speed_kmh']:.2f} km/h")
    row("Mean Overtakes / Episode", lambda m: f"{m['mean_overtakes']:.2f}")
    row("Mean Lane Changes / Episode", lambda m: f"{m['mean_lane_changes']:.2f}")
    row("Mean Episode Duration", lambda m: f"{m['mean_duration']:.1f} steps")
    print("=" * (32 + col_w * len(models)))


if __name__ == "__main__":
    main()

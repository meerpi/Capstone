"""Evaluate all fixed baselines across all ladder rungs on seeds 6000-6099.

Baselines:
- random
- always_IDLE
- always_SLOWER
- always_FASTER
- always_LANE_RIGHT
- IDM_MOBIL

Rungs:
- R0 (official highway-fast-v0)
- R2 (density 1.4)
- R3 (lanes 4, vehicles 14)
- R4 (duration 100)
- R5 (5 Hz)
- R6 (our reward)
- R7_GT (27-dim slot grid)
"""

import json
from pathlib import Path
import sys
import gymnasium as gym

sys.path.insert(0, ".")
from parity.env_factory import make_ladder_env
from parity.evaluator import (
    RandomPolicy,
    ConstantPolicy,
    IDMMobilPolicy,
    evaluate_policy,
)

RUNGS = ["R0", "R2", "R3", "R4", "R5", "R6", "R7_GT"]

BASELINES = [
    ("random", RandomPolicy(5), False),
    ("always_IDLE", ConstantPolicy(1), False),
    ("always_SLOWER", ConstantPolicy(4), False),
    ("always_FASTER", ConstantPolicy(3), False),
    ("always_LANE_RIGHT", ConstantPolicy(2), False),
    ("IDM_MOBIL", IDMMobilPolicy(), True),
]


def run_all_baselines():
    out_dir = Path("parity/baseline_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    master_results = {}

    for rung in RUNGS:
        print(f"\n==================== RUNNING BASELINES FOR {rung} ====================")
        master_results[rung] = {}
        env_fn = lambda r=rung: make_ladder_env(r)

        for name, policy, is_aware in BASELINES:
            json_file = out_dir / f"{rung}_{name}.json"
            if json_file.exists():
                print(f"[Cached] {rung} {name}")
                with open(json_file) as f:
                    master_results[rung][name] = json.load(f)
                continue

            print(f"[Evaluating] {rung} {name}...")
            res = evaluate_policy(env_fn, policy, seed_start=6000, n_episodes=100, is_env_aware=is_aware)
            res["rung"] = rung
            res["baseline_name"] = name
            with open(json_file, "w") as f:
                json.dump(res, f, indent=2)

            master_results[rung][name] = res
            print(f"  -> Crash: {res['ci_str']} | Speed: {res['mean_speed_ms']:.2f} m/s | Return: {res['mean_return']:.2f}")

    # Save summary
    summary_file = Path("parity/all_baselines_summary.json")
    with open(summary_file, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\n[Saved] All baseline results to {summary_file}")


if __name__ == "__main__":
    run_all_baselines()

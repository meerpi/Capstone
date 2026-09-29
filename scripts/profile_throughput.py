#!/usr/bin/env python3
"""Standalone Environment Stepping Throughput Profiler.

Profiles local environment-stepping throughput for HighwayEnv under:
1. Full wrapper stack (env_config.py factory -> TacticalLaneObservationWrapper
   -> ObservationStackWrapper / FrameStackTacticalWrapper k=3, 90-dim observation).
2. Bare environment (highway-fast-v0 with identical underlying scenario config, no wrappers).

Measures:
- Worker count sweep: 1, 2, 4, 8, 12, 16 parallel workers (AsyncVectorEnv).
- Steady-state throughput over at least 20,000 steps following a 1,000-step warmup.
- Wrapper overhead as percentage of bare-env time.
- Sustained 10-minute throughput profiling on the top-performing worker configuration
  logging every 30 seconds to detect thermal throttling and degradation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import env_config

# ObservationStackWrapper is the frame stacking wrapper used by the optimal overtaking agent
ObservationStackWrapper = env_config.FrameStackTacticalWrapper


def get_cpu_temp() -> float | None:
    """Read CPU temperature in degrees Celsius from sysfs hardware monitoring."""
    try:
        hwmon_dir = "/sys/class/hwmon"
        if os.path.exists(hwmon_dir):
            for h in os.listdir(hwmon_dir):
                name_file = os.path.join(hwmon_dir, h, "name")
                if os.path.exists(name_file):
                    with open(name_file, "r") as f:
                        name = f.read().strip()
                    if "k10temp" in name or "coretemp" in name:
                        for temp_input in ["temp1_input", "temp2_input", "temp3_input"]:
                            path = os.path.join(hwmon_dir, h, temp_input)
                            if os.path.exists(path):
                                with open(path, "r") as f:
                                    return float(f.read().strip()) / 1000.0
        tz_path = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(tz_path):
            with open(tz_path, "r") as f:
                return float(f.read().strip()) / 1000.0
    except Exception:
        pass
    return None


def make_wrapped_env_thunk(
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    seed: int = 42,
    frame_stack_k: int = 3,
) -> Callable[[], gym.Env]:
    """Create a thunk for instantiating the full wrapped environment stack."""
    def thunk() -> gym.Env:
        return env_config.make_optimal_env(
            lanes_count=lanes_count,
            vehicles_density=vehicles_density,
            vehicles_count=vehicles_count,
            seed=seed,
            frame_stack_k=frame_stack_k,
        )
    return thunk


def make_bare_env_thunk(
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    seed: int = 42,
    duration: int = 100,
) -> Callable[[], gym.Env]:
    """Create a thunk for instantiating bare highway-fast-v0 with identical underlying scenario config."""
    def thunk() -> gym.Env:
        cfg = copy.deepcopy(env_config.DEFAULT_ENV_CONFIG)
        cfg.update({
            "lanes_count": lanes_count,
            "vehicles_density": vehicles_density,
            "vehicles_count": vehicles_count,
            "duration": duration,
            "simulation_frequency": 5,
            "collision_reward": 0.0,
            "high_speed_reward": 0.0,
            "right_lane_reward": 0.0,
            "on_road_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
            "reward_speed_range": [18, 30],
        })
        env = gym.make("highway-fast-v0", config=cfg)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env
    return thunk


def profile_worker_configuration(
    thunk_factory: Callable[[int], Callable[[], gym.Env]],
    num_workers: int,
    min_steps: int = 20000,
    warmup_steps: int = 1000,
    action_type: str = "random",
    base_seed: int = 42,
) -> dict[str, Any]:
    """Measure steady-state steps/sec for a specific worker count after warmup."""
    env_fns = [thunk_factory(base_seed + i) for i in range(num_workers)]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    try:
        envs.reset(seed=base_seed)

        # Pre-generate cyclic action lookup table to eliminate RNG latency during timing
        table_size = 2048
        if action_type == "noop":
            action_table = np.ones((table_size, num_workers), dtype=np.int64)
        else:
            action_table = np.random.randint(0, 5, size=(table_size, num_workers), dtype=np.int64)

        # 1. Warmup phase (discarded)
        warmup_iters = max(1, math.ceil(warmup_steps / num_workers))
        for step in range(warmup_iters):
            action = action_table[step % table_size]
            envs.step(action)

        actual_warmup_steps = warmup_iters * num_workers

        # 2. Steady-state measurement phase
        measure_iters = math.ceil(min_steps / num_workers)
        actual_measure_steps = measure_iters * num_workers

        t_start = time.perf_counter()
        for step in range(measure_iters):
            action = action_table[step % table_size]
            envs.step(action)
        t_end = time.perf_counter()

        elapsed = t_end - t_start
        sps = actual_measure_steps / elapsed
        time_per_step_ms = (elapsed / actual_measure_steps) * 1000.0

        return {
            "workers": num_workers,
            "warmup_steps": actual_warmup_steps,
            "measured_steps": actual_measure_steps,
            "elapsed_seconds": elapsed,
            "sps": sps,
            "time_per_step_ms": time_per_step_ms,
        }

    finally:
        envs.close()


def run_sustained_test(
    num_workers: int,
    duration_seconds: float = 600.0,
    interval_seconds: float = 30.0,
    warmup_steps: int = 1000,
    action_type: str = "random",
    base_seed: int = 42,
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    frame_stack_k: int = 3,
) -> dict[str, Any]:
    """Run sustained throughput test for thermal throttling detection."""
    print(f"\n{'='*80}")
    print(f"SUSTAINED {duration_seconds/60:.1f}-MINUTE BENCHMARK ON BEST WORKER COUNT ({num_workers} Workers)")
    print(f"Logging throughput and CPU temperature every {interval_seconds:.0f} seconds...")
    print(f"{'='*80}\n", flush=True)

    def thunk_factory(seed: int) -> Callable[[], gym.Env]:
        return make_wrapped_env_thunk(
            lanes_count=lanes_count,
            vehicles_density=vehicles_density,
            vehicles_count=vehicles_count,
            seed=seed,
            frame_stack_k=frame_stack_k,
        )

    env_fns = [thunk_factory(base_seed + i) for i in range(num_workers)]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    intervals = []
    table_size = 4096
    if action_type == "noop":
        action_table = np.ones((table_size, num_workers), dtype=np.int64)
    else:
        action_table = np.random.randint(0, 5, size=(table_size, num_workers), dtype=np.int64)

    try:
        envs.reset(seed=base_seed)

        # Warmup
        warmup_iters = max(1, math.ceil(warmup_steps / num_workers))
        for step in range(warmup_iters):
            envs.step(action_table[step % table_size])

        print(f"Warmup of {warmup_iters * num_workers} steps complete. Starting {duration_seconds:.0f}s timer...\n", flush=True)

        overall_start = time.perf_counter()
        next_interval_target = overall_start + interval_seconds
        test_end_time = overall_start + duration_seconds

        interval_idx = 1
        interval_start_time = overall_start
        interval_step_count = 0
        total_steps = 0
        step_idx = 0

        header = f"{'Interval':<10} | {'Elapsed (s)':<12} | {'Window Steps':<14} | {'Window SPS':<12} | {'Cumul SPS':<12} | {'CPU Temp (°C)':<14}"
        print(header)
        print("-" * len(header), flush=True)

        while True:
            action = action_table[step_idx % table_size]
            envs.step(action)
            step_idx += 1
            interval_step_count += num_workers
            total_steps += num_workers

            now = time.perf_counter()
            if now >= next_interval_target or now >= test_end_time:
                interval_duration = now - interval_start_time
                cumul_duration = now - overall_start
                interval_sps = interval_step_count / interval_duration if interval_duration > 0 else 0.0
                cumul_sps = total_steps / cumul_duration if cumul_duration > 0 else 0.0
                cpu_temp = get_cpu_temp()
                temp_str = f"{cpu_temp:.1f}" if cpu_temp is not None else "N/A"

                interval_record = {
                    "interval": interval_idx,
                    "elapsed_seconds": round(cumul_duration, 2),
                    "window_steps": interval_step_count,
                    "window_duration_seconds": round(interval_duration, 3),
                    "window_sps": round(interval_sps, 2),
                    "cumulative_sps": round(cumul_sps, 2),
                    "cpu_temp_c": round(cpu_temp, 1) if cpu_temp is not None else None,
                }
                intervals.append(interval_record)

                print(
                    f"{interval_idx:<10} | {cumul_duration:>10.1f}s | {interval_step_count:>14,d} | "
                    f"{interval_sps:>12.1f} | {cumul_sps:>12.1f} | {temp_str:>14}",
                    flush=True,
                )

                if now >= test_end_time:
                    break

                interval_idx += 1
                interval_start_time = now
                interval_step_count = 0
                next_interval_target += interval_seconds

        overall_elapsed = time.perf_counter() - overall_start
        overall_sps = total_steps / overall_elapsed

        first_window_sps = intervals[0]["window_sps"] if intervals else overall_sps
        last_window_sps = intervals[-1]["window_sps"] if intervals else overall_sps
        decay_pct = ((first_window_sps - last_window_sps) / first_window_sps) * 100.0 if first_window_sps > 0 else 0.0
        throttled = decay_pct > 10.0

        return {
            "workers": num_workers,
            "duration_seconds": overall_elapsed,
            "total_steps": total_steps,
            "overall_sps": overall_sps,
            "first_window_sps": first_window_sps,
            "last_window_sps": last_window_sps,
            "decay_pct": decay_pct,
            "throttling_detected": throttled,
            "intervals": intervals,
        }

    finally:
        envs.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile local environment-stepping throughput for Capstone RL project."
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 12, 16],
        help="Worker counts to sweep (default: 1 2 4 8 12 16).",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=20000,
        help="Minimum steady-state environment steps to measure per worker count (default: 20000).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Warmup steps to discard before measuring (default: 1000).",
    )
    parser.add_argument(
        "--sustained-duration",
        type=float,
        default=600.0,
        help="Duration of sustained thermal-throttling test in seconds (default: 600.0s = 10 min).",
    )
    parser.add_argument(
        "--sustained-interval",
        type=float,
        default=30.0,
        help="Logging interval for sustained test in seconds (default: 30.0s).",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        choices=["random", "noop"],
        default="random",
        help="Action selection during profiling: 'random' or 'noop' (default: random).",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=4,
        help="Highway lanes count (default: 4).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.4,
        help="Traffic density parameter (default: 1.4).",
    )
    parser.add_argument(
        "--vehicles",
        type=int,
        default=14,
        help="NPC vehicles count (default: 14).",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=3,
        help="Number of observation frames stacked (default: 3 -> 90-dim obs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--skip-sustained",
        action="store_true",
        default=False,
        help="Skip the sustained 10-minute test.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmark_throughput_results.json",
        help="File path to save JSON results (default: benchmark_throughput_results.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("CAPSTONE LOCAL ENVIRONMENT-STEPPING THROUGHPUT PROFILER")
    print(f"Scenario: highway-fast-v0 (Lanes: {args.lanes}, Density: {args.density}, Vehicles: {args.vehicles})")
    print(f"Observation: TacticalLaneObservationWrapper + FrameStackTacticalWrapper (k={args.frame_stack} -> 90-dim)")
    print(f"Sensor Tier: Full-ADAS (360-degree, all lanes)")
    print(f"Action Policy: {args.action_type.upper()}")
    print(f"Worker Sweep: {args.workers}")
    print(f"Measurement Budget: Warmup >= {args.warmup_steps:,} steps | Steady-state >= {args.min_steps:,} steps")
    print("=" * 80)

    sweep_results = []

    for w in args.workers:
        print(f"\n>>> Profiling Worker Count: {w} workers...", flush=True)

        # 1. Profile wrapped stack
        wrapped_thunk_factory = lambda s: make_wrapped_env_thunk(
            lanes_count=args.lanes,
            vehicles_density=args.density,
            vehicles_count=args.vehicles,
            seed=s,
            frame_stack_k=args.frame_stack,
        )
        print(f"  [1/2] Wrapped Stack: Running {args.warmup_steps:,} warmup + >= {args.min_steps:,} measurement...", end="", flush=True)
        res_wrapped = profile_worker_configuration(
            thunk_factory=wrapped_thunk_factory,
            num_workers=w,
            min_steps=args.min_steps,
            warmup_steps=args.warmup_steps,
            action_type=args.action_type,
            base_seed=args.seed,
        )
        print(f" Done! -> {res_wrapped['sps']:.1f} steps/sec ({res_wrapped['time_per_step_ms']:.3f} ms/step)", flush=True)

        # 2. Profile bare highway-fast-v0
        bare_thunk_factory = lambda s: make_bare_env_thunk(
            lanes_count=args.lanes,
            vehicles_density=args.density,
            vehicles_count=args.vehicles,
            seed=s,
        )
        print(f"  [2/2] Bare Environment: Running {args.warmup_steps:,} warmup + >= {args.min_steps:,} measurement...", end="", flush=True)
        res_bare = profile_worker_configuration(
            thunk_factory=bare_thunk_factory,
            num_workers=w,
            min_steps=args.min_steps,
            warmup_steps=args.warmup_steps,
            action_type=args.action_type,
            base_seed=args.seed,
        )
        print(f" Done! -> {res_bare['sps']:.1f} steps/sec ({res_bare['time_per_step_ms']:.3f} ms/step)", flush=True)

        # Wrapper overhead calculation
        # Time per step: T_wrapped vs T_bare
        # Overhead % = (T_wrapped - T_bare) / T_bare * 100% = (sps_bare / sps_wrapped - 1) * 100%
        sps_w = res_wrapped["sps"]
        sps_b = res_bare["sps"]
        t_w_ms = res_wrapped["time_per_step_ms"]
        t_b_ms = res_bare["time_per_step_ms"]
        overhead_pct = ((sps_b / sps_w) - 1.0) * 100.0

        row = {
            "workers": w,
            "sps_wrapped": round(sps_w, 2),
            "sps_bare": round(sps_b, 2),
            "time_wrapped_ms": round(t_w_ms, 4),
            "time_bare_ms": round(t_b_ms, 4),
            "overhead_pct": round(overhead_pct, 2),
            "measured_steps_wrapped": res_wrapped["measured_steps"],
            "measured_steps_bare": res_bare["measured_steps"],
        }
        sweep_results.append(row)

    # Print Sweep Summary Table
    print("\n" + "=" * 80)
    print("THROUGHPUT SWEEP SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Worker Count':<14} | {'Steps/sec (Wrapper)':<22} | {'Steps/sec (Bare Env)':<22} | {'Wrapper Overhead (%)':<20}"
    print(header)
    print("-" * len(header))
    for r in sweep_results:
        print(
            f"{r['workers']:<14} | {r['sps_wrapped']:>22.1f} | {r['sps_bare']:>22.1f} | {r['overhead_pct']:>19.2f}%"
        )
    print("=" * 80)

    # Determine best worker count
    best_config = max(sweep_results, key=lambda x: x["sps_wrapped"])
    best_workers = best_config["workers"]
    print(f"\nBest-performing worker count (Wrapper): {best_workers} workers ({best_config['sps_wrapped']:.1f} steps/sec)")

    sustained_results = None
    if not args.skip_sustained:
        sustained_results = run_sustained_test(
            num_workers=best_workers,
            duration_seconds=args.sustained_duration,
            interval_seconds=args.sustained_interval,
            warmup_steps=args.warmup_steps,
            action_type=args.action_type,
            base_seed=args.seed,
            lanes_count=args.lanes,
            vehicles_density=args.density,
            vehicles_count=args.vehicles,
            frame_stack_k=args.frame_stack,
        )

        print("\n" + "=" * 80)
        print("SUSTAINED TEST TREND ANALYSIS")
        print("=" * 80)
        print(f"Target Duration:    {args.sustained_duration/60:.1f} minutes ({args.sustained_duration:.0f}s)")
        print(f"Actual Elapsed:     {sustained_results['duration_seconds']:.1f}s")
        print(f"Total Steps Taken:  {sustained_results['total_steps']:,d}")
        print(f"Overall Mean SPS:   {sustained_results['overall_sps']:.1f} steps/sec")
        print(f"Initial Window SPS: {sustained_results['first_window_sps']:.1f} steps/sec")
        print(f"Final Window SPS:   {sustained_results['last_window_sps']:.1f} steps/sec")
        print(f"Throughput Decay:   {sustained_results['decay_pct']:+.2f}%")
        if sustained_results["throttling_detected"]:
            print(">>> WARNING: Thermal-throttling degradation detected (> 10% decay from initial window)!")
        else:
            print(">>> PASS: Throughput remained stable within <= 10% variation (No thermal throttling degradation detected).")
        print("=" * 80)

    # Save to JSON
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "cpu": "AMD Ryzen 7 5800H (8c/16t)",
            "gpu": "RTX 3060 Laptop 6GB",
        },
        "config": vars(args),
        "sweep_results": sweep_results,
        "best_workers": best_workers,
        "sustained_results": sustained_results,
    }

    output_path = os.path.join(REPO_ROOT, args.output_json)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nAll profiling results written to: {output_path}")


if __name__ == "__main__":
    main()

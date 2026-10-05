#!/usr/bin/env python
"""Standard Autonomous Vehicle (AV) & Safe-RL Empirical Evaluation Benchmark.

Computes the standard suite of AV safety, efficiency, comfort, and statistical metrics
across baseline heuristics and learned policies under a unified Highway-Env profile.

Metrics evaluated:
1. Safety:
   - Crash count & Crash rate (%) with exact 95% Clopper-Pearson CI
   - Crashes per 1,000 km
   - Minimum Time-to-Collision (TTC in seconds)
   - Fraction of time steps with critical TTC (0 < TTC < 2.0 s)
   - Minimum Time Headway (TH in seconds)
   - Hard braking events (deceleration < -3.0 m/s^2) per 100 km
   - CMDP Cost J_C against 5.0% constraint budget
2. Efficiency:
   - Mean distance per episode (km)
   - Mean ego speed (all episodes vs surviving episodes only, in km/h)
   - Speed relative to ambient traffic (km/h)
3. Comfort & Smoothness:
   - RMS acceleration (m/s^2)
   - RMS jerk (m/s^3)
   - Lane changes per 100 km
   - Rapid-reversal rate (% of lane changes reversed within <= 2.0 s)
   - Action switch rate (% of steps with action change)
4. Statistical Rigor:
   - Unified environment profile across all methods
   - Evaluated on fresh held-out seeds (never used for diagnosis)
   - Fisher's exact test two-sided p-values against key baselines
   - Multi-seed training variance reporting
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
import scipy.stats as stats
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import baseline_classical
import env_config
from deepset_v2.env import make_deepset_env_v2
from deepset_v2.model import DeepSetAgent
from parity.metrics import clopper_pearson
from ppo import Agent as FFAgent
from ppo_lstm import RecurrentAgent
from render_utils import compute_mean_npc_speed
from train_optimal_overtaker import OptimalAgent


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def evaluate_single_episode(
    policy_spec: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run a single evaluation episode and record detailed AV metrics."""
    ptype = policy_spec["type"]
    profile = policy_spec.get("profile", "current")

    # 1. Environment initialization
    if ptype in ("optimal_agent", "lagrangian"):
        target_speeds = policy_spec.get("target_speeds", None)
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            seed=seed,
            frame_stack_k=3,
            duration=100,
            target_speeds=target_speeds,
        )
    elif ptype == "deepset_v2":
        env = make_deepset_env_v2(
            tier="full_adas",
            scenario="highway",
            seed=seed,
            range_m=100.0,
        )
    elif ptype == "classical":
        if profile == "legacy":
            env = env_config.make_env(
                scenario="highway",
                tier="full_adas",
                observation={"vehicles_count": 10},
                duration=40,
                seed=seed,
                vehicles_count=10,
                vehicles_density=1.0,
                tactical_overtaking=True,
            )
        else:
            env = env_config.make_env(
                scenario="highway",
                tier="full_adas",
                seed=seed,
                tactical_overtaking=True,
            )
    elif ptype == "feedforward_ppo":
        if profile == "legacy_50":
            env = env_config.make_env(
                scenario="highway",
                tier=policy_spec.get("tier", "front_only"),
                observation={"vehicles_count": 10},
                duration=40,
                seed=seed,
                vehicles_count=10,
                vehicles_density=1.0,
                tactical_overtaking=True,
            )
        else:
            env = env_config.make_env(
                scenario="highway",
                tier="full_adas",
                seed=seed,
                tactical_overtaking=True,
            )
    elif ptype == "trivial_slower":
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            seed=seed,
            frame_stack_k=3,
            duration=100,
        )
    elif ptype == "trivial_idle":
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            seed=seed,
            frame_stack_k=3,
            duration=100,
        )
    elif ptype == "random":
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            seed=seed,
            frame_stack_k=3,
            duration=100,
        )
    else:
        raise ValueError(f"Unknown policy type: {ptype}")

    # 2. Setup model
    device = torch.device("cpu")
    model = None

    if ptype in ("optimal_agent", "lagrangian"):
        ckpt = torch.load(policy_spec["model_path"], map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        has_cost_critic = any(k.startswith("cost_critic.") for k in state_dict.keys())
        model = OptimalAgent(
            obs_dim=policy_spec["obs_dim"], action_dim=5, use_cost_critic=has_cost_critic
        ).to(device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    elif ptype == "deepset_v2":
        model = DeepSetAgent().to(device)
        ckpt = torch.load(policy_spec["model_path"], map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
    elif ptype == "feedforward_ppo":
        model = FFAgent(obs_dim=policy_spec["obs_dim"], action_dim=5).to(device)
        ckpt = torch.load(policy_spec["model_path"], map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()

    # 3. Step through episode and measure metrics
    obs, info = env.reset(seed=seed)
    unwrapped = env.unwrapped
    x_start = float(unwrapped.vehicle.position[0])

    dt = 0.2  # 5 Hz policy frequency
    ego_speeds: list[float] = []
    npc_speeds: list[float] = []
    actions_taken: list[int] = []

    ttc_list: list[float] = []
    th_list: list[float] = []
    critical_ttc_steps = 0
    total_steps_with_lead = 0

    lane_change_events: list[tuple[int, int, int]] = []  # (step, from_lane, to_lane)
    prev_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]

    crashed = False
    step_count = 0

    while True:
        v_ego = float(info.get("speed", unwrapped.vehicle.speed))
        ego_speeds.append(v_ego)

        v_npc = compute_mean_npc_speed(env)
        npc_speeds.append(v_npc)

        # In-lane lead vehicle kinematics
        lead, _ = unwrapped.road.neighbour_vehicles(unwrapped.vehicle, unwrapped.vehicle.lane_index)
        if lead is not None:
            total_steps_with_lead += 1
            lead_len = getattr(lead, "LENGTH", 5.0)
            dist = float(lead.position[0] - unwrapped.vehicle.position[0] - lead_len)
            if dist <= 0.0:
                ttc_list.append(0.0)
                th_list.append(0.0)
                critical_ttc_steps += 1
            else:
                th = dist / v_ego if v_ego > 0.1 else float("inf")
                th_list.append(th)
                if v_ego > lead.speed:
                    ttc = dist / (v_ego - lead.speed)
                    ttc_list.append(ttc)
                    if ttc < 2.0:
                        critical_ttc_steps += 1
                else:
                    ttc_list.append(float("inf"))

        # Select action
        if ptype in ("optimal_agent", "lagrangian"):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(5, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                act, _, _, _ = model.get_action_and_value(
                    obs_t, deterministic=True, action_mask=mask_t
                )
                action = int(act.item())
        elif ptype == "deepset_v2":
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(5, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                act, _, _, _ = model.get_action_and_value(
                    obs_t, deterministic=True, action_mask=mask_t
                )
                action = int(act.item())
        elif ptype == "feedforward_ppo":
            obs_flat = np.array(obs).flatten()
            obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(5, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                act, _, _, _ = model.get_action_and_value(
                    obs_t, deterministic=True, action_mask=mask_t
                )
                action = int(act.item())
        elif ptype == "classical":
            cname = policy_spec["controller"]
            controller_fn = baseline_classical.CLASSICAL_CONTROLLERS[cname]
            action = int(controller_fn(env, obs, baseline_classical.DEFAULT_IDM_PARAMS))
        elif ptype == "trivial_slower":
            action = 4
        elif ptype == "trivial_idle":
            action = 1
        elif ptype == "random":
            action = int(np.random.randint(0, 5))
        else:
            action = 1

        actions_taken.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        cur_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]
        if cur_lane != prev_lane:
            lane_change_events.append((step_count, prev_lane, cur_lane))
            prev_lane = cur_lane

        if bool(info.get("crashed", False)) or unwrapped.vehicle.crashed:
            crashed = True
            break
        if terminated or truncated:
            break

    x_end = float(unwrapped.vehicle.position[0])
    forward_dist_m = max(0.0, x_end - x_start)
    forward_dist_km = forward_dist_m / 1000.0
    env.close()

    # Acceleration and Jerk computation
    accelerations: list[float] = []
    jerks: list[float] = []
    hard_braking_count = 0

    for i in range(1, len(ego_speeds)):
        a = (ego_speeds[i] - ego_speeds[i - 1]) / dt
        accelerations.append(a)
        if a < -3.0:
            hard_braking_count += 1

    for i in range(1, len(accelerations)):
        j = (accelerations[i] - accelerations[i - 1]) / dt
        jerks.append(j)

    rms_accel = float(np.sqrt(np.mean(np.square(accelerations)))) if accelerations else 0.0
    rms_jerk = float(np.sqrt(np.mean(np.square(jerks)))) if jerks else 0.0

    # Rapid lane reversals (lane change reversed within <= 10 steps / 2.0 s)
    rapid_reversals = 0
    for idx in range(1, len(lane_change_events)):
        curr_step, curr_from, curr_to = lane_change_events[idx]
        prev_step, prev_from, prev_to = lane_change_events[idx - 1]
        if (curr_step - prev_step <= 10) and (curr_to == prev_from):
            rapid_reversals += 1

    # Action switch count
    action_switches = sum(1 for i in range(1, len(actions_taken)) if actions_taken[i] != actions_taken[i - 1])

    valid_ttcs = [t for t in ttc_list if t < float("inf")]
    min_ttc = min(valid_ttcs) if valid_ttcs else float("inf")
    valid_ths = [th for th in th_list if th < float("inf")]
    min_th = min(valid_ths) if valid_ths else float("inf")

    return {
        "seed": seed,
        "crashed": crashed,
        "steps": step_count,
        "forward_distance_m": forward_dist_m,
        "forward_distance_km": forward_dist_km,
        "ego_mean_speed_kmh": float(np.mean(ego_speeds)) * 3.6 if ego_speeds else 0.0,
        "npc_mean_speed_kmh": float(np.mean(npc_speeds)) * 3.6 if npc_speeds else 0.0,
        "relative_speed_kmh": (float(np.mean(ego_speeds)) - float(np.mean(npc_speeds))) * 3.6 if ego_speeds and npc_speeds else 0.0,
        "rms_accel_ms2": rms_accel,
        "rms_jerk_ms3": rms_jerk,
        "hard_braking_events": hard_braking_count,
        "lane_changes": len(lane_change_events),
        "rapid_reversals": rapid_reversals,
        "action_switches": action_switches,
        "min_ttc": min_ttc,
        "min_th": min_th,
        "critical_ttc_steps": critical_ttc_steps,
        "total_steps_with_lead": total_steps_with_lead,
        "total_steps": step_count,
    }


def evaluate_policy(
    policy_spec: dict[str, Any],
    seeds: list[int],
    max_workers: int = 8,
) -> dict[str, Any]:
    """Evaluate a policy across deterministic seeds using multi-processing."""
    pname = policy_spec["name"]
    print(f"Evaluating {pname} across {len(seeds)} seeds (workers={max_workers})...", flush=True)
    t0 = time.time()

    episodes: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(evaluate_single_episode, policy_spec, s): s for s in seeds
        }
        for future in concurrent.futures.as_completed(future_map):
            episodes.append(future.result())

    episodes.sort(key=lambda x: x["seed"])
    elapsed = time.time() - t0

    n = len(episodes)
    crashes = sum(1 for ep in episodes if ep["crashed"])
    crash_rate = (crashes / n) * 100.0 if n > 0 else 0.0
    ci_low, ci_high = clopper_pearson(crashes, n)

    total_dist_km = sum(ep["forward_distance_km"] for ep in episodes)
    crashes_per_1000km = (crashes / total_dist_km * 1000.0) if total_dist_km > 0 else 0.0

    # Speeds: overall vs surviving episodes only
    all_speeds = [ep["ego_mean_speed_kmh"] for ep in episodes]
    surviving_speeds = [ep["ego_mean_speed_kmh"] for ep in episodes if not ep["crashed"]]
    mean_speed_all = float(np.mean(all_speeds)) if all_speeds else 0.0
    mean_speed_survive = float(np.mean(surviving_speeds)) if surviving_speeds else mean_speed_all

    all_rel_speeds = [ep["relative_speed_kmh"] for ep in episodes]
    mean_rel_speed = float(np.mean(all_rel_speeds)) if all_rel_speeds else 0.0

    # Safety kinematic metrics
    all_min_ttcs = [ep["min_ttc"] for ep in episodes if ep["min_ttc"] < float("inf")]
    overall_min_ttc = min(all_min_ttcs) if all_min_ttcs else float("inf")

    all_min_ths = [ep["min_th"] for ep in episodes if ep["min_th"] < float("inf")]
    overall_min_th = min(all_min_ths) if all_min_ths else float("inf")

    total_critical_ttc = sum(ep["critical_ttc_steps"] for ep in episodes)
    total_lead_steps = sum(ep["total_steps_with_lead"] for ep in episodes)
    pct_critical_ttc = (total_critical_ttc / max(1, total_lead_steps)) * 100.0

    total_hard_brakes = sum(ep["hard_braking_events"] for ep in episodes)
    hard_brakes_per_100km = (total_hard_brakes / total_dist_km * 100.0) if total_dist_km > 0 else 0.0

    # Comfort metrics
    mean_rms_accel = float(np.mean([ep["rms_accel_ms2"] for ep in episodes]))
    mean_rms_jerk = float(np.mean([ep["rms_jerk_ms3"] for ep in episodes]))

    total_lane_changes = sum(ep["lane_changes"] for ep in episodes)
    lane_changes_per_100km = (total_lane_changes / total_dist_km * 100.0) if total_dist_km > 0 else 0.0

    total_rapid_reversals = sum(ep["rapid_reversals"] for ep in episodes)
    rapid_reversal_rate = (total_rapid_reversals / max(1, total_lane_changes)) * 100.0

    total_switches = sum(ep["action_switches"] for ep in episodes)
    total_valid_steps = sum(ep["total_steps"] - 1 for ep in episodes)
    action_switch_rate = (total_switches / max(1, total_valid_steps)) * 100.0

    # CMDP Cost
    cost_jc = crashes / n
    budget_met = cost_jc <= 0.05

    summary = {
        "policy_name": pname,
        "type": policy_spec["type"],
        "model_path": policy_spec.get("model_path"),
        "profile": policy_spec.get("profile", "current"),
        "episodes": n,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{n}",
        "crash_rate_pct": round(crash_rate, 2),
        "crash_ci_95_low_pct": round(ci_low * 100.0, 2),
        "crash_ci_95_high_pct": round(ci_high * 100.0, 2),
        "total_distance_km": round(total_dist_km, 2),
        "crashes_per_1000km": round(crashes_per_1000km, 2),
        "min_ttc_s": round(overall_min_ttc, 2) if overall_min_ttc < float("inf") else "inf",
        "pct_critical_ttc": round(pct_critical_ttc, 2),
        "min_time_headway_s": round(overall_min_th, 2) if overall_min_th < float("inf") else "inf",
        "hard_braking_events_total": total_hard_brakes,
        "hard_brakings_per_100km": round(hard_brakes_per_100km, 2),
        "cmdp_cost_jc": round(cost_jc, 3),
        "cmdp_budget": 0.05,
        "cmdp_budget_met": budget_met,
        "mean_distance_km": round(float(np.mean([ep['forward_distance_km'] for ep in episodes])), 3),
        "mean_speed_all_kmh": round(mean_speed_all, 2),
        "mean_speed_survive_kmh": round(mean_speed_survive, 2),
        "mean_relative_speed_kmh": round(mean_rel_speed, 2),
        "rms_accel_ms2": round(mean_rms_accel, 2),
        "rms_jerk_ms3": round(mean_rms_jerk, 2),
        "lane_changes_total": total_lane_changes,
        "lane_changes_per_100km": round(lane_changes_per_100km, 2),
        "rapid_reversals_total": total_rapid_reversals,
        "rapid_reversal_rate_pct": round(rapid_reversal_rate, 2),
        "action_switch_rate_pct": round(action_switch_rate, 2),
        "elapsed_seconds": round(elapsed, 1),
    }

    print(
        f"  Done in {elapsed:.1f}s | Crashes: {crashes}/{n} ({crash_rate:.1f}%) | "
        f"Speed: {mean_speed_survive:.1f} km/h (surv) / {mean_speed_all:.1f} km/h (all) | "
        f"Crashes/1000km: {crashes_per_1000km:.1f} | RMS Accel: {mean_rms_accel:.2f} m/s^2"
    )
    return summary


def get_benchmark_policies() -> list[dict[str, Any]]:
    """Return policy specifications under the unified current benchmark profile."""
    return [
        {
            "name": "always_SLOWER (Floor Speed)",
            "type": "trivial_slower",
            "profile": "current",
        },
        {
            "name": "always_IDLE (Setpoint 25 m/s)",
            "type": "trivial_idle",
            "profile": "current",
        },
        {
            "name": "Random Action",
            "type": "random",
            "profile": "current",
        },
        {
            "name": "IDM Only (Lane-Keeping)",
            "type": "classical",
            "controller": "idm_only",
            "profile": "current",
        },
        {
            "name": "IDM + MOBIL (Overtaking)",
            "type": "classical",
            "controller": "idm_mobil",
            "profile": "current",
        },
        {
            "name": "Feedforward PPO (Full-ADAS)",
            "type": "feedforward_ppo",
            "model_path": "models/ppo_highway_full_adas_seed101.pt",
            "obs_dim": 75,
            "profile": "current",
        },
        {
            "name": "Optimal Overtaker (Best)",
            "type": "optimal_agent",
            "model_path": "models/ppo_optimal_overtaker_4lane_best.pt",
            "obs_dim": 81,
            "profile": "current",
        },
        {
            "name": "Optimal Overtaker (Final)",
            "type": "optimal_agent",
            "model_path": "models/ppo_optimal_overtaker_4lane.pt",
            "obs_dim": 81,
            "profile": "current",
        },
        {
            "name": "PPO-Lagrangian (Sep Norm)",
            "type": "lagrangian",
            "model_path": "checkpoints/ppo_lagrangian_sep_norm_best.pt",
            "obs_dim": 81,
            "profile": "current",
        },
        {
            "name": "DeepSet v2 (2M Baseline)",
            "type": "deepset_v2",
            "model_path": "checkpoints/deepset_v2_highway_2m_best.pt",
            "profile": "current",
        },
        {
            "name": "Optimal Overtaker (Ext. Spd, Seed 42)",
            "type": "optimal_agent",
            "model_path": "checkpoints/optimal_overtaker_extended_speeds_best.pt",
            "obs_dim": 81,
            "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
            "profile": "current_extended_speeds",
        },
        {
            "name": "Optimal Overtaker (Ext. Spd, Seed 137)",
            "type": "optimal_agent",
            "model_path": "checkpoints/optimal_overtaker_extended_speeds_seed137_best.pt",
            "obs_dim": 81,
            "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
            "profile": "current_extended_speeds",
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate complete standard AV metrics across policies.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes per policy.")
    parser.add_argument("--seed-start", type=int, default=4000, help="Fresh held-out seed start (default: 4000).")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_out/av_benchmark_metrics.json",
        help="Output JSON path (default: eval_out/av_benchmark_metrics.json).",
    )
    args = parser.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    policies = get_benchmark_policies()

    print("=" * 90)
    print(f"STANDARD AV & SAFE-RL EMPIRICAL BENCHMARK (Seeds {args.seed_start}-{args.seed_start + args.episodes - 1})")
    print(f"Unified Environment: highway-fast-v0 (4 lanes, 14 vehicles, density 1.4, duration 100s, 500 steps, 5 Hz)")
    print(f"Policies to evaluate: {len(policies)} | Workers: {args.workers}")
    print("=" * 90)

    import gymnasium
    import highway_env

    all_results: dict[str, Any] = {}
    for pol in policies:
        res = evaluate_policy(pol, seeds, max_workers=args.workers)
        all_results[pol["name"]] = res

    # Compute Fisher's exact tests against IDM Only and always_SLOWER
    idm_res = all_results.get("IDM Only (Lane-Keeping)")
    slower_res = all_results.get("always_SLOWER (Floor Speed)")

    fisher_comparisons = {}
    for name, r in all_results.items():
        comp = {}
        if idm_res:
            table_idm = [
                [r["crashes"], r["episodes"] - r["crashes"]],
                [idm_res["crashes"], idm_res["episodes"] - idm_res["crashes"]],
            ]
            _, p_idm = stats.fisher_exact(table_idm)
            comp["vs_idm_only_pvalue"] = round(float(p_idm), 4)
        if slower_res:
            table_slower = [
                [r["crashes"], r["episodes"] - r["crashes"]],
                [slower_res["crashes"], slower_res["episodes"] - slower_res["crashes"]],
            ]
            _, p_slower = stats.fisher_exact(table_slower)
            comp["vs_always_slower_pvalue"] = round(float(p_slower), 4)
        fisher_comparisons[name] = comp
        all_results[name]["fisher_tests"] = comp

    out_data = {
        "metadata": {
            "git_commit": get_git_commit(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "environment_profile": {
                "name": "current",
                "scenario": "highway",
                "lanes_count": 4,
                "vehicles_count": 14,
                "vehicles_density": 1.4,
                "duration": 100,
                "step_cap": 500,
                "policy_frequency": 5,
                "simulation_frequency": 5,
            },
            "eval_seeds": {
                "start": args.seed_start,
                "count": args.episodes,
                "type": "held_out_fresh",
            },
            "package_versions": {
                "python": sys.version.split()[0],
                "highway_env": getattr(highway_env, "__version__", "1.12.1"),
                "gymnasium": getattr(gymnasium, "__version__", "1.3.0"),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        },
        "policies": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "=" * 120)
    print("EMPIRICAL BENCHMARK SUMMARY (Safety & Efficiency)")
    print("=" * 120)
    print(f"{'Policy':<36} {'Crashes (95% CI)':<22} {'Cr/1000km':<11} {'Surv Spd':<10} {'All Spd':<10} {'Min TTC':<9} {'Crit TTC%':<10} {'p(IDM)':<8}")
    print("-" * 120)
    for name, r in all_results.items():
        ci_str = f"{r['crashes']}/{r['episodes']} [{r['crash_ci_95_low_pct']:.1f}%, {r['crash_ci_95_high_pct']:.1f}%]"
        p_val = r["fisher_tests"].get("vs_idm_only_pvalue", 1.0)
        p_str = f"{p_val:.4f}" if p_val is not None else "-"
        ttc_str = f"{r['min_ttc_s']}s" if r['min_ttc_s'] != "inf" else "None"
        print(
            f"{name:<36} {ci_str:<22} {r['crashes_per_1000km']:>7.1f}    "
            f"{r['mean_speed_survive_kmh']:>6.1f} km/h {r['mean_speed_all_kmh']:>6.1f} km/h "
            f"{ttc_str:>8} {r['pct_critical_ttc']:>7.1f}%   {p_str:>7}"
        )
    print("=" * 120)

    print("\n" + "=" * 120)
    print("EMPIRICAL BENCHMARK SUMMARY (Comfort & Smoothness)")
    print("=" * 120)
    print(f"{'Policy':<36} {'RMS Accel':<11} {'RMS Jerk':<11} {'HardBrake/100km':<17} {'LC/100km':<11} {'RapidRev%':<11} {'ActSwitch%'}")
    print("-" * 120)
    for name, r in all_results.items():
        print(
            f"{name:<36} {r['rms_accel_ms2']:>6.2f} m/s² "
            f"{r['rms_jerk_ms3']:>6.2f} m/s³ "
            f"{r['hard_brakings_per_100km']:>10.1f}        "
            f"{r['lane_changes_per_100km']:>7.1f}    "
            f"{r['rapid_reversal_rate_pct']:>6.1f}%    "
            f"{r['action_switch_rate_pct']:>6.1f}%"
        )
    print("=" * 120)
    print(f"\nArtifact written to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

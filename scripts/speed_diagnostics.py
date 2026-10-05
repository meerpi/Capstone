#!/usr/bin/env python
"""Speed Diagnostics: Measure whether agent slowness is real and systemic.

Evaluates all available checkpoints and classical baselines across benchmark
seeds 2000-2099 on their native environment profiles.

Measures:
- Ego mean speed (km/h and m/s)
- Mean NPC traffic flow speed (km/h and m/s)
- Mean in-lane lead vehicle speed (km/h and m/s)
- Relative speed (ego speed minus traffic flow speed)
- Total forward distance traveled (meters)
- Percentage of steps at each target-speed setpoint (e.g. 20, 25, 30 m/s)
- Action histogram (0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER)
- Episode duration and crash outcome with Clopper-Pearson 95% CIs

Outputs:
- JSON artifact to eval_out/speed_diagnostics.json
- Console summary table
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
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
from render_utils import compute_in_lane_leader_speed, compute_mean_npc_speed
from train_optimal_overtaker import OptimalAgent

SEEDS_START = 2000
SEEDS_COUNT = 100


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
    """Run a single evaluation episode and return detailed speed telemetry."""
    ptype = policy_spec["type"]
    profile = policy_spec.get("profile", "current")

    # 1. Instantiate environment based on policy type and profile
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
    elif ptype == "ppo_lstm":
        env = env_config.make_env(
            scenario="highway",
            tier=policy_spec.get("tier", "full_adas"),
            observation={"vehicles_count": 10},
            duration=40,
            seed=seed,
            vehicles_count=10,
            vehicles_density=1.0,
            tactical_overtaking=True,
        )
    else:
        raise ValueError(f"Unknown policy type: {ptype}")

    # 2. Setup agent / controller
    device = torch.device("cpu")
    model = None
    lstm_state = None

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
    elif ptype == "ppo_lstm":
        model = RecurrentAgent(
            obs_dim=policy_spec["obs_dim"], action_dim=5, hidden_dim=128
        ).to(device)
        ckpt = torch.load(policy_spec["model_path"], map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        model.eval()
        lstm_state = (
            torch.zeros(1, 1, 128, device=device),
            torch.zeros(1, 1, 128, device=device),
        )

    # 3. Step through episode and collect metrics
    obs, info = env.reset(seed=seed)
    unwrapped = env.unwrapped
    x_start = float(unwrapped.vehicle.position[0])

    ego_speeds: list[float] = []
    npc_speeds: list[float] = []
    leader_speeds: list[float] = []
    actions_taken: list[int] = []
    target_speeds_taken: list[float] = []

    crashed = False
    step_count = 0
    total_overtakes = 0
    lane_changes = 0
    prev_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]

    while True:
        # Measure speeds before action step
        v_ego = float(info.get("speed", unwrapped.vehicle.speed))
        ego_speeds.append(v_ego)

        v_npc = compute_mean_npc_speed(env)
        npc_speeds.append(v_npc)

        v_leader = compute_in_lane_leader_speed(env)
        if v_leader is not None:
            leader_speeds.append(float(v_leader))

        t_spd = float(getattr(unwrapped.vehicle, "target_speed", 25.0))
        target_speeds_taken.append(t_spd)

        # Action selection
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
        elif ptype == "ppo_lstm":
            obs_flat = np.array(obs).flatten()
            obs_t = torch.tensor(obs_flat, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(5, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)
            with torch.no_grad():
                hidden = model.actor.feature_net(obs_t)
                h_step, lstm_state = model.actor.lstm(
                    hidden.unsqueeze(0),
                    (lstm_state[0], lstm_state[1]),
                )
                logits = model.actor.actor_head(h_step.squeeze(0))
                logits = torch.where(
                    mask_t,
                    logits,
                    torch.tensor(-1e8, device=device, dtype=logits.dtype),
                )
                action = int(torch.argmax(logits, dim=-1).item())
        elif ptype == "classical":
            cname = policy_spec["controller"]
            controller_fn = baseline_classical.CLASSICAL_CONTROLLERS[cname]
            action = int(controller_fn(env, obs, baseline_classical.DEFAULT_IDM_PARAMS))
        else:
            action = 1

        actions_taken.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        total_overtakes += int(info.get("overtake_count", 0))

        cur_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]
        if cur_lane != prev_lane:
            lane_changes += 1
            prev_lane = cur_lane

        if bool(info.get("crashed", False)) or unwrapped.vehicle.crashed:
            crashed = True
            break
        if terminated or truncated:
            break

    x_end = float(unwrapped.vehicle.position[0])
    forward_dist = max(0.0, x_end - x_start)
    env.close()

    # Compute episode summary metrics
    mean_ego_ms = float(np.mean(ego_speeds)) if ego_speeds else 0.0
    mean_npc_ms = float(np.mean(npc_speeds)) if npc_speeds else 0.0
    mean_leader_ms = float(np.mean(leader_speeds)) if leader_speeds else mean_npc_ms
    rel_speed_kmh = (mean_ego_ms - mean_npc_ms) * 3.6

    # Target-speeds histogram
    floor_spd = 20.0 if not policy_spec.get("target_speeds") else min(policy_spec["target_speeds"])
    floor_steps = sum(1 for s in target_speeds_taken if s <= floor_spd + 0.1)
    floor_pct = (floor_steps / max(1, len(target_speeds_taken))) * 100.0

    action_counts = {str(a): actions_taken.count(a) for a in range(5)}

    return {
        "seed": seed,
        "crashed": crashed,
        "steps": step_count,
        "ego_mean_speed_ms": round(mean_ego_ms, 3),
        "ego_mean_speed_kmh": round(mean_ego_ms * 3.6, 2),
        "npc_mean_speed_ms": round(mean_npc_ms, 3),
        "npc_mean_speed_kmh": round(mean_npc_ms * 3.6, 2),
        "leader_mean_speed_ms": round(mean_leader_ms, 3),
        "leader_mean_speed_kmh": round(mean_leader_ms * 3.6, 2),
        "relative_speed_kmh": round(rel_speed_kmh, 2),
        "forward_distance_m": round(forward_dist, 1),
        "pct_steps_at_floor_setpoint": round(floor_pct, 1),
        "action_histogram": action_counts,
        "overtakes": total_overtakes,
        "lane_changes": lane_changes,
    }


def evaluate_policy(
    policy_spec: dict[str, Any],
    seeds: list[int],
    max_workers: int = 8,
) -> dict[str, Any]:
    """Evaluate a policy across deterministic benchmark seeds using parallel workers."""
    pname = policy_spec["name"]
    print(f"\nEvaluating policy: {pname} across {len(seeds)} seeds...")
    t0 = time.time()

    episodes = []
    # Use ProcessPoolExecutor for parallel simulation
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_episode, policy_spec, seed): seed
            for seed in seeds
        }
        for future in concurrent.futures.as_completed(futures):
            ep_res = future.result()
            episodes.append(ep_res)

    episodes.sort(key=lambda x: x["seed"])
    elapsed = time.time() - t0

    # Aggregate statistics
    n = len(episodes)
    crashes = sum(1 for ep in episodes if ep["crashed"])
    crash_rate = (crashes / n) * 100.0 if n > 0 else 0.0
    ci_low, ci_high = clopper_pearson(crashes, n)

    ego_speeds = [ep["ego_mean_speed_kmh"] for ep in episodes]
    flow_speeds = [ep["npc_mean_speed_kmh"] for ep in episodes]
    leader_speeds = [ep["leader_mean_speed_kmh"] for ep in episodes]
    rel_speeds = [ep["relative_speed_kmh"] for ep in episodes]
    distances = [ep["forward_distance_m"] for ep in episodes]
    floor_pcts = [ep["pct_steps_at_floor_setpoint"] for ep in episodes]
    overtakes = [ep["overtakes"] for ep in episodes]
    lane_changes = [ep["lane_changes"] for ep in episodes]

    action_totals = {str(a): sum(ep["action_histogram"][str(a)] for ep in episodes) for a in range(5)}
    total_actions = sum(action_totals.values())
    action_pcts = {
        k: round((v / max(1, total_actions)) * 100.0, 1) for k, v in action_totals.items()
    }

    summary = {
        "policy_name": pname,
        "type": policy_spec["type"],
        "profile": policy_spec.get("profile", "current"),
        "model_path": policy_spec.get("model_path"),
        "episodes": n,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{n}",
        "crash_rate_pct": round(crash_rate, 2),
        "crash_ci_95_low_pct": round(ci_low * 100.0, 2),
        "crash_ci_95_high_pct": round(ci_high * 100.0, 2),
        "mean_ego_speed_kmh": round(float(np.mean(ego_speeds)), 2),
        "std_ego_speed_kmh": round(float(np.std(ego_speeds)), 2),
        "mean_flow_speed_kmh": round(float(np.mean(flow_speeds)), 2),
        "mean_leader_speed_kmh": round(float(np.mean(leader_speeds)), 2),
        "mean_relative_speed_kmh": round(float(np.mean(rel_speeds)), 2),
        "mean_forward_distance_m": round(float(np.mean(distances)), 1),
        "mean_pct_at_floor_setpoint": round(float(np.mean(floor_pcts)), 1),
        "mean_overtakes": round(float(np.mean(overtakes)), 2),
        "mean_lane_changes": round(float(np.mean(lane_changes)), 2),
        "action_percentages": action_pcts,
        "elapsed_seconds": round(elapsed, 1),
        "episodes_detail": episodes,
    }

    print(
        f"  Done in {elapsed:.1f}s | Crashes: {crashes}/{n} ({crash_rate:.1f}%) | "
        f"Ego: {summary['mean_ego_speed_kmh']:.1f} km/h | Flow: {summary['mean_flow_speed_kmh']:.1f} km/h | "
        f"Rel: {summary['mean_relative_speed_kmh']:+.1f} km/h | Floor: {summary['mean_pct_at_floor_setpoint']:.1f}%"
    )
    return summary


def get_all_policy_specs() -> list[dict[str, Any]]:
    """Define all policies to be diagnosed."""
    return [
        {
            "name": "IDM Only (Current)",
            "type": "classical",
            "controller": "idm_only",
            "profile": "current",
        },
        {
            "name": "IDM + MOBIL (Current)",
            "type": "classical",
            "controller": "idm_mobil",
            "profile": "current",
        },
        {
            "name": "IDM Only (Legacy)",
            "type": "classical",
            "controller": "idm_only",
            "profile": "legacy",
        },
        {
            "name": "IDM + MOBIL (Legacy)",
            "type": "classical",
            "controller": "idm_mobil",
            "profile": "legacy",
        },
        {
            "name": "Feedforward PPO (Full-ADAS)",
            "type": "feedforward_ppo",
            "model_path": "models/ppo_highway_full_adas_seed101.pt",
            "obs_dim": 75,
            "profile": "current_75",
        },
        {
            "name": "Feedforward PPO (Front-Only)",
            "type": "feedforward_ppo",
            "model_path": "models/ppo_highway_front_only_seed101.pt",
            "obs_dim": 50,
            "tier": "front_only",
            "profile": "legacy_50",
        },
        {
            "name": "PPO-LSTM (Full-ADAS)",
            "type": "ppo_lstm",
            "model_path": "models/ppo_lstm_highway_full_adas_seed101.pt",
            "obs_dim": 50,
            "tier": "full_adas",
            "profile": "legacy_50",
        },
        {
            "name": "PPO-LSTM (Front-Only)",
            "type": "ppo_lstm",
            "model_path": "models/ppo_lstm_highway_front_only_seed101.pt",
            "obs_dim": 50,
            "tier": "front_only",
            "profile": "legacy_50",
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
            "name": "Optimal Overtaker (Extended Speeds)",
            "type": "optimal_agent",
            "model_path": "scratch/optimal_overtaker_extended_speeds/ppo_optimal_overtaker_4lane_best.pt",
            "obs_dim": 81,
            "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
            "profile": "current_extended_speeds",
        },
        {
            "name": "DeepSet v2 (Retrained Seed 42)",
            "type": "deepset_v2",
            "model_path": "scratch/deepset_v2_highway_retrain_seed42/deepset_v2_highway_best.pt",
            "profile": "current",
        },
        {
            "name": "DeepSet v2 (2M Baseline)",
            "type": "deepset_v2",
            "model_path": "scratch/deepset_v2_highway_2m/deepset_v2_highway_best.pt",
            "profile": "current",
        },
        {
            "name": "PPO-Lagrangian (Sep Norm)",
            "type": "lagrangian",
            "model_path": "scratch/ppo_lagrangian_1m_sep_norm/ppo_optimal_overtaker_4lane_best.pt",
            "obs_dim": 81,
            "profile": "current",
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run speed diagnostics across all policies.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of seeds (default: 100).")
    parser.add_argument("--seed-start", type=int, default=2000, help="Start seed (default: 2000).")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes (default: 8).")
    parser.add_argument(
        "--output",
        type=str,
        default="eval_out/speed_diagnostics.json",
        help="Path to output JSON file (default: eval_out/speed_diagnostics.json).",
    )
    args = parser.parse_args(argv)

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    policies = get_all_policy_specs()

    print("=" * 80)
    print(f"SPEED DIAGNOSTICS BENCHMARK (Seeds {args.seed_start}-{args.seed_start + args.episodes - 1})")
    print(f"Policies to evaluate: {len(policies)} | Workers: {args.workers}")
    print("=" * 80)

    import gymnasium
    import highway_env
    import numpy
    import torch

    metadata = {
        "git_commit": get_git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "package_versions": {
            "python": sys.version.split()[0],
            "highway_env": getattr(highway_env, "__version__", "1.12.1"),
            "gymnasium": getattr(gymnasium, "__version__", "1.3.0"),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
        },
        "seed_start": args.seed_start,
        "episodes_per_policy": args.episodes,
    }

    all_results = {}
    for pol in policies:
        res = evaluate_policy(pol, seeds, max_workers=args.workers)
        all_results[pol["name"]] = res

    out_data = {
        "metadata": metadata,
        "policies": all_results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "=" * 105)
    print(f"{'Policy':<34} {'Crashes (95% CI)':<22} {'Ego Spd':<11} {'Flow Spd':<11} {'Rel Spd':<11} {'Floor %'}")
    print("-" * 105)
    for name, r in all_results.items():
        ci_str = f"{r['crashes']}/{r['episodes']} [{r['crash_ci_95_low_pct']:.1f}%, {r['crash_ci_95_high_pct']:.1f}%]"
        print(
            f"{name:<34} {ci_str:<22} {r['mean_ego_speed_kmh']:>5.1f} km/h  "
            f"{r['mean_flow_speed_kmh']:>5.1f} km/h  {r['mean_relative_speed_kmh']:>+5.1f} km/h  "
            f"{r['mean_pct_at_floor_setpoint']:>5.1f}%"
        )
    print("=" * 105)
    print(f"Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Observation-driven classical motion planning baselines (IDM and IDM+MOBIL).

Implements observation-only Intelligent Driver Model (IDM) car-following and
Minimizing Overall Braking Induced by Lane changes (MOBIL) lateral planning
without privileged simulator access. Evaluates performance across sensor tiers.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Callable

import gymnasium as gym
import numpy as np

import env_config


# Default IDM and MOBIL parameters aligned with literature and highway-env
DEFAULT_IDM_PARAMS: dict[str, float] = {
    "v0": 30.0,  # Desired target speed (m/s)
    "a_max": 3.0,  # Maximum comfortable acceleration (m/s^2)
    "b_max": 5.0,  # Comfortable braking deceleration (m/s^2)
    "s0": 10.0,  # Minimum standstill jam distance (m)
    "T": 1.5,  # Safe time headway (s)
    "delta": 4.0,  # Acceleration exponent
    "l_veh": 5.0,  # Effective vehicle length (m)
    "b_safe": 2.0,  # Maximum safe deceleration imposed on follower (m/s^2)
    "acc_gain_th": 0.2,  # Minimum acceleration incentive threshold (m/s^2)
    "politeness": 0.0,  # Politeness factor (0.0 = selfish)
    "lane_width": 4.0,  # Standard lane width (m)
    "acc_reg_th": 0.5,  # Threshold for FASTER/SLOWER speed adjustment (m/s^2)
    "emergency_acc": -3.0,  # Deceleration threshold for emergency braking (m/s^2)
}


def parse_observation_neighbors(
    obs: np.ndarray,
    rx: float,
    ry: float,
    rv: float = 80.0,
    lane_width: float = 4.0,
) -> dict[str, Any]:
    """Parse relative observation matrix into categorized neighbor vehicles.

    Extracts ego state and identifies closest leading and lagging vehicles
    in the current lane, left adjacent lane, and right adjacent lane without
    privileged access to simulator road networks.

    Args:
        obs: Observation array of shape (N, 5), default (15, 5).
        rx: Longitudinal normalization scale in meters.
        ry: Lateral normalization scale in meters.
        rv: Velocity normalization scale in m/s (default: 80.0).
        lane_width: Nominal lane width in meters (default: 4.0).

    Returns:
        Dictionary containing ego state and categorized neighbor dictionaries.
    """
    ego_speed = float(obs[0, 3] * rv)
    ego_y = float(obs[0, 2] * ry)
    ego_lane = int(np.round(ego_y / lane_width))

    neighbors: dict[str, dict[str, float] | None] = {
        "lead_current": None,
        "lag_current": None,
        "lead_left": None,
        "lag_left": None,
        "lead_right": None,
        "lag_right": None,
    }

    min_lead_curr = float("inf")
    min_lag_curr = float("inf")
    min_lead_left = float("inf")
    min_lag_left = float("inf")
    min_lead_right = float("inf")
    min_lag_right = float("inf")

    num_vehicles = obs.shape[0]
    for i in range(1, num_vehicles):
        if obs[i, 0] <= 0.0:
            continue

        dx = float(obs[i, 1] * rx)
        dy = float(obs[i, 2] * ry)
        dvx = float(obs[i, 3] * rv)  # v_target - v_ego
        target_v = ego_speed + dvx

        lane_offset = int(np.round(dy / lane_width))

        # Current lane (lane_offset == 0)
        if lane_offset == 0:
            if dx > 0.0 and dx < min_lead_curr:
                min_lead_curr = dx
                neighbors["lead_current"] = {
                    "dx": dx,
                    "dy": dy,
                    "v": target_v,
                    "dv": -dvx,
                }
            elif dx < 0.0 and abs(dx) < min_lag_curr:
                min_lag_curr = abs(dx)
                neighbors["lag_current"] = {
                    "dx": dx,
                    "dy": dy,
                    "v": target_v,
                    "dv": dvx,
                }

        # Left lane (lane_offset == -1)
        elif lane_offset == -1:
            if dx > 0.0 and dx < min_lead_left:
                min_lead_left = dx
                neighbors["lead_left"] = {
                    "dx": dx,
                    "dy": dy,
                    "v": target_v,
                    "dv": -dvx,
                }
            elif dx < 0.0 and abs(dx) < min_lag_left:
                min_lag_left = abs(dx)
                neighbors["lag_left"] = {
                    "dx": dx,
                    "dy": dy,
                    "v": target_v,
                    "dv": dvx,
                }

        # Right lane (lane_offset == +1)
        elif lane_offset == 1:
            if dx > 0.0 and dx < min_lead_right:
                min_lead_right = dx
                neighbors["lead_right"] = {
                    "dx": dx,
                    "dy": dy,
                    "v": target_v,
                    "dv": -dvx,
                }
            elif dx < 0.0 and abs(dx) < min_lag_right:
                min_lag_right = abs(dx)
                neighbors["lag_right"] = {
                    "dx": dx,
                    "dy": dy,
                    "v": target_v,
                    "dv": dvx,
                }

    return {
        "ego_speed": ego_speed,
        "ego_y": ego_y,
        "ego_lane": ego_lane,
        "neighbors": neighbors,
    }


def compute_idm_acceleration(
    ego_speed: float,
    lead_speed: float | None = None,
    gap: float | None = None,
    target_speed: float = 30.0,
    params: dict[str, float] | None = None,
) -> float:
    """Compute continuous longitudinal acceleration using the Intelligent Driver Model.

    Args:
        ego_speed: Current longitudinal speed of ego vehicle in m/s.
        lead_speed: Longitudinal speed of preceding vehicle in m/s.
        gap: Center-to-center distance to preceding vehicle in meters.
        target_speed: Desired free-flow speed setpoint in m/s.
        params: IDM parameter dictionary.

    Returns:
        Computed acceleration in m/s^2.
    """
    cfg = DEFAULT_IDM_PARAMS if params is None else params
    v = max(float(ego_speed), 0.0)
    v0 = max(float(target_speed), 0.1)
    a = float(cfg["a_max"])
    b = float(cfg["b_max"])
    s0 = float(cfg["s0"])
    t_headway = float(cfg["T"])
    delta = float(cfg["delta"])
    l_veh = float(cfg["l_veh"])

    # Free acceleration term
    free_ratio = v / v0
    a_free = a * (1.0 - np.power(free_ratio, delta))

    # Interaction deceleration term
    if lead_speed is not None and gap is not None:
        net_distance = max(gap - l_veh, 0.1)
        delta_v = v - lead_speed
        ab_term = 2.0 * np.sqrt(a * b)
        s_star = s0 + v * t_headway + (v * delta_v) / ab_term
        a_int = -a * np.power(s_star / net_distance, 2.0)
        return float(a_free + a_int)

    return float(a_free)


def evaluate_mobil_lane_change(
    ego_speed: float,
    ego_lane: int,
    neighbors: dict[str, Any],
    lanes_count: int,
    params: dict[str, float] | None = None,
) -> int | None:
    """Evaluate candidate lane changes using the MOBIL decision model.

    Evaluates safety criterion (limiting deceleration imposed on the new follower)
    and incentive criterion (acceleration advantage exceeding hysteresis threshold).

    Args:
        ego_speed: Current ego speed in m/s.
        ego_lane: Current discrete lane index of ego vehicle.
        neighbors: Parsed neighbor vehicles dictionary.
        lanes_count: Total available lanes in the current road scenario.
        params: MOBIL parameter dictionary.

    Returns:
        Action index (0 for LANE_LEFT, 2 for LANE_RIGHT), or None if no change.
    """
    cfg = DEFAULT_IDM_PARAMS if params is None else params
    b_safe = float(cfg["b_safe"])
    acc_gain_th = float(cfg["acc_gain_th"])
    politeness = float(cfg["politeness"])
    s0 = float(cfg["s0"])
    v0 = float(cfg["v0"])

    lead_curr = neighbors.get("lead_current")
    a_ego_curr = compute_idm_acceleration(
        ego_speed=ego_speed,
        lead_speed=lead_curr["v"] if lead_curr else None,
        gap=lead_curr["dx"] if lead_curr else None,
        target_speed=v0,
        params=cfg,
    )

    candidates: list[tuple[int, float]] = []

    # Check Left Lane (target lane = ego_lane - 1, action = 0)
    if ego_lane > 0:
        lead_left = neighbors.get("lead_left")
        lag_left = neighbors.get("lag_left")

        # Physical clearance guard
        clear_lead = lead_left is None or lead_left["dx"] > s0
        clear_lag = lag_left is None or abs(lag_left["dx"]) > s0

        if clear_lead and clear_lag:
            # Safety criterion for new follower
            if lag_left is not None:
                lag_v = lag_left["v"]
                lag_gap = abs(lag_left["dx"])
                a_new_lag_pred = compute_idm_acceleration(
                    ego_speed=lag_v,
                    lead_speed=ego_speed,
                    gap=lag_gap,
                    target_speed=v0,
                    params=cfg,
                )
                safe_lag = a_new_lag_pred >= -b_safe
            else:
                a_new_lag_pred = 0.0
                safe_lag = True  # Vacuously true if unobserved / no follower

            # Ego safety in target lane
            a_ego_left_pred = compute_idm_acceleration(
                ego_speed=ego_speed,
                lead_speed=lead_left["v"] if lead_left else None,
                gap=lead_left["dx"] if lead_left else None,
                target_speed=v0,
                params=cfg,
            )
            safe_ego = a_ego_left_pred >= -b_safe

            if safe_lag and safe_ego:
                ego_gain = a_ego_left_pred - a_ego_curr
                incentive = ego_gain
                if incentive > acc_gain_th:
                    candidates.append((0, incentive))

    # Check Right Lane (target lane = ego_lane + 1, action = 2)
    if ego_lane < lanes_count - 1:
        lead_right = neighbors.get("lead_right")
        lag_right = neighbors.get("lag_right")

        clear_lead = lead_right is None or lead_right["dx"] > s0
        clear_lag = lag_right is None or abs(lag_right["dx"]) > s0

        if clear_lead and clear_lag:
            if lag_right is not None:
                lag_v = lag_right["v"]
                lag_gap = abs(lag_right["dx"])
                a_new_lag_pred = compute_idm_acceleration(
                    ego_speed=lag_v,
                    lead_speed=ego_speed,
                    gap=lag_gap,
                    target_speed=v0,
                    params=cfg,
                )
                safe_lag = a_new_lag_pred >= -b_safe
            else:
                a_new_lag_pred = 0.0
                safe_lag = True

            a_ego_right_pred = compute_idm_acceleration(
                ego_speed=ego_speed,
                lead_speed=lead_right["v"] if lead_right else None,
                gap=lead_right["dx"] if lead_right else None,
                target_speed=v0,
                params=cfg,
            )
            safe_ego = a_ego_right_pred >= -b_safe

            if safe_lag and safe_ego:
                ego_gain = a_ego_right_pred - a_ego_curr
                incentive = ego_gain
                if incentive > acc_gain_th:
                    candidates.append((2, incentive))

    if not candidates:
        return None

    # Choose candidate with highest acceleration gain
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates[0][0]


def compute_idm_discrete_action(
    env: gym.Env,
    obs: np.ndarray,
    params: dict[str, float] | None = None,
) -> int:
    """Compute discrete action using pure IDM longitudinal car-following.

    Maintains lane position without lane changing. Commands FASTER or SLOWER
    based on IDM acceleration thresholds.

    Args:
        env: Gymnasium environment instance.
        obs: Observation array of shape (N, 5), default (15, 5).
        params: IDM parameter dictionary.

    Returns:
        Discrete action index (IDLE, FASTER, or SLOWER).
    """
    cfg = DEFAULT_IDM_PARAMS if params is None else params
    rx, ry = env_config.get_scaling_factors(env)
    parsed = parse_observation_neighbors(
        obs=obs, rx=rx, ry=ry, lane_width=cfg["lane_width"]
    )

    ego_speed = parsed["ego_speed"]
    lead_curr = parsed["neighbors"]["lead_current"]

    a_idm = compute_idm_acceleration(
        ego_speed=ego_speed,
        lead_speed=lead_curr["v"] if lead_curr else None,
        gap=lead_curr["dx"] if lead_curr else None,
        target_speed=cfg["v0"],
        params=cfg,
    )

    is_intersection = env.action_space.n == 3
    action_slower = 0 if is_intersection else 4
    action_idle = 1
    action_faster = 2 if is_intersection else 3

    acc_th = cfg["acc_reg_th"]
    if a_idm > acc_th:
        return action_faster
    if a_idm < -acc_th:
        return action_slower
    return action_idle


def compute_idm_mobil_discrete_action(
    env: gym.Env,
    obs: np.ndarray,
    params: dict[str, float] | None = None,
) -> int:
    """Compute discrete action combining MOBIL lateral decisions with IDM car-following.

    Evaluates emergency braking first, then checks MOBIL lane changes, and falls
    back to IDM speed regulation.

    Args:
        env: Gymnasium environment instance.
        obs: Observation array of shape (N, 5), default (15, 5).
        params: IDM+MOBIL parameter dictionary.

    Returns:
        Discrete action index (LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER).
    """
    cfg = DEFAULT_IDM_PARAMS if params is None else params
    rx, ry = env_config.get_scaling_factors(env)
    parsed = parse_observation_neighbors(
        obs=obs, rx=rx, ry=ry, lane_width=cfg["lane_width"]
    )

    ego_speed = parsed["ego_speed"]
    ego_lane = parsed["ego_lane"]
    lead_curr = parsed["neighbors"]["lead_current"]

    a_idm = compute_idm_acceleration(
        ego_speed=ego_speed,
        lead_speed=lead_curr["v"] if lead_curr else None,
        gap=lead_curr["dx"] if lead_curr else None,
        target_speed=cfg["v0"],
        params=cfg,
    )

    is_intersection = env.action_space.n == 3
    action_slower = 0 if is_intersection else 4
    action_idle = 1
    action_faster = 2 if is_intersection else 3

    # Priority 1: Emergency longitudinal braking override
    emergency_brake = a_idm < cfg["emergency_acc"]
    close_headway = lead_curr is not None and lead_curr["dx"] < cfg["s0"]
    if emergency_brake or close_headway:
        return action_slower

    # Priority 2: Lateral lane change evaluation (for 5-action environments)
    if not is_intersection:
        lanes_count = int(env.unwrapped.config.get("lanes_count", 4))
        lane_action = evaluate_mobil_lane_change(
            ego_speed=ego_speed,
            ego_lane=ego_lane,
            neighbors=parsed["neighbors"],
            lanes_count=lanes_count,
            params=cfg,
        )
        if lane_action is not None:
            return lane_action

    # Priority 3: Nominal IDM longitudinal speed adjustment
    acc_th = cfg["acc_reg_th"]
    if a_idm > acc_th:
        return action_faster
    if a_idm < -acc_th:
        return action_slower
    return action_idle


CLASSICAL_CONTROLLERS: dict[str, Callable[[gym.Env, np.ndarray, dict | None], int]] = {
    "idm_only": compute_idm_discrete_action,
    "idm_mobil": compute_idm_mobil_discrete_action,
}


def evaluate_controller(
    scenario: str,
    tier: str,
    controller_name: str,
    episodes: int = 30,
    seed_start: int = 2000,
    params: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate a classical controller on a specific scenario and sensor tier.

    Args:
        scenario: Scenario identifier.
        tier: Sensor tier ('full_adas', 'front_only', 'short_range', 'no_velocity').
        controller_name: Key in CLASSICAL_CONTROLLERS dispatch table.
        episodes: Number of evaluation episodes (default: 30).
        seed_start: Starting random seed (default: 2000).
        params: Optional controller parameters.

    Returns:
        Dictionary summarizing crash rate, duration, speed, and lane changes.
    """
    controller_fn = CLASSICAL_CONTROLLERS[controller_name]
    env = env_config.make_env(scenario=scenario, tier=tier)

    crashes = 0
    step1_crashes = 0
    durations: list[int] = []
    speeds_mps: list[float] = []
    total_lane_changes: list[int] = []

    for ep in range(episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        ep_steps = 0
        ep_speeds: list[float] = []
        ep_lane_changes = 0
        crashed = False

        while True:
            action = controller_fn(env, obs, params)
            if action in (0, 2):  # LANE_LEFT or LANE_RIGHT
                ep_lane_changes += 1

            obs, reward, terminated, truncated, info = env.step(action)
            ep_steps += 1

            speed_val = float(info.get("speed", obs[0, 3] * 80.0))
            ep_speeds.append(speed_val)

            if bool(info.get("crashed", False)) or terminated:
                crashed = bool(info.get("crashed", False))
                if ep_steps == 1 and crashed:
                    step1_crashes += 1
                break
            if truncated:
                break

        if crashed:
            crashes += 1

        durations.append(ep_steps)
        speeds_mps.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        total_lane_changes.append(ep_lane_changes)

    env.close()

    mean_dur = float(np.mean(durations)) if durations else 0.0
    std_dur = float(np.std(durations)) if durations else 0.0
    mean_speed_mps = float(np.mean(speeds_mps)) if speeds_mps else 0.0
    mean_speed_kmh = mean_speed_mps * 3.6
    mean_lc = float(np.mean(total_lane_changes)) if total_lane_changes else 0.0

    return {
        "scenario": scenario,
        "tier": tier,
        "controller": controller_name,
        "episodes": episodes,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{episodes}",
        "crash_rate_pct": (crashes / episodes) * 100.0 if episodes > 0 else 0.0,
        "step1_crashes": step1_crashes,
        "mean_duration": mean_dur,
        "std_duration": std_dur,
        "mean_speed_kmh": mean_speed_kmh,
        "mean_lane_changes": mean_lc,
    }


def run_stage2_evaluation(
    scenarios: list[str],
    tiers: list[str],
    controllers: list[str],
    episodes: int = 30,
    seed_start: int = 2000,
) -> list[dict[str, Any]]:
    """Execute classical baseline evaluation across scenarios, tiers, and controllers.

    Args:
        scenarios: List of scenario names.
        tiers: List of sensor tiers.
        controllers: List of controller names.
        episodes: Number of episodes per condition.
        seed_start: Starting random seed.

    Returns:
        List of result dictionaries.
    """
    results: list[dict[str, Any]] = []

    for scenario in scenarios:
        for tier in tiers:
            for controller in controllers:
                res = evaluate_controller(
                    scenario=scenario,
                    tier=tier,
                    controller_name=controller,
                    episodes=episodes,
                    seed_start=seed_start,
                )
                results.append(res)
                dur_str = f"{res['mean_duration']:5.1f}+/-{res['std_duration']:4.1f}"
                print(
                    f"[RESULT] scenario={res['scenario']:<13} tier={res['tier']:<12} "
                    f"controller={res['controller']:<10} "
                    f"crashes={res['crash_fraction']} ({res['crash_rate_pct']:5.1f}%) "
                    f"duration={dur_str} "
                    f"speed={res['mean_speed_kmh']:5.1f} km/h "
                    f"lane_changes={res['mean_lane_changes']:.2f}"
                )

    return results


def print_stage2_summary_table(results: list[dict[str, Any]]) -> None:
    """Print formatted summary table of classical baseline results.

    Args:
        results: List of evaluation result dictionaries.
    """
    print("\n" + "=" * 94)
    print("CLASSICAL MOTION PLANNING BASELINE BENCHMARK TABLE")
    print("=" * 94)
    header = (
        f"| {'Scenario':<12} | {'Tier':<12} | {'Controller':<10} | "
        f"{'Crash Rate':<9} | {'Duration':<14} | {'Mean Speed':<12} | {'Lane Chg':<8} |"
    )
    separator = (
        f"|:{'-'*12}-|-{'-'*12}:|-{'-'*10}:|-{'-'*9}:|-{'-'*14}:|-{'-'*12}:|-{'-'*8}:|"
    )
    print(header)
    print(separator)

    for r in results:
        dur_str = f"{r['mean_duration']:.1f} +/- {r['std_duration']:.1f}"
        spd_str = f"{r['mean_speed_kmh']:.1f} km/h"
        row = (
            f"| {r['scenario']:<12} | {r['tier']:>12} | {r['controller']:>10} | "
            f"{r['crash_fraction']:>6} ({r['crash_rate_pct']:4.1f}%) | "
            f"{dur_str:>14} | {spd_str:>12} | {r['mean_lane_changes']:>8.2f} |"
        )
        print(row)

    print("=" * 94)


def main() -> None:
    """Parse CLI arguments and execute classical baseline benchmarks."""
    parser = argparse.ArgumentParser(
        description="Observation-Driven Classical Baseline (IDM/MOBIL)."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help="Number of episodes per evaluation condition (default: 30).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=2000,
        help="Starting evaluation seed (default: 2000).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all"] + env_config.get_scenario_names(),
        help="Scenario to benchmark or 'all' (default: all).",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["all"] + env_config.get_tier_names(),
        help="Sensor tier to benchmark or 'all' (default: all).",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="all",
        choices=["all"] + list(CLASSICAL_CONTROLLERS.keys()),
        help="Controller to benchmark or 'all' (default: all).",
    )
    args = parser.parse_args()

    scenarios = (
        env_config.get_scenario_names() if args.scenario == "all" else [args.scenario]
    )
    tiers = env_config.get_tier_names() if args.tier == "all" else [args.tier]
    controllers = (
        list(CLASSICAL_CONTROLLERS.keys())
        if args.controller == "all"
        else [args.controller]
    )

    print(
        f"Starting Classical Baseline: scenarios={scenarios}, "
        f"tiers={tiers}, controllers={controllers}, episodes={args.episodes}, "
        f"seed_start={args.seed_start}"
    )

    results = run_stage2_evaluation(
        scenarios=scenarios,
        tiers=tiers,
        controllers=controllers,
        episodes=args.episodes,
        seed_start=args.seed_start,
    )
    print_stage2_summary_table(results)


if __name__ == "__main__":
    main()

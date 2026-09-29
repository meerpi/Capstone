"""Continuous-control ground-truth reward function and environment wrapper for HighwayEnv.

Implements the calibrated continuous reward function specified in Sprint 1:
1. R_speed: Trapezoidal speed incentive [v_min, v_target, v_max, k_v]
2. R_jerk: Control derivative penalty on a_dot and delta_dot
3. R_TTC & R_headway: Closing-only time-to-collision and headway penalties
4. R_overtake: Anti-farming dwell-checked overtake credit
5. R_control: Action change magnitude penalty ||u_t - u_{t-1}||^2
6. R_terminal: Catastrophic crash penalty sized to dominate discounted future returns

All terms are computed strictly from simulator ground-truth states (unwrapped road
and vehicle physics), never from perception-filtered observations.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np


@dataclass
class ContinuousRewardConfig:
    """Configuration container for all continuous reward parameters, weights, and thresholds."""

    # --- Speed Trapezoid Parameters (m/s) ---
    v_min: float = 20.0        # Speed below which R_speed is 0.0 (72 km/h)
    v_target: float = 27.5     # Speed at which R_speed reaches 1.0 (99 km/h)
    v_max: float = 30.0        # Max desired speed (108 km/h); above this R_speed decays
    k_v: float = 0.2           # Linear decay slope above v_max (floored at 0.0)

    # --- Control / Jerk Parameters ---
    policy_frequency: float = 5.0     # Action frequency in Hz (dt = 0.2s)
    # Action physical ranges:
    acc_min: float = -5.0             # Minimum acceleration in m/s^2 (hard braking)
    acc_max: float = 3.0              # Maximum acceleration in m/s^2 (full throttle)
    steering_max_rad: float = math.pi / 12.0  # Max steering angle (+/- 15 degrees = 0.2618 rad)

    # Jerk normalization ceilings:
    # Max possible Delta a over dt is (acc_max - acc_min) = 8.0 m/s^2 -> a_dot_max = 8.0 / 0.2 = 40.0 m/s^3
    max_a_dot: float = 40.0
    # Max possible Delta delta over dt is 2 * steering_max_rad = 0.5236 rad -> delta_dot_max = 0.5236 / 0.2 = 2.618 rad/s^2
    max_delta_dot: float = (2.0 * math.pi / 12.0) / 0.2  # ~2.618 rad/s^2
    lambda_jerk_acc: float = 0.5      # Internal weight for acceleration jerk
    lambda_jerk_steer: float = 0.5    # Internal weight for steering jerk

    # Control change normalization ceiling:
    # u in [-1, 1]^2, max ||Delta u||^2 = (2)^2 + (2)^2 = 8.0
    max_delta_u_sq: float = 8.0

    # --- TTC & Headway Parameters ---
    ttc_min: float = 2.0              # Minimum safe TTC in seconds
    ttc_epsilon: float = 0.1          # TTC floor in seconds to prevent log divergence
    headway_min: float = 1.0          # Minimum safe time headway in seconds
    corridor_lateral_margin: float = 2.0  # Lateral envelope around ego center to search for lead obstacle (m)

    # --- Overtake Parameters ---
    r_ot: float = 1.0                 # Base credit per overtake
    max_overtake_lat_lanes: float = 1.5  # Max lateral separation in lane units (1.5 * 4m = 6m)
    max_overtake_lon_dist: float = 30.0  # Max longitudinal distance behind ego (|x_rel| <= 30m)
    overtake_dwell_steps: int = 5     # Minimum consecutive steps x_rel < 0 required before credit (1.0s at 5Hz)

    # --- Terminal Crash Penalty ---
    crash_penalty: float = -200.0     # Dominates discounted upper bound (V_max = 200.0 at gamma=0.99)

    # --- Term Weights (Applied after [-1, 1] / [0, 1] normalization) ---
    w_speed: float = 1.0
    w_jerk: float = 0.10
    w_ttc: float = 1.0
    w_headway: float = 0.30
    w_overtake: float = 1.0
    w_control: float = 0.05
    w_crash: float = 1.0


class ContinuousRewardFunction:
    """Computes calibrated continuous-control reward terms from simulator ground truth."""

    def __init__(self, config: ContinuousRewardConfig | None = None) -> None:
        self.config = config or ContinuousRewardConfig()

        # Episode tracking state
        self.prev_action: np.ndarray | None = None  # Normalized action u_{t-1} in [-1, 1]^2
        self.seen_ahead: set[int] = set()           # Vehicle IDs seen ahead of ego (x_rel > 0)
        self.credited_overtakes: set[int] = set()   # Vehicle IDs already credited this episode
        self.dwell_counters: dict[int, int] = {}    # v_id -> consecutive steps x_rel < 0

    def reset(self) -> None:
        """Reset internal episode state."""
        self.prev_action = None
        self.seen_ahead.clear()
        self.credited_overtakes.clear()
        self.dwell_counters.clear()

    def compute_reward(
        self,
        env: gym.Env,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute the composite continuous reward from simulator ground truth.

        Args:
            env: Gymnasium environment instance (wrapped or unwrapped).
            action: Current normalized continuous action array [acc_norm, steer_norm] in [-1, 1].

        Returns:
            Tuple of (total_reward, components_dict).
        """
        cfg = self.config
        unwrapped = env.unwrapped
        ego = unwrapped.vehicle
        road = unwrapped.road

        # ---------------------------------------------------------------------
        # 1. R_speed: Trapezoid on ego ground-truth speed
        # ---------------------------------------------------------------------
        v = float(ego.speed)
        if v <= cfg.v_min:
            r_speed_norm = 0.0
        elif v < cfg.v_target:
            r_speed_norm = (v - cfg.v_min) / (cfg.v_target - cfg.v_min)
        elif v <= cfg.v_max:
            r_speed_norm = 1.0
        else:
            r_speed_norm = max(0.0, 1.0 - cfg.k_v * (v - cfg.v_max))

        # ---------------------------------------------------------------------
        # 2. R_jerk: Control derivative penalty on a_dot and delta_dot
        # ---------------------------------------------------------------------
        # Map normalized action to physical units:
        # action[0] is acceleration in [acc_min, acc_max]
        # action[1] is steering in [-steering_max_rad, steering_max_rad]
        dt = 1.0 / cfg.policy_frequency
        u_curr = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        if self.prev_action is not None and dt > 0:
            u_prev = self.prev_action

            # Physical acceleration
            a_curr = cfg.acc_min + 0.5 * (u_curr[0] + 1.0) * (cfg.acc_max - cfg.acc_min)
            a_prev = cfg.acc_min + 0.5 * (u_prev[0] + 1.0) * (cfg.acc_max - cfg.acc_min)
            a_dot = (a_curr - a_prev) / dt

            # Physical steering
            delta_curr = u_curr[1] * cfg.steering_max_rad
            delta_prev = u_prev[1] * cfg.steering_max_rad
            delta_dot = (delta_curr - delta_prev) / dt

            # Normalized jerk components in [0, 1]
            a_jerk_norm = min(1.0, (a_dot / cfg.max_a_dot) ** 2)
            delta_jerk_norm = min(1.0, (delta_dot / cfg.max_delta_dot) ** 2)
            jerk_norm = cfg.lambda_jerk_acc * a_jerk_norm + cfg.lambda_jerk_steer * delta_jerk_norm
            r_jerk_norm = -float(np.clip(jerk_norm, 0.0, 1.0))
        else:
            r_jerk_norm = 0.0

        # ---------------------------------------------------------------------
        # 3. R_control: Action change magnitude ||u_t - u_{t-1}||^2
        # ---------------------------------------------------------------------
        if self.prev_action is not None:
            delta_u_sq = float(np.sum((u_curr - self.prev_action) ** 2))
            # Normalized to [0, 1] where max ||Delta u||^2 = 8.0
            r_control_norm = -float(np.clip(delta_u_sq / cfg.max_delta_u_sq, 0.0, 1.0))
        else:
            r_control_norm = 0.0

        # Update previous action
        self.prev_action = u_curr.copy()

        # ---------------------------------------------------------------------
        # 4. R_TTC & R_headway: Closing-only TTC and time headway from ground truth
        # ---------------------------------------------------------------------
        # Search all vehicles on road for the closest lead obstacle in the travel corridor
        ego_x = float(ego.position[0])
        ego_y = float(ego.position[1])
        lead_veh = None
        min_lead_gap = float("inf")

        for other in road.vehicles:
            if other is ego:
                continue
            dx = float(other.position[0]) - ego_x
            dy = abs(float(other.position[1]) - ego_y)

            # Check if other vehicle is ahead and within the lateral envelope
            if dx > 0 and dy <= cfg.corridor_lateral_margin:
                # Bumper-to-bumper gap (subtracting vehicle length)
                veh_len = getattr(ego, "LENGTH", 5.0)
                gap = max(0.001, dx - veh_len)
                if gap < min_lead_gap:
                    min_lead_gap = gap
                    lead_veh = other

        # Also cross-check with neighbour_vehicles on ego.lane_index
        if hasattr(road, "neighbour_vehicles") and hasattr(ego, "lane_index"):
            front_veh, _ = road.neighbour_vehicles(ego, ego.lane_index)
            if front_veh is not None and front_veh is not ego:
                dx_front = float(front_veh.position[0]) - ego_x
                if dx_front > 0:
                    veh_len = getattr(ego, "LENGTH", 5.0)
                    gap_front = max(0.001, dx_front - veh_len)
                    if gap_front < min_lead_gap:
                        min_lead_gap = gap_front
                        lead_veh = front_veh

        r_ttc_norm = 0.0
        r_headway_norm = 0.0

        if lead_veh is not None and min_lead_gap < float("inf"):
            v_lead = float(lead_veh.speed)
            v_rel = v - v_lead  # Closing speed (positive = ego faster)

            # Closing-only TTC
            if v_rel > 0:
                ttc = min_lead_gap / v_rel
                if ttc <= cfg.ttc_min:
                    ttc_clipped = max(cfg.ttc_epsilon, ttc)
                    # log(ttc_clipped / ttc_min) ranges from log(epsilon / ttc_min) to 0.0
                    # Normalizing by log(epsilon / ttc_min) produces a value in [0, 1]
                    norm_denom = math.log(cfg.ttc_epsilon / cfg.ttc_min)  # negative number
                    raw_log = math.log(ttc_clipped / cfg.ttc_min)         # negative number
                    penalty_norm = raw_log / norm_denom                  # in [0, 1], 1.0 at epsilon
                    r_ttc_norm = -float(np.clip(penalty_norm, 0.0, 1.0))

            # Headway: h = gap / v_ego
            h = min_lead_gap / max(v, 1.0)
            if h < cfg.headway_min:
                h_penalty = (cfg.headway_min - h) / cfg.headway_min
                r_headway_norm = -float(np.clip(h_penalty, 0.0, 1.0))

        # ---------------------------------------------------------------------
        # 5. R_overtake: Anti-farming dwell-checked overtake credit
        # ---------------------------------------------------------------------
        # Extract lane width from ego lane or default to 4.0m
        lane_width = 4.0
        if hasattr(ego, "lane") and ego.lane is not None and hasattr(ego.lane, "width"):
            lane_width = float(ego.lane.width)

        r_overtake_norm = 0.0
        current_candidates: list[tuple[float, int]] = []  # (lateral_separation, v_id)

        for other in road.vehicles:
            if other is ego:
                continue
            v_id = id(other)
            if v_id in self.credited_overtakes:
                continue

            x_rel = float(other.position[0]) - ego_x
            lat_sep = abs(float(other.position[1]) - ego_y)

            # Condition 1: Seen ahead
            if x_rel > 0:
                self.seen_ahead.add(v_id)
                self.dwell_counters[v_id] = 0

            # Condition 2: x_rel flipped to negative, was seen ahead, and within lateral/longitudinal bounds
            elif x_rel < 0 and v_id in self.seen_ahead:
                lat_lanes = lat_sep / lane_width
                if lat_lanes <= cfg.max_overtake_lat_lanes and abs(x_rel) <= cfg.max_overtake_lon_dist:
                    self.dwell_counters[v_id] = self.dwell_counters.get(v_id, 0) + 1
                    if self.dwell_counters[v_id] >= cfg.overtake_dwell_steps:
                        current_candidates.append((lat_sep, v_id))
                else:
                    self.dwell_counters[v_id] = 0
            else:
                self.dwell_counters[v_id] = 0

        # Exploit Prevention: At most one overtake credit per step
        # If multiple vehicles qualify simultaneously, credit the one with smallest lateral separation
        if current_candidates:
            current_candidates.sort(key=lambda item: item[0])
            best_lat, best_id = current_candidates[0]
            self.credited_overtakes.add(best_id)
            r_overtake_norm = 1.0

        # ---------------------------------------------------------------------
        # 6. R_terminal: Catastrophic crash / terminal offroad penalty
        # ---------------------------------------------------------------------
        crashed = bool(getattr(ego, "crashed", False))
        on_road = bool(getattr(ego, "on_road", True))
        offroad_terminal = bool(getattr(unwrapped, "config", {}).get("offroad_terminal", False))
        is_terminal_failure = crashed or (offroad_terminal and not on_road)

        if is_terminal_failure:
            r_crash_norm = -1.0
        else:
            r_crash_norm = 0.0

        # ---------------------------------------------------------------------
        # TODO (Sprint 3): Insert tau_stale feature and staleness-gated
        # lane-change regularization penalty here once sensor noise model is added.
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Weighted Composite Reward
        # ---------------------------------------------------------------------
        weighted_speed = cfg.w_speed * r_speed_norm
        weighted_jerk = cfg.w_jerk * r_jerk_norm
        weighted_ttc = cfg.w_ttc * r_ttc_norm
        weighted_headway = cfg.w_headway * r_headway_norm
        weighted_overtake = cfg.w_overtake * r_overtake_norm
        weighted_control = cfg.w_control * r_control_norm
        weighted_crash = abs(cfg.crash_penalty) * r_crash_norm  # -200 if terminal failure, else 0

        total_reward = (
            weighted_speed
            + weighted_jerk
            + weighted_ttc
            + weighted_headway
            + weighted_overtake
            + weighted_control
            + weighted_crash
        )

        components = {
            "reward_speed": float(weighted_speed),
            "reward_jerk": float(weighted_jerk),
            "reward_ttc": float(weighted_ttc),
            "reward_headway": float(weighted_headway),
            "reward_overtake": float(weighted_overtake),
            "reward_control": float(weighted_control),
            "reward_crash": float(weighted_crash),
            "raw_speed_norm": float(r_speed_norm),
            "raw_jerk_norm": float(r_jerk_norm),
            "raw_ttc_norm": float(r_ttc_norm),
            "raw_headway_norm": float(r_headway_norm),
            "raw_overtake_norm": float(r_overtake_norm),
            "raw_control_norm": float(r_control_norm),
            "total_reward": float(total_reward),
        }

        return float(total_reward), components


class ContinuousHighwayWrapper(gym.Wrapper):
    """Gymnasium wrapper applying the calibrated ground-truth continuous reward function."""

    def __init__(
        self,
        env: gym.Env,
        reward_config: ContinuousRewardConfig | None = None,
    ) -> None:
        super().__init__(env)
        self.reward_config = reward_config or ContinuousRewardConfig()
        self.reward_fn = ContinuousRewardFunction(config=self.reward_config)

    @property
    def prev_action(self) -> np.ndarray | None:
        """Expose previous action u_{t-1} = (a_{t-1}, delta_{t-1})."""
        return self.reward_fn.prev_action

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.reward_fn.reset()

        # Continuous actions bypass discrete action masks cleanly
        info["action_mask"] = None
        info["prev_action"] = None
        info["reward_components"] = {
            "reward_speed": 0.0,
            "reward_jerk": 0.0,
            "reward_ttc": 0.0,
            "reward_headway": 0.0,
            "reward_overtake": 0.0,
            "reward_control": 0.0,
            "reward_crash": 0.0,
            "total_reward": 0.0,
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Enforce non-negative speed bound (E1-1: prevents numerical undershoot below 0.0 m/s)
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "vehicle") and unwrapped.vehicle is not None:
            if unwrapped.vehicle.speed < 0.0:
                unwrapped.vehicle.speed = 0.0

        # Compute calibrated ground-truth continuous reward
        reward, components = self.reward_fn.compute_reward(self.env, action)

        # Invalidate discrete action mask
        info["action_mask"] = None

        # Expose previous action u_{t-1}
        info["prev_action"] = (
            self.reward_fn.prev_action.copy()
            if self.reward_fn.prev_action is not None
            else None
        )

        # Expose individual reward terms in info dict
        info["reward_components"] = components
        for k, v in components.items():
            info[k] = v

        return obs, float(reward), terminated, truncated, info

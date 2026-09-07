"""Environment configuration and sensor tier modeling for autonomous driving benchmarks.

Single source of truth for highway-env scenario instantiation, realistic ADAS
radar sensor coverage tiers, and observation filtering across benchmark tasks.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np


SCENARIO_ENV_IDS: dict[str, str] = {
    "highway": "highway-fast-v0",
    "merge": "merge-v1",
    "roundabout": "roundabout-v1",
    "intersection": "intersection-v2",
}

DEFAULT_OBSERVATION_CONFIG: dict[str, Any] = {
    "type": "Kinematics",
    "features": ["presence", "x", "y", "vx", "vy"],
    "vehicles_count": 15,
    "normalize": True,
    "absolute": False,
    "see_behind": True,
}

DEFAULT_ENV_CONFIG: dict[str, Any] = {
    "observation": DEFAULT_OBSERVATION_CONFIG,
    "policy_frequency": 5,
    "simulation_frequency": 15,
    "duration": 100,
}

SCENARIO_CONFIGS: dict[str, dict[str, Any]] = {
    "highway": {
        "lanes_count": 4,
        "vehicles_density": 1.4,
        "vehicles_count": 14,
        "collision_reward": -30.0,
        "high_speed_reward": 1.2,
        "right_lane_reward": 0.0,
        "on_road_reward": 0.2,
        "lane_change_reward": 0.0,
        "reward_speed_range": [18, 30],
        "normalize_reward": False,
    },
    "merge": {},
    "roundabout": {},
    "intersection": {},
}


def get_scaling_factors(env: gym.Env | None) -> tuple[float, float]:
    """Extract physical metric scaling factors for x and y features from an environment.

    Args:
        env: Optional Gymnasium environment instance.

    Returns:
        Tuple of (x_scale, y_scale) in meters per normalized unit.
    """
    if env is not None and hasattr(env.unwrapped, "observation_type"):
        obs_type = env.unwrapped.observation_type
        if (
            getattr(obs_type, "normalize", True)
            and hasattr(obs_type, "features_range")
            and obs_type.features_range
        ):
            f_range = obs_type.features_range
            x_scale = (f_range["x"][1] - f_range["x"][0]) / 2.0
            y_scale = (f_range["y"][1] - f_range["y"][0]) / 2.0
            return float(x_scale), float(y_scale)
    return 200.0, 16.0


def filter_full_adas(obs: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """Tier 0: Full 360 degree radar, 100m range, position + velocity.

    Passes through all vehicles unmodified.

    Args:
        obs: Kinematics observation array of shape (N, F).
        rx: Longitudinal normalization scale factor in meters.
        ry: Lateral normalization scale factor in meters.

    Returns:
        Unmodified copy of observation array.
    """
    return obs.copy()


def filter_front_only(obs: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """Tier 1: Forward radar only (120 degree forward cone).

    Zeroes out traffic rows (indices 1..) where relative longitudinal position
    x < 0 (behind ego) or where absolute bearing angle |atan2(y, x)| > 60 degrees.

    Args:
        obs: Kinematics observation array of shape (N, F).
        rx: Longitudinal normalization scale factor in meters.
        ry: Lateral normalization scale factor in meters.

    Returns:
        Filtered observation array.
    """
    filtered = obs.copy()
    half_cone_rad = math.radians(60.0)

    for i in range(1, len(filtered)):
        if filtered[i, 0] == 0.0:
            continue
        x_rel = filtered[i, 1] * rx
        y_rel = filtered[i, 2] * ry
        if x_rel < 0.0 or abs(math.atan2(y_rel, x_rel)) > half_cone_rad:
            filtered[i, :] = 0.0

    return filtered


def filter_short_range(obs: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """Tier 2: Forward radar with short range (35m cutoff).

    Applies the Tier 1 angular cone filter and zeroes out traffic rows (indices 1..)
    whose Euclidean distance exceeds 35 meters.

    Args:
        obs: Kinematics observation array of shape (N, F).
        rx: Longitudinal normalization scale factor in meters.
        ry: Lateral normalization scale factor in meters.

    Returns:
        Filtered observation array.
    """
    filtered = filter_front_only(obs, rx, ry)
    max_range = 35.0

    for i in range(1, len(filtered)):
        if filtered[i, 0] == 0.0:
            continue
        x_rel = filtered[i, 1] * rx
        y_rel = filtered[i, 2] * ry
        if math.hypot(x_rel, y_rel) > max_range:
            filtered[i, :] = 0.0

    return filtered


def filter_no_velocity(obs: np.ndarray, rx: float, ry: float) -> np.ndarray:
    """Tier 3: Positions only with no relative speed (Doppler radar ablation).

    Applies the Tier 2 short-range filter and zeroes out relative velocity columns
    (vx at index 3, vy at index 4) for all traffic vehicles (indices 1..).

    Args:
        obs: Kinematics observation array of shape (N, F).
        rx: Longitudinal normalization scale factor in meters.
        ry: Lateral normalization scale factor in meters.

    Returns:
        Filtered observation array.
    """
    filtered = filter_short_range(obs, rx, ry)
    filtered[1:, 3] = 0.0
    filtered[1:, 4] = 0.0
    return filtered


TIER_FILTERS: dict[str, Callable[[np.ndarray, float, float], np.ndarray]] = {
    "full_adas": filter_full_adas,
    "front_only": filter_front_only,
    "short_range": filter_short_range,
    "no_velocity": filter_no_velocity,
}


def apply_sensor_tier(
    obs: np.ndarray,
    tier: str,
    env: gym.Env | None = None,
) -> np.ndarray:
    """Apply sensor tier mask to a Kinematics observation matrix.

    Args:
        obs: Observation array of shape (N, F).
        tier: Sensor tier identifier.
        env: Optional environment to retrieve normalization bounds.

    Returns:
        Filtered observation array of identical shape and dtype.

    Raises:
        ValueError: If tier is not recognized.
    """
    if tier not in TIER_FILTERS:
        raise ValueError(
            f"Unknown sensor tier '{tier}'. Available: {get_tier_names()}"
        )

    rx, ry = get_scaling_factors(env)
    return TIER_FILTERS[tier](obs, rx, ry)


class SensorTierWrapper(gym.ObservationWrapper):
    """Observation wrapper applying radar sensor tier masking to observations."""

    def __init__(self, env: gym.Env, tier: str) -> None:
        """Initialize the observation wrapper.

        Args:
            env: Gymnasium environment to wrap.
            tier: Sensor tier identifier.
        """
        super().__init__(env)
        if tier not in TIER_FILTERS:
            raise ValueError(
                f"Unknown sensor tier '{tier}'. Available: {get_tier_names()}"
            )
        self.tier = tier

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """Filter observation according to configured tier.

        Args:
            observation: Raw observation array from the simulator.

        Returns:
            Tier-masked observation array.
        """
        return apply_sensor_tier(observation, self.tier, self.env)


def get_tier_names() -> list[str]:
    """Return the list of supported sensor tier names.

    Returns:
        List containing ['full_adas', 'front_only', 'short_range', 'no_velocity'].
    """
    return list(TIER_FILTERS.keys())


def get_scenario_names() -> list[str]:
    """Return the list of supported benchmark scenario names.

    Returns:
        List containing ['highway', 'merge', 'roundabout', 'intersection'].
    """
    return list(SCENARIO_ENV_IDS.keys())


class TacticalOvertakingWrapper(gym.Wrapper):
    """Augments highway-env with overtaking bonus, headway penalty, and action masks.

    Mechanisms implemented:
    - Mechanism A (Headway penalty): Penalizes tailgating when blocked in-lane
      within 25m of a lead vehicle at positive or matching relative speed.
    - Mechanism B (Overtake bonus): Rewards overtaking vehicles (+1.0 per vehicle).
    - Mechanism C (Action masking): Exposes valid discrete action mask in info dict.
    """

    def __init__(
        self,
        env: gym.Env,
        headway_threshold: float = 40.0,
        headway_penalty_coef: float = 0.35,
        overtake_reward: float = 5.0,
    ) -> None:
        super().__init__(env)
        self.headway_threshold = headway_threshold
        self.headway_penalty_coef = headway_penalty_coef
        self.overtake_reward = overtake_reward
        self._vehicles_ahead: set[int] = set()

    def _get_action_mask(self) -> np.ndarray:
        """Return boolean mask of shape (5,) indicating valid discrete actions."""
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_available_actions"):
            avail = unwrapped.get_available_actions()
            mask = np.zeros(5, dtype=bool)
            for a in avail:
                if 0 <= a < 5:
                    mask[a] = True
            return mask
        return np.ones(5, dtype=bool)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._vehicles_ahead.clear()
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "vehicle") and hasattr(unwrapped, "road"):
            ego = unwrapped.vehicle
            for v in unwrapped.road.vehicles:
                if v is not ego and v.position[0] > ego.position[0]:
                    lane_diff = abs(v.lane_index[2] - ego.lane_index[2]) if hasattr(v, "lane_index") and hasattr(ego, "lane_index") else 0
                    if lane_diff <= 1:
                        self._vehicles_ahead.add(id(v))
        info["action_mask"] = self._get_action_mask()
        info["overtake_count"] = 0
        info["overtake_bonus"] = 0.0
        info["headway_penalty"] = 0.0
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped

        bonus = 0.0
        penalty = 0.0
        overtake_count = 0

        if hasattr(unwrapped, "vehicle") and hasattr(unwrapped, "road"):
            ego = unwrapped.vehicle
            # Mechanism B: Overtake bonus (only for same or adjacent lane vehicles)
            current_ahead: set[int] = set()
            for v in unwrapped.road.vehicles:
                if v is not ego:
                    v_id = id(v)
                    lane_diff = abs(v.lane_index[2] - ego.lane_index[2]) if hasattr(v, "lane_index") and hasattr(ego, "lane_index") else 0
                    if lane_diff <= 1:
                        if v.position[0] > ego.position[0]:
                            current_ahead.add(v_id)
                        elif v_id in self._vehicles_ahead:
                            overtake_count += 1
                            bonus += self.overtake_reward
            self._vehicles_ahead = current_ahead

            # Mechanism A: Headway penalty for tailgating a slower lead car in lane
            if not ego.crashed and hasattr(unwrapped.road, "neighbour_vehicles"):
                front, _ = unwrapped.road.neighbour_vehicles(
                    ego, ego.lane_index
                )
                if front is not None:
                    dist = front.position[0] - ego.position[0]
                    if (
                        0.0 < dist < self.headway_threshold
                        and ego.speed >= front.speed
                    ):
                        frac = 1.0 - (dist / self.headway_threshold)
                        penalty = self.headway_penalty_coef * frac

        shaped_reward = reward + bonus - penalty
        info["action_mask"] = self._get_action_mask()
        info["overtake_count"] = overtake_count
        info["overtake_bonus"] = bonus
        info["headway_penalty"] = penalty

        return obs, shaped_reward, terminated, truncated, info


class TacticalLaneObservationWrapper(gym.Wrapper):
    """Structured lane-slot observation wrapper for tactical overtaking.

    Produces a 27-dimensional observation vector organized from the driver's
    tactical perspective:
    - Ego State (5): speed, target_speed, lane_position, can_left, can_right
    - Lane Slots (16): 4 lanes × (lead_dist, lead_dv, lag_dist, lag_dv)
    - Tactical Signals (6): is_blocked, left_safe, right_safe, left_adv, right_adv, ttc

    Also implements calibrated reward shaping:
    - Persistent blockage penalty (cannot be escaped by braking)
    - Calibrated collision penalty (-10.0)
    - Active overtake bonus (+2.5)
    - Lane change regularization (-0.05)
    - Speed incentive scaled by (v - 20) / 10
    """

    TACTICAL_OBS_DIM = 27
    NUM_LANES = 4
    MAX_LEAD_DIST = 100.0
    MAX_LAG_DIST = 100.0
    MAX_DV = 20.0
    BLOCK_DIST = 55.0
    BLOCK_SPEED = 27.5
    SAFE_MARGIN = 18.0
    SPEED_NORM = 30.0
    TARGET_SPEED = 30.0

    def __init__(
        self,
        env: gym.Env,
        collision_penalty: float = -50.0,
        overtake_bonus: float = 1.0,
        blockage_coef: float = 0.35,
        lane_change_penalty: float = 0.05,
        danger_zone: float = 8.0,
        danger_penalty: float = 0.3,
        jitter_penalty: float = 0.12,
    ) -> None:
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.overtake_bonus = overtake_bonus
        self.blockage_coef = blockage_coef
        self.lane_change_penalty = lane_change_penalty
        self.danger_zone = danger_zone
        self.danger_penalty = danger_penalty
        self.jitter_penalty = jitter_penalty

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.TACTICAL_OBS_DIM,), dtype=np.float32
        )

        self._vehicles_ahead: set[int] = set()
        self._prev_lane: int | None = None
        self._prev_action: int | None = None
        self._steps_blocked: int = 0

    def _build_tactical_obs(self) -> np.ndarray:
        """Build the 27-dim structured tactical observation vector."""
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        road = unwrapped.road

        obs = np.zeros(self.TACTICAL_OBS_DIM, dtype=np.float32)

        # --- Ego State (5 features) ---
        ego_speed = ego.speed
        ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

        obs[0] = np.clip(ego_speed / self.SPEED_NORM, 0.0, 1.0)
        obs[1] = self.TARGET_SPEED / self.SPEED_NORM
        obs[2] = np.clip(ego_lane / max(self.NUM_LANES - 1, 1), 0.0, 1.0)
        obs[3] = 1.0 if ego_lane > 0 else 0.0  # can_lane_left
        obs[4] = 1.0 if ego_lane < self.NUM_LANES - 1 else 0.0  # can_lane_right

        # --- Lane Slots (16 features: 4 lanes × 4) ---
        # For each lane, find closest lead and lag vehicle
        lane_leads = {}  # lane_idx -> (dist, dv)
        lane_lags = {}   # lane_idx -> (dist, dv)

        ego_x = ego.position[0]

        for v in road.vehicles:
            if v is ego:
                continue
            v_lane = v.lane_index[2] if hasattr(v, "lane_index") else -1
            if v_lane < 0 or v_lane >= self.NUM_LANES:
                continue

            dx = v.position[0] - ego_x
            dv = v.speed - ego_speed  # positive = other faster

            if dx > 0:  # lead vehicle
                if v_lane not in lane_leads or dx < lane_leads[v_lane][0]:
                    lane_leads[v_lane] = (dx, dv)
            else:  # lag vehicle
                lag_dist = abs(dx)
                if v_lane not in lane_lags or lag_dist < lane_lags[v_lane][0]:
                    lane_lags[v_lane] = (lag_dist, dv)

        for lane_idx in range(self.NUM_LANES):
            base = 5 + lane_idx * 4

            if lane_idx in lane_leads:
                d_lead, dv_lead = lane_leads[lane_idx]
                obs[base + 0] = np.clip(d_lead / self.MAX_LEAD_DIST, 0.0, 1.0)
                obs[base + 1] = np.clip(dv_lead / self.MAX_DV, -1.0, 1.0)
            else:
                obs[base + 0] = 1.0  # max distance (empty)
                obs[base + 1] = 0.0

            if lane_idx in lane_lags:
                d_lag, dv_lag = lane_lags[lane_idx]
                obs[base + 2] = np.clip(d_lag / self.MAX_LAG_DIST, 0.0, 1.0)
                obs[base + 3] = np.clip(dv_lag / self.MAX_DV, -1.0, 1.0)
            else:
                obs[base + 2] = 1.0  # max distance (empty)
                obs[base + 3] = 0.0

        # --- Tactical Action Signals (6 features) ---
        # Current lane lead info
        cur_lead = lane_leads.get(ego_lane, (self.MAX_LEAD_DIST, 0.0))
        cur_lead_dist, cur_lead_dv = cur_lead

        # is_blocked: lead < BLOCK_DIST and lead is slower
        is_blocked = (cur_lead_dist < self.BLOCK_DIST and
                      (ego_speed + cur_lead_dv) < self.BLOCK_SPEED)
        obs[21] = 1.0 if is_blocked else 0.0

        # left_lane_safe
        left_lane = ego_lane - 1
        if left_lane >= 0:
            l_lead = lane_leads.get(left_lane, (self.MAX_LEAD_DIST, 0.0))
            l_lag = lane_lags.get(left_lane, (self.MAX_LAG_DIST, 0.0))
            obs[22] = 1.0 if (l_lead[0] > self.SAFE_MARGIN and
                              l_lag[0] > self.SAFE_MARGIN) else 0.0
        else:
            obs[22] = 0.0

        # right_lane_safe
        right_lane = ego_lane + 1
        if right_lane < self.NUM_LANES:
            r_lead = lane_leads.get(right_lane, (self.MAX_LEAD_DIST, 0.0))
            r_lag = lane_lags.get(right_lane, (self.MAX_LAG_DIST, 0.0))
            obs[23] = 1.0 if (r_lead[0] > self.SAFE_MARGIN and
                              r_lag[0] > self.SAFE_MARGIN) else 0.0
        else:
            obs[23] = 0.0

        # left_lane_advantage (speed of left lead vs current lead)
        if left_lane >= 0 and left_lane in lane_leads:
            left_lead_speed = ego_speed + lane_leads[left_lane][1]
            cur_lead_speed = ego_speed + cur_lead_dv
            obs[24] = np.clip((left_lead_speed - cur_lead_speed) / self.MAX_DV, -1.0, 1.0)
        else:
            obs[24] = 0.0

        # right_lane_advantage
        if right_lane < self.NUM_LANES and right_lane in lane_leads:
            right_lead_speed = ego_speed + lane_leads[right_lane][1]
            cur_lead_speed = ego_speed + cur_lead_dv
            obs[25] = np.clip((right_lead_speed - cur_lead_speed) / self.MAX_DV, -1.0, 1.0)
        else:
            obs[25] = 0.0

        # time_to_collision (normalized)
        if cur_lead_dv < 0:  # closing in
            ttc = cur_lead_dist / abs(cur_lead_dv)
            obs[26] = np.clip(ttc / 10.0, 0.0, 1.0)  # normalize to 10s
        else:
            obs[26] = 1.0  # no collision risk

        return obs

    def _get_action_mask(self) -> np.ndarray:
        """Return boolean mask of shape (5,) indicating valid discrete actions."""
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_available_actions"):
            avail = unwrapped.get_available_actions()
            mask = np.zeros(5, dtype=bool)
            for a in avail:
                if 0 <= a < 5:
                    mask[a] = True
            return mask
        return np.ones(5, dtype=bool)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        self._vehicles_ahead.clear()
        self._steps_blocked = 0
        self._prev_action = None

        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        self._prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

        # Track vehicles ahead in adjacent corridor (|lane_diff| <= 1)
        ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
        for v in unwrapped.road.vehicles:
            if v is not ego and v.position[0] > ego.position[0]:
                v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                if abs(v_lane - ego_lane) <= 1:
                    self._vehicles_ahead.add(id(v))

        obs = self._build_tactical_obs()
        info["action_mask"] = self._get_action_mask()
        info["overtake_count"] = 0
        info["speed"] = ego.speed
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        _, base_reward, terminated, truncated, info = self.env.step(action)

        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle

        # --- Build tactical observation ---
        obs = self._build_tactical_obs()

        # --- Calibrated Reward Shaping (v3: crash-averse + anti-jitter + impatience) ---
        reward = 0.0
        overtake_count = 0

        if ego.crashed:
            reward = self.collision_penalty  # -50: must outweigh any speed gains
        else:
            # Speed incentive: scaled to 0.5 max (was 1.0)
            speed_reward = 0.5 * np.clip((ego.speed - 20.0) / 10.0, 0.0, 1.0)
            reward += speed_reward

            # Survival bonus: +0.15 per step for staying alive
            reward += 0.15

            # Proximity danger penalty: penalize getting too close to others
            min_dist = float('inf')
            for v in unwrapped.road.vehicles:
                if v is not ego:
                    dx = v.position[0] - ego.position[0]
                    dy = v.position[1] - ego.position[1]
                    d = np.sqrt(dx**2 + dy**2)
                    if d < min_dist:
                        min_dist = d
            if min_dist < self.danger_zone:
                # Quadratic penalty: gets severe as distance → 0
                proximity_frac = 1.0 - (min_dist / self.danger_zone)
                reward -= self.danger_penalty * (proximity_frac ** 2)

            # Persistent blockage penalty with escalating impatience
            front, _ = unwrapped.road.neighbour_vehicles(ego, ego.lane_index)
            if front is not None:
                dist = front.position[0] - ego.position[0]
                if 0.0 < dist < self.BLOCK_DIST and front.speed < self.BLOCK_SPEED:
                    self._steps_blocked += 1
                    # Impatience escalates from 1.0x to 2.5x after 10 steps (2 seconds)
                    impatience_factor = 1.0 + min(max(self._steps_blocked - 10, 0) / 10.0, 1.5)
                    frac = 1.0 - (dist / self.BLOCK_DIST)
                    reward -= self.blockage_coef * frac * impatience_factor
                else:
                    self._steps_blocked = 0
            else:
                self._steps_blocked = 0

            # Overtake bonus (strictly adjacent corridor: |lane_diff| <= 1)
            current_ahead: set[int] = set()
            ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
            for v in unwrapped.road.vehicles:
                if v is not ego:
                    v_id = id(v)
                    v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                    lane_diff = abs(v_lane - ego_lane)
                    if lane_diff <= 1:
                        if v.position[0] > ego.position[0]:
                            current_ahead.add(v_id)
                        elif v_id in self._vehicles_ahead:
                            overtake_count += 1
            self._vehicles_ahead = current_ahead
            reward += overtake_count * self.overtake_bonus

            # Lane change regularization
            new_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
            if self._prev_lane is not None and new_lane != self._prev_lane:
                reward -= self.lane_change_penalty
            self._prev_lane = new_lane

            # Action jitter penalty (discourage bang-bang oscillation between FASTER and SLOWER)
            if self._prev_action is not None:
                if (action == 3 and self._prev_action == 4) or (action == 4 and self._prev_action == 3):
                    reward -= self.jitter_penalty
                elif action in [3, 4] and self._prev_action in [1, 3, 4] and action != self._prev_action:
                    reward -= self.jitter_penalty * 0.4
            self._prev_action = action

        info["action_mask"] = self._get_action_mask()
        info["overtake_count"] = overtake_count if not ego.crashed else 0
        info["speed"] = ego.speed
        info["crashed"] = ego.crashed

        return obs, float(reward), terminated, truncated, info


class FrameStackTacticalWrapper(gym.Wrapper):
    """Frame stacking wrapper for tactical observations.

    Stacks K consecutive tactical observation frames to provide temporal
    derivatives (closing rates, acceleration signals) without LSTM complexity.
    """

    def __init__(self, env: gym.Env, k: int = 3) -> None:
        super().__init__(env)
        self.k = k
        inner_dim = env.observation_space.shape[0]
        self.stacked_dim = inner_dim * k
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.stacked_dim,), dtype=np.float32
        )
        self._frames: list[np.ndarray] = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames = [obs.copy() for _ in range(self.k)]
        return self._get_stacked_obs(), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.pop(0)
        self._frames.append(obs.copy())
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=0).astype(np.float32)


def make_env(
    scenario: str,
    tier: str,
    seed: int | None = None,
    render_mode: str | None = None,
    tactical_overtaking: bool = False,
    **overrides: Any,
) -> gym.Env:
    """Create and return a configured, tier-wrapped, and seeded benchmark environment.

    Args:
        scenario: Scenario name ('highway', 'merge', 'roundabout', 'intersection').
        tier: Sensor tier ('full_adas', 'front_only', 'short_range', 'no_velocity').
        seed: Optional random seed for reproducible initialization.
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        tactical_overtaking: If True, wraps with TacticalOvertakingWrapper (Mechanisms A, B, C).
        **overrides: Optional configuration overrides to merge into the environment
            config.

    Returns:
        Configured Gymnasium environment wrapped with SensorTierWrapper.

    Raises:
        ValueError: If scenario or tier is invalid.
    """
    if scenario not in SCENARIO_ENV_IDS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Available: {get_scenario_names()}"
        )
    if tier not in TIER_FILTERS:
        raise ValueError(
            f"Unknown sensor tier '{tier}'. Available tiers: {get_tier_names()}"
        )

    env_config = copy.deepcopy(DEFAULT_ENV_CONFIG)
    env_config.update(SCENARIO_CONFIGS[scenario])

    if "observation" in overrides and isinstance(overrides["observation"], dict):
        env_config["observation"].update(overrides.pop("observation"))

    env_config.update(overrides)

    env = gym.make(
        SCENARIO_ENV_IDS[scenario],
        render_mode=render_mode,
        config=env_config,
    )
    wrapped_env: gym.Env = SensorTierWrapper(env, tier=tier)

    if tactical_overtaking:
        wrapped_env = TacticalOvertakingWrapper(wrapped_env)

    if seed is not None:
        wrapped_env.reset(seed=seed)
        wrapped_env.action_space.seed(seed)

    return wrapped_env


def make_optimal_env(
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    seed: int | None = None,
    render_mode: str | None = None,
    frame_stack_k: int = 3,
    duration: int = 100,
) -> gym.Env:
    """Create an environment with TacticalLaneObservation + FrameStack wrappers.

    This is the factory function for the optimal overtaking agent pipeline.

    Args:
        lanes_count: Number of highway lanes.
        vehicles_density: Traffic density parameter.
        vehicles_count: Total NPC vehicles.
        seed: Random seed.
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        frame_stack_k: Number of frames to stack (0 = no stacking).
        duration: Episode duration in simulated seconds (100s = 500 steps).

    Returns:
        Wrapped environment producing 81-dim (K=3) or 27-dim (K=0) observations.
    """
    cfg = copy.deepcopy(DEFAULT_ENV_CONFIG)
    cfg.update({
        "lanes_count": lanes_count,
        "vehicles_density": vehicles_density,
        "vehicles_count": vehicles_count,
        "duration": duration,
        "simulation_frequency": 5,  # 1:1 with policy_freq → +33% SPS
        "collision_reward": 0.0,  # reward shaping handled by wrapper
        "high_speed_reward": 0.0,
        "right_lane_reward": 0.0,
        "on_road_reward": 0.0,
        "lane_change_reward": 0.0,
        "normalize_reward": False,
        "reward_speed_range": [18, 30],
    })

    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config=cfg,
    )

    env = TacticalLaneObservationWrapper(env)

    if frame_stack_k > 0:
        env = FrameStackTacticalWrapper(env, k=frame_stack_k)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def _run_smoke_test() -> None:
    """Execute smoke test across all 16 scenario and sensor tier combinations."""
    scenarios = get_scenario_names()
    tiers = get_tier_names()
    total_tests = len(scenarios) * len(tiers)
    passed_tests = 0

    print("=" * 70)
    print(
        f"Running Stage 1 Smoke Test ({total_tests} combinations: "
        f"{len(scenarios)} scenarios x {len(tiers)} tiers)"
    )
    print("=" * 70)

    for s_idx, scenario in enumerate(scenarios, start=1):
        for t_idx, tier in enumerate(tiers, start=1):
            comb_idx = (s_idx - 1) * len(tiers) + t_idx
            env = make_env(scenario=scenario, tier=tier, seed=42)
            obs, info = env.reset(seed=42)
            assert obs.shape == (15, 5), f"Expected shape (15, 5), got {obs.shape}"

            for _ in range(5):
                action = 1
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()

            env.close()
            passed_tests += 1
            presences = int(np.sum(obs[:, 0]))
            print(
                f"[{comb_idx:02d}/{total_tests:02d}] "
                f"scenario={scenario:<13} tier={tier:<12} "
                f"shape={obs.shape} detected_vehicles={presences}/10 -> PASS"
            )

    print("=" * 70)
    print(f"Smoke test complete: {passed_tests}/{total_tests} passed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    _run_smoke_test()

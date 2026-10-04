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
    "simulation_frequency": 5,
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
    "roundabout": {
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "duration": 11,
        "action": {
            "type": "DiscreteMetaAction",
            "target_speeds": [0, 8, 16],
        },
        "observation": {
            "type": "Kinematics",
            "absolute": True,
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-15, 15],
                "vy": [-15, 15],
            },
        },
        "collision_reward": -1.0,
        "high_speed_reward": 0.2,
        "lane_change_reward": -0.05,
        "right_lane_reward": 0.0,
    },
    "intersection": {
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "duration": 13,
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": False,
            "target_speeds": [0, 4.5, 9],
        },
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 15,
            "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
            "features_range": {
                "x": [-100, 100],
                "y": [-100, 100],
                "vx": [-20, 20],
                "vy": [-20, 20],
            },
            "absolute": True,
            "flatten": False,
            "observe_intentions": False,
        },
        "collision_reward": -5.0,
        "arrived_reward": 1.0,
        "high_speed_reward": 1.0,
        "reward_speed_range": [7.0, 9.0],
        "offroad_terminal": False,
    },
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
    - Mechanism B (Overtake bonus): Rewards overtaking vehicles (+5.0 per vehicle).
    - Mechanism C (Action masking): Exposes valid discrete action mask in info dict.
    """

    def __init__(
        self,
        env: gym.Env,
        headway_threshold: float = 40.0,
        headway_penalty_coef: float = 0.35,
        overtake_reward: float = 5.0,
        overtake_dwell_steps: int = 5,
        max_overtake_lon_dist: float = 30.0,
    ) -> None:
        super().__init__(env)
        self.headway_threshold = headway_threshold
        self.headway_penalty_coef = headway_penalty_coef
        self.overtake_reward = overtake_reward
        self.overtake_dwell_steps = overtake_dwell_steps
        self.max_overtake_lon_dist = max_overtake_lon_dist
        self._vehicles_ahead: set[int] = set()
        self._credited_overtakes: set[int] = set()
        self._dwell_counters: dict[int, int] = {}

    def _get_action_mask(self) -> np.ndarray:
        """Return boolean mask indicating valid discrete actions."""
        n_actions = getattr(self.action_space, "n", 5)
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_available_actions"):
            avail = unwrapped.get_available_actions()
            mask = np.zeros(n_actions, dtype=bool)
            for a in avail:
                if 0 <= a < n_actions:
                    mask[a] = True
            return mask
        return np.ones(n_actions, dtype=bool)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._vehicles_ahead.clear()
        self._credited_overtakes.clear()
        self._dwell_counters.clear()
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
            # Mechanism B: Overtake bonus (only for same or adjacent lane vehicles, dwell >= 5 steps, once per episode)
            current_candidates: list[tuple[float, int]] = []
            ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
            for v in unwrapped.road.vehicles:
                if v is not ego:
                    v_id = id(v)
                    if v_id in self._credited_overtakes:
                        continue
                    v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                    lane_diff = abs(v_lane - ego_lane)
                    dx = float(v.position[0] - ego.position[0])
                    dy = abs(float(v.position[1] - ego.position[1]))

                    if dx > 0:
                        if lane_diff <= 1:
                            self._vehicles_ahead.add(v_id)
                        self._dwell_counters[v_id] = 0
                    elif dx < 0 and v_id in self._vehicles_ahead:
                        if lane_diff <= 1 and abs(dx) <= self.max_overtake_lon_dist:
                            self._dwell_counters[v_id] = self._dwell_counters.get(v_id, 0) + 1
                            if self._dwell_counters[v_id] >= self.overtake_dwell_steps:
                                current_candidates.append((dy, v_id))
                        else:
                            self._dwell_counters[v_id] = 0
                    else:
                        self._dwell_counters[v_id] = 0

            if current_candidates:
                current_candidates.sort(key=lambda item: item[0])
                _, best_id = current_candidates[0]
                self._credited_overtakes.add(best_id)
                overtake_count = 1
                bonus += self.overtake_reward

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


def wrap_to_pi(x: float) -> float:
    """Normalize an angle in radians to the interval [-pi, pi].

    Args:
        x: Angle in radians.

    Returns:
        Normalized angle in [-pi, pi].
    """
    return float((x + np.pi) % (2.0 * np.pi) - np.pi)


class TacticalLaneObservationWrapper(gym.Wrapper):
    """Structured lane-slot observation wrapper for tactical overtaking and continuous control.

    Produces a 30-dimensional observation vector (or 27-dimensional legacy vector)
    organized from the driver's tactical and continuous control perspective:
    - Ego State (5): speed, target_speed, lane_position, can_left, can_right
    - Lane Slots (16): 4 lanes × (lead_dist, lead_dv, lag_dist, lag_dv)
    - Tactical Signals (6): is_blocked, left_safe, right_safe, left_adv, right_adv, ttc
    - Continuous Control Stability Features (3):
      - psi_err: heading error relative to current lane reference direction (normalized by pi/4)
      - y_lane: lateral offset from current lane centerline (normalized by 4.0m lane width)
      - yaw_rate: heading angular velocity dpsi/dt via finite difference (normalized by 1.0 rad/s)

    Also implements calibrated reward shaping:
    - Persistent blockage penalty (cannot be escaped by braking)
    - Collision penalty (-50.0)
    - Overtake bonus (+1.0)
    - Lane change regularization (-0.05)
    - Speed incentive scaled by (v - 20) / 10
    """

    BASE_TACTICAL_OBS_DIM = 27
    CONTINUOUS_FEATURES_DIM = 3
    TACTICAL_OBS_DIM = 30
    NUM_LANES = 4
    MAX_LEAD_DIST = 100.0
    MAX_LAG_DIST = 100.0
    MAX_DV = 20.0
    BLOCK_DIST = 55.0
    BLOCK_SPEED = 27.5
    SAFE_MARGIN = 18.0
    SPEED_NORM = 30.0
    TARGET_SPEED = 30.0
    MAX_HEADING_ERROR = np.pi / 4.0  # 45 degrees in radians
    MAX_LANE_OFFSET = 4.0            # 1 standard HighwayEnv lane width in meters
    MAX_YAW_RATE = 1.0               # 1.0 rad/s (~57.3 deg/s)

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
        include_continuous_features: bool = True,
        apply_reward_shaping: bool = True,
        constrained_mode: bool = False,
        overtake_dwell_steps: int = 5,
        max_overtake_lon_dist: float = 30.0,
        lateral_debounce_steps: int = 0,
    ) -> None:
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.overtake_bonus = overtake_bonus
        self.blockage_coef = blockage_coef
        self.lane_change_penalty = lane_change_penalty
        self.danger_zone = danger_zone
        self.danger_penalty = danger_penalty
        self.jitter_penalty = jitter_penalty
        self.include_continuous_features = include_continuous_features
        self.apply_reward_shaping = apply_reward_shaping
        self.constrained_mode = constrained_mode
        self.overtake_dwell_steps = overtake_dwell_steps
        self.max_overtake_lon_dist = max_overtake_lon_dist
        self.lateral_debounce_steps = lateral_debounce_steps
        self._was_crashed = False

        self.obs_dim = (
            self.TACTICAL_OBS_DIM
            if self.include_continuous_features
            else self.BASE_TACTICAL_OBS_DIM
        )
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )

        self._vehicles_ahead: set[int] = set()
        self._credited_overtakes: set[int] = set()
        self._dwell_counters: dict[int, int] = {}
        self._prev_lane: int | None = None
        self._prev_action: int | None = None
        self._lateral_lock_dir: int | None = None
        self._lateral_lock_timer: int = 0
        self._steps_blocked: int = 0
        self._prev_heading: float | None = None
        self._current_yaw_rate: float = 0.0

    def _build_tactical_obs(self) -> np.ndarray:
        """Build the structured tactical observation vector (30-dim or 27-dim)."""
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        road = unwrapped.road

        obs = np.zeros(self.obs_dim, dtype=np.float32)

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

        # --- Continuous Control Stability Features (3 features) ---
        if self.include_continuous_features:
            lane = getattr(ego, "lane", None)
            if lane is None and hasattr(unwrapped, "road") and hasattr(ego, "lane_index"):
                lane = unwrapped.road.network.get_lane(ego.lane_index)

            if lane is not None:
                s, lat = lane.local_coordinates(ego.position)
                lane_heading = lane.heading_at(s)
                psi_err = wrap_to_pi(float(ego.heading) - float(lane_heading))
                y_lane = float(lat)
            else:
                psi_err = 0.0
                y_lane = 0.0

            self._last_psi_err = psi_err
            self._last_y_lane = y_lane

            obs[27] = np.clip(psi_err / self.MAX_HEADING_ERROR, -1.0, 1.0)
            obs[28] = np.clip(y_lane / self.MAX_LANE_OFFSET, -1.0, 1.0)
            obs[29] = np.clip(self._current_yaw_rate / self.MAX_YAW_RATE, -1.0, 1.0)

        return obs

    def _get_action_mask(self) -> np.ndarray | None:
        """Return boolean mask indicating valid discrete actions, or None for continuous actions."""
        if isinstance(self.action_space, gym.spaces.Box):
            return None
        n_actions = getattr(self.action_space, "n", 5)
        unwrapped = self.env.unwrapped
        mask = np.ones(n_actions, dtype=bool)
        if hasattr(unwrapped, "get_available_actions"):
            try:
                avail = unwrapped.get_available_actions()
                mask = np.zeros(n_actions, dtype=bool)
                for a in avail:
                    if 0 <= a < n_actions:
                        mask[a] = True
            except (NotImplementedError, AttributeError):
                mask = np.ones(n_actions, dtype=bool)

        # Apply lateral action debounce lockout if active (0: LANE_LEFT, 2: LANE_RIGHT)
        if self.lateral_debounce_steps > 0 and self._lateral_lock_timer > 0 and self._lateral_lock_dir is not None:
            opp = 2 if self._lateral_lock_dir == 0 else 0
            if opp < n_actions:
                mask[opp] = False
        return mask

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        _, info = self.env.reset(seed=seed, options=options)
        self._vehicles_ahead.clear()
        self._credited_overtakes.clear()
        self._dwell_counters.clear()
        self._steps_blocked = 0
        self._prev_action = None
        self._lateral_lock_dir = None
        self._lateral_lock_timer = 0

        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        self._prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

        self._prev_heading = float(ego.heading) if hasattr(ego, "heading") else 0.0
        self._current_yaw_rate = 0.0
        self._last_psi_err = 0.0
        self._last_y_lane = 0.0

        # Track vehicles ahead in adjacent corridor (|lane_diff| <= 1)
        ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
        for v in unwrapped.road.vehicles:
            if v is not ego and v.position[0] > ego.position[0]:
                v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                if abs(v_lane - ego_lane) <= 1:
                    self._vehicles_ahead.add(id(v))

        obs = self._build_tactical_obs()
        self._was_crashed = False
        info["action_mask"] = self._get_action_mask()
        info["overtake_count"] = 0
        info["speed"] = ego.speed
        info["cost"] = 0.0
        if self.include_continuous_features:
            info["heading_error"] = self._last_psi_err
            info["lateral_offset"] = self._last_y_lane
            info["yaw_rate"] = self._current_yaw_rate
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Debounce rapid opposing lateral action if within lock window
        filtered_action = action
        if self.lateral_debounce_steps > 0 and self._lateral_lock_timer > 0 and self._lateral_lock_dir is not None:
            opp = 2 if self._lateral_lock_dir == 0 else 0
            if action == opp:
                filtered_action = 1  # Debounce opposing reversal to IDLE

        if self.lateral_debounce_steps > 0:
            if filtered_action in (0, 2):
                self._lateral_lock_dir = filtered_action
                self._lateral_lock_timer = self.lateral_debounce_steps
            elif self._lateral_lock_timer > 0:
                self._lateral_lock_timer -= 1
                if self._lateral_lock_timer == 0:
                    self._lateral_lock_dir = None

        _, base_reward, terminated, truncated, info = self.env.step(filtered_action)

        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle

        # --- Compute yaw rate via finite difference ---
        dt = 1.0 / unwrapped.config.get("policy_frequency", 5)
        current_heading = float(ego.heading) if hasattr(ego, "heading") else 0.0
        if self._prev_heading is not None and dt > 0:
            heading_diff = wrap_to_pi(current_heading - self._prev_heading)
            self._current_yaw_rate = float(heading_diff / dt)
        else:
            self._current_yaw_rate = 0.0
        self._prev_heading = current_heading

        # --- Build tactical observation ---
        obs = self._build_tactical_obs()

        # --- Calibrated Reward Shaping (only for discrete meta-actions) ---
        reward = base_reward
        overtake_count = 0

        is_newly_crashed = bool(ego.crashed and not self._was_crashed)
        if ego.crashed:
            self._was_crashed = True
        cost = 1.0 if is_newly_crashed else 0.0

        if self.apply_reward_shaping and not isinstance(self.action_space, gym.spaces.Box):
            reward = 0.0
            if ego.crashed:
                # In constrained mode, strip collision_penalty so R_t only contains speed/overtake terms
                reward = 0.0 if self.constrained_mode else self.collision_penalty
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

                # Overtake bonus (strictly adjacent corridor: |lane_diff| <= 1, debounced with dwell and once-per-episode credit)
                current_candidates: list[tuple[float, int]] = []
                ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
                for v in unwrapped.road.vehicles:
                    if v is not ego:
                        v_id = id(v)
                        if v_id in self._credited_overtakes:
                            continue
                        v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                        lane_diff = abs(v_lane - ego_lane)
                        dx = float(v.position[0] - ego.position[0])
                        dy = abs(float(v.position[1] - ego.position[1]))

                        if dx > 0:
                            if lane_diff <= 1:
                                self._vehicles_ahead.add(v_id)
                            self._dwell_counters[v_id] = 0
                        elif dx < 0 and v_id in self._vehicles_ahead:
                            if lane_diff <= 1 and abs(dx) <= self.max_overtake_lon_dist:
                                self._dwell_counters[v_id] = self._dwell_counters.get(v_id, 0) + 1
                                if self._dwell_counters[v_id] >= self.overtake_dwell_steps:
                                    current_candidates.append((dy, v_id))
                            else:
                                self._dwell_counters[v_id] = 0
                        else:
                            self._dwell_counters[v_id] = 0

                if current_candidates:
                    current_candidates.sort(key=lambda item: item[0])
                    _, best_id = current_candidates[0]
                    self._credited_overtakes.add(best_id)
                    overtake_count = 1
                    reward += self.overtake_bonus

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
        info["cost"] = cost
        if self.include_continuous_features:
            info["heading_error"] = self._last_psi_err
            info["lateral_offset"] = self._last_y_lane
            info["yaw_rate"] = self._current_yaw_rate

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


# Canonical alias for FrameStackTacticalWrapper
ObservationStackWrapper = FrameStackTacticalWrapper


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
    include_continuous_features: bool = False,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
    constrained_mode: bool = False,
    overtake_dwell_steps: int = 5,
    max_overtake_lon_dist: float = 30.0,
    target_speeds: list[float] | None = None,
    lateral_debounce_steps: int = 0,
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
        include_continuous_features: If True, computes and appends 3 continuous control
            stability features (psi_err, y_lane, yaw_rate) to yield 30-dim base
            (or 90-dim stacked at K=3). If False, outputs legacy 27-dim base (81-dim stacked).
        collision_penalty: Terminal collision penalty for reward shaping (default: -50.0).
        overtake_bonus: Overtake bonus per vehicle for reward shaping (default: 1.0).
        constrained_mode: If True, strips collision penalty from reward and emits info['cost'].
        overtake_dwell_steps: Dwell steps required before crediting overtake (default: 5).
        max_overtake_lon_dist: Maximum longitudinal distance behind ego for overtake credit (default: 30.0).
        target_speeds: Optional list of target speeds for DiscreteMetaAction (default: None = [20, 25, 30]).
        lateral_debounce_steps: Minimum lock steps preventing opposing lateral reversals (default: 5).

    Returns:
        Wrapped environment producing 90-dim (K=3) or 30-dim (K=0) observations
        (or 81/27 if include_continuous_features=False).
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
    if target_speeds is not None:
        cfg["action"] = {
            "type": "DiscreteMetaAction",
            "target_speeds": target_speeds,
        }

    env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config=cfg,
    )

    env = TacticalLaneObservationWrapper(
        env,
        collision_penalty=collision_penalty,
        overtake_bonus=overtake_bonus,
        include_continuous_features=include_continuous_features,
        constrained_mode=constrained_mode,
        overtake_dwell_steps=overtake_dwell_steps,
        max_overtake_lon_dist=max_overtake_lon_dist,
        lateral_debounce_steps=lateral_debounce_steps,
    )

    if frame_stack_k > 0:
        env = FrameStackTacticalWrapper(env, k=frame_stack_k)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def make_continuous_env(
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    seed: int | None = None,
    render_mode: str | None = None,
    frame_stack_k: int = 3,
    duration: int = 100,
    include_continuous_features: bool = True,
    reward_config: Any = None,
    offroad_terminal: bool = True,
    tier: str = "full_adas",
    **overrides: Any,
) -> gym.Env:
    """Create an environment with ContinuousAction, ground-truth reward, and 30-dim observation wrapper.

    Factory function for continuous-control overtaking agent pipeline (Sprint 1).

    Args:
        lanes_count: Number of highway lanes.
        vehicles_density: Traffic density parameter.
        vehicles_count: Total NPC vehicles.
        seed: Random seed.
        render_mode: Rendering mode ('human', 'rgb_array', or None).
        frame_stack_k: Number of frames to stack (0 = no stacking).
        duration: Episode duration in simulated seconds (100s = 500 steps).
        include_continuous_features: If True, computes and appends 3 continuous control
            stability features (psi_err, y_lane, yaw_rate) to yield 30-dim base
            (or 90-dim stacked at K=3).
        reward_config: Optional ContinuousRewardConfig instance.
        offroad_terminal: If True, terminates episode on leaving road.
        tier: Sensor tier for observation filtering ('full_adas', 'front_only', etc.).
        **overrides: Additional config overrides merged into highway-env config.

    Returns:
        Wrapped continuous environment producing 90-dim (K=3) or 30-dim (K=0) observations.
    """
    from continuous_reward import ContinuousHighwayWrapper, ContinuousRewardConfig

    cfg = copy.deepcopy(DEFAULT_ENV_CONFIG)
    cfg.update({
        "action": {
            "type": "ContinuousAction",
            "acceleration_range": (-5.0, 3.0),
            "steering_range": (-math.pi / 12.0, math.pi / 12.0),
            "speed_range": (0.0, 30.0),
            "longitudinal": True,
            "lateral": True,
            "dynamical": False,
            "clip": True,
        },
        "lanes_count": lanes_count,
        "vehicles_density": vehicles_density,
        "vehicles_count": vehicles_count,
        "duration": duration,
        "policy_frequency": 5,
        "simulation_frequency": 15,
        "offroad_terminal": offroad_terminal,
        "collision_reward": 0.0,
        "high_speed_reward": 0.0,
        "right_lane_reward": 0.0,
        "on_road_reward": 0.0,
        "lane_change_reward": 0.0,
        "normalize_reward": False,
    })
    cfg.update(overrides)

    base_env = gym.make(
        "highway-fast-v0",
        render_mode=render_mode,
        config=cfg,
    )

    if tier != "full_adas":
        base_env = SensorTierWrapper(base_env, tier=tier)

    # Wrap with 30-dim observation wrapper (discrete reward shaping disabled)
    env = TacticalLaneObservationWrapper(
        base_env,
        include_continuous_features=include_continuous_features,
        apply_reward_shaping=False,
    )

    # Wrap with ground-truth continuous reward wrapper
    reward_cfg = reward_config or ContinuousRewardConfig()
    env = ContinuousHighwayWrapper(env, reward_config=reward_cfg)

    # Frame stacking
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

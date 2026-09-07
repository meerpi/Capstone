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

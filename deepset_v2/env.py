"""DeepSet v2 environment wrapper for highway-fast-v0.

Provides set-structured policy observations with sensor-tier filtering,
while reusing TacticalLaneObservationWrapper for reward and info.
"""

from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np

from env_config import (
    DEFAULT_OBSERVATION_CONFIG,
    TacticalLaneObservationWrapper,
    apply_sensor_tier,
    get_scaling_factors,
)

# ---- Feature scaling constants (physical units → normalised) ----
X_SCALE = 100.0    # metres, observation horizon (MAX_LEAD_DIST / MAX_LAG_DIST)
Y_SCALE = 4.0      # metres, one lane width (AbstractLane.DEFAULT_WIDTH)
VX_SCALE = 30.0    # m/s, SPEED_NORM
VY_SCALE = 30.0    # m/s, SPEED_NORM
SPEED_SCALE = 30.0  # m/s

N_VEHICLES = 14     # other-vehicle slots in the observation
N_FEATURES = 5      # presence, x, y, vx, vy per vehicle
EGO_DIM = 4         # speed, lane_pos, can_left, can_right


class _KinematicsStashWrapper(gym.Wrapper):
    """Thin wrapper below TacticalLaneObservationWrapper that stashes the
    tier-filtered Kinematics array in info['_kinematics'] so the outer
    DeepSet wrapper can read it without a second observe() call.

    Design justification: TacticalLaneObservationWrapper discards the env
    observation and rebuilds from unwrapped.road.vehicles. Placing this
    wrapper *between* the base env and the tactical wrapper lets us capture
    the Kinematics obs on both reset and step, tier-filter it, and pass it
    through info. The tactical wrapper ignores both the obs and any extra
    info keys, so this is transparent.
    """

    def __init__(self, env: gym.Env, tier: str) -> None:
        super().__init__(env)
        self.tier = tier

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        filtered = apply_sensor_tier(obs, self.tier, self.env)
        info["_kinematics"] = filtered
        return obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        filtered = apply_sensor_tier(obs, self.tier, self.env)
        info["_kinematics"] = filtered
        return obs, reward, terminated, truncated, info


class DeepSetObsWrapper(gym.ObservationWrapper):
    """Converts the tactical wrapper's output into a flat DeepSet observation.

    Output shape: (EGO_DIM + N_VEHICLES * N_FEATURES,) = (4 + 14*5,) = (74,)
    Layout: ego(4) ++ 14 × [presence, x/100, y/4, vx/30, vy/30]

    Masked rows (presence=0 or |x_rel|>range_m) are exactly zero.
    """

    def __init__(self, env: gym.Env, tier: str, range_m: float = 100.0) -> None:
        super().__init__(env)
        self.tier = tier
        self.range_m = range_m
        self.obs_dim = EGO_DIM + N_VEHICLES * N_FEATURES
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        # Normalization scales from highway-env (set after first reset)
        self._rx: float | None = None
        self._ry: float | None = None

    def _ensure_scales(self) -> None:
        """Lazily read the normalization scales from the env."""
        if self._rx is None:
            self._rx, self._ry = get_scaling_factors(self.env)

    def _build_deepset_obs(self, info: dict[str, Any]) -> np.ndarray:
        """Build the flat DeepSet observation from stashed Kinematics + ego state."""
        self._ensure_scales()
        rx, ry = self._rx, self._ry

        kin = info["_kinematics"]  # (15, 5) tier-filtered, normalized [-1,1]

        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
        num_lanes = unwrapped.config.get("lanes_count", 4)

        # Ego features: speed/30, lane_pos in [0,1], can_left, can_right
        ego_obs = np.zeros(EGO_DIM, dtype=np.float32)
        ego_obs[0] = ego.speed / SPEED_SCALE
        ego_obs[1] = ego_lane / max(num_lanes - 1, 1)
        ego_obs[2] = 1.0 if ego_lane > 0 else 0.0
        ego_obs[3] = 1.0 if ego_lane < num_lanes - 1 else 0.0

        # Vehicle rows: convert from normalized [-1,1] back to metres, then rescale
        veh_obs = np.zeros((N_VEHICLES, N_FEATURES), dtype=np.float32)
        others = kin[1:]  # skip ego row
        for i in range(min(len(others), N_VEHICLES)):
            row = others[i]
            presence = row[0]
            if presence < 0.5:
                continue
            # Convert from normalized to metres
            x_m = row[1] * rx
            y_m = row[2] * ry
            vx_m = row[3] * (2.0 * 40.0)  # vx range is [-80, 80]
            vy_m = row[4] * (2.0 * 40.0)  # vy range is [-80, 80]

            # Range gate: mask if |x_rel| > range_m
            if abs(x_m) > self.range_m:
                continue

            veh_obs[i, 0] = 1.0  # presence
            veh_obs[i, 1] = x_m / X_SCALE
            veh_obs[i, 2] = y_m / Y_SCALE
            veh_obs[i, 3] = vx_m / VX_SCALE
            veh_obs[i, 4] = vy_m / VY_SCALE

        flat = np.concatenate([ego_obs, veh_obs.reshape(-1)], axis=0)
        return flat.astype(np.float32)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        # This is called by gym.ObservationWrapper, but we need info.
        # We override reset/step instead.
        return observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        ds_obs = self._build_deepset_obs(info)
        info["tier"] = self.tier
        return ds_obs, info

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        ds_obs = self._build_deepset_obs(info)
        info["tier"] = self.tier
        return ds_obs, reward, terminated, truncated, info


def make_deepset_env_v2(
    tier: str = "full_adas",
    scenario: str = "highway",
    seed: int | None = None,
    range_m: float = 100.0,
    render_mode: str | None = None,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
    target_speeds: list[float] | None = None,
    lateral_debounce_steps: int = 0,
) -> gym.Env:
    """Create a DeepSet v2 environment for highway-fast-v0 or merge-v1.

    Reward, info, and action mask come from
    TacticalLaneObservationWrapper (identical code path).
    Policy observation is built from tier-filtered Kinematics.

    Args:
        tier: Sensor tier ('full_adas', 'front_only', 'short_range', 'no_velocity').
        scenario: Scenario name ('highway' or 'merge').
        seed: Random seed.
        range_m: Range gate in metres (vehicles beyond this are masked).
        render_mode: Rendering mode.
        collision_penalty: Penalty for collisions in reward shaping.
        overtake_bonus: Bonus per vehicle overtaken in reward shaping.
        target_speeds: Override DiscreteMetaAction target speeds. For highway, defaults
            to the validated floor-speed fix [10, 15, 20, 25, 30]. For merge, defaults
            to highway-env's built-in [20, 25, 30] unless specified.
        lateral_debounce_steps: Minimum lock steps preventing opposing lateral reversals (default: 0).

    Returns:
        Gymnasium env producing flat (74,) observations.
    """
    if scenario == "highway":
        env_id = "highway-fast-v0"
        speed_ladder = target_speeds if target_speeds is not None else [10.0, 15.0, 20.0, 25.0, 30.0]
        cfg: dict[str, Any] = {
            "observation": copy.deepcopy(DEFAULT_OBSERVATION_CONFIG),
            "policy_frequency": 5,
            "simulation_frequency": 5,
            "duration": 100,
            "lanes_count": 4,
            "vehicles_density": 1.4,
            "vehicles_count": 14,
            "collision_reward": 0.0,
            "high_speed_reward": 0.0,
            "right_lane_reward": 0.0,
            "on_road_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
            "reward_speed_range": [18, 30],
            "action": {
                "type": "DiscreteMetaAction",
                "target_speeds": speed_ladder,
            },
        }
    elif scenario == "merge":
        env_id = "merge-v1"
        cfg = {
            "observation": copy.deepcopy(DEFAULT_OBSERVATION_CONFIG),
            "policy_frequency": 1,
            "simulation_frequency": 15,
            "collision_reward": 0.0,
            "high_speed_reward": 0.0,
            "right_lane_reward": 0.0,
            "on_road_reward": 0.0,
            "lane_change_reward": 0.0,
            "normalize_reward": False,
        }
        if target_speeds is not None:
            cfg["action"] = {
                "type": "DiscreteMetaAction",
                "target_speeds": target_speeds,
            }
    else:
        raise ValueError(f"Unsupported scenario '{scenario}'")

    env = gym.make(env_id, render_mode=render_mode, config=cfg)

    # Insert tier-filtering stash wrapper below the tactical wrapper
    env = _KinematicsStashWrapper(env, tier=tier)

    # Tactical reward/info wrapper (reward shaping ON, continuous features OFF)
    env = TacticalLaneObservationWrapper(
        env,
        collision_penalty=collision_penalty,
        overtake_bonus=overtake_bonus,
        apply_reward_shaping=True,
        include_continuous_features=False,
        lateral_debounce_steps=lateral_debounce_steps,
    )

    # DeepSet observation wrapper on top
    env = DeepSetObsWrapper(env, tier=tier, range_m=range_m)

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env

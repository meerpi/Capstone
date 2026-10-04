"""Reward-only wrapper that applies our exact reward shaping to any environment
without changing its observation space (preserves Kinematics observation for R6).

Reward elements:
- Collision penalty: -50.0
- Survival bonus: +0.15
- Speed incentive: up to +0.5
- Proximity danger penalty (quadratic)
- Blockage penalty with impatience
- Overtake bonus: +1.0 (with dwell & debouncing)
- Lane change penalty: -0.05
- Jitter penalty: -0.1
"""

from typing import Any
import gymnasium as gym
import numpy as np


class OurRewardWrapper(gym.Wrapper):
    """Applies our project reward shaping while preserving original observation space."""

    def __init__(
        self,
        env: gym.Env,
        collision_penalty: float = -50.0,
        overtake_bonus: float = 1.0,
        danger_zone: float = 12.0,
        danger_penalty: float = 0.5,
        blockage_dist: float = 25.0,
        blockage_speed: float = 22.0,
        blockage_coef: float = 0.25,
        lane_change_penalty: float = 0.05,
        jitter_penalty: float = 0.1,
    ):
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.overtake_bonus = overtake_bonus
        self.danger_zone = danger_zone
        self.danger_penalty = danger_penalty
        self.BLOCK_DIST = blockage_dist
        self.BLOCK_SPEED = blockage_speed
        self.blockage_coef = blockage_coef
        self.lane_change_penalty = lane_change_penalty
        self.jitter_penalty = jitter_penalty

        self._was_crashed = False
        self._steps_blocked = 0
        self._vehicles_ahead: set[int] = set()
        self._credited_overtakes: set[int] = set()
        self._dwell_counters: dict[int, int] = {}
        self._prev_lane: int | None = None
        self._prev_action: int | None = None

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._was_crashed = False
        self._steps_blocked = 0
        self._vehicles_ahead = set()
        self._credited_overtakes = set()
        self._dwell_counters = {}
        ego = self.env.unwrapped.vehicle
        self._prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
        self._prev_action = None

        # Prime vehicles ahead
        for v in self.env.unwrapped.road.vehicles:
            if v is not ego:
                if v.position[0] > ego.position[0]:
                    v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                    if abs(v_lane - self._prev_lane) <= 1:
                        self._vehicles_ahead.add(id(v))
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle

        reward = 0.0
        if ego.crashed:
            reward = self.collision_penalty
        else:
            # Speed incentive: max 0.5
            speed_reward = 0.5 * np.clip((ego.speed - 20.0) / 10.0, 0.0, 1.0)
            reward += speed_reward

            # Survival bonus
            reward += 0.15

            # Proximity danger penalty
            min_dist = float("inf")
            for v in unwrapped.road.vehicles:
                if v is not ego:
                    dx = v.position[0] - ego.position[0]
                    dy = v.position[1] - ego.position[1]
                    d = np.sqrt(dx**2 + dy**2)
                    if d < min_dist:
                        min_dist = d
            if min_dist < self.danger_zone:
                proximity_frac = 1.0 - (min_dist / self.danger_zone)
                reward -= self.danger_penalty * (proximity_frac**2)

            # Blockage penalty
            front, _ = unwrapped.road.neighbour_vehicles(ego, ego.lane_index)
            if front is not None:
                dist = front.position[0] - ego.position[0]
                if 0.0 < dist < self.BLOCK_DIST and front.speed < self.BLOCK_SPEED:
                    self._steps_blocked += 1
                    impatience_factor = 1.0 + min(max(self._steps_blocked - 10, 0) / 10.0, 1.5)
                    frac = 1.0 - (dist / self.BLOCK_DIST)
                    reward -= self.blockage_coef * frac * impatience_factor
                else:
                    self._steps_blocked = 0
            else:
                self._steps_blocked = 0

            # Overtake bonus
            current_candidates = []
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
                        if lane_diff <= 1 and abs(dx) <= 30.0:
                            self._dwell_counters[v_id] = self._dwell_counters.get(v_id, 0) + 1
                            if self._dwell_counters[v_id] >= 5:
                                current_candidates.append((dy, v_id))
                        else:
                            self._dwell_counters[v_id] = 0
                    else:
                        self._dwell_counters[v_id] = 0

            if current_candidates:
                current_candidates.sort(key=lambda item: item[0])
                _, best_id = current_candidates[0]
                self._credited_overtakes.add(best_id)
                reward += self.overtake_bonus

            # Lane change penalty
            new_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
            if self._prev_lane is not None and new_lane != self._prev_lane:
                reward -= self.lane_change_penalty
            self._prev_lane = new_lane

            # Action jitter penalty
            if self._prev_action is not None:
                if (action == 3 and self._prev_action == 4) or (action == 4 and self._prev_action == 3):
                    reward -= self.jitter_penalty
                elif action in [3, 4] and self._prev_action in [1, 3, 4] and action != self._prev_action:
                    reward -= self.jitter_penalty * 0.4
            self._prev_action = action

        return obs, float(reward), terminated, truncated, info

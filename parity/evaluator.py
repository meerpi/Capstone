"""Evaluation harness for SOTA parity ladder.

Runs 100 evaluation episodes on fixed seeds 6000-6099.
Computes:
- k/n + Clopper-Pearson 95% CI for crashes
- mean speed (m/s and km/h)
- mean lane changes per episode
- mean episodic return
- action histogram and dominant action share
- ego-vs-traffic relative speed
- constant policy flag (>90% single action share)

HARD RULE: No info["overtake_count"] anywhere.
"""

from typing import Any, Callable
import numpy as np
import gymnasium as gym

from parity.metrics import summarize_episodes, ACTION_NAMES


class RandomPolicy:
    def __init__(self, action_space_n: int = 5):
        self.n = action_space_n

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        return int(np.random.randint(0, self.n))


class ConstantPolicy:
    def __init__(self, action_id: int):
        self.action_id = action_id

    def predict(self, obs: Any, deterministic: bool = True) -> int:
        return self.action_id


class IDMMobilPolicy:
    """Rule-based baseline using IDM gap/speed thresholds and MOBIL lateral checks."""

    def predict(self, obs: Any, env: gym.Env, deterministic: bool = True) -> int:
        unwrapped = env.unwrapped
        ego = unwrapped.vehicle
        front, _ = unwrapped.road.neighbour_vehicles(ego, ego.lane_index)

        # Check if front vehicle is blocking
        if front is not None and (front.position[0] - ego.position[0] < 28.0 or front.speed < ego.speed - 1.0):
            lane_id = ego.lane_index[2]
            n_lanes = len(unwrapped.road.network.graph[ego.lane_index[0]][ego.lane_index[1]])

            # Try passing right first (if available and safe)
            if lane_id + 1 < n_lanes:
                r_target = (ego.lane_index[0], ego.lane_index[1], lane_id + 1)
                f_r, r_r = unwrapped.road.neighbour_vehicles(ego, r_target)
                safe_right = True
                if f_r is not None and f_r.position[0] - ego.position[0] < 16.0:
                    safe_right = False
                if r_r is not None and ego.position[0] - r_r.position[0] < 14.0:
                    safe_right = False
                if safe_right:
                    return 2  # LANE_RIGHT

            # Try passing left
            if lane_id - 1 >= 0:
                l_target = (ego.lane_index[0], ego.lane_index[1], lane_id - 1)
                f_l, r_l = unwrapped.road.neighbour_vehicles(ego, l_target)
                safe_left = True
                if f_l is not None and f_l.position[0] - ego.position[0] < 16.0:
                    safe_left = False
                if r_l is not None and ego.position[0] - r_l.position[0] < 14.0:
                    safe_left = False
                if safe_left:
                    return 0  # LANE_LEFT

            # If can't change lane and too close to front, brake
            if front.position[0] - ego.position[0] < 20.0:
                return 4  # SLOWER

        # Clear road ahead: accelerate to target speed
        if front is None or (front.position[0] - ego.position[0] > 35.0):
            if ego.speed < 29.0:
                return 3  # FASTER

        return 1  # IDLE


def evaluate_policy(
    env_fn: Callable[[], gym.Env],
    policy: Any,
    seed_start: int = 6000,
    n_episodes: int = 100,
    is_env_aware: bool = False,
) -> dict[str, Any]:
    """Evaluate a policy on seeds [seed_start, seed_start + n_episodes)."""
    episodes = []
    env = env_fn()

    try:
        for ep in range(n_episodes):
            seed = seed_start + ep
            obs, info = env.reset(seed=seed)
            unwrapped = env.unwrapped
            ego = unwrapped.vehicle

            crashed = False
            episode_speeds = []
            episode_rel_speeds = []
            actions = []
            total_return = 0.0
            lane_changes = 0
            prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0

            done = False
            truncated = False
            while not (done or truncated):
                if is_env_aware:
                    action = policy.predict(obs, env=env, deterministic=True)
                elif hasattr(policy, "predict"):
                    res = policy.predict(obs, deterministic=True)
                    action = int(res[0]) if isinstance(res, tuple) else int(res)
                elif hasattr(policy, "act"):
                    action = int(policy.act(obs))
                elif callable(policy):
                    action = int(policy(obs))
                else:
                    raise ValueError("Unknown policy interface")

                actions.append(action)
                obs, reward, done, truncated, info = env.step(action)
                total_return += float(reward)

                cur_speed = float(ego.speed)
                episode_speeds.append(cur_speed)

                # Traffic relative speed
                traffic_speeds = [float(v.speed) for v in unwrapped.road.vehicles if v is not ego]
                if traffic_speeds:
                    episode_rel_speeds.append(cur_speed - float(np.mean(traffic_speeds)))

                # Lane change
                cur_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
                if cur_lane != prev_lane:
                    lane_changes += 1
                    prev_lane = cur_lane

                if ego.crashed:
                    crashed = True

            episodes.append({
                "seed": seed,
                "crashed": crashed,
                "return": total_return,
                "mean_speed": float(np.mean(episode_speeds)) if episode_speeds else 0.0,
                "mean_rel_speed": float(np.mean(episode_rel_speeds)) if episode_rel_speeds else 0.0,
                "lane_changes": lane_changes,
                "actions": actions,
            })
    finally:
        env.close()

    summary = summarize_episodes(episodes)
    return summary

"""Unit tests validating the overtake-credit flicker exploit fix in discrete wrappers.

Tests:
1. Single genuine pass credits exactly once (after >= 5 dwell steps, no repeat credits).
   Tested on both TacticalLaneObservationWrapper and TacticalOvertakingWrapper.
2. Rapid oscillation near dx=0 credits at most once per vehicle per episode.
3. Preservation of |lane_diff| <= 1 and |dx| <= 30m adjacent-corridor constraints.
"""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import highway_env
import env_config
from env_config import TacticalLaneObservationWrapper, TacticalOvertakingWrapper


def _get_wrapper(env: gym.Env) -> TacticalLaneObservationWrapper:
    curr = env
    while curr is not None:
        if isinstance(curr, TacticalLaneObservationWrapper):
            return curr
        curr = getattr(curr, "env", None)
    raise RuntimeError("TacticalLaneObservationWrapper not found in environment stack")


class TestDiscreteOvertakeDebounce(unittest.TestCase):
    """Test suite validating anti-flicker and dwell debouncing in discrete overtake wrappers."""

    def test_single_genuine_pass_credits_once_tactical_obs(self) -> None:
        """Test 1A: Single genuine pass credits exactly once in TacticalLaneObservationWrapper."""
        env = env_config.make_optimal_env(vehicles_count=1, seed=42)
        env.reset(seed=42)
        wrapper = _get_wrapper(env)
        ego = env.unwrapped.vehicle
        other = [v for v in env.unwrapped.road.vehicles if v is not ego][0]

        # Align other to adjacent lane of ego, placed ahead
        ego_lane = ego.lane_index[2]
        adj_lane = ego_lane + 1 if ego_lane < 3 else ego_lane - 1
        other.lane_index = ("0", "1", adj_lane)
        other.target_lane_index = ("0", "1", adj_lane)
        other.speed = ego.speed
        other.position = np.array([ego.position[0] + 10.0, adj_lane * 4.0])

        # Step 0: other is ahead -> added to _vehicles_ahead
        _, _, _, _, info = env.step(1)  # IDLE
        self.assertEqual(info["overtake_count"], 0)
        self.assertIn(id(other), wrapper._vehicles_ahead)
        self.assertNotIn(id(other), wrapper._credited_overtakes)

        # Steps 1 to 4: Dwell steps 1 through 4 behind ego (dwell < 5, should NOT credit)
        for dwell_step in range(1, 5):
            other.position = np.array([ego.position[0] - 5.0, adj_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", adj_lane)
            other.target_lane_index = ("0", "1", adj_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(
                info["overtake_count"],
                0,
                f"Premature overtake credit fired at dwell step {dwell_step}",
            )
            self.assertEqual(wrapper._dwell_counters.get(id(other), 0), dwell_step)
            self.assertNotIn(id(other), wrapper._credited_overtakes)

        # Step 5: 5th consecutive step behind -> credits exactly once
        other.position = np.array([ego.position[0] - 5.0, adj_lane * 4.0])
        other.speed = ego.speed
        other.lane_index = ("0", "1", adj_lane)
        other.target_lane_index = ("0", "1", adj_lane)
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["overtake_count"], 1, "Overtake credit did not fire at dwell step 5")
        self.assertIn(id(other), wrapper._credited_overtakes)

        # Steps 6 to 15: Continued following behind -> NO further credits
        for step_idx in range(6, 16):
            other.position = np.array([ego.position[0] - 5.0, adj_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", adj_lane)
            other.target_lane_index = ("0", "1", adj_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(
                info["overtake_count"],
                0,
                f"Repeated credit fired at step {step_idx} for already-credited vehicle",
            )

        self.assertEqual(len(wrapper._credited_overtakes), 1)
        env.close()

    def test_single_genuine_pass_credits_once_tactical_overtaking(self) -> None:
        """Test 1B: Single genuine pass credits exactly once in TacticalOvertakingWrapper."""
        base_env = gym.make("highway-fast-v0", config={"vehicles_count": 1, "duration": 50, "lanes_count": 4})
        env = TacticalOvertakingWrapper(base_env, overtake_dwell_steps=5, max_overtake_lon_dist=30.0)
        env.reset(seed=42)
        ego = env.unwrapped.vehicle
        other = [v for v in env.unwrapped.road.vehicles if v is not ego][0]

        ego_lane = ego.lane_index[2]
        adj_lane = ego_lane + 1 if ego_lane < 3 else ego_lane - 1
        other.lane_index = ("0", "1", adj_lane)
        other.target_lane_index = ("0", "1", adj_lane)
        other.speed = ego.speed
        other.position = np.array([ego.position[0] + 10.0, adj_lane * 4.0])

        # Step 0: other is ahead
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["overtake_count"], 0)
        self.assertIn(id(other), env._vehicles_ahead)

        # Steps 1 to 4: Dwell 1 to 4 steps behind
        for dwell_step in range(1, 5):
            other.position = np.array([ego.position[0] - 5.0, adj_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", adj_lane)
            other.target_lane_index = ("0", "1", adj_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["overtake_count"], 0)
            self.assertEqual(env._dwell_counters.get(id(other), 0), dwell_step)

        # Step 5: dwell step 5 -> credit
        other.position = np.array([ego.position[0] - 5.0, adj_lane * 4.0])
        other.speed = ego.speed
        other.lane_index = ("0", "1", adj_lane)
        other.target_lane_index = ("0", "1", adj_lane)
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["overtake_count"], 1)
        self.assertIn(id(other), env._credited_overtakes)

        # Further steps -> no repeat
        for _ in range(5):
            other.position = np.array([ego.position[0] - 5.0, adj_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", adj_lane)
            other.target_lane_index = ("0", "1", adj_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["overtake_count"], 0)

        self.assertEqual(len(env._credited_overtakes), 1)
        env.close()

    def test_rapid_oscillation_near_dx_zero_credits_at_most_once(self) -> None:
        """Test 2: Two vehicles oscillating near dx=0 for 300 steps credit at most once per vehicle."""
        env = env_config.make_optimal_env(vehicles_count=2, seed=123)
        env.reset(seed=123)
        wrapper = _get_wrapper(env)
        ego = env.unwrapped.vehicle
        others = [v for v in env.unwrapped.road.vehicles if v is not ego]
        self.assertEqual(len(others), 2)

        ego.speed = 25.0
        # Assign ego to lane 1, others to lane 0 and lane 2
        ego.lane_index = ("0", "1", 1)
        ego.target_lane_index = ("0", "1", 1)
        ego.position[1] = 4.0

        others[0].lane_index = ("0", "1", 0)
        others[0].target_lane_index = ("0", "1", 0)
        others[1].lane_index = ("0", "1", 2)
        others[1].target_lane_index = ("0", "1", 2)
        others[0].speed = 25.0
        others[1].speed = 25.0

        rng = np.random.RandomState(42)
        total_credits = 0

        # Run 300 steps with dx oscillating across 0
        for step in range(300):
            # Oscillate dx within [-1.5, +1.5]
            dx0 = 1.5 * np.sin(step * 0.4) + rng.normal(0, 0.2)
            dx1 = 1.5 * np.cos(step * 0.4) + rng.normal(0, 0.2)

            others[0].position = np.array([ego.position[0] + dx0, 0.0])
            others[1].position = np.array([ego.position[0] + dx1, 8.0])
            others[0].speed = ego.speed
            others[1].speed = ego.speed
            others[0].lane_index = ("0", "1", 0)
            others[0].target_lane_index = ("0", "1", 0)
            others[1].lane_index = ("0", "1", 2)
            others[1].target_lane_index = ("0", "1", 2)

            _, _, term, trunc, info = env.step(1)  # IDLE
            c = info.get("overtake_count", 0)
            if c > 0:
                total_credits += c

        # Under the fix, no vehicle may ever be credited more than once per episode!
        self.assertLessEqual(
            len(wrapper._credited_overtakes),
            2,
            f"More than 2 unique vehicles credited: {len(wrapper._credited_overtakes)}",
        )
        self.assertLessEqual(
            total_credits,
            2,
            f"Exploit detected! Total credits fired: {total_credits} for 2 vehicles across 300 steps",
        )
        env.close()

    def test_adjacent_corridor_preservation(self) -> None:
        """Test 3: Verify |lane_diff| <= 1 and |dx| <= 30m corridor constraints are preserved."""
        env = env_config.make_optimal_env(vehicles_count=1, seed=99)
        env.reset(seed=99)
        wrapper = _get_wrapper(env)
        ego = env.unwrapped.vehicle
        other = [v for v in env.unwrapped.road.vehicles if v is not ego][0]

        ego_lane = ego.lane_index[2]

        # --- Scenario A: Passing 2 lanes away (|lane_diff| = 2) ---
        # Choose two_away_lane so |two_away_lane - ego_lane| == 2
        two_away_lane = ego_lane + 2 if ego_lane <= 1 else ego_lane - 2
        other.lane_index = ("0", "1", two_away_lane)
        other.target_lane_index = ("0", "1", two_away_lane)
        other.speed = ego.speed
        other.position = np.array([ego.position[0] + 10.0, two_away_lane * 4.0])

        # Step once to observe
        env.step(1)
        # Now place behind for 10 steps
        for _ in range(10):
            other.position = np.array([ego.position[0] - 5.0, two_away_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", two_away_lane)
            other.target_lane_index = ("0", "1", two_away_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["overtake_count"], 0, "Vehicles 2 lanes away must NEVER receive credit")

        self.assertEqual(len(wrapper._credited_overtakes), 0)

        # --- Scenario B: Passing outside 30m longitudinal window (dx = -35m) ---
        adj_lane = ego_lane + 1 if ego_lane < 3 else ego_lane - 1
        other.lane_index = ("0", "1", adj_lane)
        other.target_lane_index = ("0", "1", adj_lane)
        other.speed = ego.speed
        other.position = np.array([ego.position[0] + 10.0, adj_lane * 4.0])
        env.step(1)
        self.assertIn(id(other), wrapper._vehicles_ahead)

        for _ in range(10):
            # 35m behind ego (> max_overtake_lon_dist=30.0)
            other.position = np.array([ego.position[0] - 35.0, adj_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", adj_lane)
            other.target_lane_index = ("0", "1", adj_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["overtake_count"], 0, "Vehicles > 30m behind must NOT increment dwell or credit")

        self.assertEqual(len(wrapper._credited_overtakes), 0)

        # --- Scenario C: Passing within adjacent corridor (|lane_diff| <= 1, |dx| <= 30m) ---
        for dwell_step in range(1, 5):
            other.position = np.array([ego.position[0] - 15.0, adj_lane * 4.0])
            other.speed = ego.speed
            other.lane_index = ("0", "1", adj_lane)
            other.target_lane_index = ("0", "1", adj_lane)
            _, _, _, _, info = env.step(1)
            self.assertEqual(info["overtake_count"], 0)

        # 5th step within corridor
        other.position = np.array([ego.position[0] - 15.0, adj_lane * 4.0])
        other.speed = ego.speed
        other.lane_index = ("0", "1", adj_lane)
        other.target_lane_index = ("0", "1", adj_lane)
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["overtake_count"], 1, "Vehicle in adjacent corridor must receive credit on 5th step")
        self.assertIn(id(other), wrapper._credited_overtakes)

        env.close()


if __name__ == "__main__":
    unittest.main()

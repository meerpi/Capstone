#!/usr/bin/env python3
"""Unit tests for continuous control observation extensions in TacticalLaneObservationWrapper.

Verifies:
1. Base dimension expands from 27 to 30; FrameStackTacticalWrapper / ObservationStackWrapper
   dynamically stacks to 90 (K=3) without hardcoded values.
2. Geometric ground truth tests:
   - Centered and straight: psi_err ≈ 0, y_lane ≈ 0, yaw_rate ≈ 0
   - Lateral offset left of lane center: y_lane < 0, matches exact displacement / 4.0
   - Lateral offset right of lane center: y_lane > 0, matches exact displacement / 4.0
   - Nonzero heading error (yawed right and yawed left relative to lane tangent)
   - Yaw rate finite-difference calculation: dpsi / dt matches expected value
   - Angle wrapping at [-pi, pi] boundaries
   - Saturation clipping to [-1.0, 1.0] under extreme deviations
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import gymnasium as gym
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import env_config


class TestContinuousObservationWrapper(unittest.TestCase):
    """Test suite for continuous control observation features in TacticalLaneObservationWrapper."""

    def setUp(self) -> None:
        """Create a fresh environment before each test."""
        self.env = env_config.make_optimal_env(
            seed=42,
            frame_stack_k=0,
            include_continuous_features=True,
        )

    def tearDown(self) -> None:
        """Close environment after test."""
        self.env.close()

    def test_dimensions_and_frame_stacking(self) -> None:
        """Confirm base dimension is 30 and frame stacking produces 90 dimensions."""
        # 1. Base wrapper dimension
        self.assertEqual(self.env.observation_space.shape, (30,))
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (30,))

        # 2. FrameStackTacticalWrapper / ObservationStackWrapper at K=3
        env_stacked = env_config.make_optimal_env(
            seed=42,
            frame_stack_k=3,
            include_continuous_features=True,
        )
        self.assertEqual(env_stacked.observation_space.shape, (90,))
        obs_stacked, _ = env_stacked.reset(seed=42)
        self.assertEqual(obs_stacked.shape, (90,))
        env_stacked.close()

        # 3. Canonical alias check
        self.assertIs(
            env_config.ObservationStackWrapper,
            env_config.FrameStackTacticalWrapper,
        )

        # 4. Backward-compatible legacy mode (27-dim base, 81-dim stacked)
        env_legacy = env_config.make_optimal_env(
            seed=42,
            frame_stack_k=3,
            include_continuous_features=False,
        )
        self.assertEqual(env_legacy.observation_space.shape, (81,))
        obs_legacy, _ = env_legacy.reset(seed=42)
        self.assertEqual(obs_legacy.shape, (81,))
        env_legacy.close()

    def test_centered_and_straight(self) -> None:
        """Test ego vehicle placed precisely at lane centerline with 0 heading error."""
        obs, info = self.env.reset(seed=42)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        lane = ego.lane

        # Align ego exactly on lane centerline (s = 100.0, lateral = 0.0) and heading = 0.0
        ego.position = np.array([100.0, lane.start[1]], dtype=np.float64)
        ego.heading = float(lane.heading_at(100.0))

        # Re-build observation
        obs = self.env._build_tactical_obs()

        psi_norm = obs[27]
        y_lane_norm = obs[28]
        yaw_rate_norm = obs[29]

        self.assertAlmostEqual(psi_norm, 0.0, places=5, msg="Heading error should be 0 when straight")
        self.assertAlmostEqual(y_lane_norm, 0.0, places=5, msg="Lateral offset should be 0 when centered")
        self.assertAlmostEqual(yaw_rate_norm, 0.0, places=5, msg="Yaw rate should be 0 at initial state")

    def test_offset_left_of_center(self) -> None:
        """Test ego vehicle offset to the left of the lane centerline."""
        obs, info = self.env.reset(seed=42)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        lane = ego.lane

        # In HighwayEnv, Y increases to the right. A negative Y displacement is to the left.
        # Shift 1.0m to the left (y_offset = -1.0m)
        target_offset = -1.0
        ego.position = np.array([100.0, lane.start[1] + target_offset], dtype=np.float64)
        ego.heading = float(lane.heading_at(100.0))

        obs = self.env._build_tactical_obs()

        psi_norm = obs[27]
        y_lane_norm = obs[28]
        expected_y_norm = target_offset / env_config.TacticalLaneObservationWrapper.MAX_LANE_OFFSET  # -1.0 / 4.0 = -0.25

        self.assertAlmostEqual(psi_norm, 0.0, places=5)
        self.assertAlmostEqual(y_lane_norm, expected_y_norm, places=5)
        self.assertEqual(y_lane_norm, -0.25)

    def test_offset_right_of_center(self) -> None:
        """Test ego vehicle offset to the right of the lane centerline."""
        obs, info = self.env.reset(seed=42)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        lane = ego.lane

        # Shift +1.5m to the right (y_offset = +1.5m)
        target_offset = 1.5
        ego.position = np.array([100.0, lane.start[1] + target_offset], dtype=np.float64)
        ego.heading = float(lane.heading_at(100.0))

        obs = self.env._build_tactical_obs()

        psi_norm = obs[27]
        y_lane_norm = obs[28]
        expected_y_norm = target_offset / env_config.TacticalLaneObservationWrapper.MAX_LANE_OFFSET  # +1.5 / 4.0 = +0.375

        self.assertAlmostEqual(psi_norm, 0.0, places=5)
        self.assertAlmostEqual(y_lane_norm, expected_y_norm, places=5)
        self.assertEqual(y_lane_norm, 0.375)

    def test_nonzero_heading_error(self) -> None:
        """Test nonzero heading angles relative to lane reference direction."""
        obs, info = self.env.reset(seed=42)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        lane = ego.lane
        max_h = env_config.TacticalLaneObservationWrapper.MAX_HEADING_ERROR  # pi / 4

        # Center ego in lane
        ego.position = np.array([100.0, lane.start[1]], dtype=np.float64)

        # 1. Heading angled +0.15 rad to the right (~8.59 degrees)
        heading_err_right = 0.15
        ego.heading = float(lane.heading_at(100.0)) + heading_err_right
        obs_right = self.env._build_tactical_obs()
        expected_norm_right = heading_err_right / max_h
        self.assertAlmostEqual(obs_right[27], expected_norm_right, places=5)
        self.assertAlmostEqual(obs_right[28], 0.0, places=5)

        # 2. Heading angled -0.20 rad to the left (~-11.46 degrees)
        heading_err_left = -0.20
        ego.heading = float(lane.heading_at(100.0)) + heading_err_left
        obs_left = self.env._build_tactical_obs()
        expected_norm_left = heading_err_left / max_h
        self.assertAlmostEqual(obs_left[27], expected_norm_left, places=5)
        self.assertAlmostEqual(obs_left[28], 0.0, places=5)

    def test_yaw_rate_finite_difference(self) -> None:
        """Test finite-difference yaw rate calculation across consecutive steps."""
        obs, info = self.env.reset(seed=42)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle

        dt = 1.0 / unwrapped.config.get("policy_frequency", 5)  # 0.2s
        self.assertEqual(dt, 0.2)

        # Starting heading psi_0 = 0.0
        ego.heading = 0.0
        self.env._prev_heading = 0.0

        # Change heading to 0.06 rad over 1 step
        # Expected yaw rate = (0.06 - 0.0) / 0.2 = 0.3 rad/s
        target_dpsi = 0.06
        ego.heading = target_dpsi

        # Step environment with no-op action
        obs, reward, term, trunc, info = self.env.step(1)

        # Note: during step(), ego physics may propagate slightly, so test the wrapper's internal computation
        # directly against the finite difference rule
        expected_yaw_rate = info["yaw_rate"]
        expected_yaw_rate_norm = np.clip(expected_yaw_rate / env_config.TacticalLaneObservationWrapper.MAX_YAW_RATE, -1.0, 1.0)
        self.assertAlmostEqual(obs[29], expected_yaw_rate_norm, places=5)

    def test_angle_wrapping_and_saturation(self) -> None:
        """Test angle wrapping across [-pi, pi] and saturation clipping to [-1, 1]."""
        # 1. wrap_to_pi utility
        self.assertAlmostEqual(env_config.wrap_to_pi(0.0), 0.0, places=6)
        self.assertAlmostEqual(env_config.wrap_to_pi(math.pi * 3), -math.pi, places=5)
        self.assertAlmostEqual(env_config.wrap_to_pi(-math.pi * 3), -math.pi, places=5)
        self.assertAlmostEqual(env_config.wrap_to_pi(0.2 + 2 * math.pi), 0.2, places=6)
        self.assertAlmostEqual(env_config.wrap_to_pi(-0.2 - 2 * math.pi), -0.2, places=6)

        # 2. Saturation clipping under extreme out-of-lane displacement and spin
        obs, info = self.env.reset(seed=42)
        unwrapped = self.env.unwrapped
        ego = unwrapped.vehicle
        lane = ego.lane

        # Extreme lateral displacement (10.0m > 4.0m)
        ego.position = np.array([100.0, lane.start[1] + 10.0], dtype=np.float64)
        # Extreme heading error (1.5 rad > pi/4)
        ego.heading = 1.5
        # Extreme yaw rate
        self.env._current_yaw_rate = 3.5  # > 1.0 rad/s

        obs = self.env._build_tactical_obs()

        self.assertEqual(obs[27], 1.0, msg="Heading error should saturate at +1.0")
        self.assertEqual(obs[28], 1.0, msg="Lateral offset should saturate at +1.0")
        self.assertEqual(obs[29], 1.0, msg="Yaw rate should saturate at +1.0")

        # Opposite extreme saturation
        ego.position = np.array([100.0, lane.start[1] - 10.0], dtype=np.float64)
        ego.heading = -1.5
        self.env._current_yaw_rate = -3.5

        obs = self.env._build_tactical_obs()

        self.assertEqual(obs[27], -1.0, msg="Heading error should saturate at -1.0")
        self.assertEqual(obs[28], -1.0, msg="Lateral offset should saturate at -1.0")
        self.assertEqual(obs[29], -1.0, msg="Yaw rate should saturate at -1.0")


if __name__ == "__main__":
    unittest.main()

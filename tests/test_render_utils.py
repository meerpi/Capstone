"""Unit tests for render_utils and temporal aliasing detection."""

from __future__ import annotations

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import gymnasium as gym
import highway_env  # noqa: F401
from highway_env.road.graphics import LaneGraphics

import env_config
from render_utils import (
    ANTI_ALIAS_STRIPE_LENGTH,
    ANTI_ALIAS_STRIPE_SPACING,
    FAST_FRAME_DURATION_MS,
    REALTIME_FRAME_DURATION_MS,
    apply_anti_alias_stripes,
    compute_in_lane_leader_speed,
    compute_mean_npc_speed,
    frame_duration_ms,
    get_rendered_frame,
)
from scripts.check_render_aliasing import (
    analyze_gif_aliasing,
    estimate_dash_period,
    find_dashed_lane_rows,
)


class TestRenderUtils(unittest.TestCase):
    def test_apply_anti_alias_stripes(self):
        apply_anti_alias_stripes()
        self.assertEqual(LaneGraphics.STRIPE_SPACING, ANTI_ALIAS_STRIPE_SPACING)
        self.assertEqual(LaneGraphics.STRIPE_LENGTH, ANTI_ALIAS_STRIPE_LENGTH)
        self.assertEqual(ANTI_ALIAS_STRIPE_SPACING, 30.0)
        self.assertEqual(ANTI_ALIAS_STRIPE_LENGTH, 15.0)

    def test_frame_durations(self):
        self.assertEqual(frame_duration_ms(realtime=True), REALTIME_FRAME_DURATION_MS)
        self.assertEqual(frame_duration_ms(realtime=False), FAST_FRAME_DURATION_MS)
        self.assertEqual(REALTIME_FRAME_DURATION_MS, 200)
        self.assertEqual(FAST_FRAME_DURATION_MS, 67)

    def test_get_rendered_frame_non_empty(self):
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.0,
            vehicles_count=5,
            seed=42,
            render_mode="rgb_array",
        )
        env.unwrapped.config["offscreen_rendering"] = True
        env.reset(seed=42)
        frame = get_rendered_frame(env)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertGreater(frame.max(), 0, "Rendered frame must not be blank")
        env.close()

    def test_compute_npc_speeds(self):
        env = env_config.make_optimal_env(
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=10,
            seed=42,
        )
        env.reset(seed=42)
        mean_speed = compute_mean_npc_speed(env)
        self.assertGreater(mean_speed, 15.0, "NPC mean speed should be positive highway speed")
        self.assertLess(mean_speed, 35.0, "NPC mean speed should not exceed road limit")
        env.close()

    def test_dash_period_estimation(self):
        # Create synthetic dashed line pattern of period 24
        frame = np.zeros((150, 600, 3), dtype=np.uint8)
        # Row 42 has dashes every 24 px
        for x in range(0, 600):
            if (x % 24) < 12:
                frame[42, x] = [255, 255, 255]
        period = estimate_dash_period(frame, row_idx=42)
        self.assertAlmostEqual(period, 24.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()

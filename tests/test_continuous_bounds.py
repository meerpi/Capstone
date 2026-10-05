"""Tests for continuous-control environment bounds and safety termination.

Verifies:
1. Negative speed prevention: The ego vehicle cannot reach negative speed (cannot drive backwards)
   even under sustained maximum deceleration.
2. Off-road termination: Leaving the road surface terminates the episode immediately when
   offroad_terminal=True, and does not alter default discrete configs.
"""

import math
import unittest
import numpy as np
import gymnasium as gym

import env_config


class TestContinuousBounds(unittest.TestCase):
    def test_ego_cannot_reach_negative_speed(self) -> None:
        """Under maximum deceleration, ego speed cannot drop below 0.0 m/s."""
        env = env_config.make_continuous_env(seed=42)
        obs, info = env.reset(seed=42)
        
        ego = env.unwrapped.vehicle
        initial_speed = ego.speed
        self.assertGreater(initial_speed, 0.0, f"Expected positive initial speed, got {initial_speed}")

        # Apply maximum deceleration: action = [-1.0, 0.0] (braking at -5 m/s^2, zero steering)
        max_brake_action = np.array([-1.0, 0.0], dtype=np.float32)
        
        speeds = []
        for step in range(50):
            obs, reward, terminated, truncated, info = env.step(max_brake_action)
            speed = float(ego.speed)
            speeds.append(speed)
            self.assertGreaterEqual(speed, 0.0, f"Step {step}: ego speed became negative: {speed} m/s")
            if terminated or truncated:
                break

        env.close()

        # Confirm speed actually reached 0.0 and stayed floored at 0.0
        final_speed = speeds[-1]
        self.assertAlmostEqual(final_speed, 0.0, delta=1e-5, msg=f"Expected vehicle to stop at 0.0 m/s, got {final_speed}")
        self.assertTrue(all(s >= 0.0 for s in speeds), "Encountered negative speed during braking")

    def test_leaving_road_terminates_episode(self) -> None:
        """Leaving the road boundary terminates the episode immediately with offroad_terminal=True."""
        env = env_config.make_continuous_env(seed=42, offroad_terminal=True)
        obs, info = env.reset(seed=42)
        ego = env.unwrapped.vehicle

        # Steer hard left/right: [0.0, 1.0] (zero throttle, full steering angle)
        steer_action = np.array([0.0, 1.0], dtype=np.float32)

        terminated_due_to_offroad = False
        for step in range(100):
            obs, reward, terminated, truncated, info = env.step(steer_action)
            on_road = ego.on_road
            if not on_road:
                self.assertTrue(terminated, f"Step {step}: vehicle left the road (on_road=False) but terminated was False!")
                terminated_due_to_offroad = True
                break
            if terminated or truncated:
                break

        env.close()
        self.assertTrue(terminated_due_to_offroad, "Test did not observe vehicle leaving road within 100 steps")

    def test_legacy_configs_do_not_set_offroad_terminal(self) -> None:
        """Verify discrete configs do not set offroad_terminal."""
        self.assertNotIn("offroad_terminal", env_config.DEFAULT_ENV_CONFIG)
        self.assertNotIn("offroad_terminal", env_config.SCENARIO_CONFIGS["highway"])


if __name__ == "__main__":
    unittest.main()

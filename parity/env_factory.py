"""Environment factory for the SOTA parity ladder.

Defines the exact cumulative rungs from official highway-fast-v0 to our pipeline:
R0: official highway-fast-v0 default config, official reward & Kinematics obs
R1: same as R0 (used for implementation parity check between SB3 and our PPO)
R2: density 1.0 -> 1.4
R3: lanes 3 -> 4, vehicles_count 20 -> 14
R4: duration 30 s -> 100 s (still 1 Hz)
R5: policy_frequency 1 -> 5 Hz (duration 100 s = 500 steps, sim_freq = 5)
R6: official reward -> our shaped reward (-50/+1.0, survival, danger zone, blockage, jitter)
R7_gt: Kinematics obs -> our 27-dim tactical lane-slot observation (ground-truth)
R7_deepset: DeepSet v2 observation
"""

import copy
import sys
from typing import Any
import gymnasium as gym
import highway_env  # noqa: F401

# Import wrappers from existing code without modifying them
sys.path.insert(0, ".")
from env_config import TacticalLaneObservationWrapper


def get_base_highway_fast_config() -> dict[str, Any]:
    """Return a pristine copy of highway-fast-v0 default config."""
    return {
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 3,
        "vehicles_count": 20,
        "controlled_vehicles": 1,
        "initial_lane_id": None,
        "duration": 30,  # [s]
        "ego_spacing": 1.5,
        "vehicles_density": 1.0,
        "collision_reward": -1.0,
        "right_lane_reward": 0.1,
        "high_speed_reward": 0.4,
        "lane_change_reward": 0.0,
        "reward_speed_range": [20, 30],
        "normalize_reward": True,
        "offroad_terminal": False,
        "simulation_frequency": 5,
        "policy_frequency": 1,
    }


def make_ladder_env(rung: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create the exact environment for a given ladder rung."""
    cfg = get_base_highway_fast_config()

    rung_clean = rung.upper().replace("-", "_")

    if rung_clean in ("R0", "R1"):
        # Stock highway-fast-v0
        pass

    elif rung_clean == "R2":
        # Density 1.0 -> 1.4
        cfg["vehicles_density"] = 1.4

    elif rung_clean == "R3":
        # Cumulative: density 1.4, lanes 3 -> 4, vehicles 20 -> 14
        cfg["vehicles_density"] = 1.4
        cfg["lanes_count"] = 4
        cfg["vehicles_count"] = 14

    elif rung_clean == "R4":
        # Cumulative: R3 + duration 30 -> 100 s (still 1 Hz policy)
        cfg["vehicles_density"] = 1.4
        cfg["lanes_count"] = 4
        cfg["vehicles_count"] = 14
        cfg["duration"] = 100

    elif rung_clean in ("R5", "R5_956", "R5_995"):
        # Cumulative: R4 + policy_frequency 1 -> 5 Hz
        cfg["vehicles_density"] = 1.4
        cfg["lanes_count"] = 4
        cfg["vehicles_count"] = 14
        cfg["duration"] = 100
        cfg["policy_frequency"] = 5
        cfg["simulation_frequency"] = 5

    elif rung_clean == "R6":
        # Cumulative: R5 + our shaped reward
        cfg["vehicles_density"] = 1.4
        cfg["lanes_count"] = 4
        cfg["vehicles_count"] = 14
        cfg["duration"] = 100
        cfg["policy_frequency"] = 5
        cfg["simulation_frequency"] = 5
        cfg["normalize_reward"] = False
        cfg["collision_reward"] = 0.0
        cfg["high_speed_reward"] = 0.0
        cfg["right_lane_reward"] = 0.0
        cfg["reward_speed_range"] = [18, 30]

    elif rung_clean in ("R7", "R7_GT"):
        # Cumulative: R6 + 27-dim ground truth slot observation
        cfg["vehicles_density"] = 1.4
        cfg["lanes_count"] = 4
        cfg["vehicles_count"] = 14
        cfg["duration"] = 100
        cfg["policy_frequency"] = 5
        cfg["simulation_frequency"] = 5
        cfg["normalize_reward"] = False
        cfg["collision_reward"] = 0.0
        cfg["high_speed_reward"] = 0.0
        cfg["right_lane_reward"] = 0.0
        cfg["reward_speed_range"] = [18, 30]

    elif rung_clean == "R7_DEEPSET":
        # Cumulative: R6 + DeepSet v2 observation
        cfg["vehicles_density"] = 1.4
        cfg["lanes_count"] = 4
        cfg["vehicles_count"] = 14
        cfg["duration"] = 100
        cfg["policy_frequency"] = 5
        cfg["simulation_frequency"] = 5
        cfg["normalize_reward"] = False
        cfg["collision_reward"] = 0.0
        cfg["high_speed_reward"] = 0.0
        cfg["right_lane_reward"] = 0.0
        cfg["reward_speed_range"] = [18, 30]

    else:
        raise ValueError(f"Unknown rung: {rung}")

    env = gym.make("highway-fast-v0", config=cfg, render_mode=render_mode)

    if rung_clean == "R6":
        # Apply our reward shaping via TacticalLaneObservationWrapper, but keep Kinematics obs
        # By wrapping with TacticalLaneObservationWrapper we get the reward shaping:
        # Wait, TacticalLaneObservationWrapper changes obs to 27-dim slot grid.
        # For R6, we want official Kinematics obs + our reward!
        # Let's create a reward-only wrapper that applies our exact reward shaping while preserving Kinematics obs!
        from parity.reward_wrapper import OurRewardWrapper
        env = OurRewardWrapper(env)

    elif rung_clean in ("R7", "R7_GT"):
        env = TacticalLaneObservationWrapper(
            env,
            collision_penalty=-50.0,
            overtake_bonus=1.0,
            include_continuous_features=False,
            constrained_mode=False,
        )

    elif rung_clean == "R7_DEEPSET":
        from deepset_v2.env import make_deepset_env
        # Use deepset env builder
        env.close()
        env = make_deepset_env(
            scenario="highway",
            tier="full_adas",
            seed=seed,
            lanes_count=4,
            vehicles_density=1.4,
            vehicles_count=14,
            duration=100,
        )

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env

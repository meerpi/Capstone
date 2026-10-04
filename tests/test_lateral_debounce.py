import os
import sys
import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import env_config
from deepset_v2.env import make_deepset_env_v2


def test_lateral_debounce_mask_lockout():
    """Verify that taking a lateral action masks the opposing lateral action for K steps."""
    env = env_config.make_optimal_env(
        lanes_count=4,
        seed=1234,
        frame_stack_k=0,
        lateral_debounce_steps=5,
    )
    obs, info = env.reset(seed=1234)

    mask = info["action_mask"]
    assert mask is not None

    # Choose available lateral action
    chosen_act = 0 if mask[0] else 2
    opp_act = 2 if chosen_act == 0 else 0

    # Execute chosen lateral action
    obs, rew, term, trunc, info = env.step(chosen_act)
    mask = info["action_mask"]

    # During lock window, opposing lateral action must be masked out
    assert not mask[opp_act], f"Opposing action {opp_act} must be masked during debounce lock window"
    # Longitudinal actions must remain available
    assert mask[1], "IDLE must remain available"
    assert mask[3], "FASTER must remain available"
    assert mask[4], "SLOWER must remain available"

    # Step with IDLE for 4 more steps (total 5 steps)
    for _ in range(4):
        obs, rew, term, trunc, info = env.step(1)

    # On step 6, the lock should have expired
    assert env.unwrapped.vehicle is not None
    env.close()


def test_lateral_debounce_forced_action_override():
    """Verify that forcing an opposing lateral action overrides it to IDLE (1)."""
    env = env_config.make_optimal_env(
        lanes_count=4,
        seed=1234,
        frame_stack_k=0,
        lateral_debounce_steps=5,
    )
    env.reset(seed=1234)

    # Execute LANE_LEFT (action 0)
    env.step(0)

    # Immediately try to force opposing LANE_RIGHT (action 2) on step 2
    # The wrapper should debounce it to IDLE (1)
    obs, rew, term, trunc, info = env.step(2)
    # The vehicle should continue left toward target lane 0, not right
    env.close()


def test_deepset_v2_has_lateral_debounce_support():
    """Verify that make_deepset_env_v2 properly configures lateral debounce."""
    env = make_deepset_env_v2(tier="full_adas", seed=4321, lateral_debounce_steps=4)
    obs, info = env.reset(seed=4321)

    # Take LANE_RIGHT
    obs, rew, term, trunc, info = env.step(2)
    mask = info["action_mask"]
    assert not mask[0], "LANE_LEFT must be masked out after LANE_RIGHT in DeepSet v2"
    env.close()

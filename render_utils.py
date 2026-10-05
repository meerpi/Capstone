"""Rendering utilities for highway-env visualization.

Centralises anti-aliasing stripe constants and the headless black-frame
workaround so that all render scripts share a single source of truth.
"""

from __future__ import annotations

import numpy as np
from highway_env.road.graphics import LaneGraphics


# --- Anti-aliasing stripe constants ---
# At STRIPE_SPACING = 4.33 m (default), one GIF frame (0.2 s at 5 Hz) advances
# the dash pattern by v*0.2 m. At v ≈ 21.65 m/s the pattern advances exactly
# one period per frame, causing it to look frozen.  Below that speed the dashes
# drift *forward*, which the eye reads as the car going backward (temporal
# aliasing / wagon-wheel effect).
#
# Setting STRIPE_SPACING to 30 m and STRIPE_LENGTH to 15 m means per-frame
# ego travel at 8-40 m/s is 1.6-8.0 m — well under half of 30 m — so no
# aliasing occurs at any realistic speed.
ANTI_ALIAS_STRIPE_SPACING: float = 30.0
ANTI_ALIAS_STRIPE_LENGTH: float = 15.0

# Real-time playback: 5 Hz policy → 200 ms per frame
REALTIME_FRAME_DURATION_MS: int = 200
# Default fast playback: fps=15 → ~67 ms per frame (~3× real time)
FAST_FRAME_DURATION_MS: int = 67
FAST_FPS: int = 15


def apply_anti_alias_stripes() -> None:
    """Monkey-patch LaneGraphics stripe constants to eliminate temporal aliasing.

    Must be called **before** the first ``env.render()`` in any render script.
    """
    LaneGraphics.STRIPE_SPACING = ANTI_ALIAS_STRIPE_SPACING
    LaneGraphics.STRIPE_LENGTH = ANTI_ALIAS_STRIPE_LENGTH


def get_rendered_frame(env) -> np.ndarray:
    """Safely render a frame, working around the headless black-frame bug.

    In a headless container the first ``env.render()`` can return an all-zero
    (black) frame until ``viewer.enabled`` is set to True.  This helper detects
    and fixes that situation, then asserts the frame is not blank.

    Returns:
        RGB frame as uint8 ndarray of shape (H, W, 3).
    """
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None — check render_mode")

    unwrapped = env.unwrapped
    if hasattr(unwrapped, "viewer") and unwrapped.viewer is not None:
        if not unwrapped.viewer.enabled:
            unwrapped.viewer.enabled = True
            frame = env.render()

    assert frame.max() > 0, (
        "Rendered frame is all-black after viewer workaround. "
        "Ensure render_mode='rgb_array' and offscreen_rendering=True."
    )
    return frame


def compute_mean_npc_speed(env) -> float:
    """Return the mean speed of all NPC vehicles (m/s)."""
    unwrapped = env.unwrapped
    ego = unwrapped.vehicle
    npcs = [v for v in unwrapped.road.vehicles if v is not ego]
    if not npcs:
        return 0.0
    return float(np.mean([v.speed for v in npcs]))


def compute_in_lane_leader_speed(env) -> float | None:
    """Return the speed of the closest lead vehicle in the ego's lane (m/s), or None if no leader."""
    unwrapped = env.unwrapped
    ego = unwrapped.vehicle
    ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
    ego_x = ego.position[0]

    best_dx = float("inf")
    best_speed = None
    for v in unwrapped.road.vehicles:
        if v is ego:
            continue
        v_lane = v.lane_index[2] if hasattr(v, "lane_index") else -1
        if v_lane != ego_lane:
            continue
        dx = v.position[0] - ego_x
        if 0 < dx < best_dx:
            best_dx = dx
            best_speed = v.speed
    return best_speed


def frame_duration_ms(realtime: bool) -> int:
    """Return GIF frame duration in ms based on playback mode."""
    return REALTIME_FRAME_DURATION_MS if realtime else FAST_FRAME_DURATION_MS

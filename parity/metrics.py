"""Metrics calculation and evaluation utilities for highway-env SOTA parity ladder.

Calculates:
- k/n + Clopper-Pearson 95% CI for crash rate via scipy.stats.beta
- mean speed (m/s and km/h)
- mean lane changes per episode
- mean return
- action histogram and dominant action share
- ego-vs-traffic relative speed
- constant policy flag (dominant action share > 90%)

HARD RULE: No info["overtake_count"] anywhere.
"""

from typing import Any
import numpy as np
from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Compute exact Clopper-Pearson 95% confidence interval for a binomial proportion via scipy."""
    if n == 0:
        return 0.0, 0.0
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2, k + 1, n - k))
    return low, high


ACTION_NAMES = {
    0: "LANE_LEFT",
    1: "IDLE",
    2: "LANE_RIGHT",
    3: "FASTER",
    4: "SLOWER",
}


def summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a list of episode outcome dictionaries."""
    n = len(episodes)
    if n == 0:
        return {}

    k = sum(1 for ep in episodes if ep["crashed"])
    ci_low, ci_high = clopper_pearson(k, n)

    speeds = [ep["mean_speed"] for ep in episodes]
    returns = [ep["return"] for ep in episodes]
    lane_changes = [ep["lane_changes"] for ep in episodes]
    rel_speeds = [ep["mean_rel_speed"] for ep in episodes if ep["mean_rel_speed"] is not None]

    all_actions = []
    for ep in episodes:
        all_actions.extend(ep["actions"])

    total_actions = len(all_actions)
    action_counts = {a: all_actions.count(a) for a in range(5)}
    action_histogram = {
        ACTION_NAMES[a]: {
            "count": action_counts[a],
            "share": action_counts[a] / max(total_actions, 1),
        }
        for a in range(5)
    }

    dominant_action_id = max(action_counts, key=action_counts.get) if total_actions > 0 else 1
    dominant_action_name = ACTION_NAMES[dominant_action_id]
    dominant_action_frac = action_counts[dominant_action_id] / max(total_actions, 1)

    is_constant = bool(dominant_action_frac > 0.90)

    return {
        "n_episodes": n,
        "crashes": k,
        "crash_rate": k / n,
        "ci_95": (ci_low, ci_high),
        "ci_str": f"{k}/{n} ({k/n:.1%}) [{ci_low:.1%}, {ci_high:.1%}]",
        "mean_speed_ms": float(np.mean(speeds)),
        "std_speed_ms": float(np.std(speeds)),
        "mean_speed_kmh": float(np.mean(speeds) * 3.6),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_lane_changes": float(np.mean(lane_changes)),
        "std_lane_changes": float(np.std(lane_changes)),
        "mean_rel_speed_ms": float(np.mean(rel_speeds)) if rel_speeds else 0.0,
        "std_rel_speed_ms": float(np.std(rel_speeds)) if rel_speeds else 0.0,
        "action_histogram": action_histogram,
        "dominant_action": dominant_action_name,
        "dominant_action_frac": dominant_action_frac,
        "is_constant": is_constant,
    }

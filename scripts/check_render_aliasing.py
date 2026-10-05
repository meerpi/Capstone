#!/usr/bin/env python
"""Check temporal aliasing (wagon-wheel / reverse driving illusion) in rendered GIFs.

For a highway-env rendered GIF:
- The ego vehicle camera is ego-centred (moving forward, left-to-right in world).
- When the car moves forward, road dashes should appear to move backward
  (right-to-left on screen, negative x shift).
- At STRIPE_SPACING = 4.33 m (period ~23.8 px at 5.5 px/m), at speeds below
  21.65 m/s the dashes advance less than one period per frame, causing the
  apparent phase shift modulo the stripe period to be positive (rightward
  drift). The human eye reads this as the road moving forward, or the car
  reversing.
- When STRIPE_SPACING = 30 m (period 165 px), per-frame travel at 8-40 m/s
  is 8.8 - 44 px, well below half the period (82.5 px), completely eliminating
  aliasing (apparent drift is always negative, 0% reverse illusion).

This script measures the phase shift between consecutive frames using circular
cross-correlation, reports the fraction of reversing frames, and validates
against expected baselines.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence
import numpy as np
from PIL import Image


DEFAULT_BASELINES = {
    "ppo_top_1_seed_3094.gif": 0.0,
    "ppo_top_2_seed_3115.gif": 28.5,
    "ppo_top_3_seed_3010.gif": 37.7,
    "ppo_top_4_seed_3000.gif": 4.0,
    "ppo_top_5_seed_3019.gif": 68.5,
    "ppo_top_6_seed_3092.gif": 65.7,
}


def find_dashed_lane_rows(frame: np.ndarray) -> list[int]:
    """Locate dashed lane row indices in a 150x600 highway frame."""
    # Dashed lanes have periodic high-intensity dashes
    h, w = frame.shape[:2]
    # Check rows below banner (y > 28)
    candidates = []
    for y in range(30, h - 10):
        row = (frame[y, 50:w - 50, 0] > 200).astype(float)
        # Periodic autocorrelation check
        if row.sum() > 20 and row.sum() < (w - 100) * 0.9:
            row_norm = row - row.mean()
            corr = np.correlate(row_norm, row_norm, mode="full")
            center = len(corr) // 2
            # Check for strong peak in lag 15-35 (standard) or lag 140-180 (anti-aliased)
            lags_std = corr[center + 15 : center + 35]
            lags_wide = corr[center + 140 : center + 190]
            if len(lags_std) > 0 and lags_std.max() > 0.4 * corr[center]:
                candidates.append(y)
            elif len(lags_wide) > 0 and lags_wide.max() > 0.4 * corr[center]:
                candidates.append(y)
    # Default to known row 42 if detection doesn't find candidates
    return candidates if candidates else [42]


def estimate_dash_period(frame: np.ndarray, row_idx: int = 42) -> float:
    """Estimate the spatial period of dashes in pixels using autocorrelation."""
    w = frame.shape[1]
    row = (frame[row_idx, 50:w - 50, 0] > 200).astype(float)
    row_norm = row - row.mean()
    if np.std(row_norm) < 1e-3:
        return 23.815  # Default standard period
    corr = np.correlate(row_norm, row_norm, mode="full")
    center = len(corr) // 2
    # Search for peak in standard range [20, 28] or wide range [150, 180]
    p_std = np.argmax(corr[center + 18 : center + 30]) + 18 if center + 30 < len(corr) else 24
    std_val = corr[center + p_std] if center + p_std < len(corr) else 0

    p_wide = np.argmax(corr[center + 140 : center + 185]) + 140 if center + 185 < len(corr) else 165
    wide_val = corr[center + p_wide] if center + p_wide < len(corr) else 0

    if wide_val > std_val and wide_val > 0.3 * corr[center]:
        return float(p_wide)
    return float(p_std) if std_val > 0.3 * corr[center] else 23.815


def analyze_gif_aliasing(
    gif_path: str,
    dash_row: int | None = None,
    period: float | None = None,
    threshold_px: float = 0.5,
) -> dict[str, float | int]:
    """Measure the fraction of frames exhibiting forward dash drift (reverse illusion).

    Args:
        gif_path: Path to the GIF file.
        dash_row: Specific row to analyze, or None to auto-detect.
        period: Known dash period in px, or None to auto-estimate.
        threshold_px: Minimum positive phase shift (px) to count as reverse drift.

    Returns:
        Dictionary with:
          - reverse_fraction_pct: Percentage of frames with reverse illusion.
          - total_frames: Total analyzed frame transitions.
          - reverse_frames: Count of reverse frames.
          - estimated_period: Spatial period of dashes in pixels.
          - analyzed_row: Row index analyzed.
    """
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF file not found: {gif_path}")

    img = Image.open(gif_path)
    frames: list[np.ndarray] = []
    try:
        while True:
            frames.append(np.array(img.convert("RGB")))
            img.seek(len(frames))
    except EOFError:
        pass

    if len(frames) < 2:
        return {
            "reverse_fraction_pct": 0.0,
            "total_frames": 0,
            "reverse_frames": 0,
            "estimated_period": 0.0,
            "analyzed_row": 0,
        }

    mid_frame = frames[len(frames) // 2]
    if dash_row is not None:
        row = dash_row
    else:
        detected_rows = find_dashed_lane_rows(mid_frame)
        row = detected_rows[0] if detected_rows else 42

    # Determine period: standard is 4.33m * 5.5 px/m = 23.815 px;
    # anti-aliased is 30m * 5.5 px/m = 165.0 px
    if period is not None:
        T = float(period)
    else:
        # Detect whether this is standard (~24 px) or wide (~165 px) stripes
        detected = estimate_dash_period(mid_frame, row)
        if detected > 80.0:
            T = 165.0  # 30m * 5.5 px/m
        else:
            T = 23.815  # 4.33m * 5.5 px/m

    reverse_count = 0
    total_transitions = len(frames) - 1
    w = mid_frame.shape[1]
    x_start, x_end = 80, min(520, w)

    for i in range(total_transitions):
        f1 = frames[i][row, x_start:x_end, 0].astype(float)
        f2 = frames[i + 1][row, x_start:x_end, 0].astype(float)

        f1_norm = f1 - f1.mean()
        f2_norm = f2 - f2.mean()
        # Only analyze frames where lane dashes are actually present
        # (stripe vs road has std ~77, plain asphalt or occlusion has std < 10)
        if np.std(f1) < 15.0 or np.std(f2) < 15.0:
            continue

        # FFT circular cross-correlation
        N = len(f1_norm)
        F1 = np.fft.fft(f1_norm)
        F2 = np.fft.fft(f2_norm)
        corr = np.real(np.fft.ifft(F2 * np.conj(F1)))

        shift = int(np.argmax(corr))
        if shift > N // 2:
            shift -= N

        # Wrap shift modulo period T into (-T/2, T/2]
        app_shift = ((shift + T / 2.0) % T) - T / 2.0

        if app_shift > threshold_px:
            reverse_count += 1

    pct = (reverse_count / total_transitions * 100.0) if total_transitions > 0 else 0.0

    return {
        "reverse_fraction_pct": round(pct, 1),
        "total_frames": total_transitions,
        "reverse_frames": reverse_count,
        "estimated_period": round(float(T), 1),
        "analyzed_row": row,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check GIF render aliasing.")
    parser.add_argument("gifs", nargs="*", help="Paths to GIFs to analyze.")
    parser.add_argument(
        "--dir", type=str, default=None, help="Directory containing GIFs to analyze."
    )
    parser.add_argument(
        "--verify-baseline",
        action="store_true",
        help="Verify against verified_facts baseline percentages.",
    )
    parser.add_argument(
        "--max-allowed-reverse",
        type=float,
        default=0.0,
        help="Maximum allowed reverse frame percentage (default: 0.0%%).",
    )
    args = parser.parse_args(argv)

    gif_paths: list[str] = list(args.gifs)
    if args.dir and os.path.exists(args.dir):
        for f in sorted(os.listdir(args.dir)):
            if f.endswith(".gif"):
                gif_paths.append(os.path.join(args.dir, f))

    if not gif_paths and args.verify_baseline:
        gif_paths = [
            os.path.join("visualizations/top10", name)
            for name in DEFAULT_BASELINES
        ]

    if not gif_paths:
        print("No GIF paths provided. Use paths as arguments or --dir.")
        return 1

    print("=" * 75)
    print("TEMPORAL ALIASING AUDIT (Reverse Illusion Measurement)")
    print("=" * 75)
    print(f"{'GIF Filename':<32} {'Period':<8} {'Rev Frames':<12} {'Rev %':<8} {'Status'}")
    print("-" * 75)

    all_passed = True
    for path in gif_paths:
        basename = os.path.basename(path)
        try:
            row_to_use = 42 if args.verify_baseline else None
            res = analyze_gif_aliasing(path, dash_row=row_to_use)
            pct = float(res["reverse_fraction_pct"])
            period = float(res["estimated_period"])
            rev_cnt = int(res["reverse_frames"])
            tot_cnt = int(res["total_frames"])

            if args.verify_baseline and basename in DEFAULT_BASELINES:
                expected = DEFAULT_BASELINES[basename]
                # Within 3 percentage points of expected baseline
                matches = abs(pct - expected) <= 3.0
                status = f"BASELINE OK (exp ~{expected:.0f}%)" if matches else f"MISMATCH (exp {expected}%)"
                if not matches:
                    all_passed = False
            else:
                passed = pct <= args.max_allowed_reverse
                status = "PASS (no aliasing)" if passed else f"FAIL (> {args.max_allowed_reverse}%)"
                if not passed:
                    all_passed = False

            print(
                f"{basename:<32} {period:<8.1f} {rev_cnt}/{tot_cnt:<8} {pct:<8.1f}% {status}"
            )
        except Exception as e:
            print(f"{basename:<32} ERROR: {e}")
            all_passed = False

    print("=" * 75)
    if all_passed:
        print("Result: ALL GIFS PASSED ✓")
        return 0
    else:
        print("Result: ALIASING CHECK FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())

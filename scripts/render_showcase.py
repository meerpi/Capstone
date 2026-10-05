#!/usr/bin/env python3
"""Showcase Figure Collection Renderer.

Renders high-definition demonstration GIFs from gifs/manifest.json using existing
visual renderers, verifies simulation determinism, enforces size limits, and generates
gifs/INDEX.md.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import json
import math
import os
import sys
from typing import Any

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from highway_env.envs.highway_env import HighwayEnvFast
from render_utils import apply_anti_alias_stripes
import record_optimal_overtaker
import record_visual_driving

POLICY_RENDER_CONFIG: dict[str, dict[str, Any]] = {
    "classical_mobil_current_fa": {
        "renderer": "classical",
        "controller": "mobil",
        "tier": "full_adas",
        "checkpoint_path": None,
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 20,
    },
    "classical_mobil_current_fo": {
        "renderer": "classical",
        "controller": "mobil",
        "tier": "front_only",
        "checkpoint_path": None,
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 20,
    },
    "ppo_ff_current_fa": {
        "renderer": "ppo_ff",
        "tier": "full_adas",
        "checkpoint_path": "models/ppo_highway_full_adas_seed101.pt",
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 20,
    },
    "overtaker_baseline": {
        "renderer": "optimal",
        "checkpoint_path": "models/ppo_optimal_overtaker_4lane_best.pt",
        "target_speeds": [20.0, 25.0, 30.0],
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 15,
    },
    "overtaker_ext_seed42_best": {
        "renderer": "optimal",
        "checkpoint_path": "scratch/optimal_overtaker_extended_speeds/ppo_optimal_overtaker_4lane_best.pt",
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 15,
    },
    "overtaker_ext_seed137_best": {
        "renderer": "optimal",
        "checkpoint_path": "scratch/optimal_overtaker_extended_speeds_seed137/ppo_optimal_overtaker_4lane_best.pt",
        "target_speeds": [10.0, 15.0, 20.0, 25.0, 30.0],
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 15,
    },
    "ppo_lagrangian_sep_norm": {
        "renderer": "optimal",
        "checkpoint_path": "scratch/ppo_lagrangian_1m_sep_norm/ppo_optimal_overtaker_4lane_best.pt",
        "target_speeds": [20.0, 25.0, 30.0],
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "fps": 15,
    },
    "deepset_v2_retrain_seed42": {
        "renderer": None,
        "default_density": 1.4,
        "default_vehicles": 14,
        "default_duration": 100,
        "reason": "Existing script scripts/diagnose_crashes.py only renders crash clips and lacks support for surviving episodes; DeepSet v2 had 0% crashes in scanned seeds.",
    },
}


def compute_non_default_tag(run: dict[str, Any], cfg: dict[str, Any]) -> str:
    params = run["parameters"]
    parts = []
    if params["vehicles_density"] != cfg["default_density"]:
        parts.append(f"_dens{params['vehicles_density']}")
    if params["vehicles_count"] != cfg["default_vehicles"]:
        parts.append(f"_veh{params['vehicles_count']}")
    if params["duration"] != cfg["default_duration"]:
        parts.append(f"_dur{params['duration']}")
    return "".join(parts)


def compute_param_display_str(run: dict[str, Any], cfg: dict[str, Any]) -> str:
    params = run["parameters"]
    parts = []
    if params["vehicles_density"] != cfg["default_density"]:
        parts.append(f"density={params['vehicles_density']}")
    if params["vehicles_count"] != cfg["default_vehicles"]:
        parts.append(f"vehicles={params['vehicles_count']}")
    if params["duration"] != cfg["default_duration"]:
        parts.append(f"duration={params['duration']}s")
    return ", ".join(parts) if parts else "default"


def _render_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Execute a single episode rendering job using the designated existing renderer."""
    apply_anti_alias_stripes()
    run = task["run"]
    policy = run["policy"]
    seed = run["seed"]
    params = run["parameters"]
    out_gif = task["dest_path"]
    cfg = POLICY_RENDER_CONFIG.get(policy)

    if not cfg or not cfg.get("renderer"):
        reason = cfg.get("reason", f"No existing renderer for {policy}") if cfg else f"Unknown policy {policy}"
        return {
            "task": task,
            "status": "skipped",
            "reason": reason,
        }

    os.makedirs(os.path.dirname(out_gif), exist_ok=True)

    # Check if existing GIF is already rendered and valid
    if not task.get("force", False) and os.path.exists(out_gif) and os.path.getsize(out_gif) <= 3 * 1024 * 1024:
        try:
            with Image.open(out_gif) as im:
                n_frames = im.n_frames
            if n_frames > 1:
                return {
                    "task": task,
                    "status": "kept",
                    "file_size": os.path.getsize(out_gif),
                    "n_frames": n_frames,
                    "replay_steps": run["steps"],
                    "replay_crashed": run["crashed"],
                    "replay_mean_spd": run["mean_speed_kmh"],
                    "overtakes": run.get("overtakes", 0),
                    "lane_changes": run.get("lane_changes", 0),
                }
        except Exception:
            pass

    # Temporary speed tracking via HighwayEnvFast step hook
    speeds: list[float] = []
    orig_step = HighwayEnvFast.step

    def _hooked_step(self, action):
        res = orig_step(self, action)
        if hasattr(self, "vehicle") and self.vehicle is not None:
            speeds.append(float(self.vehicle.speed) * 3.6)
        return res

    HighwayEnvFast.step = _hooked_step

    try:
        renderer_type = cfg["renderer"]
        if renderer_type == "classical":
            res = record_visual_driving.record_classical_episode(
                tier=cfg["tier"],
                seed=seed,
                use_mobil=(cfg["controller"] == "mobil"),
                output_gif=out_gif,
                output_mp4=None,
                fps=cfg["fps"],
                max_steps=int(params["duration"] * 5),
                vehicles_density=params["vehicles_density"],
                vehicles_count=params["vehicles_count"],
                duration=params["duration"],
            )
        elif renderer_type == "ppo_ff":
            res = record_visual_driving.record_ppo_episode(
                model_path=cfg["checkpoint_path"],
                tier=cfg["tier"],
                seed=seed,
                output_gif=out_gif,
                output_mp4=None,
                fps=cfg["fps"],
                max_steps=int(params["duration"] * 5),
                vehicles_density=params["vehicles_density"],
                vehicles_count=params["vehicles_count"],
                duration=params["duration"],
            )
        elif renderer_type == "optimal":
            res = record_optimal_overtaker.record_optimal_episode(
                model_path=cfg["checkpoint_path"],
                lanes_count=4,
                vehicles_density=params["vehicles_density"],
                vehicles_count=params["vehicles_count"],
                seed=seed,
                max_steps=int(params["duration"] * 5),
                output_gif=out_gif,
                output_mp4=None,
                artifact_dir=None,
                fps=cfg["fps"],
                device="cpu",
                target_speeds=cfg.get("target_speeds"),
                duration=params["duration"],
            )
        else:
            return {"task": task, "status": "skipped", "reason": f"Unhandled renderer: {renderer_type}"}

    except Exception as exc:
        if os.path.exists(out_gif):
            try:
                os.remove(out_gif)
            except OSError:
                pass
        return {"task": task, "status": "failed", "reason": str(exc)}
    finally:
        HighwayEnvFast.step = orig_step

    # Verify simulation metrics
    replay_steps = res.get("steps", 0)
    replay_crashed = res.get("crashed", False)
    replay_mean_spd = round(float(np.mean(speeds)), 2) if speeds else 0.0

    exp_steps = run["steps"]
    exp_crashed = run["crashed"]
    exp_spd = run["mean_speed_kmh"]

    diff_crash = (replay_crashed != exp_crashed)
    diff_steps = (replay_steps != exp_steps)
    diff_spd = (abs(replay_mean_spd - exp_spd) > 0.20)

    if diff_crash or diff_steps or diff_spd:
        if os.path.exists(out_gif):
            os.remove(out_gif)
        return {
            "task": task,
            "status": "deleted_discrepancy",
            "reason": f"Replay mismatch: crashed ({replay_crashed} vs {exp_crashed}), steps ({replay_steps} vs {exp_steps}), speed ({replay_mean_spd} vs {exp_spd})",
        }

    # Verify GIF file size and frame count
    if not os.path.exists(out_gif):
        return {"task": task, "status": "failed", "reason": "Output GIF was not created"}

    file_size_bytes = os.path.getsize(out_gif)
    if file_size_bytes > 3 * 1024 * 1024:
        return {
            "task": task,
            "status": "oversized",
            "file_size": file_size_bytes,
            "reason": f"GIF exceeds 3 MB limit ({file_size_bytes / (1024*1024):.2f} MB)",
        }

    with Image.open(out_gif) as im:
        n_frames = im.n_frames

    return {
        "task": task,
        "status": "kept",
        "file_size": file_size_bytes,
        "n_frames": n_frames,
        "replay_steps": replay_steps,
        "replay_crashed": replay_crashed,
        "replay_mean_spd": replay_mean_spd,
        "overtakes": run.get("overtakes", 0),
        "lane_changes": run.get("lane_changes", 0),
    }


def main() -> None:
    apply_anti_alias_stripes()
    parser = argparse.ArgumentParser(description="Render showcase figures from gifs/manifest.json.")
    parser.add_argument("--manifest", type=str, default="gifs/manifest.json",
                        help="Path to manifest JSON (default: gifs/manifest.json).")
    parser.add_argument("--output-dir", type=str, default="gifs",
                        help="Base output folder (default: gifs).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2),
                        help="Parallel worker count (default: CPU count - 2).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without rendering.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-rendering even if output GIF already exists.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of tasks to render (useful for testing).")

    args = parser.parse_args()

    with open(args.manifest, "r") as fp:
        manifest = json.load(fp)

    scan_cfg = manifest["scan_configuration"]
    pool_seeds = scan_cfg["num_seeds"]
    seed_start = scan_cfg["seed_start"]
    seed_end = seed_start + pool_seeds - 1
    categories = manifest["categories"]

    render_tasks: list[dict[str, Any]] = []

    for cat_name, cat_data in categories.items():
        ranking_rule = cat_data["ranking_rule"]
        eligible_count = cat_data["eligible_count"]

        # Check if category is per-policy
        is_per_policy = cat_name in ("policy_best_clean", "policy_best_crash")
        is_paired = cat_name.startswith("paired_")

        for rank_idx, entry in enumerate(cat_data["entries"], 1):
            if is_paired:
                pair_rank = entry["rank"]
                crashed_run = entry["crashed_run"]
                surv_run = entry["surviving_run"]

                for run_type, run_obj in [("crashed", crashed_run), ("surviving", surv_run)]:
                    pol = run_obj["policy"]
                    prof = run_obj["profile"]
                    s = run_obj["seed"]
                    cfg = POLICY_RENDER_CONFIG.get(pol, {})
                    param_tag = compute_non_default_tag(run_obj, cfg) if cfg else ""
                    filename = f"pair{pair_rank:02d}_{run_type}_{pol}_{prof}_seed{s}{param_tag}.gif"
                    dest_path = os.path.join(args.output_dir, cat_name, filename)

                    caption = f"Selected from {pool_seeds} seeds ({seed_start}-{seed_end}), rank {pair_rank} of {eligible_count} by {ranking_rule}"

                    render_tasks.append({
                        "category": cat_name,
                        "rank": pair_rank,
                        "eligible_count": eligible_count,
                        "ranking_rule": ranking_rule,
                        "run": run_obj,
                        "dest_path": dest_path,
                        "filename": filename,
                        "caption": caption,
                        "is_paired": True,
                        "pair_role": run_type,
                        "force": args.force,
                    })

            elif is_per_policy:
                pol = entry["policy"]
                prof = entry["profile"]
                s = entry["seed"]
                # In per policy, rank is within that policy pool (rank 1 or 2)
                rank_in_policy = 1
                if "rank2" in entry.get("file_name", ""):
                    rank_in_policy = 2
                cfg = POLICY_RENDER_CONFIG.get(pol, {})
                param_tag = compute_non_default_tag(entry, cfg) if cfg else ""
                filename = f"{rank_in_policy:02d}_{pol}_{prof}_seed{s}{param_tag}.gif"
                dest_path = os.path.join(args.output_dir, cat_name, pol, filename)

                caption = f"Selected from {pool_seeds} seeds ({seed_start}-{seed_end}), rank {rank_in_policy} of {eligible_count} by {ranking_rule}"

                render_tasks.append({
                    "category": cat_name,
                    "rank": rank_in_policy,
                    "eligible_count": eligible_count,
                    "ranking_rule": ranking_rule,
                    "run": entry,
                    "dest_path": dest_path,
                    "filename": filename,
                    "caption": caption,
                    "is_per_policy": True,
                    "force": args.force,
                })

            else:
                pol = entry["policy"]
                prof = entry["profile"]
                s = entry["seed"]
                cfg = POLICY_RENDER_CONFIG.get(pol, {})
                param_tag = compute_non_default_tag(entry, cfg) if cfg else ""
                filename = f"{rank_idx:02d}_{pol}_{prof}_seed{s}{param_tag}.gif"
                dest_path = os.path.join(args.output_dir, cat_name, filename)

                caption = f"Selected from {pool_seeds} seeds ({seed_start}-{seed_end}), rank {rank_idx} of {eligible_count} by {ranking_rule}"

                render_tasks.append({
                    "category": cat_name,
                    "rank": rank_idx,
                    "eligible_count": eligible_count,
                    "ranking_rule": ranking_rule,
                    "run": entry,
                    "dest_path": dest_path,
                    "filename": filename,
                    "caption": caption,
                    "is_global": True,
                    "force": args.force,
                })

    if args.limit is not None:
        render_tasks = render_tasks[:args.limit]

    print(f"Total entries to process: {len(render_tasks)}")

    if args.dry_run:
        print("\nDry run requested. Listing planned destinations:")
        for t in render_tasks[:10]:
            print(f"  [{t['category']}] -> {t['dest_path']}")
        print(f"... ({len(render_tasks)} total tasks)")
        return

    # Execute rendering
    results = []
    print(f"Executing rendering across {args.workers} workers...", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, res in enumerate(executor.map(_render_worker, render_tasks, chunksize=1)):
            results.append(res)
            st = res["status"]
            fn = res["task"]["filename"]
            cat = res["task"]["category"]
            if (i + 1) % 10 == 0 or (i + 1) == len(render_tasks):
                print(f"Progress: {i + 1}/{len(render_tasks)} ({st}: {cat}/{fn})", flush=True)

    # Classify results
    kept_items = [r for r in results if r["status"] == "kept"]
    skipped_items = [r for r in results if r["status"] == "skipped"]
    deleted_items = [r for r in results if r["status"] == "deleted_discrepancy"]
    oversized_items = [r for r in results if r["status"] == "oversized"]
    failed_items = [r for r in results if r["status"] == "failed"]

    print(f"\nRendering Summary:")
    print(f"  Total planned: {len(render_tasks)}")
    print(f"  Kept GIFs:     {len(kept_items)}")
    print(f"  Skipped:       {len(skipped_items)}")
    print(f"  Deleted:       {len(deleted_items)}")
    print(f"  Oversized:     {len(oversized_items)}")
    print(f"  Failed:        {len(failed_items)}")

    # Generate gifs/INDEX.md
    index_md_path = os.path.join(args.output_dir, "INDEX.md")
    print(f"\nWriting figure index to: {index_md_path}...")

    # Group kept items by category
    by_category: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for k in kept_items:
        by_category[k["task"]["category"]].append(k)

    skipped_by_category: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for s in skipped_items:
        skipped_by_category[s["task"]["category"]].append(s)

    with open(index_md_path, "w") as fp:
        fp.write("# Showcase Figure Collection Index\n\n")

        for cat_name in categories.keys():
            items = by_category.get(cat_name, [])
            fp.write(f"## {cat_name}\n\n")
            fp.write("| File | Category | Policy | Profile | Seed | Outcome | Mean Speed (km/h) | Overtakes | Lane Changes | Parameters (if non-default) | Caption |\n")
            fp.write("|---|---|---|---|---|---|---|---|---|---|---|\n")

            for item in items:
                task = item["task"]
                run = task["run"]
                rel_file = os.path.relpath(task["dest_path"], args.output_dir)
                pol = run["policy"]
                prof = run["profile"]
                seed = run["seed"]
                outcome = "Crashed" if item["replay_crashed"] else "Surviving"
                mean_spd = f"{item['replay_mean_spd']:.2f}"
                ots = item["overtakes"]
                lcs = item["lane_changes"]
                cfg = POLICY_RENDER_CONFIG.get(pol, {})
                params_str = compute_param_display_str(run, cfg)
                caption = task["caption"]

                fp.write(f"| `{rel_file}` | {cat_name} | {pol} | {prof} | {seed} | {outcome} | {mean_spd} | {ots} | {lcs} | {params_str} | {caption} |\n")

            fp.write("\n")

            if cat_name in skipped_by_category:
                fp.write("### Skipped Runs\n\n")
                for sk in skipped_by_category[cat_name]:
                    task = sk["task"]
                    run = task["run"]
                    pol = run["policy"]
                    seed = run["seed"]
                    reason = sk["reason"]
                    fp.write(f"- `{pol}` (seed {seed}): Skipped. Reason: {reason}\n")
                fp.write("\n")

    print(f"✓ Figure index successfully written to {index_md_path}")


if __name__ == "__main__":
    main()

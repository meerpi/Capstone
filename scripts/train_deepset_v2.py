#!/usr/bin/env python3
"""DeepSet v2 PPO Training and Evaluation Pipeline.

Matches project hyperparameter standards from train_optimal_overtaker.py:
- 8 parallel AsyncVectorEnv environments
- 512 rollout steps per environment (batch size 4096)
- 4 minibatches (minibatch size 1024), 4 update epochs
- Optimizer: Adam (lr 2.5e-4 -> 1e-5 linear decay, eps 1e-5)
- GAE: gamma 0.995, lambda 0.95
- Entropy annealing: 0.02 -> 0.005
- Gradient clipping: max_grad_norm 0.5
- Value loss clipping: clip_coef 0.2, vf_coef 0.5
- Calibrated reward shaping: collision penalty -50.0, overtake bonus 1.0

Supports scenarios:
- 'highway' (highway-fast-v0, 500-step episodes)
- 'merge' (merge-v1, ramp-merge coordination)

CRITICAL SAFETY: All checkpoints and evaluation outputs are written strictly
to scratch/ or user-specified output directories. models/ is strictly read-only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from typing import Any, Callable

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from deepset_v2.env import make_deepset_env_v2
from deepset_v2.model import DeepSetAgent
from scripts.evaluate_comparison_100seeds import clopper_pearson_ci


def make_env_thunk(
    scenario: str,
    tier: str,
    seed: int,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
    range_m: float = 100.0,
    target_speeds: list[float] | None = None,
) -> Callable[[], gym.Env]:
    """Create a thunk for AsyncVectorEnv."""
    def thunk() -> gym.Env:
        env = make_deepset_env_v2(
            tier=tier,
            scenario=scenario,
            seed=seed,
            range_m=range_m,
            collision_penalty=collision_penalty,
            overtake_bonus=overtake_bonus,
            target_speeds=target_speeds,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def is_better_candidate(
    candidate: dict[str, Any],
    best: dict[str, Any] | None,
    reversal_threshold: float = 2.0,
) -> bool:
    """Lexicographic safety and behavioral sanity comparator:
    (crashes == 0, chatter_ok, -crashes, -reversal_rate, merge_successes, mean_speed_kmh).
    """
    if best is None:
        return True

    def _extract_reversal(d: dict[str, Any]) -> float:
        for k in (
            "action_reversal_rate",
            "reversal_rate",
            "rapid_reversals_per_ep",
            "mean_reversals",
            "mean_rapid_reversals",
            "chatter_rate",
        ):
            if k in d:
                return float(d[k])
        return 0.0

    c_crashes = float(candidate["crashes"])
    b_crashes = float(best["crashes"])
    c_rev = _extract_reversal(candidate)
    b_rev = _extract_reversal(best)
    c_chatter_rank = 1.0 if c_rev <= reversal_threshold else -c_rev
    b_chatter_rank = 1.0 if b_rev <= reversal_threshold else -b_rev
    c_succ = float(candidate.get("merge_successes", 0))
    b_succ = float(best.get("merge_successes", 0))
    c_speed = float(candidate.get("mean_speed_kmh", 0.0))
    b_speed = float(best.get("mean_speed_kmh", 0.0))

    return (c_crashes == 0, c_chatter_rank, -c_crashes, c_succ, c_speed) > (
        b_crashes == 0,
        b_chatter_rank,
        -b_crashes,
        b_succ,
        b_speed,
    )


def evaluate_deepset(
    agent: DeepSetAgent,
    scenario: str = "highway",
    tier: str = "full_adas",
    episodes: int = 100,
    seed_start: int = 2000,
    device: torch.device = torch.device("cpu"),
    deterministic: bool = True,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
    range_m: float = 100.0,
    target_speeds: list[float] | None = None,
) -> dict[str, Any]:
    """Evaluate DeepSetAgent on fixed deterministic seeds and compute exact Clopper-Pearson CIs."""
    agent.eval()
    env = make_deepset_env_v2(
        tier=tier,
        scenario=scenario,
        seed=seed_start,
        range_m=range_m,
        collision_penalty=collision_penalty,
        overtake_bonus=overtake_bonus,
        target_speeds=target_speeds,
    )

    action_dim = int(env.action_space.n)
    crashes = 0
    merge_successes = 0
    durations: list[int] = []
    speeds_mps: list[float] = []
    total_lane_changes: list[int] = []
    total_rapid_reversals: list[int] = []
    total_overtakes: list[int] = []

    for ep in range(episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        ep_steps = 0
        ep_speeds: list[float] = []
        ep_lane_changes = 0
        ep_rapid_reversals = 0
        ep_lc_events: list[tuple[int, int, int]] = []
        ep_overtakes = 0
        crashed = False
        current_lane = (
            env.unwrapped.vehicle.lane_index[2]
            if hasattr(env.unwrapped, "vehicle") and hasattr(env.unwrapped.vehicle, "lane_index")
            else None
        )

        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_tensor, deterministic=deterministic, action_mask=mask_t
                )
                action_idx = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action_idx)
            ep_steps += 1

            if hasattr(env.unwrapped, "vehicle") and hasattr(env.unwrapped.vehicle, "lane_index"):
                new_lane = env.unwrapped.vehicle.lane_index[2]
                if current_lane is not None and new_lane != current_lane:
                    ep_lane_changes += 1
                    if ep_lc_events:
                        prev_step, prev_from, _ = ep_lc_events[-1]
                        if new_lane == prev_from and (ep_steps - prev_step) <= 5:
                            ep_rapid_reversals += 1
                    ep_lc_events.append((ep_steps, current_lane, new_lane))
                current_lane = new_lane

            ep_speeds.append(float(info.get("speed", 0.0)))
            ep_overtakes += int(info.get("overtake_count", 0))

            if info.get("crashed", False) or getattr(env.unwrapped.vehicle, "crashed", False):
                crashed = True

            if terminated or truncated:
                break

        if crashed:
            crashes += 1
        else:
            if scenario == "merge":
                # In merge-v1, successful merge passes access ramp: pos_x > 370
                pos_x = getattr(env.unwrapped.vehicle, "position", [0.0])[0]
                if pos_x > 370.0:
                    merge_successes += 1
            else:
                merge_successes += 1

        durations.append(ep_steps)
        speeds_mps.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        total_lane_changes.append(ep_lane_changes)
        total_rapid_reversals.append(ep_rapid_reversals)
        total_overtakes.append(ep_overtakes)

    env.close()

    mean_speed_kmh = float(np.mean(speeds_mps) * 3.6) if speeds_mps else 0.0
    crash_rate = (crashes / episodes) * 100.0
    crash_ci_low, crash_ci_high = clopper_pearson_ci(crashes, episodes, 0.95)
    mean_duration = float(np.mean(durations)) if durations else 0.0
    mean_lc = float(np.mean(total_lane_changes)) if total_lane_changes else 0.0
    mean_rev = float(np.mean(total_rapid_reversals)) if total_rapid_reversals else 0.0
    mean_ot = float(np.mean(total_overtakes)) if total_overtakes else 0.0

    results: dict[str, Any] = {
        "scenario": scenario,
        "tier": tier,
        "episodes": episodes,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{episodes}",
        "crash_rate": crash_rate,
        "crash_ci_low": crash_ci_low * 100.0,
        "crash_ci_high": crash_ci_high * 100.0,
        "mean_speed_kmh": mean_speed_kmh,
        "mean_duration": mean_duration,
        "mean_lane_changes": mean_lc,
        "action_reversal_rate": mean_rev,
        "reversal_rate": mean_rev,
        "mean_reversals": mean_rev,
        "rapid_reversals_per_ep": mean_rev,
        "mean_overtakes": mean_ot,
    }

    if scenario == "merge":
        succ_rate = (merge_successes / episodes) * 100.0
        succ_ci_low, succ_ci_high = clopper_pearson_ci(merge_successes, episodes, 0.95)
        results["merge_successes"] = merge_successes
        results["merge_success_rate"] = succ_rate
        results["merge_success_ci_low"] = succ_ci_low * 100.0
        results["merge_success_ci_high"] = succ_ci_high * 100.0

    return results


def train_deepset(args: argparse.Namespace) -> str:
    """Execute standard PPO training loop for DeepSet v2."""
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = int(args.total_timesteps // batch_size)

    run_name = f"deepset_v2_{args.scenario}_{args.tier}_seed{args.seed}_{int(time.time())}"
    log_dir = os.path.join(args.output_dir, "tb_logs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Parallel environments
    target_speeds = [float(x) for x in args.target_speeds] if args.target_speeds else None
    env_fns = [
        make_env_thunk(
            scenario=args.scenario,
            tier=args.tier,
            seed=args.seed + i * 1000,
            collision_penalty=args.collision_penalty,
            overtake_bonus=args.overtake_bonus,
            range_m=args.range_m,
            target_speeds=target_speeds,
        )
        for i in range(args.num_envs)
    ]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(envs.single_action_space.n)

    print("=" * 65)
    print(f"=== DeepSet v2 PPO Training: {args.scenario.upper()} ({args.tier}) ===")
    print(f"Observation dim: {obs_dim} (Ego=4, Vehicles=14×5)")
    print(f"Budget: {args.total_timesteps:,} steps | {num_iterations} iterations")
    print(f"Batch size: {batch_size} | Minibatch size: {minibatch_size} | Epochs: {args.update_epochs}")
    print(f"Device: {device} | Parallel Envs: {args.num_envs} (Async)")
    print(f"LR: {args.learning_rate} -> {args.lr_end} | Entropy: {args.ent_coef} -> {args.ent_end}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 65, flush=True)

    agent = DeepSetAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout buffers
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    masks = torch.zeros((args.num_steps, args.num_envs, action_dim), dtype=torch.bool, device=device)

    global_step = 0
    start_time = time.time()
    next_obs_np, next_info = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    init_mask = next_info.get("action_mask", np.ones((args.num_envs, action_dim), dtype=bool))
    next_mask = torch.tensor(init_mask, dtype=torch.bool, device=device)

    best_val_candidate: dict[str, Any] | None = None
    best_model_path = os.path.join(args.output_dir, f"deepset_v2_{args.scenario}_best.pt")
    final_model_path = os.path.join(args.output_dir, f"deepset_v2_{args.scenario}_final.pt")

    episode_returns: list[float] = []
    episode_lengths: list[int] = []

    for iteration in range(1, num_iterations + 1):
        progress = (iteration - 1.0) / max(num_iterations, 1)

        # Learning rate annealing
        lr_now = args.learning_rate + (args.lr_end - args.learning_rate) * progress
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Entropy coefficient annealing
        ent_now = args.ent_coef + (args.ent_end - args.ent_coef) * progress

        # Rollout collection
        agent.eval()
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            masks[step] = next_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, action_mask=next_mask
                )
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            step_obs, step_reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(step_reward, dtype=torch.float32, device=device)
            next_obs = torch.tensor(step_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(next_done_np, dtype=torch.float32, device=device)
            step_mask = infos.get("action_mask", np.ones((args.num_envs, action_dim), dtype=bool))
            next_mask = torch.tensor(step_mask, dtype=torch.bool, device=device)

            if "episode" in infos and "_episode" in infos:
                for idx, is_done in enumerate(infos["_episode"]):
                    if is_done:
                        r = float(infos["episode"]["r"][idx])
                        l_val = int(infos["episode"]["l"][idx])
                        episode_returns.append(r)
                        episode_lengths.append(l_val)
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l_val, global_step)

        # Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(-1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = masks.reshape((-1, action_dim))

        # PPO Update
        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs: list[float] = []
        pg_losses: list[float] = []
        v_losses: list[float] = []
        entropy_losses: list[float] = []
        approx_kls: list[float] = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    b_actions.long()[mb_inds],
                    action_mask=b_masks[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    approx_kls.append(approx_kl.item())
                    clipfrac = ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    clipfracs.append(clipfrac)

                # Advantage normalization per minibatch
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_now * entropy_loss + args.vf_coef * v_loss

                assert not torch.isnan(loss), f"Encountered NaN loss at global_step={global_step}"

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())

        elapsed_time = time.time() - start_time
        sps = int(global_step / max(elapsed_time, 1e-5))

        # Log to TensorBoard
        writer.add_scalar("losses/policy_loss", float(np.mean(pg_losses)), global_step)
        writer.add_scalar("losses/value_loss", float(np.mean(v_losses)), global_step)
        writer.add_scalar("losses/entropy", float(np.mean(entropy_losses)), global_step)
        writer.add_scalar("losses/approx_kl", float(np.mean(approx_kls)), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("charts/learning_rate", lr_now, global_step)
        writer.add_scalar("charts/entropy_coef", ent_now, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)

        # Console logging
        if iteration % args.log_frequency == 0 or iteration == num_iterations:
            recent_ret = f"{np.mean(episode_returns[-20:]):.2f}" if episode_returns else "N/A"
            recent_len = f"{np.mean(episode_lengths[-20:]):.1f}" if episode_lengths else "N/A"
            print(
                f"[{iteration:4d}/{num_iterations}] Step: {global_step:7d} | "
                f"SPS: {sps:4d} | "
                f"Ret: {recent_ret:>7} | "
                f"Len: {recent_len:>5} | "
                f"PG: {np.mean(pg_losses):6.3f} | "
                f"VF: {np.mean(v_losses):6.3f} | "
                f"Ent: {np.mean(entropy_losses):5.3f} | "
                f"KL: {np.mean(approx_kls):6.4f}",
                flush=True,
            )

        # Periodic validation & checkpointing
        if (
            args.val_frequency > 0
            and (global_step % args.val_frequency < batch_size or iteration == num_iterations)
            and global_step >= args.val_start_step
        ):
            print(f"--> Validating checkpoint at step {global_step} on {args.val_seeds} seeds...", flush=True)
            val_results = evaluate_deepset(
                agent=agent,
                scenario=args.scenario,
                tier=args.tier,
                episodes=args.val_seeds,
                seed_start=args.val_seed_start,
                device=device,
                deterministic=True,
                collision_penalty=args.collision_penalty,
                overtake_bonus=args.overtake_bonus,
                range_m=args.range_m,
                target_speeds=target_speeds,
            )
            val_crash_str = f"{val_results['crashes']}/{val_results['episodes']} ({val_results['crash_rate']:.1f}%)"
            val_speed_str = f"{val_results['mean_speed_kmh']:.1f} km/h"
            val_rev_str = f"{val_results.get('action_reversal_rate', 0.0):.1f}"
            extra_str = f" | Succ: {val_results.get('merge_success_rate', 0.0):.1f}%" if args.scenario == "merge" else ""
            print(f"    Val Result: crashes={val_crash_str} | speed={val_speed_str} | rev_rate={val_rev_str}{extra_str}", flush=True)

            writer.add_scalar("val/crash_rate", val_results["crash_rate"], global_step)
            writer.add_scalar("val/mean_speed_kmh", val_results["mean_speed_kmh"], global_step)
            writer.add_scalar("val/action_reversal_rate", val_results.get("action_reversal_rate", 0.0), global_step)
            if args.scenario == "merge":
                writer.add_scalar("val/merge_success_rate", val_results.get("merge_success_rate", 0.0), global_step)

            if is_better_candidate(val_results, best_val_candidate):
                best_val_candidate = val_results
                torch.save(
                    {
                        "model_state_dict": agent.state_dict(),
                        "global_step": global_step,
                        "iteration": iteration,
                        "config": agent.config,
                        "scenario": args.scenario,
                        "tier": args.tier,
                        "val_results": val_results,
                    },
                    best_model_path,
                )
                print(f"    ✓ New BEST candidate saved to {best_model_path}!", flush=True)

    envs.close()

    # Save final model
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "global_step": global_step,
            "iteration": num_iterations,
            "config": agent.config,
            "scenario": args.scenario,
            "tier": args.tier,
        },
        final_model_path,
    )
    print(f"✓ Final model checkpoint saved to {final_model_path}")
    writer.close()

    # Return model path for evaluation (best if exists, else final)
    return best_model_path if os.path.exists(best_model_path) else final_model_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSet v2 PPO Training and Evaluation Pipeline")
    # Task / Scenario
    parser.add_argument("--scenario", type=str, choices=["highway", "merge"], default="highway")
    parser.add_argument("--tier", type=str, default="full_adas", choices=["full_adas", "front_only", "short_range", "no_velocity"])
    parser.add_argument("--range-m", type=float, default=100.0)

    # PPO Hyperparameters matching train_optimal_overtaker.py
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--lr-end", type=float, default=1e-5)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.02)
    parser.add_argument("--ent-end", type=float, default=0.005)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--cuda", action="store_true", default=True)

    # Reward shaping parameters
    parser.add_argument("--collision-penalty", type=float, default=-50.0)
    parser.add_argument("--overtake-bonus", type=float, default=1.0)
    parser.add_argument("--target-speeds", type=float, nargs="+", default=None,
                        help="Override DiscreteMetaAction target speeds, e.g. --target-speeds 10 15 20 25 30")

    # Output / Validation
    parser.add_argument("--output-dir", type=str, default="scratch/deepset_v2")
    parser.add_argument("--log-frequency", type=int, default=5)
    parser.add_argument("--val-frequency", type=int, default=50000)
    parser.add_argument("--val-start-step", type=int, default=50000)
    parser.add_argument("--val-seeds", type=int, default=50, help="Validation seeds (default: 50, Clopper-Pearson 95%% CI upper bound 7.1%%)")
    parser.add_argument("--val-seed-start", type=int, default=4000)

    # Evaluation
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--eval-seeds", type=int, default=100)
    parser.add_argument("--eval-seed-start", type=int, default=2000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    target_speeds = [float(x) for x in args.target_speeds] if args.target_speeds else None

    if args.eval_only:
        assert args.model_path is not None, "--model-path must be specified for --eval-only"
        print(f"Loading DeepSetAgent from {args.model_path}...")
        agent = DeepSetAgent().to(device)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        agent.load_state_dict(state_dict)

        results = evaluate_deepset(
            agent=agent,
            scenario=args.scenario,
            tier=args.tier,
            episodes=args.eval_seeds,
            seed_start=args.eval_seed_start,
            device=device,
            deterministic=True,
            collision_penalty=args.collision_penalty,
            overtake_bonus=args.overtake_bonus,
            range_m=args.range_m,
            target_speeds=target_speeds,
        )
        print("\n" + "=" * 65)
        print(f"EVALUATION RESULTS ({args.scenario}, {args.tier}, {args.eval_seeds} seeds: {args.eval_seed_start}-{args.eval_seed_start+args.eval_seeds-1})")
        print("=" * 65)
        for k, v in results.items():
            print(f"  {k:<25}: {v}")
        print("=" * 65)
        return

    # Train DeepSetAgent
    model_to_eval = train_deepset(args)

    # Final evaluation on requested seeds
    print("\n" + "=" * 65)
    print(f"FINAL 100-SEED BENCHMARK ({args.scenario}, {args.tier}, seeds {args.eval_seed_start}-{args.eval_seed_start+args.eval_seeds-1})")
    print(f"Model: {model_to_eval}")
    print("=" * 65)
    agent = DeepSetAgent().to(device)
    ckpt = torch.load(model_to_eval, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    agent.load_state_dict(state_dict)

    results = evaluate_deepset(
        agent=agent,
        scenario=args.scenario,
        tier=args.tier,
        episodes=args.eval_seeds,
        seed_start=args.eval_seed_start,
        device=device,
        deterministic=True,
        collision_penalty=args.collision_penalty,
        overtake_bonus=args.overtake_bonus,
        range_m=args.range_m,
        target_speeds=target_speeds,
    )
    for k, v in results.items():
        print(f"  {k:<25}: {v}")
    print("=" * 65)

    # Save results to output json
    eval_json_path = os.path.join(args.output_dir, f"benchmark_{args.scenario}_{args.tier}_{args.eval_seeds}seeds.json")
    with open(eval_json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Benchmark results written to {eval_json_path}")


if __name__ == "__main__":
    main()

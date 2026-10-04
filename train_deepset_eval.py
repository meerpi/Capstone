"""Training and evaluation script for DeepSet permutation-invariant observation encoder.

Evaluates DeepSetAgent against the standard PPO tactical overtaking baseline:
- Exactly matches PPO hyperparameters: 8 envs, 512 steps (4096 batch), lr=2.5e-4, gamma=0.995, gae=0.95
- Matches calibrated reward shaping: collision_penalty=-50.0, overtake_bonus=1.0 (or 2.5)
- Tests cross-scenario compatibility on highway-fast-v0 and merge-v1 without slot redesign
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deepset_encoder import DeepSetAgent, make_deepset_env


def make_env_thunk(
    scenario: str,
    seed: int,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
) -> Callable[[], gym.Env]:
    """Create a thunk for SyncVectorEnv."""
    def thunk() -> gym.Env:
        env = make_deepset_env(
            scenario=scenario,
            seed=seed,
            collision_penalty=collision_penalty,
            overtake_bonus=overtake_bonus,
        )
        return gym.wrappers.RecordEpisodeStatistics(env)
    return thunk


def clopper_pearson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute exact Clopper-Pearson binomial confidence interval [p_lower, p_upper]."""
    if n == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    if k == 0:
        p_lower = 0.0
    else:
        target = alpha / 2.0
        low, high = 0.0, float(k) / n
        for _ in range(80):
            mid = (low + high) / 2.0
            tail_prob = sum(math.comb(n, j) * (mid ** j) * ((1.0 - mid) ** (n - j)) for j in range(k, n + 1))
            if tail_prob < target:
                low = mid
            else:
                high = mid
        p_lower = (low + high) / 2.0
    if k == n:
        p_upper = 1.0
    else:
        target = alpha / 2.0
        low, high = float(k) / n, 1.0
        for _ in range(80):
            mid = (low + high) / 2.0
            tail_prob = sum(math.comb(n, j) * (mid ** j) * ((1.0 - mid) ** (n - j)) for j in range(k + 1))
            if tail_prob > target:
                low = mid
            else:
                high = mid
        p_upper = (low + high) / 2.0
    return p_lower, p_upper


def evaluate_deepset(
    agent: DeepSetAgent,
    scenario: str = "highway",
    episodes: int = 30,
    seed_start: int = 2000,
    device: str = "cpu",
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
) -> dict[str, Any]:
    """Evaluate DeepSet agent deterministically across benchmark seeds."""
    agent.eval()
    crashes = 0
    durations: list[int] = []
    speeds: list[float] = []
    overtakes: list[int] = []
    lane_changes: list[int] = []

    for ep in range(episodes):
        seed = seed_start + ep
        env = make_deepset_env(
            scenario=scenario,
            seed=seed,
            collision_penalty=collision_penalty,
            overtake_bonus=overtake_bonus,
        )
        obs, info = env.reset(seed=seed)
        ep_steps = 0
        ep_speeds: list[float] = []
        ep_overtakes = 0
        ep_lc = 0
        crashed = False

        ego = getattr(env.unwrapped, "vehicle", None)
        prev_lane = ego.lane_index[2] if (ego and hasattr(ego, "lane_index")) else None

        while True:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask = info.get("action_mask", np.ones(5, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_t, deterministic=True, action_mask=mask_t
                )
                action_idx = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action_idx)
            ep_steps += 1
            ep_speeds.append(float(info.get("speed", 0.0)))
            ep_overtakes += int(info.get("overtake_count", 0))

            if ego and hasattr(ego, "lane_index"):
                new_lane = ego.lane_index[2]
                if prev_lane is not None and new_lane != prev_lane:
                    ep_lc += 1
                prev_lane = new_lane

            if info.get("crashed", False):
                crashed = True

            if terminated or truncated:
                break

        if crashed:
            crashes += 1
        durations.append(ep_steps)
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        overtakes.append(ep_overtakes)
        lane_changes.append(ep_lc)
        env.close()

    mean_speed_kmh = float(np.mean(speeds) * 3.6) if speeds else 0.0
    crash_rate = (crashes / episodes) * 100.0
    mean_duration = float(np.mean(durations))
    mean_ot = float(np.mean(overtakes))
    mean_lc = float(np.mean(lane_changes))
    ci_lower, ci_upper = clopper_pearson_ci(crashes, episodes)

    results = {
        "episodes": episodes,
        "crashes": crashes,
        "crash_rate": crash_rate,
        "ci_lower": ci_lower * 100.0,
        "ci_upper": ci_upper * 100.0,
        "mean_speed_kmh": mean_speed_kmh,
        "mean_duration": mean_duration,
        "mean_overtakes": mean_ot,
        "mean_lane_changes": mean_lc,
    }
    return results


def train_deepset(
    scenario: str = "highway",
    total_timesteps: int = 100000,
    seed: int = 42,
    num_envs: int = 8,
    num_steps: int = 512,
    learning_rate: float = 2.5e-4,
    lr_end: float = 1e-5,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.02,
    ent_end: float = 0.005,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    num_minibatches: int = 4,
    update_epochs: int = 4,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "scratch/deepset_eval",
) -> tuple[DeepSetAgent, dict[str, Any]]:
    output_dir = os.path.join(output_dir, scenario)
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    batch_size = num_envs * num_steps
    num_iterations = total_timesteps // batch_size
    minibatch_size = batch_size // num_minibatches

    print(f"\n============================================================")
    print(f"DeepSet PPO Training: {scenario.upper()}")
    print(f"Timesteps: {total_timesteps:,} ({num_iterations} iters) | Envs: {num_envs} | Batch: {batch_size}")
    print(f"Device: {device} | Reward: collision={collision_penalty}, overtake={overtake_bonus}")
    print(f"============================================================")

    env_fns = [
        make_env_thunk(
            scenario=scenario,
            seed=seed + i * 100,
            collision_penalty=collision_penalty,
            overtake_bonus=overtake_bonus,
        )
        for i in range(num_envs)
    ]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    agent = DeepSetAgent(veh_dim=5, ego_dim=5, action_dim=5, latent_dim=128, pool="both").to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Observation shape from vector env: (num_envs, N, 5)
    obs_shape = envs.single_observation_space.shape
    num_vehicles = obs_shape[0]
    feat_dim = obs_shape[1]

    # Rollout buffers
    obs_buf = torch.zeros((num_steps, num_envs, num_vehicles, feat_dim), device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), device=device)
    dones_buf = torch.zeros((num_steps, num_envs), device=device)
    values_buf = torch.zeros((num_steps, num_envs), device=device)
    masks_buf = torch.zeros((num_steps, num_envs, 5), dtype=torch.bool, device=device)

    # Initial state
    next_obs, next_info = envs.reset(seed=seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(num_envs, device=device)
    next_mask = torch.tensor(
        next_info.get("action_mask", np.ones((num_envs, 5), dtype=bool)),
        dtype=torch.bool,
        device=device,
    )

    global_step = 0
    start_time = time.time()
    best_candidate: dict[str, Any] | None = None

    for iteration in range(1, num_iterations + 1):
        # Linear annealing
        frac = 1.0 - (iteration - 1.0) / max(num_iterations, 1)
        cur_lr = lr_end + frac * (learning_rate - lr_end)
        cur_ent = ent_end + frac * (ent_coef - ent_end)
        for param_group in optimizer.param_groups:
            param_group["lr"] = cur_lr

        agent.eval()
        ep_returns: list[float] = []

        # --- Rollout collection ---
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            masks_buf[step] = next_mask

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, action_mask=next_mask
                )
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            step_actions = action.cpu().numpy()
            step_obs, step_rewards, step_terms, step_truncs, step_infos = envs.step(step_actions)

            rewards_buf[step] = torch.tensor(step_rewards, dtype=torch.float32, device=device)
            next_obs = torch.tensor(step_obs, dtype=torch.float32, device=device)
            next_done = torch.tensor(
                np.logical_or(step_terms, step_truncs), dtype=torch.float32, device=device
            )

            if "final_info" in step_infos:
                for f_info in step_infos["final_info"]:
                    if f_info and "episode" in f_info:
                        ep_returns.append(float(f_info["episode"]["r"]))

            mask_arr = step_infos.get("action_mask", np.ones((num_envs, 5), dtype=bool))
            next_mask = torch.tensor(mask_arr, dtype=torch.bool, device=device)

        # --- GAE Advantage computation ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # Flatten batch
        b_obs = obs_buf.reshape(-1, num_vehicles, feat_dim)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)
        b_masks = masks_buf.reshape(-1, 5)

        # Standardize advantage
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # --- Optimization Epochs ---
        agent.train()
        b_inds = np.arange(batch_size)
        clipfracs: list[float] = []

        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    action=b_actions[mb_inds],
                    action_mask=b_masks[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    clipfracs.append(float(((ratio - 1.0).abs() > clip_coef).float().mean().item()))

                mb_advantages = b_advantages[mb_inds]

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds], -clip_coef, clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cur_ent * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        # Explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        mean_ret = float(np.mean(ep_returns)) if ep_returns else 0.0

        if iteration % 4 == 0 or iteration == num_iterations:
            print(
                f"[{iteration:3d}/{num_iterations:3d}] Step: {global_step:7,d} | SPS: {sps:4d} | "
                f"Return: {mean_ret:6.1f} | PgLoss: {pg_loss.item():+.4f} | VLoss: {v_loss.item():6.2f} | "
                f"Ent: {entropy_loss.item():.3f} | ExplVar: {explained_var:.3f} | LR: {cur_lr:.2e}",
                flush=True,
            )

        # Periodic evaluation every 10 iterations
        if iteration % 10 == 0 or iteration == num_iterations:
            val_results = evaluate_deepset(
                agent=agent,
                scenario=scenario,
                episodes=10,
                seed_start=3000,
                device=device,
                collision_penalty=collision_penalty,
                overtake_bonus=overtake_bonus,
            )
            print(
                f"  [EVAL] crashes={val_results['crashes']}/{val_results['episodes']} "
                f"({val_results['crash_rate']:5.1f}%) "
                f"speed={val_results['mean_speed_kmh']:5.1f} km/h "
                f"duration={val_results['mean_duration']:5.1f} "
                f"LC={val_results['mean_lane_changes']:.1f} OT={val_results['mean_overtakes']:.1f}",
                flush=True,
            )
            # Lexicographic selection
            if best_candidate is None or (
                val_results["crashes"] < best_candidate["crashes"]
                or (
                    val_results["crashes"] == best_candidate["crashes"]
                    and val_results["mean_speed_kmh"] > best_candidate["mean_speed_kmh"]
                )
            ):
                best_candidate = val_results.copy()
                best_path = os.path.join(output_dir, f"deepset_{scenario}_best.pt")
                torch.save({
                    "model_state_dict": agent.state_dict(),
                    "eval_metrics": best_candidate,
                    "scenario": scenario,
                }, best_path)
                print(f"  ★ New best model saved to {best_path}", flush=True)

    envs.close()

    # Final evaluation: 100 seeds for >= 500k runs, 30 seeds for shorter runs
    final_episodes = 100 if total_timesteps >= 500000 else 30
    print(f"\n============================================================")
    print(f"FINAL {final_episodes}-SEED EVALUATION: {scenario.upper()} (Seeds 2000-{2000 + final_episodes - 1})")
    print(f"============================================================")
    final_results = evaluate_deepset(
        agent=agent,
        scenario=scenario,
        episodes=final_episodes,
        seed_start=2000,
        device=device,
        collision_penalty=collision_penalty,
        overtake_bonus=overtake_bonus,
    )
    print(
        f"  [FINAL] crashes={final_results['crashes']}/{final_results['episodes']} "
        f"({final_results['crash_rate']:5.1f}%) [95% CI: {final_results['ci_lower']:4.1f}%, {final_results['ci_upper']:4.1f}%] "
        f"speed={final_results['mean_speed_kmh']:5.1f} km/h "
        f"duration={final_results['mean_duration']:5.1f} "
        f"LC={final_results['mean_lane_changes']:.1f} OT={final_results['mean_overtakes']:.1f}",
        flush=True,
    )

    final_path = os.path.join(output_dir, f"deepset_{scenario}_final.pt")
    torch.save({
        "model_state_dict": agent.state_dict(),
        "final_metrics": final_results,
        "scenario": scenario,
    }, final_path)

    return agent, final_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSet PPO Training and Evaluation")
    parser.add_argument("--scenario", type=str, default="highway", choices=["highway", "merge", "roundabout", "intersection"])
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--collision-penalty", type=float, default=-50.0)
    parser.add_argument("--overtake-bonus", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default="scratch/deepset_eval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint for evaluation")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Number of evaluation episodes")
    parser.add_argument("--eval-seed-start", type=int, default=2000, help="Starting seed for evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.eval_only:
        if args.model_path is None:
            raise ValueError("--model-path must be specified when --eval-only is set")
        ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        agent = DeepSetAgent().to(args.device)
        agent.load_state_dict(state_dict)
        print(f"=== EVALUATING DEEPSET: {args.scenario.upper()} ({args.eval_episodes} episodes, seeds {args.eval_seed_start}-{args.eval_seed_start + args.eval_episodes - 1}) ===")
        res = evaluate_deepset(
            agent=agent,
            scenario=args.scenario,
            episodes=args.eval_episodes,
            seed_start=args.eval_seed_start,
            device=args.device,
            collision_penalty=args.collision_penalty,
            overtake_bonus=args.overtake_bonus,
        )
        print(f"Crashes: {res['crashes']}/{res['episodes']} ({res['crash_rate']:.1f}%) [95% CI: {res['ci_lower']:.1f}%, {res['ci_upper']:.1f}%]")
        print(f"Mean Duration: {res['mean_duration']:.1f} steps")
        print(f"Mean Speed: {res['mean_speed_kmh']:.1f} km/h")
        print(f"Mean Overtakes: {res['mean_overtakes']:.2f} OT/ep")
        print(f"Mean Lane Changes: {res['mean_lane_changes']:.1f} LC/ep")
    else:
        train_deepset(
            scenario=args.scenario,
            total_timesteps=args.total_timesteps,
            seed=args.seed,
            num_envs=args.num_envs,
            num_steps=args.num_steps,
            learning_rate=args.learning_rate,
            collision_penalty=args.collision_penalty,
            overtake_bonus=args.overtake_bonus,
            device=args.device,
            output_dir=args.output_dir,
        )

"""Optimal Tactical Overtaking PPO Agent.

Trains a PPO policy using structured lane-slot observations (81-dim with K=3
frame stacking) and calibrated reward shaping on highway-fast-v0.

Key design decisions (backed by research):
- Structured 27-dim tactical observation per frame (permutation-invariant lane slots)
- K=3 frame stacking → 81-dim input (provides temporal derivatives)
- 3-layer MLP with 256 hidden units + Tanh activations
- Persistent blockage penalty (eliminates brake exploit)
- Calibrated collision penalty (-10.0, not paralyzing -30.0)
- 8-core AsyncVectorEnv for ~4x throughput
- Entropy annealing from 0.02 → 0.005

Usage:
    # Train on 4-lane highway (default):
    python train_optimal_overtaker.py

    # Train on 3-lane highway:
    python train_optimal_overtaker.py --lanes 3

    # Evaluate a trained model:
    python train_optimal_overtaker.py --eval-only --model-path models/ppo_optimal_overtaker.pt
"""

import argparse
import os
import random
import time
from typing import Any, Callable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import env_config


def make_env_thunk(
    lanes_count: int,
    vehicles_density: float,
    vehicles_count: int,
    seed: int,
    frame_stack_k: int = 3,
) -> Callable[[], gym.Env]:
    """Create a thunk for parallel environment instantiation."""

    def thunk() -> gym.Env:
        env = env_config.make_optimal_env(
            lanes_count=lanes_count,
            vehicles_density=vehicles_density,
            vehicles_count=vehicles_count,
            seed=seed,
            frame_stack_k=frame_stack_k,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    """Initialize linear layer with orthogonal weights."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class OptimalAgent(nn.Module):
    """3-layer Actor-Critic MLP with 256 hidden units for tactical overtaking."""

    def __init__(self, obs_dim: int, action_dim: int = 5) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x_flat = x.reshape(-1, self.obs_dim)
        return self.critic(x_flat)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_flat = x.reshape(-1, self.obs_dim)
        logits = self.actor(x_flat)

        if action_mask is not None:
            mask_flat = action_mask.reshape(-1, logits.shape[-1])
            logits = torch.where(
                mask_flat,
                logits,
                torch.tensor(-1e8, dtype=logits.dtype, device=logits.device),
            )

        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x_flat)


def train(args: argparse.Namespace) -> str:
    """Execute PPO training with optimal hyperparameters."""
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = int(args.total_timesteps // batch_size)

    run_name = f"ppo_optimal_{args.lanes}lane_seed{args.seed}_{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    # Create parallel environments
    env_fns = [
        make_env_thunk(
            lanes_count=args.lanes,
            vehicles_density=args.density,
            vehicles_count=args.vehicles,
            seed=args.seed + i,
            frame_stack_k=args.frame_stack,
        )
        for i in range(args.num_envs)
    ]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(envs.single_action_space.n)

    print(f"=== Optimal Tactical Overtaking PPO ===")
    print(f"Observation dim: {obs_dim} ({args.frame_stack}×27 frame stack)")
    print(f"Lanes: {args.lanes} | Density: {args.density} | Vehicles: {args.vehicles}")
    print(f"Budget: {args.total_timesteps:,} steps | {num_iterations} iterations")
    print(f"Batch: {batch_size} | Minibatch: {minibatch_size}")
    print(f"Device: {device} | Envs: {args.num_envs} (Async)")
    print(f"LR: {args.learning_rate} → {args.lr_end} | Entropy: {args.ent_coef} → {args.ent_end}")
    print("=" * 55, flush=True)

    agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout storage
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    masks = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), dtype=torch.bool, device=device
    )

    global_step = 0
    start_time = time.time()
    next_obs, next_info = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    init_mask = next_info.get(
        "action_mask", np.ones((args.num_envs, action_dim), dtype=bool)
    )
    next_mask = torch.tensor(init_mask, dtype=torch.bool, device=device)

    # Tracking best model
    best_eval_score = -float("inf")
    episode_returns: list[float] = []
    episode_speeds: list[float] = []
    episode_overtakes: list[int] = []

    for iteration in range(1, num_iterations + 1):
        progress = (iteration - 1.0) / num_iterations

        # Anneal learning rate
        lr_now = args.learning_rate + (args.lr_end - args.learning_rate) * progress
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Anneal entropy coefficient
        ent_now = args.ent_coef + (args.ent_end - args.ent_coef) * progress

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

            action_np = action.cpu().numpy()
            step_obs, step_reward, terminations, truncations, infos = envs.step(
                action_np
            )
            next_done_np = np.logical_or(terminations, truncations)

            rewards[step] = torch.tensor(
                step_reward, dtype=torch.float32, device=device
            )
            next_obs = torch.tensor(
                step_obs, dtype=torch.float32, device=device
            )
            next_done = torch.tensor(
                next_done_np, dtype=torch.float32, device=device
            )
            step_mask = infos.get(
                "action_mask", np.ones((args.num_envs, action_dim), dtype=bool)
            )
            next_mask = torch.tensor(step_mask, dtype=torch.bool, device=device)

            # Log episode statistics
            if "episode" in infos and "_episode" in infos:
                for idx, is_done in enumerate(infos["_episode"]):
                    if is_done:
                        r = float(infos["episode"]["r"][idx])
                        l_val = int(infos["episode"]["l"][idx])
                        episode_returns.append(r)
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l_val, global_step)

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Flatten
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = masks.reshape((-1, action_dim))

        # PPO update
        b_inds = np.arange(batch_size)
        clipfracs: list[float] = []

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
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (
                    mb_advantages - mb_advantages.mean()
                ) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_now * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Logging
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("charts/learning_rate", lr_now, global_step)
        writer.add_scalar("charts/entropy_coef", ent_now, global_step)

        if iteration % 5 == 0 or iteration == num_iterations:
            recent_returns = episode_returns[-20:] if episode_returns else [0.0]
            mean_ret = np.mean(recent_returns)
            print(
                f"[{iteration:3d}/{num_iterations}] "
                f"Step: {global_step:>7,} | SPS: {sps:>4} | "
                f"Return: {mean_ret:>7.1f} | "
                f"PgLoss: {pg_loss.item():>7.4f} | "
                f"VLoss: {v_loss.item():>7.4f} | "
                f"Ent: {entropy_loss.item():>5.3f} | "
                f"ExplVar: {explained_var:>5.3f} | "
                f"LR: {lr_now:.2e} | EntC: {ent_now:.4f}",
                flush=True,
            )

        # Periodic validation and checkpoint
        if iteration % 20 == 0 or iteration == num_iterations:
            os.makedirs("models", exist_ok=True)
            ckpt_path = f"models/ppo_optimal_overtaker_{args.lanes}lane.pt"
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "iteration": iteration,
                "global_step": global_step,
                "args": vars(args),
            }, ckpt_path)

            val_results = evaluate(
                ckpt_path,
                lanes_count=args.lanes,
                vehicles_density=args.density,
                vehicles_count=args.vehicles,
                episodes=10,
                seed_start=3000,
                frame_stack_k=args.frame_stack,
                device=str(device),
            )

            eval_score = val_results["mean_speed_kmh"] - val_results["crash_rate"] * 2
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_path = f"models/ppo_optimal_overtaker_{args.lanes}lane_best.pt"
                torch.save({
                    "model_state_dict": agent.state_dict(),
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "args": vars(args),
                }, best_path)
                print(f"  ★ New best model saved: score={eval_score:.1f}", flush=True)

            writer.add_scalar("eval/crash_rate", val_results["crash_rate"], global_step)
            writer.add_scalar("eval/mean_speed_kmh", val_results["mean_speed_kmh"], global_step)
            writer.add_scalar("eval/mean_lane_changes", val_results["mean_lane_changes"], global_step)
            writer.add_scalar("eval/mean_overtakes", val_results["mean_overtakes"], global_step)

    envs.close()
    writer.close()

    # Final save
    final_path = f"models/ppo_optimal_overtaker_{args.lanes}lane.pt"
    torch.save({
        "model_state_dict": agent.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "args": vars(args),
    }, final_path)
    print(f"\n✓ Final model saved: {final_path}", flush=True)
    return final_path


def evaluate(
    model_path: str,
    lanes_count: int = 4,
    vehicles_density: float = 1.4,
    vehicles_count: int = 14,
    episodes: int = 30,
    seed_start: int = 2000,
    frame_stack_k: int = 3,
    device: str = "cpu",
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate trained optimal overtaking agent."""
    env = env_config.make_optimal_env(
        lanes_count=lanes_count,
        vehicles_density=vehicles_density,
        vehicles_count=vehicles_count,
        seed=seed_start,
        frame_stack_k=frame_stack_k,
    )

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["model_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    agent.eval()

    crashes = 0
    durations: list[int] = []
    speeds_mps: list[float] = []
    total_lane_changes: list[int] = []
    total_overtakes: list[int] = []

    for ep in range(episodes):
        seed = seed_start + ep
        obs, info = env.reset(seed=seed)
        ep_steps = 0
        ep_speeds: list[float] = []
        ep_lane_changes = 0
        ep_overtakes = 0
        crashed = False
        current_lane = (
            env.unwrapped.vehicle.lane_index[2]
            if hasattr(env.unwrapped, "vehicle")
            else None
        )

        while True:
            obs_tensor = torch.tensor(
                obs, dtype=torch.float32, device=device
            ).unsqueeze(0)
            mask = info.get("action_mask", np.ones(action_dim, dtype=bool))
            mask_t = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_tensor, deterministic=deterministic, action_mask=mask_t
                )
                action_idx = int(action.item())

            obs, reward, terminated, truncated, info = env.step(action_idx)
            ep_steps += 1

            # Track lane changes
            if hasattr(env.unwrapped, "vehicle"):
                new_lane = env.unwrapped.vehicle.lane_index[2]
                if current_lane is not None and new_lane != current_lane:
                    ep_lane_changes += 1
                current_lane = new_lane

            ep_speeds.append(float(info.get("speed", 0.0)))
            ep_overtakes += int(info.get("overtake_count", 0))

            if info.get("crashed", False):
                crashed = True

            if terminated or truncated:
                break

        if crashed:
            crashes += 1
        durations.append(ep_steps)
        speeds_mps.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        total_lane_changes.append(ep_lane_changes)
        total_overtakes.append(ep_overtakes)

    env.close()

    mean_speed_kmh = float(np.mean(speeds_mps) * 3.6) if speeds_mps else 0.0
    crash_rate = (crashes / episodes) * 100.0
    mean_duration = float(np.mean(durations))
    mean_lc = float(np.mean(total_lane_changes))
    mean_ot = float(np.mean(total_overtakes))

    results = {
        "lanes": lanes_count,
        "episodes": episodes,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{episodes}",
        "crash_rate": crash_rate,
        "mean_duration": mean_duration,
        "mean_speed_kmh": mean_speed_kmh,
        "mean_lane_changes": mean_lc,
        "mean_overtakes": mean_ot,
    }

    print(
        f"  [EVAL {lanes_count}L] crashes={crashes}/{episodes} ({crash_rate:5.1f}%) "
        f"speed={mean_speed_kmh:5.1f} km/h "
        f"duration={mean_duration:5.1f} "
        f"LC={mean_lc:.1f} OT={mean_ot:.1f}",
        flush=True,
    )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimal Tactical Overtaking PPO Agent"
    )
    # Environment
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument("--density", type=float, default=1.4)
    parser.add_argument("--vehicles", type=int, default=14)
    parser.add_argument("--frame-stack", type=int, default=3)

    # Training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-timesteps", type=int, default=600000)
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

    # Evaluation
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed-start", type=int, default=2000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.eval_only:
        if args.model_path is None:
            args.model_path = f"models/ppo_optimal_overtaker_{args.lanes}lane_best.pt"
        device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        evaluate(
            model_path=args.model_path,
            lanes_count=args.lanes,
            vehicles_density=args.density,
            vehicles_count=args.vehicles,
            episodes=args.eval_episodes,
            seed_start=args.eval_seed_start,
            frame_stack_k=args.frame_stack,
            device=device,
        )
    else:
        # Train and evaluate on specified lane config
        saved_path = train(args)
        device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

        print("\n" + "=" * 60)
        print(f"FINAL EVALUATION ({args.lanes}-lane, 30 seeds)")
        print("=" * 60)
        evaluate(
            model_path=saved_path,
            lanes_count=args.lanes,
            vehicles_density=args.density,
            vehicles_count=args.vehicles,
            episodes=30,
            seed_start=2000,
            frame_stack_k=args.frame_stack,
            device=device,
        )


if __name__ == "__main__":
    main()

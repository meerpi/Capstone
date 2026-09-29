"""Feedforward PPO (MLP) implementation for highway-env sensor ablation.

Trains and evaluates a memoryless Actor-Critic PPO policy across sensor tiers
using Gymnasium SyncVectorEnv and tier-masked observations from env_config.
"""

import argparse
import math
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
    scenario: str,
    tier: str,
    seed: int,
    tactical_overtaking: bool = True,
) -> Callable[[], gym.Env]:
    """Create a thunk for instantiating an environment in SyncVectorEnv.

    Args:
        scenario: Benchmark scenario name ('highway', 'merge', etc.).
        tier: Sensor tier ('full_adas', 'front_only', etc.).
        seed: Random seed for environment initialization.
        tactical_overtaking: If True, wraps with TacticalOvertakingWrapper.

    Returns:
        Callable returning a wrapped Gymnasium environment.
    """

    def thunk() -> gym.Env:
        env = env_config.make_env(
            scenario=scenario,
            tier=tier,
            seed=seed,
            tactical_overtaking=tactical_overtaking,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(
    layer: nn.Linear,
    std: float = np.sqrt(2),
    bias_const: float = 0.0,
) -> nn.Linear:
    """Initialize a linear layer with orthogonal weights and constant bias.

    Args:
        layer: PyTorch linear layer.
        std: Gain factor for orthogonal initialization.
        bias_const: Constant value for bias initialization.

    Returns:
        Initialized linear layer.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Two-layer feedforward Actor-Critic MLP policy for discrete action spaces."""

    def __init__(self, obs_dim: int, action_dim: int) -> None:
        """Initialize Actor and Critic networks with 256 hidden units each.

        Args:
            obs_dim: Dimension of flattened observation tensor (e.g. 50).
            action_dim: Number of discrete actions (e.g. 5 for highway).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
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
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value estimate V(s).

        Args:
            x: Observation tensor of shape (..., obs_dim) or (..., 10, 5).

        Returns:
            Estimated state values of shape (..., 1).
        """
        x_flat = x.reshape(-1, self.obs_dim)
        return self.critic(x_flat)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or evaluate actions and return log-probabilities and value.

        Args:
            x: Observation tensor.
            action: Optional action tensor to evaluate.
            deterministic: If True, selects greedy argmax action.
            action_mask: Optional boolean mask where True indicates a valid action.

        Returns:
            Tuple of (action, log_prob, entropy, value).
        """
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


def train_ppo(args: argparse.Namespace) -> str:
    """Execute PPO policy optimization and save model weights.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Path to the saved model checkpoint.
    """
    batch_size = int(args.num_envs * args.num_steps)
    minibatch_size = int(batch_size // args.num_minibatches)
    num_iterations = int(args.total_timesteps // batch_size)

    run_name = (
        f"ppo_{args.scenario}_{args.tier}_seed{args.seed}_{int(time.time())}"
    )
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    env_fns = [
        make_env_thunk(args.scenario, args.tier, args.seed + i)
        for i in range(args.num_envs)
    ]
    vec_cls = (
        gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    )
    envs = vec_cls(env_fns)

    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(envs.single_action_space.n)

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    opt_actor = optim.Adam(
        agent.actor.parameters(), lr=args.learning_rate, eps=1e-5
    )
    opt_critic = optim.Adam(
        agent.critic.parameters(), lr=args.learning_rate, eps=1e-5
    )

    # Rollout storage tensors
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)
    masks = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), dtype=torch.bool, device=device
    )
    terms = torch.zeros((args.num_steps, args.num_envs), device=device)
    truncs = torch.zeros((args.num_steps, args.num_envs), device=device)
    trunc_bootstrap_values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    next_obs, next_info = envs.reset(seed=args.seed)
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)
    init_mask = next_info.get(
        "action_mask", np.ones((args.num_envs, action_dim), dtype=bool)
    )
    next_mask = torch.tensor(init_mask, dtype=torch.bool, device=device)

    # Resume support (D1-1)
    start_iteration = 1
    last_save_step = 0
    if getattr(args, "resume", None) and os.path.exists(args.resume):
        ckpt_data = torch.load(args.resume, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt_data["model_state_dict"])
        if "optimizer_actor_state_dict" in ckpt_data:
            opt_actor.load_state_dict(ckpt_data["optimizer_actor_state_dict"])
        if "optimizer_critic_state_dict" in ckpt_data:
            opt_critic.load_state_dict(ckpt_data["optimizer_critic_state_dict"])
        global_step = ckpt_data.get("global_step", 0)
        start_iteration = ckpt_data.get("iteration", 0) + 1
        last_save_step = global_step
        if "rng_state" in ckpt_data:
            rng = ckpt_data["rng_state"]
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"].cpu())
            if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])
            if "numpy" in rng and rng["numpy"] is not None:
                np.random.set_state(rng["numpy"])
            if "python" in rng and rng["python"] is not None:
                random.setstate(rng["python"])
        print(f"✓ Resumed PPO from step {global_step} (iteration {start_iteration - 1})", flush=True)

    print(
        f"--- Training PPO: scenario={args.scenario}, tier={args.tier} ---",
        flush=True,
    )
    print(
        f"Budget: {args.total_timesteps:,} steps | Device: {device} | "
        f"Rollout: {args.num_envs} envs x {args.num_steps} steps | "
        f"Minibatch: {minibatch_size} | Epochs: {args.update_epochs}",
        flush=True,
    )

    for iteration in range(start_iteration, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            lrnow = frac * args.learning_rate
            opt_actor.param_groups[0]["lr"] = lrnow
            opt_critic.param_groups[0]["lr"] = lrnow

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

            terms[step] = torch.tensor(terminations, dtype=torch.float32, device=device)
            truncs[step] = torch.tensor(truncations, dtype=torch.float32, device=device)

            if getattr(args, "bootstrap_truncation", False):
                is_trunc = np.logical_and(truncations, np.logical_not(terminations))
                if np.any(is_trunc):
                    for env_idx in np.where(is_trunc)[0]:
                        if "final_obs" in infos and infos.get("_final_obs", [False] * args.num_envs)[env_idx]:
                            final_o = infos["final_obs"][env_idx]
                        elif "final_observation" in infos and infos.get("_final_observation", [False] * args.num_envs)[env_idx]:
                            final_o = infos["final_observation"][env_idx]
                        else:
                            final_o = step_obs[env_idx]
                        final_o_t = torch.tensor(final_o, dtype=torch.float32, device=device).unsqueeze(0)
                        with torch.no_grad():
                            trunc_bootstrap_values[step, env_idx] = agent.get_value(final_o_t).squeeze()

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
                        l = int(infos["episode"]["l"][idx])
                        writer.add_scalar("charts/episodic_return", r, global_step)
                        writer.add_scalar("charts/episodic_length", l, global_step)

        # Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if getattr(args, "bootstrap_truncation", False):
                    # Truncation Bootstrapping (B7-1 fix)
                    if t == args.num_steps - 1:
                        is_term = terms[t]
                        is_trunc = truncs[t]
                        nextnonterminal = 1.0 - is_term
                        nextvalues = torch.where(
                            (is_trunc.bool() & ~is_term.bool()),
                            trunc_bootstrap_values[t],
                            next_value.squeeze(0),
                        )
                        nextnoncutoff = 1.0 - torch.clamp(is_term + is_trunc, 0.0, 1.0)
                    else:
                        is_term = terms[t]
                        is_trunc = truncs[t]
                        nextnonterminal = 1.0 - is_term
                        nextvalues = torch.where(
                            (is_trunc.bool() & ~is_term.bool()),
                            trunc_bootstrap_values[t],
                            values[t + 1],
                        )
                        nextnoncutoff = 1.0 - torch.clamp(is_term + is_trunc, 0.0, 1.0)

                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta + args.gamma * args.gae_lambda * nextnoncutoff * lastgaelam
                    )
                else:
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

        # Flatten rollout batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_masks = masks.reshape((-1, action_dim))

        # Policy and Value Network Updates
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
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std() + 1e-8)

                # Policy surrogate loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                actor_loss = pg_loss - args.ent_coef * entropy_loss
                opt_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), args.max_grad_norm
                )
                opt_actor.step()

                opt_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.critic.parameters(), args.max_grad_norm
                )
                opt_critic.step()

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

        if iteration % 10 == 0 or iteration == num_iterations:
            print(
                f"[Iter {iteration:3d}/{num_iterations}] "
                f"Step: {global_step:,} | SPS: {sps} | "
                f"PolicyLoss: {pg_loss.item():.4f} | "
                f"ValLoss: {v_loss.item():.4f} | "
                f"ExplVar: {explained_var:.3f}",
                flush=True,
            )

        # Checkpointing (D1-1)
        if getattr(args, "checkpoint_dir", None) and (global_step - last_save_step >= getattr(args, "save_frequency", 50000)):
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            ckpt_data = {
                "model_state_dict": agent.state_dict(),
                "optimizer_actor_state_dict": opt_actor.state_dict(),
                "optimizer_critic_state_dict": opt_critic.state_dict(),
                "global_step": global_step,
                "iteration": iteration,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "args": vars(args),
                "rng_state": {
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "numpy": np.random.get_state(),
                    "python": random.getstate(),
                },
            }
            step_ckpt_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{global_step}.pt")
            latest_ckpt_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
            torch.save(ckpt_data, step_ckpt_path)
            torch.save(ckpt_data, latest_ckpt_path)
            last_save_step = global_step
            print(f"✓ Checkpoint saved: {step_ckpt_path} (step {global_step})", flush=True)

        if iteration % 25 == 0:
            val_dir = getattr(args, "checkpoint_dir", None) or "scratch/checkpoints"
            os.makedirs(val_dir, exist_ok=True)
            ckpt_path = os.path.join(val_dir, f"val_ppo_{args.scenario}_{args.tier}_seed{args.seed}.pt")
            torch.save(agent.state_dict(), ckpt_path)
            quick_val = evaluate_ppo_policy(
                ckpt_path,
                args.scenario,
                args.tier,
                episodes=5,
                seed_start=3000,
                device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
            )
            print(
                f"--- [PPO VAL @ Step {global_step:,}] Crashes: {quick_val['crash_fraction']} ({quick_val['crash_rate']:.1f}%) | "
                f"Speed: {quick_val['mean_speed_kmh']:.1f} km/h | "
                f"Duration: {quick_val['mean_duration']:.1f} | "
                f"LC: {quick_val['mean_lane_changes']:.2f} ---",
                flush=True,
            )

    envs.close()
    writer.close()

    # Tooling fix: Guard canonical models/ directory against dry-runs / smoke-tests
    if getattr(args, "output_path", None) is not None:
        model_path = args.output_path
    elif getattr(args, "total_timesteps", 1200000) < 100000 or getattr(args, "dry_run", False):
        os.makedirs("scratch/checkpoints", exist_ok=True)
        model_path = f"scratch/checkpoints/ppo_{args.scenario}_{args.tier}_seed{args.seed}.pt"
    else:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/ppo_{args.scenario}_{args.tier}_seed{args.seed}.pt"

    ckpt_to_save = {
        "model_state_dict": agent.state_dict(),
        "optimizer_actor_state_dict": opt_actor.state_dict(),
        "optimizer_critic_state_dict": opt_critic.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "iteration": num_iterations,
        "global_step": global_step,
        "args": vars(args),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(ckpt_to_save, model_path)
    print(f"Checkpoint saved successfully: {model_path}", flush=True)
    return model_path


def evaluate_ppo_policy(
    model_path: str,
    scenario: str = "highway",
    tier: str = "full_adas",
    episodes: int = 30,
    seed_start: int = 2000,
    device: str = "cpu",
    deterministic: bool = True,
) -> dict[str, Any]:
    """Evaluate a trained PPO policy across deterministic test seeds.

    Args:
        model_path: Path to the .pt checkpoint file.
        scenario: Benchmark scenario identifier.
        tier: Sensor tier ('full_adas', 'front_only', etc.).
        episodes: Number of evaluation episodes.
        seed_start: Initial random seed.
        device: PyTorch device ('cpu' or 'cuda').
        deterministic: If True, actions are selected greedily (argmax).

    Returns:
        Dictionary of summary evaluation statistics.
    """
    raw_ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = (
        raw_ckpt["model_state_dict"]
        if isinstance(raw_ckpt, dict) and "model_state_dict" in raw_ckpt
        else raw_ckpt
    )

    if isinstance(raw_ckpt, dict) and "obs_dim" in raw_ckpt:
        ckpt_obs_dim = raw_ckpt["obs_dim"]
    elif "actor.0.weight" in state_dict:
        ckpt_obs_dim = state_dict["actor.0.weight"].shape[1]
    elif "critic.0.weight" in state_dict:
        ckpt_obs_dim = state_dict["critic.0.weight"].shape[1]
    else:
        ckpt_obs_dim = 75

    env_overrides: dict[str, Any] = {}
    if ckpt_obs_dim == 50:
        env_overrides["observation"] = {"vehicles_count": 10}
        env_overrides["vehicles_count"] = 10
        env_overrides["vehicles_density"] = 1.0
        env_overrides["duration"] = 40
        obs_dim = 50
    elif ckpt_obs_dim == 75:
        env_overrides["observation"] = {"vehicles_count": 15}
        env_overrides["vehicles_count"] = 14
        env_overrides["vehicles_density"] = 1.4
        env_overrides["duration"] = 100
        obs_dim = 75
    else:
        dummy_env = env_config.make_env(scenario=scenario, tier=tier, seed=0)
        obs_dim = int(np.prod(dummy_env.observation_space.shape))
        dummy_env.close()

    env = env_config.make_env(
        scenario=scenario,
        tier=tier,
        seed=seed_start,
        tactical_overtaking=True,
        **env_overrides,
    )
    obs_shape = env.observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = int(env.action_space.n)

    agent = Agent(obs_dim=obs_dim, action_dim=action_dim).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    crashes = 0
    durations: list[int] = []
    speeds_mps: list[float] = []
    total_lane_changes: list[int] = []

    for ep in range(episodes):
        seed = seed_start + ep
        if ep > 0:
            env = env_config.make_env(
                scenario=scenario,
                tier=tier,
                seed=seed,
                tactical_overtaking=True,
                **env_overrides,
            )
        obs, info = env.reset(seed=seed)
        ep_steps = 0
        ep_speeds: list[float] = []
        ep_lane_changes = 0
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

            if hasattr(env.unwrapped, "vehicle"):
                new_lane = env.unwrapped.vehicle.lane_index[2]
                if current_lane is not None and new_lane != current_lane:
                    ep_lane_changes += 1
                current_lane = new_lane

            speed_val = float(info.get("speed", 0.0))
            ep_speeds.append(speed_val)

            if info.get("crashed", False) or (terminated and not truncated):
                crashed = True

            if terminated or truncated:
                break

        if crashed:
            crashes += 1

        durations.append(ep_steps)
        speeds_mps.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        total_lane_changes.append(ep_lane_changes)

    env.close()

    mean_speed_kmh = float(np.mean(speeds_mps) * 3.6) if speeds_mps else 0.0
    crash_rate = (crashes / episodes) * 100.0
    mean_duration = float(np.mean(durations))
    std_duration = float(np.std(durations))
    mean_lane_changes = float(np.mean(total_lane_changes))

    results = {
        "scenario": scenario,
        "tier": tier,
        "model_path": model_path,
        "episodes": episodes,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{episodes}",
        "crash_rate": crash_rate,
        "mean_duration": mean_duration,
        "std_duration": std_duration,
        "mean_speed_kmh": mean_speed_kmh,
        "mean_lane_changes": mean_lane_changes,
    }

    print(
        f"[EVAL RESULT] scenario={scenario:<10} tier={tier:<12} "
        f"crashes={crashes}/{episodes} ({crash_rate:5.1f}%) "
        f"duration={mean_duration:5.1f}+/-{std_duration:4.1f} "
        f"speed={mean_speed_kmh:5.1f} km/h "
        f"lane_changes={mean_lane_changes:.2f}",
        flush=True,
    )
    return results


def parse_args() -> argparse.Namespace:
    """Parse and return command-line configuration arguments.

    Returns:
        Namespace with parsed parameters.
    """
    parser = argparse.ArgumentParser(
        description="Feedforward PPO Baseline for Highway Driving"
    )
    parser.add_argument("--scenario", type=str, default="highway")
    parser.add_argument(
        "--tier",
        type=str,
        default="full_adas",
        choices=["full_adas", "front_only", "short_range", "no_velocity"],
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--total-timesteps", type=int, default=1200000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=512)
    parser.add_argument("--anneal-lr", action="store_true", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", action="store_true", default=True)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true", default=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument(
        "--async-envs",
        dest="async_envs",
        action="store_true",
        default=True,
        help="Use AsyncVectorEnv for multi-core rollout speedup (default: True).",
    )
    parser.add_argument(
        "--sync-envs",
        dest="async_envs",
        action="store_false",
        help="Use SyncVectorEnv instead of AsyncVectorEnv.",
    )

    # Truncation Bootstrapping (B7-1)
    parser.add_argument(
        "--bootstrap-truncation",
        action="store_true",
        default=False,
        help="Bootstrap value of final observation upon episode truncation (B7-1 fix).",
    )

    # Checkpointing and Resuming (D1-1)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save periodic training checkpoints.",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=50000,
        help="Save checkpoint every N environment steps.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint from which to resume training.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Explicit output path for model checkpoint.",
    )

    # Evaluation options
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--eval-seed-start", type=int, default=2000)
    parser.add_argument("--eval-deterministic", action="store_true", default=True)

    return parser.parse_args()


def main() -> None:
    """Main execution function for training and evaluating PPO agent."""
    args = parse_args()

    if args.eval_only:
        if args.model_path is None:
            args.model_path = (
                f"models/ppo_{args.scenario}_{args.tier}_seed{args.seed}.pt"
            )
        evaluate_ppo_policy(
            model_path=args.model_path,
            scenario=args.scenario,
            tier=args.tier,
            episodes=args.eval_episodes,
            seed_start=args.eval_seed_start,
            device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
            deterministic=args.eval_deterministic,
        )
    else:
        saved_model_path = train_ppo(args)
        evaluate_ppo_policy(
            model_path=saved_model_path,
            scenario=args.scenario,
            tier=args.tier,
            episodes=args.eval_episodes,
            seed_start=args.eval_seed_start,
            device="cuda" if torch.cuda.is_available() and args.cuda else "cpu",
            deterministic=args.eval_deterministic,
        )


if __name__ == "__main__":
    main()
"""Recurrent Proximal Policy Optimization (PPO-LSTM) for Sensor-Budgeted Highway Driving.

Implements Truncated Backpropagation Through Time (T-BPTT) with:
- Decoupled Actor and Critic recurrent networks (separate LSTMs to prevent gradient interference)
- Invalid Action Masking (Mechanism C) for boundary compliance
- Tactical Overtaking (Mechanisms A & B) via env_config.TacticalOvertakingWrapper
- Full evaluation across 30 benchmark test seeds (2000-2029)
"""

import argparse
import json
import os
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import env_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PPO-LSTM training."""
    parser = argparse.ArgumentParser(description="Recurrent PPO for Highway Driving")
    parser.add_argument("--exp_name", type=str, default="ppo_lstm", help="Experiment name")
    parser.add_argument("--scenario", type=str, default="highway", help="Driving scenario")
    parser.add_argument(
        "--tier",
        type=str,
        default="front_only",
        choices=["all", "full_adas", "front_only", "short_range", "no_velocity"],
        help="Sensor budget tier",
    )
    parser.add_argument("--seed", type=int, default=101, help="Random seed")
    parser.add_argument(
        "--total_timesteps", type=int, default=1200000, help="Total training steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Adam learning rate"
    )
    parser.add_argument(
        "--num_envs", type=int, default=12, help="Parallel environments (12 CPU workers)"
    )
    parser.add_argument(
        "--num_steps", type=int, default=256, help="Rollout steps per env (BPTT window)"
    )
    parser.add_argument(
        "--anneal_lr", type=bool, default=True, help="Toggle linear learning rate annealing"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument(
        "--gae_lambda", type=float, default=0.95, help="Lambda for GAE estimation"
    )
    parser.add_argument(
        "--num_minibatches", type=int, default=4, help="Number of minibatches per epoch"
    )
    parser.add_argument(
        "--update_epochs", type=int, default=4, help="Number of update epochs"
    )
    parser.add_argument(
        "--norm_adv", type=bool, default=True, help="Toggle advantage normalization"
    )
    parser.add_argument(
        "--clip_coef", type=float, default=0.2, help="Surrogate clipping coefficient"
    )
    parser.add_argument(
        "--clip_vloss", type=bool, default=True, help="Toggle clipped value loss"
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="Entropy bonus coefficient"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.5, help="Maximum gradient norm"
    )
    parser.add_argument("--cuda", type=bool, default=True, help="Toggle CUDA acceleration")
    return parser.parse_args()


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize linear layers with orthogonal weights and constant biases."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def lstm_init(lstm: nn.LSTM) -> nn.LSTM:
    """Initialize LSTM weights orthogonally with zero biases."""
    for name, param in lstm.named_parameters():
        if "bias" in name:
            nn.init.constant_(param, 0.0)
        elif "weight" in name:
            nn.init.orthogonal_(param, 1.0)
    return lstm


class RecurrentActor(nn.Module):
    """Recurrent policy network with independent LSTM memory for action selection."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.lstm = lstm_init(nn.LSTM(hidden_dim, hidden_dim))
        self.actor_head = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

    def forward(
        self,
        x: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute action, log probability, entropy, and updated LSTM state."""
        hidden = self.feature_net(x.reshape(-1, self.obs_dim))

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.hidden_dim))
        done = done.reshape((-1, batch_size))
        new_hidden: list[torch.Tensor] = []

        for h, d in zip(hidden, done):
            h_step, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h_step)

        flat_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        logits = self.actor_head(flat_hidden)

        if action_mask is not None:
            mask_flat = action_mask.reshape(-1, logits.shape[-1])
            logits = torch.where(
                mask_flat,
                logits,
                torch.tensor(-1e8, device=logits.device, dtype=logits.dtype),
            )

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), lstm_state


class RecurrentCritic(nn.Module):
    """Recurrent value network with independent LSTM memory for state value estimation."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.feature_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.lstm = lstm_init(nn.LSTM(hidden_dim, hidden_dim))
        self.critic_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(
        self,
        x: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Compute state value and updated LSTM state."""
        hidden = self.feature_net(x.reshape(-1, self.obs_dim))

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.hidden_dim))
        done = done.reshape((-1, batch_size))
        new_hidden: list[torch.Tensor] = []

        for h, d in zip(hidden, done):
            h_step, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden.append(h_step)

        flat_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        values = self.critic_head(flat_hidden)
        return values, lstm_state


class RecurrentAgent(nn.Module):
    """Full decoupled Actor-Critic Agent with recurrent LSTM backbones."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor = RecurrentActor(obs_dim, action_dim, hidden_dim)
        self.critic = RecurrentCritic(obs_dim, hidden_dim)

    def get_initial_states(
        self, batch_size: int, device: torch.device
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Generate initial zero hidden and cell states for both LSTMs."""
        actor_state = (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )
        critic_state = (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
        )
        return actor_state, critic_state

    def get_value(
        self,
        x: torch.Tensor,
        critic_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
    ) -> torch.Tensor:
        """Compute state value."""
        values, _ = self.critic(x, critic_state, done)
        return values

    def get_action_and_value(
        self,
        x: torch.Tensor,
        actor_state: tuple[torch.Tensor, torch.Tensor],
        critic_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        """Compute action, logprob, entropy, value, and next states."""
        action, logprob, entropy, next_actor_state = self.actor(
            x, actor_state, done, action_mask=action_mask, action=action
        )
        values, next_critic_state = self.critic(x, critic_state, done)
        return (
            action,
            logprob,
            entropy,
            values,
            next_actor_state,
            next_critic_state,
        )


def make_env_thunk(
    scenario: str,
    tier: str,
    seed: int,
    tactical_overtaking: bool = True,
):
    """Return a zero-argument callable that creates the environment."""
    def thunk() -> gym.Env:
        return env_config.make_env(
            scenario=scenario,
            tier=tier,
            seed=seed,
            tactical_overtaking=tactical_overtaking,
        )
    return thunk


def train_ppo_lstm(args: argparse.Namespace) -> str:
    """Train Recurrent PPO with T-BPTT on highway-env across specified sensor tier."""
    args.batch_size = int(args.num_envs * args.num_steps)
    assert args.num_envs % args.num_minibatches == 0, (
        f"num_envs ({args.num_envs}) must be divisible by num_minibatches ({args.num_minibatches})"
    )
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = int(args.total_timesteps // args.batch_size)

    run_name = (
        f"ppo_lstm_{args.scenario}_{args.tier}_seed{args.seed}_{int(time.time())}"
    )
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print("=" * 80, flush=True)
    print(
        f"--- Training Recurrent PPO (LSTM): scenario={args.scenario}, tier={args.tier} ---",
        flush=True,
    )
    print(
        f"Budget: {args.total_timesteps:,} steps | Device: {device} | "
        f"Rollout: {args.num_envs} envs x {args.num_steps} steps | "
        f"Batch Size: {args.batch_size:,} | Minibatches: {args.num_minibatches} "
        f"(envs per batch: {args.num_envs // args.num_minibatches})",
        flush=True,
    )
    print("=" * 80, flush=True)

    # 12 parallel environments via AsyncVectorEnv
    env_fns = [
        make_env_thunk(
            scenario=args.scenario,
            tier=args.tier,
            seed=args.seed + i * 1000,
            tactical_overtaking=True,
        )
        for i in range(args.num_envs)
    ]
    envs = gym.vector.AsyncVectorEnv(env_fns)

    obs_shape = envs.single_observation_space.shape
    obs_dim = int(np.prod(obs_shape))
    action_dim = envs.single_action_space.n

    agent = RecurrentAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(
        device
    )

    # Decoupled optimizers for actor and critic
    optimizer_actor = optim.Adam(
        agent.actor.parameters(), lr=args.learning_rate, eps=1e-5
    )
    optimizer_critic = optim.Adam(
        agent.critic.parameters(), lr=args.learning_rate, eps=1e-5
    )

    # Rollout buffer storage
    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    masks_buf = torch.zeros(
        (args.num_steps, args.num_envs, action_dim), dtype=torch.bool, device=device
    )

    global_step = 0
    start_time = time.time()

    next_obs_np, next_info = envs.reset(seed=args.seed)
    next_obs = torch.tensor(
        next_obs_np.reshape(args.num_envs, obs_dim), dtype=torch.float32, device=device
    )
    next_done = torch.zeros(args.num_envs, device=device)

    # Extract initial action mask
    if "action_mask" in next_info:
        next_mask = torch.tensor(
            next_info["action_mask"], dtype=torch.bool, device=device
        )
    else:
        next_mask = torch.ones(
            (args.num_envs, action_dim), dtype=torch.bool, device=device
        )

    next_actor_state, next_critic_state = agent.get_initial_states(
        batch_size=args.num_envs, device=device
    )

    for iteration in range(1, args.num_iterations + 1):
        # Cache initial states for T-BPTT minibatch chunking
        initial_actor_state = (
            next_actor_state[0].clone(),
            next_actor_state[1].clone(),
        )
        initial_critic_state = (
            next_critic_state[0].clone(),
            next_critic_state[1].clone(),
        )

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_actor.param_groups[0]["lr"] = lrnow
            optimizer_critic.param_groups[0]["lr"] = lrnow

        # Rollout collection loop
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            masks_buf[step] = next_mask

            with torch.no_grad():
                (
                    action,
                    logprob,
                    _,
                    value,
                    next_actor_state,
                    next_critic_state,
                ) = agent.get_action_and_value(
                    next_obs,
                    next_actor_state,
                    next_critic_state,
                    next_done,
                    action_mask=next_mask,
                )
                values_buf[step] = value.flatten()

            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, term, trunc, info = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(term, trunc)

            rewards_buf[step] = torch.tensor(
                reward, dtype=torch.float32, device=device
            ).view(-1)
            next_obs = torch.tensor(
                next_obs_np.reshape(args.num_envs, obs_dim),
                dtype=torch.float32,
                device=device,
            )
            next_done = torch.tensor(
                next_done_np, dtype=torch.float32, device=device
            )

            if "action_mask" in info:
                next_mask = torch.tensor(
                    info["action_mask"], dtype=torch.bool, device=device
                )
            else:
                next_mask = torch.ones(
                    (args.num_envs, action_dim), dtype=torch.bool, device=device
                )

        # GAE Advantage Estimation
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs, next_critic_state, next_done
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = (
                    rewards_buf[t]
                    + args.gamma * nextvalues * nextnonterminal
                    - values_buf[t]
                )
                advantages[t] = lastgaelam = (
                    delta
                    + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values_buf

        # Flatten buffers for batch indexing
        b_obs = obs_buf.reshape((-1, obs_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_dones = dones_buf.reshape(-1)
        b_masks = masks_buf.reshape((-1, action_dim))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Recurrent Minibatch Chunking: Preserving contiguous env trajectories
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                # Slice cached states for this minibatch of environments
                mb_actor_state = (
                    initial_actor_state[0][:, mbenvinds],
                    initial_actor_state[1][:, mbenvinds],
                )
                mb_critic_state = (
                    initial_critic_state[0][:, mbenvinds],
                    initial_critic_state[1][:, mbenvinds],
                )

                (
                    _,
                    newlogprob,
                    entropy,
                    newvalue,
                    _,
                    _,
                ) = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mb_actor_state,
                    mb_critic_state,
                    b_dones[mb_inds],
                    action_mask=b_masks[mb_inds],
                    action=b_actions.long()[mb_inds],
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                entropy_loss = entropy.mean()
                actor_loss = pg_loss - args.ent_coef * entropy_loss

                # Value Loss
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

                # Decoupled backward pass & independent gradient clipping
                optimizer_actor.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent.actor.parameters(), args.max_grad_norm)
                optimizer_actor.step()

                optimizer_critic.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                optimizer_critic.step()

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            np.nan if var_y == 0 else 1.0 - np.var(y_true - y_pred) / var_y
        )
        sps = int(global_step / (time.time() - start_time))

        if iteration % 20 == 0 or iteration == args.num_iterations:
            print(
                f"[Iter {iteration:3d}/{args.num_iterations}] "
                f"Step: {global_step:,} | SPS: {sps} | "
                f"PolicyLoss: {pg_loss.item():.4f} | "
                f"ValLoss: {v_loss.item():.4f} | "
                f"ExplVar: {explained_var:.3f}",
                flush=True,
            )

        if iteration % 50 == 0:
            os.makedirs("models", exist_ok=True)
            ckpt_path = f"models/ppo_lstm_{args.scenario}_{args.tier}_seed{args.seed}.pt"
            torch.save(agent.state_dict(), ckpt_path)
            quick_val = evaluate_ppo_lstm_policy(
                ckpt_path, args.scenario, args.tier, episodes=5, seed_start=3000
            )
            print(
                f"--- [VAL @ Step {global_step:,}] Crashes: {quick_val['crash_fraction']} ({quick_val['crash_rate']:.1f}%) | "
                f"Speed: {quick_val['mean_speed_kmh']:.1f} km/h | "
                f"Duration: {quick_val['mean_duration']:.1f} | "
                f"LC: {quick_val['mean_lane_changes']:.2f} ---",
                flush=True,
            )

    envs.close()
    writer.close()

    os.makedirs("models", exist_ok=True)
    model_path = f"models/ppo_lstm_{args.scenario}_{args.tier}_seed{args.seed}.pt"
    torch.save(agent.state_dict(), model_path)
    print(f"Checkpoint saved successfully: {model_path}", flush=True)
    return model_path


def evaluate_ppo_lstm_policy(
    model_path: str,
    scenario: str = "highway",
    tier: str = "full_adas",
    episodes: int = 30,
    seed_start: int = 2000,
) -> dict[str, Any]:
    """Evaluate trained Recurrent PPO on deterministic benchmark test seeds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load agent
    dummy_env = env_config.make_env(scenario=scenario, tier=tier, seed=0)
    obs_dim = int(np.prod(dummy_env.observation_space.shape))
    action_dim = dummy_env.action_space.n
    dummy_env.close()

    agent = RecurrentAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=128).to(
        device
    )
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    crashes = 0
    durations: list[float] = []
    speeds: list[float] = []
    lane_changes_list: list[int] = []

    for ep in range(episodes):
        test_seed = seed_start + ep
        env = env_config.make_env(
            scenario=scenario, tier=tier, seed=test_seed, tactical_overtaking=True
        )
        obs_np, info = env.reset(seed=test_seed)

        actor_state, critic_state = agent.get_initial_states(
            batch_size=1, device=device
        )
        done = False
        step_count = 0
        speed_sum = 0.0
        lane_changes = 0

        unwrapped = env.unwrapped
        last_lane = getattr(unwrapped.vehicle, "lane_index", [0, 0, 0])[2]

        while not done:
            obs_tensor = torch.tensor(
                obs_np.reshape(1, obs_dim), dtype=torch.float32, device=device
            )
            done_tensor = torch.zeros(1, device=device)

            if "action_mask" in info:
                mask_tensor = torch.tensor(
                    info["action_mask"], dtype=torch.bool, device=device
                ).unsqueeze(0)
            else:
                mask_tensor = torch.ones(
                    (1, action_dim), dtype=torch.bool, device=device
                )

            with torch.no_grad():
                # Greedy deterministic evaluation
                hidden = agent.actor.feature_net(obs_tensor)
                h_step, actor_state = agent.actor.lstm(
                    hidden.unsqueeze(0),
                    (actor_state[0], actor_state[1]),
                )
                logits = agent.actor.actor_head(h_step.squeeze(0))
                logits = torch.where(
                    mask_tensor,
                    logits,
                    torch.tensor(-1e8, device=device, dtype=logits.dtype),
                )
                action = int(torch.argmax(logits, dim=-1).item())

            obs_np, _, term, trunc, info = env.step(action)
            done = term or trunc
            step_count += 1

            veh = unwrapped.vehicle
            speed_sum += float(getattr(veh, "speed", 0.0))
            current_lane = getattr(veh, "lane_index", [0, 0, 0])[2]
            if current_lane != last_lane:
                lane_changes += 1
                last_lane = current_lane

            if term and getattr(veh, "crashed", False):
                crashes += 1
                break

        durations.append(float(step_count))
        speeds.append(float(speed_sum / max(1, step_count)) * 3.6)
        lane_changes_list.append(lane_changes)
        env.close()

    result = {
        "scenario": scenario,
        "tier": tier,
        "model_path": model_path,
        "episodes": episodes,
        "crashes": crashes,
        "crash_fraction": f"{crashes}/{episodes}",
        "crash_rate": (crashes / episodes) * 100.0,
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "mean_speed_kmh": float(np.mean(speeds)),
        "mean_lane_changes": float(np.mean(lane_changes_list)),
    }

    print(
        f"[LSTM EVAL RESULT] scenario={scenario:<10} tier={tier:<11} "
        f"crashes={result['crash_fraction']} ({result['crash_rate']:5.1f}%) "
        f"duration={result['mean_duration']:5.1f}+/-{result['std_duration']:4.1f} "
        f"speed={result['mean_speed_kmh']:5.1f} km/h "
        f"lane_changes={result['mean_lane_changes']:.2f}",
        flush=True,
    )
    return result


def main() -> None:
    """Execute complete recurrent PPO training and evaluation benchmark."""
    args = parse_args()
    tiers = ["full_adas", "front_only"] if args.tier == "all" else [args.tier]
    benchmark_results: dict[str, Any] = {}

    for tier in tiers:
        args.tier = tier
        t0 = time.time()
        model_path = train_ppo_lstm(args)
        train_time = time.time() - t0

        print(f"Evaluating Recurrent PPO {tier} on 30 test seeds...", flush=True)
        eval_res = evaluate_ppo_lstm_policy(
            model_path=model_path,
            scenario=args.scenario,
            tier=tier,
            episodes=30,
            seed_start=2000,
        )
        eval_res["train_duration_sec"] = train_time
        eval_res["policy"] = "PPO + LSTM"
        benchmark_results[f"ppo_lstm_{tier}"] = eval_res

    with open("ppo_lstm_results.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("\n" + "=" * 80)
    print("RECURRENT PPO (LSTM) BENCHMARK COMPLETE")
    print("=" * 80)
    for key, res in benchmark_results.items():
        print(
            f"{res['tier']:<12} | Crashes: {res['crash_fraction']} ({res['crash_rate']:.1f}%) | "
            f"Duration: {res['mean_duration']:.1f} +/- {res['std_duration']:.1f} | "
            f"Speed: {res['mean_speed_kmh']:.1f} km/h | "
            f"Lane Changes: {res['mean_lane_changes']:.2f}"
        )
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
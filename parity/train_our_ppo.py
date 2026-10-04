"""Our PPO implementation runner for SOTA parity ladder.

Uses our project's Actor-Critic MLP architecture (from ppo.py or identical PyTorch Agent),
trained with identical hyperparameters to SB3:
- num_envs: 6
- num_steps: 128 (rollout buffer: 768)
- minibatch_size: 64 (12 minibatches per epoch)
- update_epochs: 10
- learning_rate: 5e-4 (Adam eps=1e-5)
- gamma: 0.8 (R0-R4) / 0.956, 0.995 (R5)
- gae_lambda: 0.95
- clip_coef: 0.2
- ent_coef: 0.0
- vf_coef: 0.5
- max_grad_norm: 0.5
- Checkpoint saving strictly under parity/checkpoints/ (models/ is read-only)
- Evaluation on seeds 6000-6099
"""

import argparse
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

sys.path.insert(0, ".")
from parity.env_factory import make_ladder_env
from parity.evaluator import evaluate_policy


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class OurPPOAgent(nn.Module):
    """Matches ppo.py / SB3 MLP architecture: 2-layer MLP (256, 256) with Tanh()."""

    def __init__(self, obs_dim: int, action_dim: int):
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
        x_flat = x.reshape(-1, self.obs_dim)
        return self.critic(x_flat)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        x_flat = x.reshape(-1, self.obs_dim)
        logits = self.actor(x_flat)
        probs = Categorical(logits=logits)
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x_flat)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        self.eval()
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
            elif x.ndim == 2:
                # e.g. (5, 5) -> flatten to (1, 25)
                x = x.reshape(1, -1)
            action, _, _, _ = self.get_action_and_value(x, deterministic=deterministic)
            return int(action.item())


def train_and_eval_our_ppo(
    rung: str,
    seed: int,
    total_timesteps: int = 200000,
    gamma: float = 0.8,
    device: str = "cpu",
):
    save_dir = Path("parity/checkpoints") / f"our_{rung}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.pt"

    # Seed RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    num_envs = 6
    num_steps = 128
    batch_size = num_envs * num_steps  # 768
    minibatch_size = 64
    num_minibatches = batch_size // minibatch_size  # 12
    update_epochs = 10
    learning_rate = 5e-4
    gae_lambda = 0.95
    clip_coef = 0.2
    clip_vloss = True
    ent_coef = 0.0
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Environment setup
    def make_thunk(env_seed):
        def thunk():
            return make_ladder_env(rung, seed=env_seed)
        return thunk

    envs = gym.vector.AsyncVectorEnv([make_thunk(seed + i * 100) for i in range(num_envs)])
    single_obs_space = envs.single_observation_space
    single_act_space = envs.single_action_space

    obs_dim = int(np.prod(single_obs_space.shape))
    action_dim = int(single_act_space.n)

    agent = OurPPOAgent(obs_dim, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Storage buffers
    obs_buf = torch.zeros((num_steps, num_envs, obs_dim), dtype=torch.float32, device=device)
    actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    rewards_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)
    values_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=device)

    # Initialize environment
    global_step = 0
    next_obs_np, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_envs, obs_dim)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    num_updates = total_timesteps // batch_size
    print(f"\n[Our PPO] Training {rung} seed={seed} steps={total_timesteps} ({num_updates} updates) gamma={gamma}...")

    agent.train()
    for update in range(1, num_updates + 1):
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Step envs
            next_obs_np, reward_np, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards_buf[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device).reshape(num_envs, obs_dim)
            next_done = torch.as_tensor(terminations | truncations, dtype=torch.float32, device=device)

        # GAE calculation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf, device=device)
            lastgaelam = 0
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
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_logprobs = logprobs_buf.reshape(-1)
        b_actions = actions_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # Optimize policy & value network
        b_inds = np.arange(batch_size)
        for epoch in range(update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -clip_coef, clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

        if update % 50 == 0 or update == num_updates:
            print(f"[Our PPO Update {update}/{num_updates}] steps={global_step} loss={loss.item():.4f}")

    envs.close()
    torch.save(agent.state_dict(), model_path)
    print(f"[Our PPO] Saved weights to {model_path}")

    # Evaluate on seeds 6000-6099
    print(f"[Our PPO] Evaluating {rung} seed={seed} on seeds 6000-6099...")
    eval_env_fn = lambda: make_ladder_env(rung)
    results = evaluate_policy(eval_env_fn, agent, seed_start=6000, n_episodes=100)
    results["rung"] = rung
    results["seed"] = seed
    results["total_timesteps"] = total_timesteps
    results["method"] = "Our_PPO"

    print(f"[Our PPO Result] Crash: {results['ci_str']} | Speed: {results['mean_speed_ms']:.2f} m/s | Return: {results['mean_return']:.2f} | Dom: {results['dominant_action']} ({results['dominant_action_frac']:.1%})")
    
    import json
    res_path = Path("parity") / f"results_our_{rung}_s{seed}_{total_timesteps}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Our PPO] Saved results to {res_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.8)
    args = parser.parse_args()

    train_and_eval_our_ppo(args.rung, args.seed, args.steps, args.gamma)

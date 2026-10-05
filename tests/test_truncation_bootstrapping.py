"""Unit test verifying truncation bootstrapping vs terminal truncation behavior.

Tests on a synthetic time-limited environment with constant reward r = 1.0:
- True infinite-horizon analytical return: V*(s) = sum_{k=0}^infty gamma^k * 1.0 = 1 / (1 - gamma) = 100.0 (at gamma = 0.99)
- Old behavior (collapses truncation into terminal):
  nextnonterminal = 0.0 at truncation
  delta_9 = 1.0 + 0 - 100.0 = -99.0
  return_9 = 1.0 (99% error!)
  return_0 = 43.0 (with GAE lambda=0.95) or 9.56 (with lambda=1.0) << 100.0
- New behavior (bootstraps V(final_observation) upon truncation):
  nextnonterminal = 1.0, nextvalues = V(final_obs) = 100.0, nextnoncutoff = 0.0
  delta_t = 1.0 + 0.99 * 100.0 - 100.0 = 0.0
  advantage_t = 0.0
  return_t = 100.0 for ALL t in [0, 9] (exact 0.0% error!)
"""

import argparse
import unittest
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import ppo
import ppo_lstm


class SyntheticTimeLimitedEnv(gym.Env):
    """Synthetic environment yielding constant reward r=1.0 until truncation at T steps."""

    def __init__(self, max_steps: int = 10) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.steps = 0
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.steps = 0
        return np.ones(4, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        reward = 1.0
        terminated = False
        truncated = (self.steps >= self.max_steps)
        obs = np.ones(4, dtype=np.float32) * (self.steps / self.max_steps)
        info = {}
        if truncated:
            # Store final observation in info per vector-env conventions
            info["final_obs"] = obs.copy()
        return obs, reward, terminated, truncated, info


def compute_gae_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminations: torch.Tensor,
    truncations: torch.Tensor,
    v_final: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    bootstrap_truncation: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE advantages and returns under old vs new truncation logic."""
    num_steps = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0.0

    for t in reversed(range(num_steps)):
        if bootstrap_truncation:
            # New behavior: separate terminations from truncations
            is_term = terminations[t]
            is_trunc = truncations[t]
            nextnonterminal = 1.0 - is_term
            if t == num_steps - 1:
                nextvalues = v_final if (is_trunc and not is_term) else next_value
            else:
                nextvalues = v_final if (is_trunc and not is_term) else values[t + 1]
            nextnoncutoff = 1.0 - min(1.0, float(is_term + is_trunc))

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnoncutoff * lastgaelam
        else:
            # Terminal truncation behavior: collapses truncations into terminal flags
            is_done = float(terminations[t] or truncations[t])
            nextnonterminal = 1.0 - is_done
            nextvalues = next_value if t == num_steps - 1 else values[t + 1]

            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam

    returns = advantages + values
    return advantages, returns


class TestTruncationBootstrapping(unittest.TestCase):
    def setUp(self) -> None:
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.max_steps = 10
        self.env = SyntheticTimeLimitedEnv(max_steps=self.max_steps)

        # True analytical infinite-horizon value for constant r=1.0:
        self.true_v_analytical = 1.0 / (1.0 - self.gamma)  # 100.0

    def test_analytical_old_vs_new_bootstrapping(self) -> None:
        """Verify that old behavior produces massive error while new behavior matches analytical 100.0."""
        T = self.max_steps
        rewards = torch.ones(T)
        # Assume an accurate critic V(s) = 100.0
        values = torch.full((T,), self.true_v_analytical)
        v_final = torch.tensor(self.true_v_analytical)
        next_val = torch.tensor(self.true_v_analytical)

        # Episode steps: 0..8 regular, step 9 truncated (not terminated)
        terminations = torch.zeros(T)
        truncations = torch.zeros(T)
        truncations[-1] = 1.0  # Truncated at step 9

        # 1. Compute with OLD behavior (collapsing truncation into terminal)
        _, ret_old = compute_gae_returns(
            rewards=rewards,
            values=values,
            terminations=terminations,
            truncations=truncations,
            v_final=v_final,
            next_value=next_val,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            bootstrap_truncation=False,
        )

        # 2. Compute with NEW behavior (bootstrapping V(final_obs))
        _, ret_new = compute_gae_returns(
            rewards=rewards,
            values=values,
            terminations=terminations,
            truncations=truncations,
            v_final=v_final,
            next_value=next_val,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            bootstrap_truncation=True,
        )

        # Assertions on OLD behavior:
        # At step T-1 = 9, old return collapses to 1.0 (r_9 only, dropping continuation value)
        self.assertAlmostEqual(ret_old[-1].item(), 1.0, delta=1e-5)
        # At step 0, old return is dramatically lower than 100.0
        self.assertLess(ret_old[0].item(), 50.0)
        self.assertNotAlmostEqual(ret_old[0].item(), self.true_v_analytical, delta=1.0)

        # Assertions on NEW behavior:
        # At EVERY step t in [0, 9], the bootstrapped return matches the exact analytical 100.0
        for t in range(T):
            self.assertAlmostEqual(
                ret_new[t].item(),
                self.true_v_analytical,
                delta=1e-5,
                msg=f"Step {t}: new return {ret_new[t].item()} != analytical {self.true_v_analytical}",
            )

    def test_true_termination_is_not_bootstrapped(self) -> None:
        """Verify that true environmental termination (e.g. crash) sets continuation value to 0."""
        T = self.max_steps
        rewards = torch.ones(T)
        values = torch.full((T,), self.true_v_analytical)
        v_final = torch.tensor(self.true_v_analytical)
        next_val = torch.tensor(self.true_v_analytical)

        # Episode terminates (e.g. crash) at step 5
        terminations = torch.zeros(T)
        truncations = torch.zeros(T)
        terminations[5] = 1.0

        _, ret_new = compute_gae_returns(
            rewards=rewards,
            values=values,
            terminations=terminations,
            truncations=truncations,
            v_final=v_final,
            next_value=next_val,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            bootstrap_truncation=True,
        )

        # At the terminal step 5, continuation value MUST be 0 -> return is 1.0
        self.assertAlmostEqual(ret_new[5].item(), 1.0, delta=1e-5)

    def test_ppo_truncation_bootstrapping_execution(self) -> None:
        """Verify ppo.train_ppo runs end-to-end with bootstrap_truncation=True without error."""
        args = argparse.Namespace(
            scenario="highway",
            tier="front_only",
            seed=42,
            total_timesteps=64,
            learning_rate=3e-4,
            num_envs=2,
            num_steps=32,
            anneal_lr=False,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=2,
            update_epochs=1,
            norm_adv=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            cuda=False,
            async_envs=False,
            bootstrap_truncation=True,
            checkpoint_dir=None,
            save_frequency=50000,
            resume=None,
            output_path=None,
        )
        saved_path = ppo.train_ppo(args)
        self.assertTrue(saved_path.startswith("scratch/checkpoints/"))

    def test_ppo_lstm_truncation_bootstrapping_execution(self) -> None:
        """Verify ppo_lstm.train_ppo_lstm runs end-to-end with bootstrap_truncation=True without error."""
        args = argparse.Namespace(
            scenario="highway",
            tier="front_only",
            seed=42,
            total_timesteps=64,
            learning_rate=3e-4,
            num_envs=2,
            num_steps=32,
            anneal_lr=False,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=2,
            update_epochs=1,
            norm_adv=True,
            clip_coef=0.2,
            clip_vloss=True,
            ent_coef=0.01,
            max_grad_norm=0.5,
            cuda=False,
            bootstrap_truncation=True,
            checkpoint_dir=None,
            save_frequency=50000,
            resume=None,
            output_path=None,
        )
        saved_path = ppo_lstm.train_ppo_lstm(args)
        self.assertTrue(saved_path.startswith("scratch/checkpoints/"))


if __name__ == "__main__":
    unittest.main()

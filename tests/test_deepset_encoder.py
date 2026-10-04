"""Unit tests for permutation-invariant DeepSet observation encoder."""

from __future__ import annotations

import os
import sys
import unittest
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gymnasium as gym
import highway_env
from deepset_encoder import DeepSetEncoder, DeepSetAgent, DeepSetKinematicsWrapper, make_deepset_env


class TestDeepSetEncoder(unittest.TestCase):
    """Test suite verifying permutation invariance, variable count, and cross-scenario generalization."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.encoder = DeepSetEncoder(veh_dim=5, ego_dim=5, emb_dim=64, out_dim=128, pool="both")

    def test_permutation_invariance(self) -> None:
        """Verify that permuting other vehicles (rows 1..N-1) produces identical output."""
        # 1 ego + 5 other vehicles
        x = torch.randn(2, 6, 5)
        # Ensure presence is 1.0 for all
        x[..., 0] = 1.0

        out_orig = self.encoder(x)

        # Permute other vehicles for batch element 0 and 1 differently
        p1 = [0] + [3, 1, 5, 2, 4]
        p2 = [0] + [5, 4, 3, 2, 1]
        x_perm = torch.stack([x[0, p1, :], x[1, p2, :]], dim=0)

        out_perm = self.encoder(x_perm)

        diff = (out_orig - out_perm).abs().max().item()
        self.assertLess(
            diff,
            1e-5,
            f"Permutation invariance violated! Max difference: {diff}",
        )

    def test_variable_vehicle_count(self) -> None:
        """Verify encoder seamlessly processes any vehicle count N = 1, 4, 10, 25."""
        counts = [1, 4, 10, 25]
        for n in counts:
            x = torch.randn(4, n, 5)
            x[..., 0] = 1.0
            out = self.encoder(x)
            self.assertEqual(
                out.shape,
                (4, 128),
                f"Expected shape (4, 128) for N={n}, got {out.shape}",
            )

    def test_padding_invariance(self) -> None:
        """Verify that padding with absent vehicles (presence=0) does not alter output."""
        # Active vehicles: ego + 2 others
        active_obs = torch.tensor([
            [1.0, 0.0, 0.0, 0.8, 0.0],    # ego
            [1.0, 0.2, 0.1, 0.05, 0.0],   # other 1
            [1.0, -0.4, -0.1, -0.1, 0.0],  # other 2
        ]).unsqueeze(0)

        # Padded version: same 3 active + 7 absent padding rows
        padding_rows = torch.zeros(1, 7, 5)
        padded_obs = torch.cat([active_obs, padding_rows], dim=1)

        out_active = self.encoder(active_obs)
        out_padded = self.encoder(padded_obs)

        diff = (out_active - out_padded).abs().max().item()
        self.assertLess(
            diff,
            1e-5,
            f"Padding invariance violated! Max difference: {diff}",
        )

    def test_all_four_highway_scenarios(self) -> None:
        """Verify encoder accepts highway, merge, roundabout, and intersection directly."""
        scenarios = ["highway", "merge", "roundabout", "intersection"]
        for sc in scenarios:
            env = make_deepset_env(scenario=sc, seed=42)
            obs, info = env.reset(seed=42)
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

            out = self.encoder(obs_t)
            self.assertEqual(
                out.shape,
                (1, 128),
                f"Scenario '{sc}' failed encoder forward pass. Got {out.shape}",
            )

            # Step environment once
            next_obs, reward, term, trunc, info = env.step(1)
            next_obs_t = torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            next_out = self.encoder(next_obs_t)
            self.assertEqual(next_out.shape, (1, 128))
            self.assertIn("action_mask", info)
            self.assertIn("overtake_count", info)
            env.close()

    def test_agent_action_and_value(self) -> None:
        """Verify DeepSetAgent computes valid actions, values, and backpropagates gradients."""
        agent = DeepSetAgent(veh_dim=5, ego_dim=5, action_dim=5, latent_dim=128)

        # Batch of 4 with 10 vehicles
        x = torch.randn(4, 10, 5)
        x[..., 0] = 1.0

        # Action masking
        mask = torch.tensor([
            [True, True, True, False, False],
            [True, True, False, True, False],
            [True, False, True, False, True],
            [False, True, True, True, True],
        ])

        action, logprob, entropy, value = agent.get_action_and_value(x, action_mask=mask)

        self.assertEqual(action.shape, (4,))
        self.assertEqual(logprob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(value.shape, (4, 1))

        # Check masked actions are never selected
        for i in range(4):
            self.assertTrue(mask[i, action[i].item()], f"Agent selected masked action {action[i]}")

        # Check backward pass
        loss = value.mean() - logprob.mean()
        loss.backward()

        for name, param in agent.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradient")


if __name__ == "__main__":
    unittest.main()

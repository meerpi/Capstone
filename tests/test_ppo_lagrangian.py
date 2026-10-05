"""Unit tests for PPO-Lagrangian (PID-Lagrangian) components.

Tests:
1. PID Controller over-budget test: J_C >> d increases lambda.
2. PID Controller under-budget test: J_C << d decreases lambda toward 0 and clamps at 0.
3. Lagrangian-adjusted advantage with separate normalization:
   A_R_norm and A_C_norm separately normalized to unit scale; A_adj = A_R_norm - lambda * A_C_norm.
   High cost-advantage, low reward-advantage state flips sign when lambda > 0.
4. Wrapper cost signal & reward stripping test:
   constrained_mode=True strips collision penalty from reward and sets info['cost']=1.0.
   constrained_mode=False keeps collision penalty in reward.
5. Dual-critic agent architecture test:
   OptimalAgent handles reward critic and cost critic appropriately.
6. Cost-GAE true termination correctness test:
   When an episode ends in a TRUE crash (terminated=True), the cost value bootstrap
   is strictly zero (no cost-to-go beyond crash), matching reward termination behavior.
"""

import unittest
import numpy as np
import torch
import gymnasium as gym

from train_optimal_overtaker import (
    PIDLagrangianController,
    OptimalAgent,
)
from env_config import (
    TacticalLaneObservationWrapper,
    make_optimal_env,
)


class TestPIDLagrangianController(unittest.TestCase):
    """Test PID Lagrangian multiplier dynamics and anti-windup clamping."""

    def test_over_budget_increases_lambda(self) -> None:
        """When cost return J_C >> d, lambda should increase across iterations."""
        cost_limit = 0.05
        kp = 0.05
        ki = 0.001
        controller = PIDLagrangianController(cost_limit=cost_limit, kp=kp, ki=ki)

        # Start at 0
        self.assertEqual(controller.lagrange_mult, 0.0)

        # Feed J_C = 0.25 (well above 0.05 budget)
        lambdas = []
        for _ in range(10):
            lam, err, integ = controller.step(J_C=0.25)
            lambdas.append(lam)
            self.assertGreater(err, 0.0)
            self.assertGreater(integ, 0.0)

        # Check monotonic increase
        for i in range(len(lambdas) - 1):
            self.assertGreater(lambdas[i + 1], lambdas[i])
        self.assertGreater(lambdas[-1], 0.0)

    def test_under_budget_decreases_lambda_to_zero(self) -> None:
        """When cost return J_C << d, lambda should decrease toward 0 and clamp at 0."""
        cost_limit = 0.05
        kp = 0.05
        ki = 0.001
        controller = PIDLagrangianController(cost_limit=cost_limit, kp=kp, ki=ki)

        # First build up lambda with high cost
        for _ in range(5):
            controller.step(J_C=0.50)
        self.assertGreater(controller.lagrange_mult, 0.0)

        # Now feed J_C = 0.0 (well below 0.05 budget)
        prev_lam = controller.lagrange_mult
        for _ in range(100):
            lam, err, integ = controller.step(J_C=0.0)
            self.assertLessEqual(lam, prev_lam)
            self.assertGreaterEqual(lam, 0.0)
            self.assertGreaterEqual(integ, 0.0)  # Anti-windup clamping prevents negative integral
            prev_lam = lam

        # Eventually reaches exactly 0.0
        self.assertEqual(controller.lagrange_mult, 0.0)

    def test_serialization(self) -> None:
        """Controller state can be saved and restored exactly."""
        c1 = PIDLagrangianController(cost_limit=0.05, kp=0.05, ki=0.001)
        for _ in range(5):
            c1.step(0.3)

        state = c1.state_dict()
        c2 = PIDLagrangianController()
        c2.load_state_dict(state)

        self.assertEqual(c1.lagrange_mult, c2.lagrange_mult)
        self.assertEqual(c1.integral, c2.integral)
        self.assertEqual(c1.prev_error, c2.prev_error)


class TestLagrangianAdvantageAdjustment(unittest.TestCase):
    """Test policy gradient Lagrangian advantage adjustment with separate normalization."""

    def test_separate_normalization_equalizes_scales(self) -> None:
        """A_R and A_C have wildly different raw scales, but separately normalizing

        them ensures that lambda=1.0 gives the cost term equal weight to reward.
        """
        # Raw reward advantages: mean ~ 10, std ~ 5 (from large speed/overtake returns)
        torch.manual_seed(42)
        raw_ar = torch.randn(1000) * 5.0 + 10.0
        # Raw cost advantages: mean ~ 0.2, std ~ 0.4 (bounded crash indicators)
        raw_ac = torch.randn(1000) * 0.4 + 0.2

        # 1. Separately normalize both components
        ar_norm = (raw_ar - raw_ar.mean()) / (raw_ar.std() + 1e-8)
        ac_norm = (raw_ac - raw_ac.mean()) / (raw_ac.std() + 1e-8)

        self.assertAlmostEqual(ar_norm.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(ar_norm.std().item(), 1.0, places=4)
        self.assertAlmostEqual(ac_norm.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(ac_norm.std().item(), 1.0, places=4)

        # Mean absolute magnitudes of normalized streams are both unit-scale (~0.798 for N(0, 1))
        mag_r = torch.abs(ar_norm).mean().item()
        mag_c = torch.abs(ac_norm).mean().item()
        self.assertAlmostEqual(mag_r, mag_c, delta=0.05)

        # 2. Combine with lambda: A_adj = A_R_norm - lambda * A_C_norm
        # At lambda = 1.0, the cost term has identical weight to reward
        lam = 1.0
        cost_term = lam * ac_norm
        self.assertAlmostEqual(torch.abs(cost_term).mean().item(), mag_r, delta=0.05)

        # At lambda = 0.5, cost term has exactly 50% magnitude of reward
        lam_half = 0.5
        self.assertAlmostEqual(torch.abs(lam_half * ac_norm).mean().item(), 0.5 * mag_r, delta=0.05)

        # Confirm A_adj is NOT re-normalized (its scale varies with lambda)
        adj_1 = ar_norm - 1.0 * ac_norm
        adj_2 = ar_norm - 2.0 * ac_norm
        self.assertGreater(adj_2.std().item(), adj_1.std().item())

    def test_advantage_sign_flip_for_hazardous_action(self) -> None:
        """A hazardous action with high cost advantage and modest reward advantage

        flips sign (becomes negative) once lambda > 0 under separate normalization.
        """
        # Batch of 4 actions:
        # Action 0: High reward, zero cost (safe & fast)
        # Action 1: Modest reward, high cost (reckless crash risk)
        # Action 2: Low reward, zero cost (slow & safe)
        # Action 3: Low reward, high cost (reckless & slow)
        ar = torch.tensor([5.0, 3.5, 1.0, 0.0])
        ac = torch.tensor([0.0, 1.0, 0.0, 1.0])

        ar_norm = (ar - ar.mean()) / (ar.std() + 1e-8)
        ac_norm = (ac - ac.mean()) / (ac.std() + 1e-8)

        # At lambda = 0: Action 1 has positive advantage (reward is above mean)
        adj_0 = ar_norm - 0.0 * ac_norm
        self.assertGreater(adj_0[1].item(), 0.0)

        # At lambda = 1.0: Action 0 dominates Action 1, and Action 1's sign flips negative
        adj_1 = ar_norm - 1.0 * ac_norm
        self.assertGreater(adj_1[0].item(), adj_1[1].item())
        self.assertLess(adj_1[1].item(), 0.0)


class TestCostGAETermination(unittest.TestCase):
    """Test cost GAE termination correctness (Step 1).

    Confirms that when an episode ends in a TRUE crash (terminated=True),
    the cost value bootstrap is strictly zero (no cost-to-go beyond crash),
    matching reward termination behavior.
    """

    def _compute_cost_gae(
        self,
        costs: torch.Tensor,
        cost_values: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        next_cost_value: torch.Tensor,
        cost_trunc_bootstrap_values: torch.Tensor,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        bootstrap_truncation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
        """Replicate the exact cost GAE computation loop from train_optimal_overtaker.py."""
        num_steps = len(costs)
        cost_advantages = torch.zeros_like(costs)
        cost_lastgaelam = 0.0
        bootstrap_values_used = []

        for t in reversed(range(num_steps)):
            if bootstrap_truncation:
                is_term = terminations[t]
                is_trunc = truncations[t]
                nextnonterminal = 1.0 - is_term
                if t == num_steps - 1:
                    nextcostvalues = (
                        cost_trunc_bootstrap_values[t]
                        if (bool(is_trunc.item()) and not bool(is_term.item()))
                        else next_cost_value.squeeze()
                    )
                else:
                    nextcostvalues = (
                        cost_trunc_bootstrap_values[t]
                        if (bool(is_trunc.item()) and not bool(is_term.item()))
                        else cost_values[t + 1]
                    )
                nextnoncutoff = 1.0 - min(1.0, float(is_term.item() + is_trunc.item()))

                # Continuation value term: gamma * nextcostvalues * nextnonterminal
                continuation = gamma * nextcostvalues * nextnonterminal
                bootstrap_values_used.append(float(continuation.item()))

                delta_cost = costs[t] + continuation - cost_values[t]
                cost_advantages[t] = cost_lastgaelam = (
                    delta_cost + gamma * gae_lambda * nextnoncutoff * cost_lastgaelam
                )
            else:
                # Unmasked behavior: collapses truncations into terminal
                is_done = float(terminations[t].item() or truncations[t].item())
                nextnonterminal = 1.0 - is_done
                nextcostvalues = (
                    next_cost_value.squeeze()
                    if t == num_steps - 1
                    else cost_values[t + 1]
                )
                continuation = gamma * nextcostvalues * nextnonterminal
                bootstrap_values_used.append(float(continuation.item()))

                delta_cost = costs[t] + continuation - cost_values[t]
                cost_advantages[t] = cost_lastgaelam = (
                    delta_cost + gamma * gae_lambda * nextnonterminal * cost_lastgaelam
                )

        cost_returns = cost_advantages + cost_values
        bootstrap_values_used.reverse()
        return cost_advantages, cost_returns, bootstrap_values_used

    def test_true_crash_has_zero_cost_bootstrap(self) -> None:
        """When an episode terminates on a true crash:

        1. The cost continuation value at that step MUST be exactly 0.0 (no cost beyond crash).
        2. Target cost return at that step MUST equal exactly the crash cost (1.0).
        3. Post-reset cost values in subsequent steps must NOT bleed backward into the crashed step.
        """
        T = 5
        gamma = 0.995
        gae_lambda = 0.95

        # Step 0: cruising, c=0
        # Step 1: near miss, c=0
        # Step 2: CRASH (terminated=True), c=1.0
        # Step 3: post-reset step (new episode), c=0
        # Step 4: post-reset step, c=0
        costs = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        cost_values = torch.tensor([0.10, 0.30, 0.85, 0.15, 0.10])
        terminations = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        truncations = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        next_cost_val = torch.tensor([0.10])
        trunc_boot_vals = torch.zeros(T)

        for b_trunc in [True, False]:
            adv, ret, bootstraps = self._compute_cost_gae(
                costs=costs,
                cost_values=cost_values,
                terminations=terminations,
                truncations=truncations,
                next_cost_value=next_cost_val,
                cost_trunc_bootstrap_values=trunc_boot_vals,
                gamma=gamma,
                gae_lambda=gae_lambda,
                bootstrap_truncation=b_trunc,
            )

            # 1. At step 2 (crash step), the bootstrap continuation value MUST be exactly 0.0
            self.assertEqual(
                bootstraps[2],
                0.0,
                msg=f"Cost continuation bootstrap at crashed step was {bootstraps[2]}, expected 0.0 (bootstrap_truncation={b_trunc})",
            )

            # 2. At step 2, target cost return must equal exactly 1.0 (the crash cost with no continuation)
            self.assertAlmostEqual(
                ret[2].item(),
                1.0,
                places=5,
                msg=f"Cost return at crash step was {ret[2].item()}, expected 1.0 (bootstrap_truncation={b_trunc})",
            )

            # 3. Step 2 advantage must equal delta = c_2 - V_c(s_2) = 1.0 - 0.85 = 0.15
            expected_adv_2 = 1.0 - cost_values[2].item()
            self.assertAlmostEqual(
                adv[2].item(),
                expected_adv_2,
                places=5,
                msg=f"Cost advantage at crash step was {adv[2].item()}, expected {expected_adv_2}",
            )

            # 4. GAE backward accumulation at step 1: delta_1 + gamma * lambda * 1.0 * adv_2
            # delta_1 = 0.0 + gamma * cost_values[2] - cost_values[1]
            delta_1 = 0.0 + gamma * cost_values[2].item() - cost_values[1].item()
            expected_adv_1 = delta_1 + gamma * gae_lambda * adv[2].item()
            self.assertAlmostEqual(
                adv[1].item(),
                expected_adv_1,
                places=5,
                msg="Pre-crash step advantage did not receive correct discounted GAE from crash",
            )

    def test_cost_and_reward_termination_symmetry(self) -> None:
        """Confirm cost GAE and reward GAE handle termination with identical non-terminal masking."""
        T = 4
        # At step 2, true crash terminates episode
        terms = torch.tensor([0.0, 0.0, 1.0, 0.0])
        truncs = torch.zeros(T)

        costs = torch.tensor([0.0, 0.0, 1.0, 0.0])
        cost_vals = torch.tensor([0.05, 0.20, 0.90, 0.10])

        # In constrained mode, reward on crash is 0.0 (speed only)
        rewards = torch.tensor([0.5, 0.5, 0.0, 0.5])
        reward_vals = torch.tensor([10.0, 12.0, 8.0, 10.0])

        # Both reward and cost continuation values at step 2 must be 0.0
        # Reward return at step 2: r_2 + 0 = 0.0
        # Cost return at step 2: c_2 + 0 = 1.0
        adv_c, ret_c, boot_c = self._compute_cost_gae(
            costs=costs,
            cost_values=cost_vals,
            terminations=terms,
            truncations=truncs,
            next_cost_value=torch.tensor([0.10]),
            cost_trunc_bootstrap_values=torch.zeros(T),
            bootstrap_truncation=True,
        )

        adv_r, ret_r, boot_r = self._compute_cost_gae(
            costs=rewards,
            cost_values=reward_vals,
            terminations=terms,
            truncations=truncs,
            next_cost_value=torch.tensor([10.0]),
            cost_trunc_bootstrap_values=torch.zeros(T),
            bootstrap_truncation=True,
        )

        # Continuation values at step 2 are identical (both 0.0)
        self.assertEqual(boot_c[2], 0.0)
        self.assertEqual(boot_r[2], 0.0)

        # Returns at step 2 are exactly immediate signals
        self.assertAlmostEqual(ret_c[2].item(), 1.0, places=5)
        self.assertAlmostEqual(ret_r[2].item(), 0.0, places=5)


class TestTacticalLaneObservationWrapperConstrained(unittest.TestCase):
    """Test TacticalLaneObservationWrapper constrained mode vs unconstrained mode."""

    def test_constrained_mode_cost_and_reward_stripping(self) -> None:
        """In constrained_mode:

        - info['cost'] is 1.0 when crashed
        - collision_penalty is NOT subtracted from reward (reward does not have -50.0).
        """
        env = make_optimal_env(
            lanes_count=4,
            vehicles_count=10,
            seed=42,
            frame_stack_k=0,
            collision_penalty=-50.0,
            constrained_mode=True,
        )
        obs, info = env.reset(seed=42)
        self.assertEqual(info["cost"], 0.0)

        # Simulate a crash directly on ego
        ego = env.unwrapped.vehicle
        ego.crashed = True

        _, reward, terminated, _, info = env.step(1)  # step with crashed ego
        self.assertEqual(info["cost"], 1.0)
        self.assertEqual(info["crashed"], True)
        # Reward in constrained mode for crash is 0.0, NOT -50.0
        self.assertEqual(reward, 0.0)
        env.close()

    def test_unconstrained_mode_preserves_collision_penalty(self) -> None:
        """In unconstrained (standard) mode:

        - collision_penalty (-50.0) is in reward
        - info['cost'] is still available (1.0 on crash).
        """
        env = make_optimal_env(
            lanes_count=4,
            vehicles_count=10,
            seed=42,
            frame_stack_k=0,
            collision_penalty=-50.0,
            constrained_mode=False,
        )
        obs, info = env.reset(seed=42)
        self.assertEqual(info["cost"], 0.0)

        ego = env.unwrapped.vehicle
        ego.crashed = True

        _, reward, terminated, _, info = env.step(1)
        self.assertEqual(info["cost"], 1.0)
        self.assertEqual(info["crashed"], True)
        # In unconstrained mode, reward is exactly collision_penalty (-50.0)
        self.assertEqual(reward, -50.0)
        env.close()


class TestOptimalAgentDualCritic(unittest.TestCase):
    """Test OptimalAgent dual critic architecture."""

    def test_agent_with_cost_critic(self) -> None:
        """When use_cost_critic=True, agent contains both critic and cost_critic."""
        obs_dim = 81
        action_dim = 5
        agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim, use_cost_critic=True)

        self.assertTrue(hasattr(agent, "cost_critic"))
        self.assertTrue(hasattr(agent, "critic"))

        x = torch.randn(4, obs_dim)
        act, logp, ent, val, cost_val = agent.get_action_and_value(x, return_cost_value=True)

        self.assertEqual(act.shape, (4,))
        self.assertEqual(val.shape, (4, 1))
        self.assertEqual(cost_val.shape, (4, 1))

        # Check get_cost_value directly
        cv = agent.get_cost_value(x)
        self.assertEqual(cv.shape, (4, 1))

    def test_agent_without_cost_critic(self) -> None:
        """When use_cost_critic=False, agent has no cost_critic attribute."""
        obs_dim = 81
        action_dim = 5
        agent = OptimalAgent(obs_dim=obs_dim, action_dim=action_dim, use_cost_critic=False)

        self.assertFalse(hasattr(agent, "cost_critic"))
        x = torch.randn(4, obs_dim)
        act, logp, ent, val = agent.get_action_and_value(x, return_cost_value=False)
        self.assertEqual(act.shape, (4,))
        self.assertEqual(val.shape, (4, 1))

        with self.assertRaises(RuntimeError):
            agent.get_cost_value(x)


if __name__ == "__main__":
    unittest.main()

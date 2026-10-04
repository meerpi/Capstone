"""Tests for PPO-LSTM phantom sample reset timing and loss masking.

Verifies:
- T1 (contamination): First real sample of a new episode must have zero dependence
  on observations preceding it when reset timing is fixed, but demonstrates
  contamination under --keep-phantom-samples.
- T2 (consistency): Update-phase forward pass over stored rollout yields ratio == 1.0 (atol=1e-5).
"""

import unittest
import torch
import numpy as np
import gymnasium as gym

import env_config
from ppo_lstm import RecurrentAgent, make_env_thunk


class TestPPOLSTMPhantom(unittest.TestCase):
    def _collect_rollout(
        self,
        agent: RecurrentAgent,
        seed: int = 9000,
        num_envs: int = 4,
        num_steps: int = 32,
    ):
        scenario = "highway"
        tier = "full_adas"
        env_fns = [
            make_env_thunk(scenario, tier, seed=seed + i * 1000, tactical_overtaking=True)
            for i in range(num_envs)
        ]
        envs = gym.vector.SyncVectorEnv(env_fns)
        obs, _ = envs.reset(seed=seed)
        obs_dim = obs.shape[-1] * obs.shape[-2]

        obs_buf = torch.zeros(num_steps, num_envs, obs_dim)
        dones_buf = torch.zeros(num_steps, num_envs)
        actions_buf = torch.zeros(num_steps, num_envs, dtype=torch.long)
        logprobs_buf = torch.zeros(num_steps, num_envs)
        values_buf = torch.zeros(num_steps, num_envs)

        next_obs = torch.tensor(obs.reshape(num_envs, obs_dim), dtype=torch.float32)
        next_done = torch.zeros(num_envs)
        next_actor_state = (torch.zeros(1, num_envs, 128), torch.zeros(1, num_envs, 128))
        next_critic_state = (torch.zeros(1, num_envs, 128), torch.zeros(1, num_envs, 128))

        initial_actor_state = (next_actor_state[0].clone(), next_actor_state[1].clone())
        initial_critic_state = (next_critic_state[0].clone(), next_critic_state[1].clone())

        for step in range(num_steps):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            with torch.no_grad():
                action, logprob, _, value, next_actor_state, next_critic_state = (
                    agent.get_action_and_value(
                        next_obs, next_actor_state, next_critic_state, next_done
                    )
                )
            actions_buf[step] = action
            logprobs_buf[step] = logprob
            values_buf[step] = value.squeeze(-1)
            next_obs_np, _, term, trunc, _ = envs.step(action.numpy())
            next_done_np = np.logical_or(term, trunc)
            next_obs = torch.tensor(next_obs_np.reshape(num_envs, obs_dim), dtype=torch.float32)
            next_done = torch.tensor(next_done_np, dtype=torch.float32)

        envs.close()
        return (
            obs_buf,
            dones_buf,
            actions_buf,
            logprobs_buf,
            values_buf,
            initial_actor_state,
            initial_critic_state,
        )

    def test_t2_consistency(self) -> None:
        """T2 (consistency): Recomputed update log-probs equal rollout log-probs."""
        torch.manual_seed(9000)
        np.random.seed(9000)
        agent = RecurrentAgent(obs_dim=75, action_dim=5, hidden_dim=128, keep_phantom_samples=False)

        (
            obs_buf,
            dones_buf,
            actions_buf,
            logprobs_buf,
            values_buf,
            init_a,
            init_c,
        ) = self._collect_rollout(agent, seed=9000)

        num_steps, num_envs, obs_dim = obs_buf.shape
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_actions = actions_buf.reshape(-1)
        b_dones = dones_buf.reshape(-1)
        flatinds = np.arange(num_steps * num_envs).reshape(num_steps, num_envs)
        mb_inds = flatinds[:, np.arange(num_envs)].ravel()

        with torch.no_grad():
            _, newlogprob, _, newvalue, _, _ = agent.get_action_and_value(
                b_obs[mb_inds],
                init_a,
                init_c,
                b_dones[mb_inds],
                action=b_actions[mb_inds],
            )

        ratio = (newlogprob - logprobs_buf.reshape(-1)[mb_inds]).exp()
        max_ratio_err = (ratio - 1.0).abs().max().item()
        max_val_err = (newvalue.squeeze(-1) - values_buf.reshape(-1)[mb_inds]).abs().max().item()

        self.assertLess(max_ratio_err, 1e-5, f"T2 ratio error exceeded: {max_ratio_err}")
        self.assertLess(max_val_err, 1e-5, f"T2 value error exceeded: {max_val_err}")

    def test_t1_contamination_fixed(self) -> None:
        """T1 (contamination): First real sample of new episode is invariant to prior noise."""
        torch.manual_seed(9000)
        np.random.seed(9000)
        agent = RecurrentAgent(obs_dim=75, action_dim=5, hidden_dim=128, keep_phantom_samples=False)

        (
            obs_buf,
            dones_buf,
            actions_buf,
            _,
            _,
            init_a,
            init_c,
        ) = self._collect_rollout(agent, seed=9000)

        num_steps = obs_buf.shape[0]
        phantoms = torch.nonzero(dones_buf)
        self.assertGreater(len(phantoms), 0, "No episode boundaries found in rollout")

        checked = 0
        for p in phantoms:
            step, env_idx = p[0].item(), p[1].item()
            if step + 1 >= num_steps:
                continue
            first_real_step = step + 1

            env_obs = obs_buf[: first_real_step + 1, env_idx : env_idx + 1]
            env_dones = dones_buf[: first_real_step + 1, env_idx : env_idx + 1]
            env_act = actions_buf[: first_real_step + 1, env_idx : env_idx + 1]
            env_init_a = (init_a[0][:, env_idx : env_idx + 1], init_a[1][:, env_idx : env_idx + 1])
            env_init_c = (init_c[0][:, env_idx : env_idx + 1], init_c[1][:, env_idx : env_idx + 1])

            with torch.no_grad():
                _, orig_lp, _, orig_val, _, _ = agent.get_action_and_value(
                    env_obs, env_init_a, env_init_c, env_dones, action=env_act.flatten()
                )
                orig_lp_step = orig_lp.reshape(-1)[first_real_step]
                orig_val_step = orig_val.reshape(-1)[first_real_step]

                noisy_obs = env_obs.clone()
                noisy_obs[:first_real_step] = torch.randn_like(noisy_obs[:first_real_step])

                _, noisy_lp, _, noisy_val, _, _ = agent.get_action_and_value(
                    noisy_obs, env_init_a, env_init_c, env_dones, action=env_act.flatten()
                )
                noisy_lp_step = noisy_lp.reshape(-1)[first_real_step]
                noisy_val_step = noisy_val.reshape(-1)[first_real_step]

            diff_lp = (orig_lp_step - noisy_lp_step).abs().item()
            diff_val = (orig_val_step - noisy_val_step).abs().item()
            self.assertLess(diff_lp, 1e-6, f"Env {env_idx} step {first_real_step} logprob contaminated: {diff_lp}")
            self.assertLess(diff_val, 1e-6, f"Env {env_idx} step {first_real_step} value contaminated: {diff_val}")
            checked += 1

        self.assertGreater(checked, 0, "No episode transitions verified")

    def test_t1_contamination_legacy_fails(self) -> None:
        """T1 with --keep-phantom-samples: Must demonstrate contamination (fail)."""
        torch.manual_seed(9000)
        np.random.seed(9000)
        agent = RecurrentAgent(obs_dim=75, action_dim=5, hidden_dim=128, keep_phantom_samples=True)

        (
            obs_buf,
            dones_buf,
            actions_buf,
            _,
            _,
            init_a,
            init_c,
        ) = self._collect_rollout(agent, seed=9000)

        num_steps = obs_buf.shape[0]
        phantoms = torch.nonzero(dones_buf)
        max_diff_lp = 0.0
        max_diff_val = 0.0

        for p in phantoms:
            step, env_idx = p[0].item(), p[1].item()
            if step + 1 >= num_steps:
                continue
            first_real_step = step + 1

            env_obs = obs_buf[: first_real_step + 1, env_idx : env_idx + 1]
            env_dones = dones_buf[: first_real_step + 1, env_idx : env_idx + 1]
            env_act = actions_buf[: first_real_step + 1, env_idx : env_idx + 1]
            env_init_a = (init_a[0][:, env_idx : env_idx + 1], init_a[1][:, env_idx : env_idx + 1])
            env_init_c = (init_c[0][:, env_idx : env_idx + 1], init_c[1][:, env_idx : env_idx + 1])

            with torch.no_grad():
                _, orig_lp, _, orig_val, _, _ = agent.get_action_and_value(
                    env_obs, env_init_a, env_init_c, env_dones, action=env_act.flatten()
                )
                orig_lp_step = orig_lp.reshape(-1)[first_real_step]
                orig_val_step = orig_val.reshape(-1)[first_real_step]

                noisy_obs = env_obs.clone()
                noisy_obs[:first_real_step] = torch.randn_like(noisy_obs[:first_real_step])

                _, noisy_lp, _, noisy_val, _, _ = agent.get_action_and_value(
                    noisy_obs, env_init_a, env_init_c, env_dones, action=env_act.flatten()
                )
                noisy_lp_step = noisy_lp.reshape(-1)[first_real_step]
                noisy_val_step = noisy_val.reshape(-1)[first_real_step]

            diff_lp = (orig_lp_step - noisy_lp_step).abs().item()
            diff_val = (orig_val_step - noisy_val_step).abs().item()
            max_diff_lp = max(max_diff_lp, diff_lp)
            max_diff_val = max(max_diff_val, diff_val)

        # Under legacy behavior, state IS contaminated so diffs must be non-zero (> 1e-5)
        self.assertGreater(max_diff_lp, 1e-5, f"Expected contamination with keep_phantom_samples, got max_diff_lp={max_diff_lp}")
        self.assertGreater(max_diff_val, 1e-5, f"Expected contamination with keep_phantom_samples, got max_diff_val={max_diff_val}")


if __name__ == "__main__":
    unittest.main()

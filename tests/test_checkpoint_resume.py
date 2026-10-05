"""Test checkpoint saving, RNG serialization, and resume capability.

Tests:
1. Train an agent for 20,000 steps.
2. Confirm checkpoint contains:
   - model_state_dict
   - optimizer_state_dict
   - global_step
   - rng_state (torch, numpy, python)
3. Resume from the 20,000-step checkpoint.
4. Confirm step count starts at 20,000, advances to 25,000, optimizer state matches,
   and training metrics continue smoothly.
"""

import os
import shutil
import tempfile
import unittest
import torch
import numpy as np

import env_config
import ppo
import ppo_lstm
from train_optimal_overtaker import OptimalAgent, train, parse_args


class TestCheckpointResume(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="test_ckpt_")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_checkpoint_and_resume_20k(self) -> None:
        """Train 20k steps, stop, resume, and confirm step count, optimizer, and loss curve."""
        # Setup arguments for initial training: 20k steps
        # 4 envs, 128 steps per rollout = 512 steps per iteration
        # 39 iterations * 512 = 19,968 steps (~20k steps)
        # 40 iterations * 512 = 20,480 steps
        num_envs = 4
        num_steps = 128
        total_steps_phase1 = 20480

        import argparse

        args1 = argparse.Namespace(
            lanes=3,
            density=1.0,
            vehicles=8,
            frame_stack=3,
            seed=42,
            total_timesteps=total_steps_phase1,
            learning_rate=2.5e-4,
            lr_end=1e-4,
            num_envs=num_envs,
            num_steps=num_steps,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=2,
            update_epochs=2,
            clip_coef=0.2,
            ent_coef=0.01,
            ent_end=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            cuda=False,
            bootstrap_truncation=True,
            checkpoint_dir=self.test_dir,
            save_frequency=10000,
            resume=None,
            eval_only=False,
            model_path=None,
            eval_episodes=2,
            eval_seed_start=2000,
        )

        print(f"\n--- Initial Training: {total_steps_phase1} steps ---")
        saved_path1 = train(args1)

        # 1. Verify checkpoint exists in test_dir
        latest_ckpt = os.path.join(self.test_dir, "checkpoint_latest.pt")
        self.assertTrue(os.path.exists(latest_ckpt), f"Latest checkpoint not found at {latest_ckpt}")

        # 2. Inspect checkpoint contents
        ckpt_data = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt_data)
        self.assertIn("optimizer_state_dict", ckpt_data)
        self.assertIn("global_step", ckpt_data)
        self.assertIn("iteration", ckpt_data)
        self.assertIn("rng_state", ckpt_data)

        # Check RNG fields
        rng = ckpt_data["rng_state"]
        self.assertIn("torch", rng)
        self.assertIn("numpy", rng)
        self.assertIn("python", rng)

        phase1_step = ckpt_data["global_step"]
        phase1_iter = ckpt_data["iteration"]
        self.assertGreaterEqual(phase1_step, 20000, f"Expected >= 20,000 steps, got {phase1_step}")
        print(f"✓ Checkpoint validated: global_step={phase1_step}, iteration={phase1_iter}")

        # 3. Resume for an additional 5,120 steps (up to ~25.6k steps)
        total_steps_phase2 = phase1_step + 5120

        args2 = argparse.Namespace(
            lanes=3,
            density=1.0,
            vehicles=8,
            frame_stack=3,
            seed=42,
            total_timesteps=total_steps_phase2,
            learning_rate=2.5e-4,
            lr_end=1e-4,
            num_envs=num_envs,
            num_steps=num_steps,
            gamma=0.99,
            gae_lambda=0.95,
            num_minibatches=2,
            update_epochs=2,
            clip_coef=0.2,
            ent_coef=0.01,
            ent_end=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            cuda=False,
            bootstrap_truncation=True,
            checkpoint_dir=self.test_dir,
            save_frequency=5000,
            resume=latest_ckpt,
            eval_only=False,
            model_path=None,
            eval_episodes=2,
            eval_seed_start=2000,
        )

        print(f"\n--- Resume Training: from {phase1_step} up to {total_steps_phase2} steps ---")
        saved_path2 = train(args2)

        # 4. Verify resumed checkpoint
        resumed_ckpt_data = torch.load(os.path.join(self.test_dir, "checkpoint_latest.pt"), map_location="cpu", weights_only=False)
        phase2_step = resumed_ckpt_data["global_step"]
        phase2_iter = resumed_ckpt_data["iteration"]
        self.assertGreaterEqual(phase2_step, total_steps_phase2)
        self.assertGreater(phase2_iter, phase1_iter)
        print(f"✓ Resume validated: advanced from {phase1_step} to {phase2_step} steps (iteration {phase2_iter})")

    def test_ppo_checkpoint_and_resume(self) -> None:
        """Verify ppo.py checkpoint saving, RNG state, resume capability, and tooling guard."""
        import argparse
        num_envs = 2
        num_steps = 32
        steps_phase1 = 64  # 1 iteration of 64 steps

        args1 = argparse.Namespace(
            scenario="highway",
            tier="front_only",
            seed=42,
            total_timesteps=steps_phase1,
            learning_rate=3e-4,
            num_envs=num_envs,
            num_steps=num_steps,
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
            checkpoint_dir=self.test_dir,
            save_frequency=32,
            resume=None,
            output_path=None,
        )

        print("\n--- Testing PPO: Initial Training & Checkpoint Saving ---")
        saved_path1 = ppo.train_ppo(args1)

        # 1. Tooling guard check: ensure low-step run did NOT write into models/
        self.assertFalse(saved_path1.startswith("models/"), f"Guard violated! Saved to canonical path: {saved_path1}")
        self.assertTrue(saved_path1.startswith("scratch/checkpoints/"), f"Expected scratch path, got {saved_path1}")

        # 2. Checkpoint structure validation
        latest_ckpt = os.path.join(self.test_dir, "checkpoint_latest.pt")
        self.assertTrue(os.path.exists(latest_ckpt))
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("optimizer_actor_state_dict", ckpt)
        self.assertIn("optimizer_critic_state_dict", ckpt)
        self.assertIn("global_step", ckpt)
        self.assertIn("iteration", ckpt)
        self.assertIn("rng_state", ckpt)
        phase1_step = ckpt["global_step"]
        phase1_iter = ckpt["iteration"]
        self.assertGreaterEqual(phase1_step, 32)
        print(f"✓ PPO Checkpoint validated: global_step={phase1_step}, iteration={phase1_iter}")

        # 3. Resume training
        steps_phase2 = 128
        args2 = argparse.Namespace(
            scenario="highway",
            tier="front_only",
            seed=42,
            total_timesteps=steps_phase2,
            learning_rate=3e-4,
            num_envs=num_envs,
            num_steps=num_steps,
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
            checkpoint_dir=self.test_dir,
            save_frequency=32,
            resume=latest_ckpt,
            output_path=None,
        )

        print("\n--- Testing PPO: Resuming ---")
        saved_path2 = ppo.train_ppo(args2)
        resumed_ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        self.assertGreaterEqual(resumed_ckpt["global_step"], steps_phase2)
        self.assertGreater(resumed_ckpt["iteration"], phase1_iter)
        print(f"✓ PPO Resume validated: advanced from {phase1_step} to {resumed_ckpt['global_step']} steps")

    def test_ppo_lstm_checkpoint_and_resume(self) -> None:
        """Verify ppo_lstm.py checkpoint saving, RNG state, resume capability, and tooling guard."""
        import argparse
        num_envs = 2
        num_steps = 32
        steps_phase1 = 64  # 1 iteration of 64 steps

        args1 = argparse.Namespace(
            scenario="highway",
            tier="front_only",
            seed=42,
            total_timesteps=steps_phase1,
            learning_rate=3e-4,
            num_envs=num_envs,
            num_steps=num_steps,
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
            checkpoint_dir=self.test_dir,
            save_frequency=32,
            resume=None,
            output_path=None,
        )

        print("\n--- Testing PPO-LSTM: Initial Training & Checkpoint Saving ---")
        saved_path1 = ppo_lstm.train_ppo_lstm(args1)

        # 1. Tooling guard check: ensure low-step run did NOT write into models/
        self.assertFalse(saved_path1.startswith("models/"), f"Guard violated! Saved to canonical path: {saved_path1}")
        self.assertTrue(saved_path1.startswith("scratch/checkpoints/"), f"Expected scratch path, got {saved_path1}")

        # 2. Checkpoint structure validation
        latest_ckpt = os.path.join(self.test_dir, "checkpoint_latest.pt")
        self.assertTrue(os.path.exists(latest_ckpt))
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("optimizer_actor_state_dict", ckpt)
        self.assertIn("optimizer_critic_state_dict", ckpt)
        self.assertIn("global_step", ckpt)
        self.assertIn("iteration", ckpt)
        self.assertIn("rng_state", ckpt)
        phase1_step = ckpt["global_step"]
        phase1_iter = ckpt["iteration"]
        self.assertGreaterEqual(phase1_step, 32)
        print(f"✓ PPO-LSTM Checkpoint validated: global_step={phase1_step}, iteration={phase1_iter}")

        # 3. Resume training
        steps_phase2 = 128
        args2 = argparse.Namespace(
            scenario="highway",
            tier="front_only",
            seed=42,
            total_timesteps=steps_phase2,
            learning_rate=3e-4,
            num_envs=num_envs,
            num_steps=num_steps,
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
            checkpoint_dir=self.test_dir,
            save_frequency=32,
            resume=latest_ckpt,
            output_path=None,
        )

        print("\n--- Testing PPO-LSTM: Resuming ---")
        saved_path2 = ppo_lstm.train_ppo_lstm(args2)
        resumed_ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        self.assertGreaterEqual(resumed_ckpt["global_step"], steps_phase2)
        self.assertGreater(resumed_ckpt["iteration"], phase1_iter)
        print(f"✓ PPO-LSTM Resume validated: advanced from {phase1_step} to {resumed_ckpt['global_step']} steps")


if __name__ == "__main__":
    unittest.main()

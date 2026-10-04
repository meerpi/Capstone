"""SB3 PPO training and evaluation runner for SOTA parity ladder.

Supports:
- Rung selection (R0 to R7)
- Seeds: 0, 1, 2
- Total timesteps: 20k (R0 Farama script) or 200k (ladder)
- Saves checkpoints strictly under parity/checkpoints/ (models/ is read-only)
- Evaluates on fixed seeds 6000-6099
"""

import argparse
import os
from pathlib import Path
import sys
import gymnasium as gym
import numpy as np

sys.path.insert(0, ".")
from parity.env_factory import make_ladder_env
from parity.evaluator import evaluate_policy


def train_and_eval_sb3(rung: str, seed: int, total_timesteps: int = 200000, gamma: float = 0.8):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    save_dir = Path("parity/checkpoints") / f"sb3_{rung}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "model.zip"

    n_cpu = 6
    batch_size = 64
    n_steps = batch_size * 12 // n_cpu  # 128

    env_fn = lambda: make_ladder_env(rung)

    vec_env = make_vec_env(
        env_fn,
        n_envs=n_cpu,
        seed=seed,
        vec_env_cls=DummyVecEnv if n_cpu <= 1 else SubprocVecEnv,
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=gamma,
        device="cpu",
        verbose=1,
        seed=seed,
    )

    print(f"\n[SB3] Training {rung} seed={seed} steps={total_timesteps} gamma={gamma}...")
    model.learn(total_timesteps=total_timesteps)
    model.save(str(model_path.with_suffix("")))
    print(f"[SB3] Saved model to {model_path}")
    vec_env.close()

    # Evaluate on seeds 6000-6099
    print(f"[SB3] Evaluating {rung} seed={seed} on seeds 6000-6099...")
    eval_env_fn = lambda: make_ladder_env(rung)
    results = evaluate_policy(eval_env_fn, model, seed_start=6000, n_episodes=100)
    results["rung"] = rung
    results["seed"] = seed
    results["total_timesteps"] = total_timesteps
    results["method"] = "SB3_PPO"

    print(f"[SB3 Result] Crash: {results['ci_str']} | Speed: {results['mean_speed_ms']:.2f} m/s | Return: {results['mean_return']:.2f} | Dom: {results['dominant_action']} ({results['dominant_action_frac']:.1%})")
    
    import json
    res_path = Path("parity") / f"results_sb3_{rung}_s{seed}_{total_timesteps}.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SB3] Saved results to {res_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rung", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--gamma", type=float, default=0.8)
    args = parser.parse_args()

    train_and_eval_sb3(args.rung, args.seed, args.steps, args.gamma)

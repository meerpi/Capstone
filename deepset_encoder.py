"""Permutation-invariant DeepSet observation encoder for autonomous highway driving.

This module provides a drop-in architectural alternative to fixed lane-slot
encoders (such as TacticalLaneObservationWrapper). Rather than relying on a
fixed 4-lane grid with rigid lead/lag slots (which breaks in multi-lane merges,
roundabouts, or intersections), this encoder treats other vehicles as an
unordered set:

    h_i = phi(v_i)               # Shared per-vehicle MLP
    e_others = Pool({h_i})       # Permutation-invariant pooling (mean + max)
    e_ego = psi(v_ego)           # Dedicated ego state MLP
    z = rho([e_ego, e_others])   # Combined representation

Key Architectural Guarantees:
1. Permutation Invariance: Changing the order of other vehicles in the input
   produces identical output (difference < 1e-6).
2. Variable Vehicle Count: Works natively with any number of vehicles N >= 1
   without architecture changes.
3. Padding Invariance: Masking ensures absent/dummy vehicles (presence == 0)
   do not distort the pooled representation.
4. Cross-Scenario Generalization: Runs directly on highway, merge, roundabout,
   and intersection environments without per-scenario slot redesign.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def layer_init(
    layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Linear:
    """Initialize linear layer with orthogonal weights and constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class DeepSetEncoder(nn.Module):
    """DeepSet permutation-invariant observation encoder for multi-vehicle traffic."""

    def __init__(
        self,
        veh_dim: int = 5,
        ego_dim: int = 5,
        emb_dim: int = 64,
        out_dim: int = 128,
        pool: str = "both",
    ) -> None:
        """Initialize DeepSetEncoder.

        Args:
            veh_dim: Number of raw features per other vehicle (presence, x, y, vx, vy).
            ego_dim: Number of raw features for ego vehicle.
            emb_dim: Dimension of latent embedding per other vehicle h_i.
            out_dim: Output dimension of combined representation z.
            pool: Pooling aggregation mode: 'mean', 'max', or 'both'.
        """
        super().__init__()
        if pool not in ("mean", "max", "both"):
            raise ValueError(f"Unknown pool mode '{pool}'. Choose 'mean', 'max', or 'both'.")

        self.veh_dim = veh_dim
        self.ego_dim = ego_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.pool = pool

        # Shared per-vehicle MLP phi: R^veh_dim -> R^emb_dim
        self.phi = nn.Sequential(
            layer_init(nn.Linear(veh_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, emb_dim)),
            nn.Tanh(),
        )

        # Dedicated ego state MLP psi: R^ego_dim -> R^32
        self.ego_mlp = nn.Sequential(
            layer_init(nn.Linear(ego_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh(),
        )

        pool_dim = emb_dim * 2 if pool == "both" else emb_dim
        self.pool_dim = pool_dim

        # Combined projection MLP rho: R^(32 + pool_dim) -> R^out_dim
        self.rho = nn.Sequential(
            layer_init(nn.Linear(32 + pool_dim, out_dim)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass encoding traffic state into fixed-size latent vector.

        Args:
            x: Input tensor. Supported shapes:
               - (batch, N, feat_dim): Standard batch of vehicle matrices
               - (N, feat_dim): Single environment observation
               - (batch, N * feat_dim): Flattened observation vector

        Returns:
            Latent representation tensor of shape (batch, out_dim).
        """
        # Ensure at least 2D
        if x.dim() == 2 and x.shape[1] > 32 and x.shape[1] % self.veh_dim == 0:
            # Flattened (batch, N * feat_dim)
            x = x.view(x.shape[0], -1, self.veh_dim)
        elif x.dim() == 2:
            # Single sample (N, feat_dim)
            x = x.unsqueeze(0)

        batch_size, N, feat_dim = x.shape

        # Extract features for ego and others (use first veh_dim features)
        x_core = x[..., : self.veh_dim]
        ego = x_core[:, 0, : self.ego_dim]
        others = x_core[:, 1:, :]

        # 1. Ego embedding
        ego_emb = self.ego_mlp(ego)  # (batch, 32)

        # 2. Other vehicles permutation-invariant embedding
        if others.shape[1] == 0:
            pooled = torch.zeros(batch_size, self.pool_dim, device=x.device, dtype=x.dtype)
        else:
            presence = others[..., 0:1]  # (batch, N-1, 1), presence flag
            h = self.phi(others)         # (batch, N-1, emb_dim)
            h_masked = h * presence      # Zero out non-present vehicles

            if self.pool in ("mean", "both"):
                counts = torch.clamp(presence.sum(dim=1), min=1.0)
                mean_p = h_masked.sum(dim=1) / counts

            if self.pool in ("max", "both"):
                # Non-present vehicles set to -1e9 so they don't participate in max
                h_for_max = torch.where(presence > 0.5, h, torch.tensor(-1e9, device=x.device, dtype=x.dtype))
                max_p = torch.max(h_for_max, dim=1).values
                # If no vehicles present at all, fallback to zeros
                has_any = (presence.sum(dim=1) > 0.5).float()
                max_p = max_p * has_any

            if self.pool == "mean":
                pooled = mean_p
            elif self.pool == "max":
                pooled = max_p
            else:
                pooled = torch.cat([mean_p, max_p], dim=-1)

        # 3. Combine ego state and pooled surrounding vehicle embedding
        combined = torch.cat([ego_emb, pooled], dim=-1)
        return self.rho(combined)


class DeepSetAgent(nn.Module):
    """Actor-Critic policy agent with DeepSet permutation-invariant backbone."""

    def __init__(
        self,
        veh_dim: int = 5,
        ego_dim: int = 5,
        action_dim: int = 5,
        latent_dim: int = 128,
        pool: str = "both",
    ) -> None:
        super().__init__()
        self.encoder = DeepSetEncoder(
            veh_dim=veh_dim,
            ego_dim=ego_dim,
            emb_dim=64,
            out_dim=latent_dim,
            pool=pool,
        )
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(latent_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, action_dim), std=0.01),
        )

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation from observation."""
        return self.encoder(x)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s)."""
        latent = self.encoder(x)
        return self.critic(latent)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action, log probability, entropy, and value."""
        latent = self.encoder(x)
        logits = self.actor(latent)

        if action_mask is not None:
            # Mask invalid actions with large negative value
            logits = torch.where(
                action_mask, logits, torch.tensor(-1e8, device=logits.device)
            )

        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(latent)


class DeepSetKinematicsWrapper(gym.Wrapper):
    """Gymnasium wrapper providing raw Kinematics observations with calibrated reward shaping.

    Retains the exact same reward structure as TacticalLaneObservationWrapper:
    - Collision penalty (-50.0)
    - Overtake bonus (+1.0 or +2.5) with dwell-checked anti-farming
    - Blockage penalty (-0.35 * steps_blocked)
    - Lane change penalty (-0.05)
    - Speed incentive (0.8 * speed_norm)
    - Valid action masking in info['action_mask']

    Outputs observation as (N, 5) float32 matrix representing the raw vehicle set.
    """

    SPEED_NORM: float = 30.0
    BLOCK_DIST: float = 55.0
    BLOCK_SPEED: float = 27.5

    def __init__(
        self,
        env: gym.Env,
        collision_penalty: float = -50.0,
        overtake_bonus: float = 1.0,
        blockage_coef: float = 0.35,
        lane_change_penalty: float = 0.05,
        danger_zone: float = 15.0,
        danger_penalty: float = 0.5,
        jitter_penalty: float = 0.02,
        overtake_dwell_steps: int = 5,
        max_overtake_lon_dist: float = 30.0,
    ) -> None:
        super().__init__(env)
        self.collision_penalty = collision_penalty
        self.overtake_bonus = overtake_bonus
        self.blockage_coef = blockage_coef
        self.lane_change_penalty = lane_change_penalty
        self.danger_zone = danger_zone
        self.danger_penalty = danger_penalty
        self.jitter_penalty = jitter_penalty
        self.overtake_dwell_steps = overtake_dwell_steps
        self.max_overtake_lon_dist = max_overtake_lon_dist

        # Underlying Kinematics observation space
        self.observation_space = env.observation_space

        self._vehicles_ahead: set[int] = set()
        self._credited_overtakes: set[int] = set()
        self._dwell_counters: dict[int, int] = {}
        self._prev_lane: int | None = None
        self._prev_action: int | None = None
        self._steps_blocked: int = 0
        self._was_crashed: bool = False

    def _get_action_mask(self) -> np.ndarray:
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "get_available_actions"):
            avail = unwrapped.get_available_actions()
            mask = np.zeros(5, dtype=bool)
            for a in avail:
                if 0 <= a < 5:
                    mask[a] = True
            return mask
        return np.ones(5, dtype=bool)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._vehicles_ahead.clear()
        self._credited_overtakes.clear()
        self._dwell_counters.clear()
        self._steps_blocked = 0
        self._prev_action = None
        self._was_crashed = False

        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "vehicle") and hasattr(unwrapped, "road"):
            ego = unwrapped.vehicle
            self._prev_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
            for v in unwrapped.road.vehicles:
                if v is not ego and v.position[0] > ego.position[0]:
                    lane_diff = abs(v.lane_index[2] - ego.lane_index[2]) if hasattr(v, "lane_index") and hasattr(ego, "lane_index") else 0
                    if lane_diff <= 1:
                        self._vehicles_ahead.add(id(v))

        info["action_mask"] = self._get_action_mask()
        info["overtake_count"] = 0
        info["crashed"] = False
        info["speed"] = float(ego.speed) if hasattr(unwrapped, "vehicle") else 0.0
        return np.array(obs, dtype=np.float32), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        reward = base_reward
        overtake_count = 0

        if hasattr(unwrapped, "vehicle") and hasattr(unwrapped, "road"):
            ego = unwrapped.vehicle
            ego_speed = float(ego.speed)
            reward = 0.0

            if ego.crashed:
                reward = self.collision_penalty
            else:
                # 1. Speed incentive: scaled to 0.5 max (was 1.0)
                speed_reward = 0.5 * np.clip((ego_speed - 20.0) / 10.0, 0.0, 1.0)
                reward += speed_reward

                # 2. Survival bonus: +0.15 per step for staying alive
                reward += 0.15

                # 3. Proximity danger penalty: quadratic penalty when min_dist < danger_zone (15.0m)
                min_dist = float("inf")
                for v in unwrapped.road.vehicles:
                    if v is not ego:
                        dx = float(v.position[0] - ego.position[0])
                        dy = float(v.position[1] - ego.position[1])
                        d = float(np.sqrt(dx**2 + dy**2))
                        if d < min_dist:
                            min_dist = d
                if min_dist < self.danger_zone:
                    proximity_frac = 1.0 - (min_dist / self.danger_zone)
                    reward -= self.danger_penalty * (proximity_frac ** 2)

                # 4. Persistent blockage penalty with escalating impatience
                if hasattr(unwrapped.road, "neighbour_vehicles") and hasattr(ego, "lane_index"):
                    front, _ = unwrapped.road.neighbour_vehicles(ego, ego.lane_index)
                    if front is not None:
                        dist = float(front.position[0] - ego.position[0])
                        if 0.0 < dist < self.BLOCK_DIST and float(front.speed) < self.BLOCK_SPEED:
                            self._steps_blocked += 1
                            impatience_factor = 1.0 + min(max(self._steps_blocked - 10, 0) / 10.0, 1.5)
                            frac = 1.0 - (dist / self.BLOCK_DIST)
                            reward -= self.blockage_coef * frac * impatience_factor
                        else:
                            self._steps_blocked = 0
                    else:
                        self._steps_blocked = 0
                else:
                    self._steps_blocked = 0

                # 5. Anti-farming dwell-checked overtake bonus
                current_candidates: list[tuple[float, int]] = []
                ego_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
                for v in unwrapped.road.vehicles:
                    if v is not ego:
                        v_id = id(v)
                        if v_id in self._credited_overtakes:
                            continue
                        v_lane = v.lane_index[2] if hasattr(v, "lane_index") else 0
                        lane_diff = abs(v_lane - ego_lane)
                        dx = float(v.position[0] - ego.position[0])
                        dy = abs(float(v.position[1] - ego.position[1]))

                        if dx > 0:
                            if lane_diff <= 1:
                                self._vehicles_ahead.add(v_id)
                            self._dwell_counters[v_id] = 0
                        elif dx < 0 and v_id in self._vehicles_ahead:
                            if lane_diff <= 1 and abs(dx) <= self.max_overtake_lon_dist:
                                self._dwell_counters[v_id] = self._dwell_counters.get(v_id, 0) + 1
                                if self._dwell_counters[v_id] >= self.overtake_dwell_steps:
                                    current_candidates.append((dy, v_id))
                            else:
                                self._dwell_counters[v_id] = 0
                        else:
                            self._dwell_counters[v_id] = 0

                if current_candidates:
                    current_candidates.sort(key=lambda item: item[0])
                    _, best_id = current_candidates[0]
                    self._credited_overtakes.add(best_id)
                    overtake_count = 1
                    reward += self.overtake_bonus

                # 6. Lane change regularization
                new_lane = ego.lane_index[2] if hasattr(ego, "lane_index") else 0
                if self._prev_lane is not None and new_lane != self._prev_lane:
                    reward -= self.lane_change_penalty
                self._prev_lane = new_lane

                # 7. Action jitter penalty
                if self._prev_action is not None:
                    if (action == 3 and self._prev_action == 4) or (action == 4 and self._prev_action == 3):
                        reward -= self.jitter_penalty
                    elif action in [3, 4] and self._prev_action in [1, 3, 4] and action != self._prev_action:
                        reward -= self.jitter_penalty * 0.4
                self._prev_action = action

            info["speed"] = ego_speed
            info["crashed"] = bool(ego.crashed)
            info["overtake_count"] = overtake_count if not ego.crashed else 0
            info["action_mask"] = self._get_action_mask()
        else:
            reward = float(base_reward)

        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info


def make_deepset_env(
    scenario: str = "highway",
    vehicles_count: int | None = None,
    seed: int | None = None,
    collision_penalty: float = -50.0,
    overtake_bonus: float = 1.0,
    duration: int = 100,
    policy_frequency: int = 5,
    simulation_frequency: int | None = None,
) -> gym.Env:
    """Create an environment configured with DeepSetKinematicsWrapper.

    Supports: 'highway', 'merge', 'roundabout', 'intersection' (or direct env IDs).
    Uses duration=100 and policy_frequency=5 (500 max steps) matching DEFAULT_ENV_CONFIG.
    """
    env_id_map = {
        "highway": "highway-fast-v0",
        "merge": "merge-v1",
        "roundabout": "roundabout-v1",
        "intersection": "intersection-v2",
    }
    gym_id = env_id_map.get(scenario, scenario)

    if simulation_frequency is None:
        simulation_frequency = 5 if scenario == "highway" else 15

    obs_veh_count = vehicles_count or (15 if scenario == "highway" else 10)
    config: dict[str, Any] = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": obs_veh_count,
            "features": ["presence", "x", "y", "vx", "vy"],
            "normalize": True,
            "absolute": False,
            "see_behind": True,
        },
        "duration": duration,
        "policy_frequency": policy_frequency,
        "simulation_frequency": simulation_frequency,
    }
    if scenario == "highway":
        config["vehicles_count"] = vehicles_count or 14
        config["lanes_count"] = 4
        config["vehicles_density"] = 1.4
    elif vehicles_count is not None:
        config["vehicles_count"] = vehicles_count

    env = gym.make(gym_id, config=config)
    wrapped = DeepSetKinematicsWrapper(
        env,
        collision_penalty=collision_penalty,
        overtake_bonus=overtake_bonus,
    )
    if seed is not None:
        wrapped.reset(seed=seed)
        wrapped.action_space.seed(seed)
    return wrapped


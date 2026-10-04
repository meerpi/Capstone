"""DeepSet v2 actor-critic agent for highway-fast-v0.

Ego-conditioned Deep Set with separate actor/critic encoders, parameter-matched
to the legacy 3×256 OptimalAgent within ±15%.

Architecture:
    phi(v_i; ego): shared MLP over [vehicle_feats ; ego] → emb_dim
    pooling: masked (sum/N_max) concat masked max
    rho([pooled ; ego]) → 3×256 Tanh → head

Input: flat obs (B, 74) → ego (B, 4), rows (B, 14, 5).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from deepset_v2.env import EGO_DIM, N_FEATURES, N_VEHICLES

# ---- Constants ----
PHI_WIDTH = 64
PHI_LAYERS = 2
RHO_WIDTH = 256
RHO_LAYERS = 3
VEH_FEAT_DIM = N_FEATURES - 1  # x, y, vx, vy (presence stripped)
ACTION_DIM = 5


def _layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init + constant bias, matching legacy OptimalAgent."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class _DeepSetEncoder(nn.Module):
    """Ego-conditioned Deep Set encoder.

    phi takes [vehicle_feats(4) ; ego(4)] → emb_dim.
    Pooling: masked sum/N_max concat masked max.
    rho takes [pooled(2*emb_dim) ; ego(4)] → 3×256 → output.
    """

    def __init__(self, role: str = "actor") -> None:
        super().__init__()
        self.role = role
        phi_in = VEH_FEAT_DIM + EGO_DIM  # 4 + 4 = 8

        # phi: 2-layer MLP, ego-conditioned
        self.phi = nn.Sequential(
            _layer_init(nn.Linear(phi_in, PHI_WIDTH)),
            nn.Tanh(),
            _layer_init(nn.Linear(PHI_WIDTH, PHI_WIDTH)),
            nn.Tanh(),
        )

        pool_dim = PHI_WIDTH * 2  # sum + max
        rho_in = pool_dim + EGO_DIM  # pooled + raw ego

        # rho: 3×256 Tanh
        self.rho = nn.Sequential(
            _layer_init(nn.Linear(rho_in, RHO_WIDTH)),
            nn.Tanh(),
            _layer_init(nn.Linear(RHO_WIDTH, RHO_WIDTH)),
            nn.Tanh(),
            _layer_init(nn.Linear(RHO_WIDTH, RHO_WIDTH)),
            nn.Tanh(),
        )

    def forward(self, ego: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            ego: (B, EGO_DIM) ego features.
            rows: (B, N, VEH_FEAT_DIM) vehicle features (presence stripped).
            mask: (B, N) boolean, True where vehicle is present.

        Returns:
            (B, RHO_WIDTH) encoded representation.
        """
        B, N, _ = rows.shape

        # Ego-conditioned phi input: [vehicle_feats ; ego] broadcast over N
        ego_exp = ego.unsqueeze(1).expand(B, N, EGO_DIM)  # (B, N, 4)
        phi_in = torch.cat([rows, ego_exp], dim=-1)  # (B, N, 8)
        h = self.phi(phi_in)  # (B, N, PHI_WIDTH)

        # Mask: zero out non-present vehicles
        mask_f = mask.unsqueeze(-1).float()  # (B, N, 1)
        h_masked = h * mask_f

        # Sum pooling, divided by N_max (constant=14) for scale stability
        h_sum = h_masked.sum(dim=1) / N_VEHICLES  # (B, PHI_WIDTH)

        # Max pooling with safe -inf masking
        h_for_max = h.clone()
        h_for_max[~mask] = -1e9
        h_max = h_for_max.max(dim=1).values  # (B, PHI_WIDTH)
        # Zero out if no vehicles present
        any_present = mask.any(dim=1, keepdim=True).float()  # (B, 1)
        h_max = h_max * any_present

        pooled = torch.cat([h_sum, h_max], dim=-1)  # (B, 2*PHI_WIDTH)

        # rho: [pooled ; ego]
        rho_in = torch.cat([pooled, ego], dim=-1)  # (B, 2*PHI_WIDTH + EGO_DIM)
        return self.rho(rho_in)  # (B, RHO_WIDTH)


class DeepSetAgent(nn.Module):
    """Actor-Critic agent with ego-conditioned Deep Set backbone.

    Separate actor and critic encoders (no weight sharing), matching legacy
    OptimalAgent. Compatible method signatures for drop-in PPO loop usage.
    """

    def __init__(self) -> None:
        super().__init__()
        # Separate encoders
        self._actor_enc = _DeepSetEncoder(role="actor")
        self._critic_enc = _DeepSetEncoder(role="critic")

        # Heads
        self.actor_head = _layer_init(nn.Linear(RHO_WIDTH, ACTION_DIM), std=0.01)
        self.critic_head = _layer_init(nn.Linear(RHO_WIDTH, 1), std=1.0)

        # Config for serialization
        self.config = {
            "phi_width": PHI_WIDTH,
            "phi_layers": PHI_LAYERS,
            "rho_width": RHO_WIDTH,
            "rho_layers": RHO_LAYERS,
            "veh_feat_dim": VEH_FEAT_DIM,
            "ego_dim": EGO_DIM,
            "n_vehicles": N_VEHICLES,
            "n_features": N_FEATURES,
            "action_dim": ACTION_DIM,
            "pooling": "sum_div_nmax+max",
            "x_scale": 100.0,
            "y_scale": 4.0,
            "vx_scale": 30.0,
            "vy_scale": 30.0,
            "speed_scale": 30.0,
        }

    def _parse_obs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Parse flat observation into ego, vehicle rows, and mask.

        Args:
            x: (B, 74) or (74,) flat observation.

        Returns:
            ego: (B, 4)
            feats: (B, 14, 4) vehicle features (x, y, vx, vy)
            mask: (B, 14) boolean presence mask
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B = x.shape[0]
        ego = x[:, :EGO_DIM]  # (B, 4)
        veh_flat = x[:, EGO_DIM:]  # (B, 14*5)
        veh = veh_flat.view(B, N_VEHICLES, N_FEATURES)  # (B, 14, 5)
        presence = veh[:, :, 0]  # (B, 14)
        feats = veh[:, :, 1:]  # (B, 14, 4) — x, y, vx, vy
        mask = presence > 0.5
        return ego, feats, mask

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute state value V(s)."""
        ego, feats, mask = self._parse_obs(x)
        z = self._critic_enc(ego, feats, mask)
        return self.critic_head(z)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action, log probability, entropy, and value.

        Signature-compatible with legacy OptimalAgent.
        """
        ego, feats, mask = self._parse_obs(x)

        # Actor
        z_actor = self._actor_enc(ego, feats, mask)
        logits = self.actor_head(z_actor)

        if action_mask is not None:
            logits = torch.where(
                action_mask, logits, torch.tensor(-1e8, device=logits.device, dtype=logits.dtype)
            )

        probs = Categorical(logits=logits)

        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = probs.sample()

        # Critic
        z_critic = self._critic_enc(ego, feats, mask)
        value = self.critic_head(z_critic)

        return action, probs.log_prob(action), probs.entropy(), value

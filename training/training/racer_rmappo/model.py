"""Shared recurrent actor and permutation-equivariant centralized critic."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def _orthogonal_init(module: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class DepthEncoder(nn.Module):
    def __init__(self, frame_stack: int, output_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(frame_stack, 16, 5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((5, 8)),
            nn.Flatten(),
            nn.Linear(32 * 5 * 8, output_dim),
            nn.LayerNorm(output_dim),
            nn.ELU(),
        )
        self.apply(_orthogonal_init)

    def forward(self, depth: Tensor) -> Tensor:
        return self.network(depth)


class SharedRecurrentActor(nn.Module):
    """One actor shared by every UAV; execution needs no centralized state."""

    def __init__(
        self,
        frame_stack: int = 3,
        ego_dim: int = 11,
        target_dim: int = 7,
        neighbor_dim: int = 8,
        hidden_size: int = 256,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.depth_encoder = DepthEncoder(frame_stack, 128)
        self.ego_target_encoder = nn.Sequential(
            nn.Linear(ego_dim + target_dim, 96), nn.LayerNorm(96), nn.ELU()
        )
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, 64), nn.ELU(), nn.Linear(64, 64), nn.ELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 96 + 64, hidden_size), nn.LayerNorm(hidden_size), nn.ELU()
        )
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.action_mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.action_mean.weight, 0.01)

    def initial_hidden(self, batch: int, agents: int, device: torch.device | str) -> Tensor:
        return torch.zeros(batch, agents, self.hidden_size, device=device)

    def encode(self, observation: Dict[str, Tensor]) -> Tensor:
        depth = observation["depth"]
        ego = observation["ego"]
        target = observation["target"]
        neighbors = observation["neighbors"]
        leading = depth.shape[:-3]
        depth_features = self.depth_encoder(depth.reshape(-1, *depth.shape[-3:])).reshape(*leading, -1)
        ego_target = self.ego_target_encoder(torch.cat((ego, target), dim=-1))

        neighbor_features = self.neighbor_encoder(neighbors)
        valid = neighbors[..., -1:] > 0.5
        masked = neighbor_features.masked_fill(~valid, torch.finfo(neighbor_features.dtype).min)
        pooled = masked.max(dim=-2).values
        any_valid = valid.any(dim=-2)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return self.fusion(torch.cat((depth_features, ego_target, pooled), dim=-1))

    def _distribution(self, feature: Tensor) -> Normal:
        mean = self.action_mean(feature)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw_action: Tensor, action: Tensor) -> Tensor:
        correction = torch.log(1.0 - action.square() + 1e-6)
        return (distribution.log_prob(raw_action) - correction).sum(dim=-1)

    def step(
        self,
        observation: Dict[str, Tensor],
        hidden: Tensor,
        deterministic: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        feature = self.encode(observation)
        next_hidden = self.gru(feature.reshape(-1, feature.shape[-1]), hidden.reshape(-1, self.hidden_size))
        next_hidden = next_hidden.reshape(*feature.shape[:-1], self.hidden_size)
        distribution = self._distribution(next_hidden)
        raw_action = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw_action)
        log_prob = self._squashed_log_prob(distribution, raw_action, action)
        entropy = distribution.entropy().sum(dim=-1)
        return action, log_prob, entropy, next_hidden

    def evaluate_actions(
        self,
        observation: Dict[str, Tensor],
        hidden: Tensor,
        actions: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        feature = self.encode(observation)
        next_hidden = self.gru(feature.reshape(-1, feature.shape[-1]), hidden.reshape(-1, self.hidden_size))
        next_hidden = next_hidden.reshape(*feature.shape[:-1], self.hidden_size)
        distribution = self._distribution(next_hidden)
        bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = torch.atanh(bounded)
        log_prob = self._squashed_log_prob(distribution, raw_action, bounded)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, next_hidden

    def evaluate_sequence(
        self,
        observations: Dict[str, Tensor],
        initial_hidden: Tensor,
        actions: Tensor,
        episode_start: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        hidden = initial_hidden
        log_probs = []
        entropies = []
        for step in range(actions.shape[0]):
            hidden = hidden * (~episode_start[step]).unsqueeze(-1)
            step_observation = {key: value[step] for key, value in observations.items()}
            log_prob, entropy, hidden = self.evaluate_actions(step_observation, hidden, actions[step])
            log_probs.append(log_prob)
            entropies.append(entropy)
        return torch.stack(log_probs), torch.stack(entropies)


class CentralizedCritic(nn.Module):
    """Critic sees every agent during training and returns one value per agent."""

    def __init__(self, state_dim: int = 13, hidden_size: int = 256) -> None:
        super().__init__()
        self.agent_encoder = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.ELU(), nn.Linear(128, 128), nn.ELU()
        )
        self.value_head = nn.Sequential(
            nn.Linear(128 * 3 + 1, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )
        self.apply(_orthogonal_init)
        nn.init.orthogonal_(self.value_head[-1].weight, 1.0)

    def forward(self, state: Tensor, agent_mask: Tensor | None = None) -> Tensor:
        feature = self.agent_encoder(state)
        if agent_mask is None:
            agent_mask = torch.ones(state.shape[:-1], dtype=torch.bool, device=state.device)
        weights = agent_mask.to(feature.dtype).unsqueeze(-1)
        mean = (feature * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)
        masked = feature.masked_fill(~agent_mask.unsqueeze(-1), torch.finfo(feature.dtype).min)
        maximum = masked.max(dim=-2).values
        maximum = torch.where(agent_mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        coverage = state[..., -2].mean(dim=-1, keepdim=True)
        global_feature = torch.cat((mean, maximum, coverage), dim=-1)
        expanded_global = global_feature.unsqueeze(-2).expand(*feature.shape[:-1], global_feature.shape[-1])
        return self.value_head(torch.cat((feature, expanded_global), dim=-1)).squeeze(-1)

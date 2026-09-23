"""Rollout storage and generalized advantage estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import Tensor


@dataclass
class RolloutStorage:
    observations: Dict[str, List[Tensor]] = field(default_factory=dict)
    critic_states: List[Tensor] = field(default_factory=list)
    actions: List[Tensor] = field(default_factory=list)
    log_probs: List[Tensor] = field(default_factory=list)
    values: List[Tensor] = field(default_factory=list)
    rewards: List[Tensor] = field(default_factory=list)
    dones: List[Tensor] = field(default_factory=list)
    episode_starts: List[Tensor] = field(default_factory=list)
    hidden_states: List[Tensor] = field(default_factory=list)

    def add(
        self,
        observation: Dict[str, Tensor],
        critic_state: Tensor,
        action: Tensor,
        log_prob: Tensor,
        value: Tensor,
        reward: Tensor,
        done: Tensor,
        episode_start: Tensor,
        hidden: Tensor,
    ) -> None:
        for key, value_tensor in observation.items():
            self.observations.setdefault(key, []).append(value_tensor.detach())
        self.critic_states.append(critic_state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.values.append(value.detach())
        self.rewards.append(reward.detach())
        self.dones.append(done.detach())
        self.episode_starts.append(episode_start.detach())
        self.hidden_states.append(hidden.detach())

    def stack(self) -> Dict[str, object]:
        return {
            "observations": {key: torch.stack(values) for key, values in self.observations.items()},
            "critic_states": torch.stack(self.critic_states),
            "actions": torch.stack(self.actions),
            "log_probs": torch.stack(self.log_probs),
            "values": torch.stack(self.values),
            "rewards": torch.stack(self.rewards),
            "dones": torch.stack(self.dones),
            "episode_starts": torch.stack(self.episode_starts),
            "hidden_states": torch.stack(self.hidden_states),
        }


def generalized_advantage_estimate(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    bootstrap_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(bootstrap_value)
    next_value = bootstrap_value
    for step in reversed(range(rewards.shape[0])):
        nonterminal = (~dones[step]).to(rewards.dtype)
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
        next_value = values[step]
    return advantages, advantages + values

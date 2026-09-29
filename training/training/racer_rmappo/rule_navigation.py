"""Deterministic navigation baseline for Isaac lifecycle acceptance tests."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor


def proportional_navigation_action(
    observation: Mapping[str, Tensor],
    target_cfg: Mapping[str, Any],
    action_limits: Mapping[str, Any],
    *,
    position_gain: float,
    yaw_gain: float,
    max_cruise_speed_mps: float,
) -> Tensor:
    """Convert the actor target feature into a bounded hybrid action.

    This is an acceptance-test controller, not an expert policy and not a PPO
    training target. Candidate selection/residual components remain zero.
    """
    if position_gain <= 0.0 or yaw_gain <= 0.0:
        raise ValueError("controller gains must be positive")
    if max_cruise_speed_mps <= 0.0:
        raise ValueError("max_cruise_speed_mps must be positive")
    target = observation["target"]
    if target.shape[-1] != 7:
        raise ValueError("target observation must end in seven features")

    relative_body_m = target[..., :3] * float(
        target_cfg["distance_normalizer_m"]
    )
    desired_velocity = relative_body_m * float(position_gain)
    cruise_limit = min(
        float(max_cruise_speed_mps), float(action_limits["speed_norm_mps"])
    )
    speed = desired_velocity.norm(dim=-1, keepdim=True)
    desired_velocity = desired_velocity * (
        cruise_limit / speed.clamp_min(1e-6)
    ).clamp(max=1.0)

    forward_limit = torch.where(
        desired_velocity[..., 0] >= 0.0,
        torch.as_tensor(
            float(action_limits["forward_mps"]),
            dtype=target.dtype,
            device=target.device,
        ),
        torch.as_tensor(
            float(action_limits["backward_mps"]),
            dtype=target.dtype,
            device=target.device,
        ),
    )
    normalized_velocity = torch.stack(
        (
            desired_velocity[..., 0] / forward_limit,
            desired_velocity[..., 1] / float(action_limits["lateral_mps"]),
            desired_velocity[..., 2] / float(action_limits["vertical_mps"]),
        ),
        dim=-1,
    )
    yaw_error = torch.atan2(target[..., 4], target[..., 5])
    desired_yaw_rate = (float(yaw_gain) * yaw_error).clamp(
        -float(action_limits["yaw_rate_rps"]),
        float(action_limits["yaw_rate_rps"]),
    )
    normalized_yaw_rate = desired_yaw_rate / float(
        action_limits["yaw_rate_rps"]
    )

    action = torch.zeros(*target.shape[:-1], 9, dtype=target.dtype, device=target.device)
    action[..., :3] = normalized_velocity.clamp(-1.0, 1.0)
    action[..., 3] = normalized_yaw_rate.clamp(-1.0, 1.0)
    return action

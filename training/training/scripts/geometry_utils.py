"""Lightweight coordinate transforms shared by policies and environments.

This module intentionally depends only on PyTorch so geometry-only code and
unit tests do not import Isaac Sim, TorchRL rendering helpers, or wandb.
"""

from __future__ import annotations

import torch


def vec_to_new_frame(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """Express 3-D vectors in the frame whose x-axis is ``goal_direction``."""
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)

    goal_direction_x = goal_direction / goal_direction.norm(
        dim=-1, keepdim=True
    )
    z_direction = torch.tensor([0.0, 0.0, 1.0], device=vec.device)

    goal_direction_y = torch.cross(
        z_direction.expand_as(goal_direction_x), goal_direction_x, dim=-1
    )
    goal_direction_y /= goal_direction_y.norm(dim=-1, keepdim=True)

    goal_direction_z = torch.cross(
        goal_direction_x, goal_direction_y, dim=-1
    )
    goal_direction_z /= goal_direction_z.norm(dim=-1, keepdim=True)

    batch_size = vec.size(0)
    if vec.ndim == 3:
        vec_x_new = torch.bmm(
            vec.view(batch_size, vec.shape[1], 3),
            goal_direction_x.view(batch_size, 3, 1),
        )
        vec_y_new = torch.bmm(
            vec.view(batch_size, vec.shape[1], 3),
            goal_direction_y.view(batch_size, 3, 1),
        )
        vec_z_new = torch.bmm(
            vec.view(batch_size, vec.shape[1], 3),
            goal_direction_z.view(batch_size, 3, 1),
        )
    else:
        vec_x_new = torch.bmm(
            vec.view(batch_size, 1, 3),
            goal_direction_x.view(batch_size, 3, 1),
        )
        vec_y_new = torch.bmm(
            vec.view(batch_size, 1, 3),
            goal_direction_y.view(batch_size, 3, 1),
        )
        vec_z_new = torch.bmm(
            vec.view(batch_size, 1, 3),
            goal_direction_z.view(batch_size, 3, 1),
        )

    return torch.cat((vec_x_new, vec_y_new, vec_z_new), dim=-1)


def vec_to_world(vec: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
    """Convert vectors in the target-aligned frame back to world coordinates."""
    world_direction = torch.tensor(
        [1.0, 0.0, 0.0], device=vec.device
    ).expand_as(goal_direction)
    world_frame_new = vec_to_new_frame(world_direction, goal_direction)
    return vec_to_new_frame(vec, world_frame_new)

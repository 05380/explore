"""Interface contract for the high-fidelity Isaac Sim backend.

The old ``scripts/env.py`` is a single-drone LiDAR environment and must not be
silently used for RACER RMAPPO.  A future Isaac backend must expose this exact
shape contract so the tested trainer can be reused unchanged.
"""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

from torch import Tensor


REQUIRED_STEP_INFO = {
    "reward_components",
    "coverage",
    "collision",
    "obstacle_collision",
    "inter_drone_collision",
    "out_of_bounds",
    "stall",
    "safety_takeover",
    "goal_reached",
    "navigation_reached",
    "viewpoint_decisions",
    "episode_finished",
    "episode_success",
    "episode_collision",
    "episode_obstacle_collision",
    "episode_inter_drone_collision",
    "episode_timeout",
    "episode_stall",
    "episode_out_of_bounds",
    "episode_safety_takeover",
    "coverage_target_reached",
    "coverage_target_steps",
}


class MultiUAVBackend(Protocol):
    num_envs: int
    num_agents: int

    def reset(self) -> Tuple[Dict[str, Tensor], Tensor]: ...

    def step(
        self, hybrid_action: Tensor
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict[str, object]]: ...


def validate_backend_shapes(
    backend: MultiUAVBackend,
) -> Tuple[Dict[str, Tensor], Tensor]:
    observation, critic_state = backend.reset()
    expected = (backend.num_envs, backend.num_agents)
    if tuple(critic_state.shape[:2]) != expected:
        raise ValueError(f"critic state starts with {critic_state.shape[:2]}, expected {expected}")
    required = {"depth", "ego", "target", "neighbors", "candidates", "decision_mask"}
    if set(observation) != required:
        raise ValueError(f"observation keys must be {sorted(required)}, got {sorted(observation)}")
    for key, value in observation.items():
        if tuple(value.shape[:2]) != expected:
            raise ValueError(f"{key} starts with {value.shape[:2]}, expected {expected}")
    return observation, critic_state


def validate_step_info(info: Dict[str, object], num_envs: int) -> None:
    missing = REQUIRED_STEP_INFO - set(info)
    if missing:
        raise ValueError(f"backend step info is missing metrics: {sorted(missing)}")
    for key in REQUIRED_STEP_INFO - {"reward_components"}:
        value = info[key]
        if not isinstance(value, Tensor) or tuple(value.shape) != (num_envs,):
            shape = getattr(value, "shape", None)
            raise ValueError(f"step info {key} must have shape ({num_envs},), got {shape}")

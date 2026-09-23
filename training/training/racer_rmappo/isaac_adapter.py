"""Interface contract for the high-fidelity Isaac Sim backend.

The old ``scripts/env.py`` is a single-drone LiDAR environment and must not be
silently used for RACER RMAPPO.  A future Isaac backend must expose this exact
shape contract so the tested trainer can be reused unchanged.
"""

from __future__ import annotations

from typing import Dict, Protocol, Tuple

from torch import Tensor


class MultiUAVBackend(Protocol):
    num_envs: int
    num_agents: int

    def reset(self) -> Tuple[Dict[str, Tensor], Tensor]: ...

    def step(
        self, normalized_action: Tensor
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict[str, object]]: ...


def validate_backend_shapes(backend: MultiUAVBackend) -> None:
    observation, critic_state = backend.reset()
    expected = (backend.num_envs, backend.num_agents)
    if tuple(critic_state.shape[:2]) != expected:
        raise ValueError(f"critic state starts with {critic_state.shape[:2]}, expected {expected}")
    required = {"depth", "ego", "target", "neighbors"}
    if set(observation) != required:
        raise ValueError(f"observation keys must be {sorted(required)}, got {sorted(observation)}")
    for key, value in observation.items():
        if tuple(value.shape[:2]) != expected:
            raise ValueError(f"{key} starts with {value.shape[:2]}, expected {expected}")

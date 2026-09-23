"""CPU replay storage for wall-following teacher demonstrations."""

from __future__ import annotations

from typing import Dict

import torch
from tensordict.tensordict import TensorDict


class DemonstrationBuffer:
    """A fixed-capacity ring buffer containing only policy-visible inputs."""

    OBSERVATION_KEYS = ("state", "lidar", "direction", "dynamic_obstacle")

    def __init__(
        self,
        capacity: int,
        balanced_sampling: bool = True,
        phase_balance_exponent: float = 0.5,
    ):
        self.capacity = max(int(capacity), 1)
        self.balanced_sampling = bool(balanced_sampling)
        self.phase_balance_exponent = min(
            max(float(phase_balance_exponent), 0.0),
            1.0,
        )
        self._storage: Dict[str, torch.Tensor] = {}
        self._size = 0
        self._position = 0

    def __len__(self) -> int:
        return self._size

    def _initialize(self, observation, action: torch.Tensor):
        for key in self.OBSERVATION_KEYS:
            value = observation[key]
            self._storage[key] = torch.empty(
                (self.capacity, *value.shape[1:]),
                dtype=value.dtype,
                device="cpu",
            )
        self._storage["teacher_action_normalized"] = torch.empty(
            (self.capacity, *action.shape[1:]),
            dtype=action.dtype,
            device="cpu",
        )
        self._storage["teacher_confidence"] = torch.empty(
            self.capacity,
            dtype=torch.float32,
            device="cpu",
        )
        self._storage["teacher_phase"] = torch.empty(
            self.capacity,
            dtype=torch.long,
            device="cpu",
        )

    @torch.no_grad()
    def add(
        self,
        observation,
        action: torch.Tensor,
        confidence: torch.Tensor,
        phase: torch.Tensor,
        mask: torch.Tensor,
    ) -> int:
        """Append masked demonstrations and return the number written."""
        mask = mask.detach().reshape(-1).bool()
        selected = mask.nonzero(as_tuple=False).squeeze(-1)
        if selected.numel() == 0:
            return 0

        selected_observation = {
            key: observation[key].detach().index_select(0, selected).to("cpu")
            for key in self.OBSERVATION_KEYS
        }
        selected_action = action.detach().index_select(0, selected).to("cpu")
        selected_confidence = confidence.detach().reshape(-1).index_select(0, selected).float().to("cpu")
        selected_phase = phase.detach().reshape(-1).index_select(0, selected).long().to("cpu")

        if not self._storage:
            self._initialize(selected_observation, selected_action)

        count = selected.numel()
        if count > self.capacity:
            start = count - self.capacity
            selected_observation = {key: value[start:] for key, value in selected_observation.items()}
            selected_action = selected_action[start:]
            selected_confidence = selected_confidence[start:]
            selected_phase = selected_phase[start:]
            count = self.capacity

        indices = (torch.arange(count, device="cpu") + self._position) % self.capacity
        for key, value in selected_observation.items():
            self._storage[key][indices] = value
        self._storage["teacher_action_normalized"][indices] = selected_action
        self._storage["teacher_confidence"][indices] = selected_confidence
        self._storage["teacher_phase"][indices] = selected_phase

        self._position = (self._position + count) % self.capacity
        self._size = min(self._size + count, self.capacity)
        return int(count)

    def _sample_indices(self, batch_size: int) -> torch.Tensor:
        batch_size = max(int(batch_size), 1)
        if not self.balanced_sampling or self._size == 0:
            return torch.randint(self._size, (batch_size,), device="cpu")

        phases = self._storage["teacher_phase"][: self._size]
        available_phases, inverse, counts = torch.unique(
            phases,
            return_inverse=True,
            return_counts=True,
        )
        if available_phases.numel() <= 1:
            return torch.randint(self._size, (batch_size,), device="cpu")

        # exponent=0 is uniform per record; exponent=1 gives equal total mass
        # to every phase. The default square-root correction improves coverage
        # without repeatedly cloning a handful of rare transition records.
        sample_weights = counts[inverse].float().pow(-self.phase_balance_exponent)
        return torch.multinomial(sample_weights, batch_size, replacement=True)

    def sample(self, batch_size: int, device) -> TensorDict:
        if self._size == 0:
            raise RuntimeError("Cannot sample an empty demonstration buffer.")

        indices = self._sample_indices(batch_size)
        observation = {
            key: self._storage[key].index_select(0, indices).to(device)
            for key in self.OBSERVATION_KEYS
        }
        count = indices.numel()
        return TensorDict(
            {
                "agents": TensorDict(
                    {
                        "observation": TensorDict(observation, batch_size=[count], device=device),
                    },
                    batch_size=[count],
                    device=device,
                ),
                "teacher_action_normalized": self._storage["teacher_action_normalized"]
                .index_select(0, indices)
                .to(device),
                "teacher_confidence": self._storage["teacher_confidence"]
                .index_select(0, indices)
                .to(device),
                "teacher_phase": self._storage["teacher_phase"].index_select(0, indices).to(device),
            },
            batch_size=[count],
            device=device,
        )

    def phase_counts(self) -> Dict[int, int]:
        if self._size == 0:
            return {}
        phases, counts = torch.unique(
            self._storage["teacher_phase"][: self._size],
            return_counts=True,
        )
        return {int(phase): int(count) for phase, count in zip(phases, counts)}

    def state_dict(self) -> Dict[str, object]:
        return {
            "capacity": self.capacity,
            "balanced_sampling": self.balanced_sampling,
            "phase_balance_exponent": self.phase_balance_exponent,
            "size": self._size,
            "position": self._position,
            "storage": {key: value.clone() for key, value in self._storage.items()},
        }

    def load_state_dict(self, state: Dict[str, object]):
        if int(state["capacity"]) != self.capacity:
            raise ValueError("Demonstration buffer capacity does not match the saved state.")
        self.balanced_sampling = bool(state["balanced_sampling"])
        self.phase_balance_exponent = min(
            max(float(state.get("phase_balance_exponent", 0.5)), 0.0),
            1.0,
        )
        self._size = int(state["size"])
        self._position = int(state["position"])
        self._storage = {
            key: value.to("cpu").clone()
            for key, value in state["storage"].items()
        }

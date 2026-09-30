"""RACER local-navigation RMAPPO training components."""

from .config import apply_curriculum_stage, load_config
from .model import CentralizedCritic, SharedRecurrentActor, NavigationRecurrentActor
from .reward import RewardComposer

__all__ = [
    "CentralizedCritic",
    "RewardComposer",
    "SharedRecurrentActor",
    "NavigationRecurrentActor",
    "apply_curriculum_stage",
    "load_config",
]

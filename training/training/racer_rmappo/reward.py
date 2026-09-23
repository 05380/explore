"""Reward composition shared by smoke and Isaac Sim backends."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor


class RewardComposer:
    def __init__(self, cfg: Mapping[str, object]) -> None:
        self.cfg = cfg

    @staticmethod
    def _capped_count(count: Tensor, coefficient: float, cap: float) -> Tensor:
        value = count.to(torch.float32) * coefficient
        if coefficient >= 0.0:
            return value.clamp(max=cap)
        return value.clamp(min=-cap)

    def __call__(self, signals: Mapping[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        cfg = self.cfg
        progress = signals["target_progress_m"] * float(cfg["target_progress_per_m"])
        local_new = self._capped_count(
            signals["local_new_voxels"],
            float(cfg["newly_observed_local_voxel"]),
            float(cfg["local_voxel_reward_cap"]),
        )
        team_new = self._capped_count(
            signals["team_unique_new_voxels"],
            float(cfg["newly_observed_team_unique_voxel"]),
            float(cfg["team_voxel_reward_cap"]),
        )
        duplicate = self._capped_count(
            signals["duplicate_voxels"],
            float(cfg["duplicate_observation_voxel"]),
            float(cfg["duplicate_voxel_penalty_cap"]),
        )
        near_cfg = cfg["near_obstacle"]
        safe_clearance = torch.full_like(
            signals["obstacle_clearance_m"], float(near_cfg["safe_clearance_m"])
        )
        if bool(near_cfg.get("speed_adaptive", False)) and "speed_mps" in signals:
            speed = signals["speed_mps"].clamp_min(0.0)
            safe_clearance = safe_clearance + speed * float(near_cfg["reaction_time_s"])
            safe_clearance = safe_clearance + speed.square() / (
                2.0 * float(near_cfg["braking_deceleration_mps2"])
            )
            safe_clearance = safe_clearance.clamp(max=float(near_cfg["max_safe_clearance_m"]))
        clearance_ratio = (
            (safe_clearance - signals["obstacle_clearance_m"])
            / safe_clearance.clamp_min(1e-6)
        ).clamp(0.0, 1.0)
        near_obstacle = clearance_ratio * float(near_cfg["weight"])

        separation_cfg = cfg["inter_drone_separation"]
        separation_ratio = (
            (float(separation_cfg["safe_distance_m"]) - signals["nearest_drone_m"])
            / float(separation_cfg["safe_distance_m"])
        ).clamp(0.0, 1.0)
        separation = separation_ratio * float(separation_cfg["weight"])

        components = {
            "target_progress": progress,
            "goal_reached": signals["goal_reached"].float() * float(cfg["goal_reached"]),
            "local_new_voxels": local_new,
            "team_new_voxels": team_new,
            "duplicate_voxels": duplicate,
            "near_obstacle": near_obstacle,
            "separation": separation,
            "action_delta": signals["action_delta_l2"] * float(cfg["action_delta_l2"]),
            "vertical_action": signals["vertical_action_l2"] * float(cfg["vertical_action_l2"]),
            "stall": signals["stall"].float() * float(cfg["stall"]["penalty"]),
            "step": torch.full_like(progress, float(cfg["step"])),
            "collision": signals["collision"].float() * float(cfg["collision_terminal"]),
            "out_of_bounds": signals["out_of_bounds"].float() * float(cfg["out_of_bounds_terminal"]),
            "coverage_milestone": signals.get("coverage_milestone_reward", torch.zeros_like(progress)),
        }
        reward = torch.stack(tuple(components.values())).sum(dim=0)
        return reward, components

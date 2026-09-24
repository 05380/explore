"""Fast tensor-only contract environment.

This backend is deliberately not a physics simulator.  It validates the RMAPPO
data path, recurrent resets, reward bookkeeping and 1/4/8/16-agent tensor
shapes before an expensive Isaac Sim run.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import torch
from torch import Tensor

from .reward import RewardComposer


class ContractSmokeEnv:
    def __init__(self, cfg: Mapping[str, object], device: torch.device | str) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(cfg["training"]["num_parallel_swarms"])
        self.num_agents = int(cfg["experiment"]["num_agents"])
        self.control_hz = float(cfg["experiment"]["control_hz"])
        self.dt = 1.0 / self.control_hz
        self.world_size = torch.tensor(cfg["world"]["size_m"], device=self.device)
        self.minimum_z = float(cfg["world"]["min_flight_z_m"])
        self.maximum_z = min(float(cfg["world"]["max_flight_z_m"]), float(self.world_size[2]))
        self.resolution = float(cfg["world"]["voxel_resolution_m"])
        self.camera = cfg["camera"]
        self.depth_cfg = cfg["actor_observation"]["depth"]
        self.neighbor_cfg = cfg["actor_observation"]["neighbors"]
        self.candidate_cfg = cfg["actor_observation"]["racer_candidates"]
        self.selection_cfg = cfg["action"]["viewpoint_selection"]
        self.action_limits = cfg["action"]["physical_limits"]
        self.reward_composer = RewardComposer(cfg["reward"])
        self.max_steps = int(cfg["training"].get("smoke_episode_steps", 512))
        self.max_obstacles = int(cfg["training"].get("smoke_max_obstacles", 32))
        self.frame_stack = int(self.depth_cfg["frame_stack"])
        self.depth_width, self.depth_height = (int(v) for v in self.depth_cfg["resize"])
        self.max_neighbors = int(self.neighbor_cfg["max_neighbors"])
        self.max_candidates = int(self.candidate_cfg["max_candidates"])
        self.communication_radius = float(self.neighbor_cfg["communication_radius_m"])
        self.drone_radius = 0.30
        self.stall_steps = max(1, int(float(cfg["reward"]["stall"]["window_seconds"]) * self.control_hz))
        self.terminate_on_collision = bool(cfg["training"].get("terminate_team_on_agent_collision", True))

        grid = torch.ceil(self.world_size / self.resolution).to(torch.long)
        self.grid_shape = tuple(int(value) for value in grid.tolist())
        self.voxel_count = math.prod(self.grid_shape)
        self.positions = torch.zeros(self.num_envs, self.num_agents, 3, device=self.device)
        self.velocities = torch.zeros_like(self.positions)
        self.yaw = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.targets = torch.zeros_like(self.positions)
        self.target_yaw = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        self.candidate_positions = torch.zeros(
            self.num_envs, self.num_agents, self.max_candidates, 3, device=self.device
        )
        self.candidate_yaws = torch.zeros(
            self.num_envs, self.num_agents, self.max_candidates, device=self.device
        )
        self.candidate_gains = torch.zeros_like(self.candidate_yaws)
        self.candidate_valid = torch.zeros_like(self.candidate_yaws, dtype=torch.bool)
        self.task_decision = torch.zeros(
            self.num_envs, self.num_agents, dtype=torch.bool, device=self.device
        )
        self.previous_action = torch.zeros(self.num_envs, self.num_agents, 4, device=self.device)
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.stall_count = torch.zeros(self.num_envs, self.num_agents, dtype=torch.long, device=self.device)
        self.local_seen = torch.zeros(
            self.num_envs, self.num_agents, self.voxel_count, dtype=torch.bool, device=self.device
        )
        self.team_seen = torch.zeros(self.num_envs, self.voxel_count, dtype=torch.bool, device=self.device)
        self.depth_stack = torch.zeros(
            self.num_envs,
            self.num_agents,
            self.frame_stack,
            self.depth_height,
            self.depth_width,
            device=self.device,
        )
        self.obstacle_position = torch.zeros(self.num_envs, self.max_obstacles, 2, device=self.device)
        self.obstacle_radius = torch.zeros(self.num_envs, self.max_obstacles, device=self.device)
        self.obstacle_height = torch.zeros(self.num_envs, self.max_obstacles, device=self.device)
        self.milestone_paid = torch.zeros(
            self.num_envs, len(cfg["reward"]["team_coverage_milestones"]), dtype=torch.bool, device=self.device
        )
        self.reset()

    def _random_positions(self, count: int) -> Tensor:
        xy = (torch.rand(count, 2, device=self.device) - 0.5) * (self.world_size[:2] - 4.0)
        z = self.minimum_z + 0.5 + torch.rand(count, 1, device=self.device) * max(
            self.maximum_z - self.minimum_z - 1.0, 0.1
        )
        return torch.cat((xy, z), dim=-1)

    def _reset_envs(self, env_ids: Tensor) -> None:
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        spacing = 1.5
        side = math.ceil(math.sqrt(self.num_agents))
        ids = torch.arange(self.num_agents, device=self.device)
        formation = torch.stack((ids % side, ids // side), dim=-1).to(torch.float32)
        formation = (formation - formation.mean(dim=0, keepdim=True)) * spacing
        formation = formation.unsqueeze(0).expand(count, -1, -1)
        center = torch.zeros(count, 1, 2, device=self.device)
        self.positions[env_ids, :, :2] = center + formation
        self.positions[env_ids, :, 2] = 1.5
        self.velocities[env_ids] = 0.0
        self.yaw[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self.stall_count[env_ids] = 0
        self.local_seen[env_ids] = False
        self.team_seen[env_ids] = False
        self.depth_stack[env_ids] = 0.0
        self.milestone_paid[env_ids] = False

        for env_id in env_ids.tolist():
            self.obstacle_position[env_id] = (
                torch.rand(self.max_obstacles, 2, device=self.device) - 0.5
            ) * (self.world_size[:2] - 3.0)
            # Cylinders represent tree trunks and conservative building footprints.
            self.obstacle_radius[env_id] = 0.35 + torch.rand(self.max_obstacles, device=self.device) * 1.65
            self.obstacle_height[env_id] = 1.5 + torch.rand(self.max_obstacles, device=self.device) * max(
                self.maximum_z - 1.5, 0.2
            )
            # Keep the initial formation free.
            start_distance = torch.cdist(
                self.obstacle_position[env_id], self.positions[env_id, :, :2]
            ).min(dim=-1).values
            too_close = start_distance < (self.obstacle_radius[env_id] + 2.0)
            self.obstacle_position[env_id, too_close] += self.world_size[:2] * 0.35
            half = self.world_size[:2] * 0.5 - 1.0
            self.obstacle_position[env_id].clamp_(min=-half, max=half)
        task_mask = torch.zeros(
            self.num_envs, self.num_agents, dtype=torch.bool, device=self.device
        )
        task_mask[env_ids] = True
        self._generate_tasks(task_mask)

    def _generate_tasks(self, task_mask: Tensor) -> None:
        """Generate fixed RACER-like candidate sets for new decision events."""
        for env_id, agent_id in torch.nonzero(task_mask, as_tuple=False).tolist():
            points = self._random_positions(self.max_candidates)
            delta = points[:, None, :2] - self.obstacle_position[env_id, None, :, :]
            clearance = delta.norm(dim=-1) - self.obstacle_radius[env_id].unsqueeze(0)
            above = points[:, None, 2] > self.obstacle_height[env_id].unsqueeze(0)
            clearance = clearance.masked_fill(above, float(self.camera["max_depth_m"]))
            valid = clearance.min(dim=-1).values > (
                self.drone_radius + float(self.cfg["world"]["obstacle_inflation_m"])
            )
            if not bool(valid.any()):
                points[0] = self.positions[env_id, agent_id]
                valid[0] = True
            yaws = (torch.rand(self.max_candidates, device=self.device) * 2.0 - 1.0) * math.pi
            # Synthetic visible counts stand in for RLTask.candidate_visible_voxels.
            gains = 10.0 + torch.rand(self.max_candidates, device=self.device) * 90.0
            gains *= valid.float()
            self.candidate_positions[env_id, agent_id] = points
            self.candidate_yaws[env_id, agent_id] = yaws
            self.candidate_gains[env_id, agent_id] = gains
            self.candidate_valid[env_id, agent_id] = valid
            first = int(torch.nonzero(valid, as_tuple=False)[0].item())
            self.targets[env_id, agent_id] = points[first]
            self.target_yaw[env_id, agent_id] = yaws[first]
            self.task_decision[env_id, agent_id] = True

    def reset(self) -> Tuple[Dict[str, Tensor], Tensor]:
        self._reset_envs(torch.arange(self.num_envs, device=self.device))
        self._update_depth_stack()
        return self._observation(), self._critic_state()

    def _body_xy(self, vector: Tensor) -> Tensor:
        cosine = torch.cos(self.yaw)
        sine = torch.sin(self.yaw)
        x = cosine * vector[..., 0] + sine * vector[..., 1]
        y = -sine * vector[..., 0] + cosine * vector[..., 1]
        return torch.stack((x, y), dim=-1)

    def _world_xy(self, vector: Tensor) -> Tensor:
        cosine = torch.cos(self.yaw)
        sine = torch.sin(self.yaw)
        x = cosine * vector[..., 0] - sine * vector[..., 1]
        y = sine * vector[..., 0] + cosine * vector[..., 1]
        return torch.stack((x, y), dim=-1)

    def _obstacle_clearance(self) -> Tensor:
        delta = self.positions[:, :, None, :2] - self.obstacle_position[:, None, :, :]
        horizontal = delta.norm(dim=-1) - self.obstacle_radius[:, None, :]
        above = self.positions[:, :, None, 2] > self.obstacle_height[:, None, :]
        horizontal = horizontal.masked_fill(above, float(self.camera["max_depth_m"]))
        return horizontal.min(dim=-1).values

    def _nearest_drone(self) -> Tensor:
        if self.num_agents == 1:
            return torch.full(
                (self.num_envs, 1), float(self.communication_radius), device=self.device
            )
        distance = torch.cdist(self.positions, self.positions)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device).unsqueeze(0)
        return distance.masked_fill(eye, float("inf")).min(dim=-1).values

    def _render_inverse_depth(self) -> Tensor:
        result = torch.zeros(
            self.num_envs, self.num_agents, 1, self.depth_height, self.depth_width, device=self.device
        )
        horizontal_half = math.radians(float(self.camera["horizontal_fov_deg"]) * 0.5)
        vertical_half = math.radians(float(self.camera["vertical_fov_deg"]) * 0.5)
        minimum = float(self.camera["min_depth_m"])
        maximum = float(self.camera["max_depth_m"])
        inv_minimum = 1.0 / minimum
        inv_maximum = 1.0 / maximum
        for env_id in range(self.num_envs):
            for agent_id in range(self.num_agents):
                relative_xy = self.obstacle_position[env_id] - self.positions[env_id, agent_id, :2]
                yaw = self.yaw[env_id, agent_id]
                cosine, sine = torch.cos(yaw), torch.sin(yaw)
                forward = cosine * relative_xy[:, 0] + sine * relative_xy[:, 1]
                lateral = -sine * relative_xy[:, 0] + cosine * relative_xy[:, 1]
                azimuth = torch.atan2(lateral, forward)
                center_z = self.obstacle_height[env_id] * 0.5 - self.positions[env_id, agent_id, 2]
                distance = torch.sqrt(forward.square() + lateral.square()).clamp_min(1e-4)
                elevation = torch.atan2(center_z, distance)
                surface_distance = (distance - self.obstacle_radius[env_id]).clamp_min(minimum)
                visible = (
                    (forward > 0.0)
                    & (surface_distance <= maximum)
                    & (azimuth.abs() <= horizontal_half)
                    & (elevation.abs() <= vertical_half + 0.25)
                )
                for obstacle_id in torch.nonzero(visible, as_tuple=False).flatten().tolist():
                    column = int(
                        ((float(azimuth[obstacle_id]) / horizontal_half + 1.0) * 0.5)
                        * (self.depth_width - 1)
                    )
                    row = int(
                        ((1.0 - float(elevation[obstacle_id]) / vertical_half) * 0.5)
                        * (self.depth_height - 1)
                    )
                    pixel_radius = max(
                        1,
                        int(
                            self.depth_width
                            * float(self.obstacle_radius[env_id, obstacle_id])
                            / max(float(distance[obstacle_id]), 0.5)
                            / (2.0 * math.tan(horizontal_half))
                        ),
                    )
                    x0, x1 = max(0, column - pixel_radius), min(self.depth_width, column + pixel_radius + 1)
                    y0, y1 = max(0, row - pixel_radius), min(self.depth_height, row + pixel_radius + 1)
                    normalized = (
                        1.0 / float(surface_distance[obstacle_id]) - inv_maximum
                    ) / (inv_minimum - inv_maximum)
                    current = result[env_id, agent_id, 0, y0:y1, x0:x1]
                    result[env_id, agent_id, 0, y0:y1, x0:x1] = torch.maximum(
                        current, torch.full_like(current, normalized)
                    )
        return result

    def _update_depth_stack(self) -> None:
        newest = self._render_inverse_depth()
        self.depth_stack = torch.cat((self.depth_stack[:, :, 1:], newest), dim=2)

    def _neighbor_observation(self) -> Tensor:
        output = torch.zeros(
            self.num_envs, self.num_agents, self.max_neighbors, 8, device=self.device
        )
        if self.num_agents == 1:
            return output
        relative = self.positions[:, None, :, :] - self.positions[:, :, None, :]
        relative_velocity = self.velocities[:, None, :, :] - self.velocities[:, :, None, :]
        distance = relative.norm(dim=-1)
        eye = torch.eye(self.num_agents, dtype=torch.bool, device=self.device).unsqueeze(0)
        distance = distance.masked_fill(eye, float("inf"))
        count = min(self.max_neighbors, self.num_agents - 1)
        nearest_distance, indices = distance.topk(count, dim=-1, largest=False)
        gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, 3)
        rel = torch.gather(relative, 2, gather_index)
        rel_vel = torch.gather(relative_velocity, 2, gather_index)
        yaw = self.yaw.unsqueeze(-1)
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        rel_body_x = cosine * rel[..., 0] + sine * rel[..., 1]
        rel_body_y = -sine * rel[..., 0] + cosine * rel[..., 1]
        vel_body_x = cosine * rel_vel[..., 0] + sine * rel_vel[..., 1]
        vel_body_y = -sine * rel_vel[..., 0] + cosine * rel_vel[..., 1]
        valid = nearest_distance <= self.communication_radius
        drop_max = float(self.cfg["world"]["randomization"]["communication_drop_probability"][1])
        if drop_max > 0.0:
            valid &= torch.rand_like(nearest_distance) >= drop_max * 0.5
        output[:, :, :count, :3] = torch.stack((rel_body_x, rel_body_y, rel[..., 2]), dim=-1)
        output[:, :, :count, 3:6] = torch.stack((vel_body_x, vel_body_y, rel_vel[..., 2]), dim=-1)
        output[:, :, :count, 6] = 0.0
        output[:, :, :count, 7] = valid.float()
        output[:, :, :count, :7] *= valid.unsqueeze(-1)
        return output

    def _observation(self) -> Dict[str, Tensor]:
        body_velocity_xy = self._body_xy(self.velocities[..., :2])
        body_velocity = torch.cat((body_velocity_xy, self.velocities[..., 2:3]), dim=-1)
        target_relative = self.targets - self.positions
        target_xy = self._body_xy(target_relative[..., :2])
        target_body = torch.cat((target_xy, target_relative[..., 2:3]), dim=-1)
        target_distance = target_relative.norm(dim=-1, keepdim=True)
        target_yaw_error = self.target_yaw - self.yaw
        ego = torch.cat(
            (
                body_velocity / float(self.action_limits["speed_norm_mps"]),
                torch.zeros(self.num_envs, self.num_agents, 2, device=self.device),
                torch.sin(self.yaw).unsqueeze(-1),
                torch.cos(self.yaw).unsqueeze(-1),
                self.previous_action,
            ),
            dim=-1,
        )
        target = torch.cat(
            (
                target_body / float(self.depth_cfg["normalize_range_m"][1]),
                target_distance / float(self.depth_cfg["normalize_range_m"][1]),
                torch.sin(target_yaw_error).unsqueeze(-1),
                torch.cos(target_yaw_error).unsqueeze(-1),
                torch.zeros_like(target_distance),
            ),
            dim=-1,
        )
        candidate_relative = self.candidate_positions - self.positions.unsqueeze(-2)
        cosine, sine = torch.cos(self.yaw).unsqueeze(-1), torch.sin(self.yaw).unsqueeze(-1)
        candidate_body = torch.stack(
            (
                cosine * candidate_relative[..., 0] + sine * candidate_relative[..., 1],
                -sine * candidate_relative[..., 0] + cosine * candidate_relative[..., 1],
                candidate_relative[..., 2],
            ),
            dim=-1,
        )
        candidate_distance = candidate_relative.norm(dim=-1, keepdim=True)
        yaw_error = self.candidate_yaws - self.yaw.unsqueeze(-1)
        rank = torch.arange(self.max_candidates, device=self.device, dtype=torch.float32)
        rank = rank.view(1, 1, -1, 1) / max(self.max_candidates - 1, 1)
        rank = rank.expand(self.num_envs, self.num_agents, -1, -1)
        candidates = torch.cat(
            (
                candidate_body / float(self.candidate_cfg["distance_normalizer_m"]),
                candidate_distance / float(self.candidate_cfg["distance_normalizer_m"]),
                torch.sin(yaw_error).unsqueeze(-1),
                torch.cos(yaw_error).unsqueeze(-1),
                (self.candidate_gains / float(self.candidate_cfg["visible_gain_normalizer"])).unsqueeze(-1),
                rank,
                self.candidate_valid.float().unsqueeze(-1),
            ),
            dim=-1,
        )
        return {
            "depth": self.depth_stack.clone(),
            "ego": ego,
            "target": target,
            "neighbors": self._neighbor_observation(),
            "candidates": candidates,
            "decision_mask": self.task_decision.float().unsqueeze(-1),
        }

    def _coverage(self) -> Tensor:
        return self.team_seen.float().mean(dim=-1)

    def _critic_state(self) -> Tensor:
        target_relative = self.targets - self.positions
        clearance = self._obstacle_clearance().clamp(0.0, float(self.camera["max_depth_m"]))
        coverage = self._coverage().view(self.num_envs, 1, 1).expand(-1, self.num_agents, -1)
        progress = (self.step_count.float() / max(self.max_steps, 1)).view(self.num_envs, 1, 1)
        progress = progress.expand(-1, self.num_agents, -1)
        return torch.cat(
            (
                self.positions / self.world_size,
                self.velocities / float(self.action_limits["speed_norm_mps"]),
                target_relative / float(self.depth_cfg["normalize_range_m"][1]),
                target_relative.norm(dim=-1, keepdim=True) / float(self.depth_cfg["normalize_range_m"][1]),
                clearance.unsqueeze(-1) / float(self.camera["max_depth_m"]),
                coverage,
                progress,
            ),
            dim=-1,
        )

    def _voxel_indices(self) -> Tensor:
        shifted = self.positions + self.world_size * torch.tensor(
            [0.5, 0.5, 0.0], device=self.device
        )
        index = torch.floor(shifted / self.resolution).to(torch.long)
        index[..., 0].clamp_(0, self.grid_shape[0] - 1)
        index[..., 1].clamp_(0, self.grid_shape[1] - 1)
        index[..., 2].clamp_(0, self.grid_shape[2] - 1)
        return index[..., 0] + self.grid_shape[0] * (
            index[..., 1] + self.grid_shape[1] * index[..., 2]
        )

    def _update_seen(self) -> Tuple[Tensor, Tensor, Tensor]:
        indices = self._voxel_indices()
        local_new = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        team_new = torch.zeros_like(local_new)
        duplicate = torch.zeros_like(local_new)
        for env_id in range(self.num_envs):
            for agent_id in range(self.num_agents):
                voxel = int(indices[env_id, agent_id])
                was_local = bool(self.local_seen[env_id, agent_id, voxel])
                was_team = bool(self.team_seen[env_id, voxel])
                local_new[env_id, agent_id] = float(not was_local)
                team_new[env_id, agent_id] = float(not was_team)
                duplicate[env_id, agent_id] = float(was_team)
                self.local_seen[env_id, agent_id, voxel] = True
                self.team_seen[env_id, voxel] = True
        return local_new, team_new, duplicate

    def _coverage_milestone_reward(self, old_coverage: Tensor, new_coverage: Tensor) -> Tensor:
        result = torch.zeros(self.num_envs, self.num_agents, device=self.device)
        milestones = sorted(
            (float(level), float(value))
            for level, value in self.cfg["reward"]["team_coverage_milestones"].items()
        )
        for milestone_id, (level, value) in enumerate(milestones):
            crossed = (old_coverage < level) & (new_coverage >= level) & (~self.milestone_paid[:, milestone_id])
            result[crossed] += value
            self.milestone_paid[crossed, milestone_id] = True
        return result

    def step(self, hybrid_action: Tensor):
        if hybrid_action.shape[-1] != 9:
            raise ValueError(f"hybrid action must have 9 fields, got {hybrid_action.shape[-1]}")
        action = hybrid_action[..., :4].clamp(-1.0, 1.0)
        decision = self.task_decision.clone()
        requested_index = hybrid_action[..., 4].round().long().clamp(0, self.max_candidates - 1)
        requested_valid = torch.gather(
            self.candidate_valid, -1, requested_index.unsqueeze(-1)
        ).squeeze(-1)
        fallback_index = self.candidate_valid.float().argmax(dim=-1)
        selected_index = torch.where(requested_valid, requested_index, fallback_index)
        gather_position = selected_index[..., None, None].expand(-1, -1, 1, 3)
        selected_position = torch.gather(
            self.candidate_positions, -2, gather_position
        ).squeeze(-2)
        selected_yaw = torch.gather(
            self.candidate_yaws, -1, selected_index.unsqueeze(-1)
        ).squeeze(-1)
        selected_gain = torch.gather(
            self.candidate_gains, -1, selected_index.unsqueeze(-1)
        ).squeeze(-1) / float(self.candidate_cfg["visible_gain_normalizer"])
        offset_scale = torch.tensor(
            self.selection_cfg["max_position_offset_m"], device=self.device
        )
        body_offset = hybrid_action[..., 5:8].clamp(-1.0, 1.0) * offset_scale
        world_offset_xy = self._world_xy(body_offset[..., :2])
        world_offset = torch.cat((world_offset_xy, body_offset[..., 2:3]), dim=-1)
        selected_position = selected_position + world_offset
        half_xy = self.world_size[:2] * 0.5 - 0.5
        selected_position[..., 0] = selected_position[..., 0].clamp(-half_xy[0], half_xy[0])
        selected_position[..., 1] = selected_position[..., 1].clamp(-half_xy[1], half_xy[1])
        selected_position[..., 2] = selected_position[..., 2].clamp(self.minimum_z, self.maximum_z)
        selected_yaw = selected_yaw + hybrid_action[..., 8].clamp(-1.0, 1.0) * float(
            self.selection_cfg["max_yaw_offset_rad"]
        )
        self.targets = torch.where(decision.unsqueeze(-1), selected_position, self.targets)
        self.target_yaw = torch.where(decision, selected_yaw, self.target_yaw)
        viewpoint_gain_prior = decision.float() * selected_gain
        self.task_decision.zero_()
        old_distance = (self.targets - self.positions).norm(dim=-1)
        old_positions = self.positions.clone()
        old_coverage = self._coverage()

        x_action = action[..., 0]
        forward = torch.where(
            x_action >= 0.0,
            x_action * float(self.action_limits["forward_mps"]),
            x_action * float(self.action_limits["backward_mps"]),
        )
        body_velocity = torch.stack(
            (
                forward,
                action[..., 1] * float(self.action_limits["lateral_mps"]),
                action[..., 2] * float(self.action_limits["vertical_mps"]),
            ),
            dim=-1,
        )
        speed = body_velocity.norm(dim=-1, keepdim=True)
        scale = (float(self.action_limits["speed_norm_mps"]) / speed.clamp_min(1e-6)).clamp(max=1.0)
        body_velocity *= scale
        self.velocities[..., :2] = self._world_xy(body_velocity[..., :2])
        self.velocities[..., 2] = body_velocity[..., 2]
        self.yaw += action[..., 3] * float(self.action_limits["yaw_rate_rps"]) * self.dt
        self.positions += self.velocities * self.dt
        self.step_count += 1

        new_distance = (self.targets - self.positions).norm(dim=-1)
        goal_reached = new_distance <= 1.0
        self._generate_tasks(goal_reached)

        clearance = self._obstacle_clearance()
        nearest_drone = self._nearest_drone()
        collision = (clearance <= self.drone_radius) | (nearest_drone <= 2.0 * self.drone_radius)
        half_xy = self.world_size[:2] * 0.5
        out_of_bounds = (
            (self.positions[..., 0].abs() > half_xy[0])
            | (self.positions[..., 1].abs() > half_xy[1])
            | (self.positions[..., 2] < self.minimum_z)
            | (self.positions[..., 2] > self.maximum_z)
        )
        displacement = (self.positions - old_positions).norm(dim=-1)
        stalled_now = (displacement < 0.01) & ((old_distance - new_distance) < 0.001)
        self.stall_count = torch.where(stalled_now, self.stall_count + 1, torch.zeros_like(self.stall_count))
        stall = self.stall_count >= self.stall_steps

        local_new, team_new, duplicate = self._update_seen()
        new_coverage = self._coverage()
        milestone = self._coverage_milestone_reward(old_coverage, new_coverage)
        signals = {
            "target_progress_m": old_distance - new_distance,
            "goal_reached": goal_reached,
            "local_new_voxels": local_new,
            "team_unique_new_voxels": team_new,
            "duplicate_voxels": duplicate,
            "obstacle_clearance_m": clearance,
            "speed_mps": self.velocities.norm(dim=-1),
            "nearest_drone_m": nearest_drone,
            "action_delta_l2": (action - self.previous_action).square().sum(dim=-1),
            "vertical_action_l2": action[..., 2].square(),
            "stall": stall,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "coverage_milestone_reward": milestone,
            "viewpoint_gain_prior": viewpoint_gain_prior,
        }
        reward, components = self.reward_composer(signals)
        self.previous_action = action
        team_failure = (collision | out_of_bounds).any(dim=-1) if self.terminate_on_collision else torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        timeout = self.step_count >= self.max_steps
        done = team_failure | timeout

        finished_coverage = new_coverage.clone()
        finished_collision = collision.any(dim=-1).float()
        self._reset_envs(torch.nonzero(done, as_tuple=False).flatten())
        self._update_depth_stack()
        info = {
            "reward_components": components,
            "coverage": finished_coverage,
            "collision": finished_collision,
            "goal_reached": goal_reached.float().sum(dim=-1),
            "viewpoint_decisions": decision.float().sum(dim=-1),
        }
        return self._observation(), self._critic_state(), reward, done, info

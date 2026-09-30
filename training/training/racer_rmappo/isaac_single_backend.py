"""Single-agent fixed-target contract for the first Isaac navigation stage.

This module contains no eager Omniverse imports.  A caller that already owns a
``SimulationApp`` creates :class:`IsaacSingleDroneProbe` and injects it here.
The backend deliberately disables viewpoint selection: candidate zero mirrors
the fixed target, while ``decision_mask`` stays zero so PPO only optimizes the
four-dimensional navigation action during this curriculum stage.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, Mapping, Sequence, Tuple

import torch
from torch import Tensor

from .d455m_sensor import depth_to_normalized_inverse, resize_inverse_depth_for_actor
from .reward import RewardComposer


def wrap_angle(angle: Tensor) -> Tensor:
    """Wrap radians to ``[-pi, pi]`` without changing shape or device."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quaternion_rpy_wxyz(quaternion: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert normalized ``[w,x,y,z]`` quaternions to roll, pitch and yaw."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four values")
    w, x, y, z = quaternion.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x.square() + y.square()))
    pitch_sine = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch = torch.asin(pitch_sine)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    return roll, pitch, yaw


def world_to_yaw_local(vector: Tensor, yaw: Tensor) -> Tensor:
    """Rotate world XYZ vectors into yaw-local forward/left/up coordinates."""
    if vector.shape[-1] != 3:
        raise ValueError("vector must end in three values")
    yaw = torch.as_tensor(yaw, dtype=vector.dtype, device=vector.device)
    yaw = torch.broadcast_to(yaw, vector.shape[:-1])
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    forward = cosine * vector[..., 0] + sine * vector[..., 1]
    left = -sine * vector[..., 0] + cosine * vector[..., 1]
    return torch.stack((forward, left, vector[..., 2]), dim=-1)


def scale_navigation_action(action: Tensor, limits: Mapping[str, Any]) -> Tensor:
    """Scale normalized ``[forward,left,up,yaw_rate]`` into physical commands."""
    if action.shape[-1] != 4:
        raise ValueError("navigation action must end in four values")
    bounded = action.clamp(-1.0, 1.0)
    forward = torch.where(
        bounded[..., 0] >= 0.0,
        bounded[..., 0] * float(limits["forward_mps"]),
        bounded[..., 0] * float(limits["backward_mps"]),
    )
    velocity = torch.stack(
        (
            forward,
            bounded[..., 1] * float(limits["lateral_mps"]),
            bounded[..., 2] * float(limits["vertical_mps"]),
        ),
        dim=-1,
    )
    speed = velocity.norm(dim=-1, keepdim=True)
    scale = (
        float(limits["speed_norm_mps"]) / speed.clamp_min(1e-6)
    ).clamp(max=1.0)
    velocity = velocity * scale
    yaw_rate = bounded[..., 3:4] * float(limits["yaw_rate_rps"])
    return torch.cat((velocity, yaw_rate), dim=-1)


class IsaacSingleNavigationBackend:
    """One physical drone, one rule-provided target and no learned selection."""

    def __init__(
        self,
        cfg: Mapping[str, Any],
        isaac_cfg: Mapping[str, Any],
        probe: Any,
    ) -> None:
        self.cfg = cfg
        self.isaac_cfg = isaac_cfg
        self.probe = probe
        # Opt-in evaluation telemetry. Never changes the policy observation.
        self.collect_diagnostics = False
        self.last_depth_summary = {}
        self.device = torch.device(probe.device)
        self.num_envs = 1
        self.num_agents = 1
        if int(cfg["experiment"]["num_agents"]) != 1:
            raise ValueError("IsaacSingleNavigationBackend requires a one-agent stage")
        if int(cfg["training"]["num_parallel_swarms"]) != 1:
            raise ValueError("the first Isaac navigation backend supports one parallel swarm")

        self.control_hz = float(cfg["experiment"]["control_hz"])
        if not math.isclose(
            self.control_hz,
            float(isaac_cfg["control"]["control_hz"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("RMAPPO and Isaac control frequencies must match")
        self.dt = 1.0 / self.control_hz
        self.camera_cfg = isaac_cfg["camera"]
        self.depth_cfg = cfg["actor_observation"]["depth"]
        self.target_cfg = cfg["actor_observation"]["selected_target"]
        self.candidate_cfg = cfg["actor_observation"]["racer_candidates"]
        self.neighbor_cfg = cfg["actor_observation"]["neighbors"]
        self.action_limits = cfg["action"]["physical_limits"]
        self.reward_composer = RewardComposer(cfg["reward"])
        navigation = isaac_cfg["navigation_backend"]
        self.target_position = torch.tensor(
            navigation["fixed_target_position_m"],
            dtype=torch.float32,
            device=self.device,
        )
        self.target_yaw = torch.tensor(
            float(navigation["fixed_target_yaw_rad"]),
            dtype=torch.float32,
            device=self.device,
        )
        self.goal_position_m = float(navigation["goal_position_tolerance_m"])
        self.goal_yaw_rad = float(navigation["goal_yaw_tolerance_rad"])
        self.goal_tilt_rad = float(navigation["goal_tilt_tolerance_rad"])
        self.max_steps = max(
            1, int(round(float(navigation["episode_seconds"]) * self.control_hz))
        )
        self.terminate_on_stall = bool(navigation.get("terminate_on_stall", True))
        self.reset_pose_sync_control_steps = int(
            navigation.get("reset_pose_sync_control_steps", 1)
        )
        self.frame_stack = int(self.depth_cfg["frame_stack"])
        self.depth_width, self.depth_height = (
            int(value) for value in self.depth_cfg["resize"]
        )
        self.max_candidates = int(self.candidate_cfg["max_candidates"])
        self.max_neighbors = int(self.neighbor_cfg["max_neighbors"])
        stall_cfg = cfg["reward"]["stall"]
        self.stall_window_steps = max(
            1, int(round(float(stall_cfg["window_seconds"]) * self.control_hz))
        )
        self.stall_min_displacement = float(stall_cfg["min_displacement_m"])
        self.position_history: Deque[Tensor] = deque(
            maxlen=self.stall_window_steps + 1
        )
        self.previous_action = torch.zeros(1, 1, 4, device=self.device)
        self.depth_stack = torch.zeros(
            1,
            1,
            self.frame_stack,
            self.depth_height,
            self.depth_width,
            device=self.device,
        )
        self.last_clearance_m = torch.full(
            (1, 1), float(self.camera_cfg["max_depth_m"]), device=self.device
        )
        self.step_count = 0
        self.last_telemetry: Dict[str, Any] = {}
        self.episode_collision = False
        self.episode_out_of_bounds = False
        self.episode_stall = False
        self.episode_safety_takeover = False
        self._closed = False

    def _telemetry_tensors(
        self, telemetry: Mapping[str, Any]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        position = torch.tensor(
            telemetry["position_m"], dtype=torch.float32, device=self.device
        )
        orientation = torch.tensor(
            telemetry["orientation_wxyz"], dtype=torch.float32, device=self.device
        )
        velocity = torch.tensor(
            telemetry["linear_velocity_mps"],
            dtype=torch.float32,
            device=self.device,
        )
        roll, pitch, yaw = quaternion_rpy_wxyz(orientation)
        return position, orientation, velocity, roll, pitch, yaw

    def _capture_actor_depth(self, warmup_frames: int) -> Tensor:
        metric = self.probe.depth_camera.capture(warmup_frames=warmup_frames)
        plane = metric["distance_to_image_plane"]
        normalized, _ = depth_to_normalized_inverse(
            plane,
            float(self.camera_cfg["min_depth_m"]),
            float(self.camera_cfg["max_depth_m"]),
        )
        actor_depth = resize_inverse_depth_for_actor(
            normalized,
            output_width=self.depth_width,
            output_height=self.depth_height,
        ).to(self.device)

        # Use the same, geometrically validated axial depth that feeds the
        # actor. Isaac Sim 2023.1's distance_to_camera annotator can have
        # incompatible clipping semantics (for example reporting 2.35 m for a
        # wall whose verified optical-axis depth is 3.25 m). The minimum axial
        # depth is conservative off-axis and avoids mixing those conventions.
        plane_metric = torch.as_tensor(plane, dtype=torch.float32)
        plane_valid = (
            torch.isfinite(plane_metric)
            & (plane_metric >= float(self.camera_cfg["min_depth_m"]))
            & (plane_metric <= float(self.camera_cfg["max_depth_m"]))
        )
        clearance = (
            float(plane_metric[plane_valid].min().item())
            if bool(plane_valid.any())
            else float(self.camera_cfg["max_depth_m"])
        )
        self.last_clearance_m.fill_(clearance)
        if self.collect_diagnostics:
            self.last_depth_summary = {
                "valid_fraction": float(plane_valid.float().mean()),
                "too_near_fraction": float(
                    (torch.isfinite(plane_metric) & (plane_metric > 0.0)
                     & (plane_metric < float(self.camera_cfg["min_depth_m"]))).float().mean()
                ),
                "sensor_min_axial_depth_m": clearance,
            }
        return actor_depth

    def _update_depth_stack(self, newest: Tensor, reset: bool = False) -> None:
        newest = newest.reshape(1, 1, 1, self.depth_height, self.depth_width)
        if reset:
            self.depth_stack.copy_(newest.expand_as(self.depth_stack))
        else:
            self.depth_stack = torch.cat((self.depth_stack[:, :, 1:], newest), dim=2)

    def _relative_target(
        self, telemetry: Mapping[str, Any]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        position, orientation, velocity, roll, pitch, yaw = self._telemetry_tensors(
            telemetry
        )
        relative_world = self.target_position - position
        relative_body = world_to_yaw_local(relative_world, yaw)
        distance = relative_world.norm()
        yaw_error = wrap_angle(self.target_yaw - yaw)
        return (
            relative_body,
            relative_world,
            distance,
            yaw_error,
            velocity,
            roll,
            pitch,
        )

    def _observation(self) -> Dict[str, Tensor]:
        (
            relative_body,
            _,
            distance,
            yaw_error,
            velocity_world,
            roll,
            pitch,
        ) = self._relative_target(self.last_telemetry)
        _, _, _, _, _, yaw = self._telemetry_tensors(self.last_telemetry)
        body_velocity = world_to_yaw_local(velocity_world, yaw)
        ego = torch.cat(
            (
                body_velocity / float(self.action_limits["speed_norm_mps"]),
                torch.stack((roll, pitch)),
                torch.stack((torch.sin(yaw), torch.cos(yaw))),
                self.previous_action.reshape(4),
            )
        ).reshape(1, 1, 11)
        distance_normalizer = float(self.target_cfg["distance_normalizer_m"])
        target = torch.cat(
            (
                relative_body / distance_normalizer,
                (distance / distance_normalizer).reshape(1),
                torch.sin(yaw_error).reshape(1),
                torch.cos(yaw_error).reshape(1),
                torch.zeros(1, device=self.device),
            )
        ).reshape(1, 1, 7)

        candidates = torch.zeros(
            1, 1, self.max_candidates, 9, device=self.device
        )
        candidate_normalizer = float(self.candidate_cfg["distance_normalizer_m"])
        candidates[0, 0, 0, :3] = relative_body / candidate_normalizer
        candidates[0, 0, 0, 3] = distance / candidate_normalizer
        candidates[0, 0, 0, 4] = torch.sin(yaw_error)
        candidates[0, 0, 0, 5] = torch.cos(yaw_error)
        candidates[0, 0, 0, 6] = 0.0
        candidates[0, 0, 0, 7] = 0.0
        candidates[0, 0, 0, 8] = 1.0
        return {
            "depth": self.depth_stack.clone(),
            "ego": ego,
            "target": target,
            "neighbors": torch.zeros(
                1, 1, self.max_neighbors, 8, device=self.device
            ),
            "candidates": candidates,
            # Candidate zero is supplied by the rule stage. Selection and
            # residual log-probabilities must not enter PPO in this curriculum.
            "decision_mask": torch.zeros(1, 1, 1, device=self.device),
        }

    def _critic_state(self) -> Tensor:
        position, _, velocity, _, _, _ = self._telemetry_tensors(
            self.last_telemetry
        )
        _, relative_world, distance, _, _, _, _ = self._relative_target(
            self.last_telemetry
        )
        world_size = torch.tensor(
            self.isaac_cfg["scene"]["size_m"],
            dtype=torch.float32,
            device=self.device,
        )
        state = torch.cat(
            (
                position / world_size,
                velocity / float(self.action_limits["speed_norm_mps"]),
                relative_world / float(self.target_cfg["distance_normalizer_m"]),
                (distance / float(self.target_cfg["distance_normalizer_m"])).reshape(1),
                (self.last_clearance_m.reshape(1) / float(self.camera_cfg["max_depth_m"])),
                torch.zeros(1, device=self.device),
                torch.tensor(
                    [self.step_count / max(self.max_steps, 1)],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        )
        return state.reshape(1, 1, 13)

    def reset(self) -> Tuple[Dict[str, Tensor], Tensor]:
        consume_initial = getattr(
            self.probe, "consume_initial_reset_telemetry", None
        )
        initial_telemetry = (
            consume_initial() if callable(consume_initial) else None
        )
        self.last_telemetry = (
            initial_telemetry
            if initial_telemetry is not None
            else self.probe.reset()
        )
        self.last_telemetry = self.probe.synchronize_pose_to_renderer(
            self.reset_pose_sync_control_steps
        )
        self.step_count = 0
        self.previous_action.zero_()
        self.position_history.clear()
        position = torch.tensor(
            self.last_telemetry["position_m"],
            dtype=torch.float32,
            device=self.device,
        )
        self.position_history.append(position.clone())
        self.episode_collision = False
        self.episode_out_of_bounds = False
        self.episode_stall = False
        self.episode_safety_takeover = False
        newest = self._capture_actor_depth(
            int(self.camera_cfg["warmup_render_frames"])
        )
        self._update_depth_stack(newest, reset=True)
        return self._observation(), self._critic_state()

    def step(
        self, hybrid_action: Tensor
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Dict[str, object]]:
        action = torch.as_tensor(
            hybrid_action, dtype=torch.float32, device=self.device
        )
        if tuple(action.shape) != (1, 1, 9):
            raise ValueError(
                f"single Isaac hybrid action must have shape (1,1,9), got {tuple(action.shape)}"
            )
        raw_navigation = action[..., :4]
        bounded_navigation = raw_navigation.clamp(-1.0, 1.0)
        safety_takeover = bool((raw_navigation.abs() > 1.0).any().item())
        command = scale_navigation_action(
            bounded_navigation, self.action_limits
        ).reshape(4)

        _, _, old_distance, _, _, _, _ = self._relative_target(
            self.last_telemetry
        )
        telemetry, applied_command = self.probe.step(command)
        self.last_telemetry = telemetry
        self.step_count += 1
        newest = self._capture_actor_depth(1)
        self._update_depth_stack(newest)

        (
            _,
            _,
            new_distance,
            yaw_error,
            velocity,
            roll,
            pitch,
        ) = self._relative_target(telemetry)
        position = torch.tensor(
            telemetry["position_m"], dtype=torch.float32, device=self.device
        )
        self.position_history.append(position.clone())
        navigation_reached = new_distance <= self.goal_position_m
        heading_reached = yaw_error.abs() <= self.goal_yaw_rad
        tilt = torch.sqrt(roll.square() + pitch.square())
        tilt_reached = tilt <= self.goal_tilt_rad
        observation_completed = navigation_reached & heading_reached & tilt_reached
        collision = bool(telemetry["collision"])
        out_of_bounds = bool(telemetry["out_of_bounds"])
        stall = False
        if len(self.position_history) == self.position_history.maxlen:
            displacement = (self.position_history[-1] - self.position_history[0]).norm()
            stall = bool(
                (displacement < self.stall_min_displacement).item()
                and not bool(navigation_reached.item())
            )

        shape = (1, 1)
        collision_tensor = torch.full(
            shape, collision, dtype=torch.bool, device=self.device
        )
        out_of_bounds_tensor = torch.full(
            shape, out_of_bounds, dtype=torch.bool, device=self.device
        )
        stall_tensor = torch.full(
            shape, stall, dtype=torch.bool, device=self.device
        )
        signals = {
            "target_progress_m": (old_distance - new_distance).reshape(shape),
            "goal_reached": observation_completed.reshape(shape),
            "local_new_voxels": torch.zeros(shape, device=self.device),
            "team_unique_new_voxels": torch.zeros(shape, device=self.device),
            "duplicate_voxels": torch.zeros(shape, device=self.device),
            "obstacle_clearance_m": self.last_clearance_m,
            "speed_mps": velocity.norm().reshape(shape),
            "nearest_drone_m": torch.full(
                shape,
                float(self.neighbor_cfg["communication_radius_m"]),
                device=self.device,
            ),
            "action_delta_l2": (
                bounded_navigation - self.previous_action
            ).square().sum(dim=-1),
            "vertical_action_l2": bounded_navigation[..., 2].square(),
            "stall": stall_tensor,
            "collision": collision_tensor,
            "out_of_bounds": out_of_bounds_tensor,
            "coverage_milestone_reward": torch.zeros(shape, device=self.device),
            "viewpoint_gain_prior": torch.zeros(shape, device=self.device),
        }
        reward, components = self.reward_composer(signals)
        self.previous_action.copy_(bounded_navigation)

        self.episode_collision |= collision
        self.episode_out_of_bounds |= out_of_bounds
        self.episode_stall |= stall
        self.episode_safety_takeover |= safety_takeover
        timeout = self.step_count >= self.max_steps
        done_value = (
            collision
            or out_of_bounds
            or bool(observation_completed.item())
            or timeout
            or (stall and self.terminate_on_stall)
        )
        success = (
            done_value
            and bool(observation_completed.item())
            and not self.episode_collision
            and not self.episode_out_of_bounds
        )
        done = torch.tensor([done_value], dtype=torch.bool, device=self.device)
        episode_finished = float(done_value)
        info: Dict[str, object] = {
            "reward_components": components,
            "coverage": torch.zeros(1, device=self.device),
            "collision": torch.tensor([float(collision)], device=self.device),
            "obstacle_collision": torch.tensor([float(collision)], device=self.device),
            "inter_drone_collision": torch.zeros(1, device=self.device),
            "out_of_bounds": torch.tensor([float(out_of_bounds)], device=self.device),
            "stall": torch.tensor([float(stall)], device=self.device),
            "safety_takeover": torch.tensor([float(safety_takeover)], device=self.device),
            "goal_reached": observation_completed.float().reshape(1),
            "navigation_reached": navigation_reached.float().reshape(1),
            "viewpoint_decisions": torch.zeros(1, device=self.device),
            "episode_finished": torch.tensor([episode_finished], device=self.device),
            "episode_success": torch.tensor([float(success)], device=self.device),
            "episode_collision": torch.tensor(
                [float(done_value and self.episode_collision)], device=self.device
            ),
            "episode_obstacle_collision": torch.tensor(
                [float(done_value and self.episode_collision)], device=self.device
            ),
            "episode_inter_drone_collision": torch.zeros(1, device=self.device),
            "episode_timeout": torch.tensor(
                [float(done_value and timeout and not success)], device=self.device
            ),
            "episode_stall": torch.tensor(
                [float(done_value and self.episode_stall)], device=self.device
            ),
            "episode_out_of_bounds": torch.tensor(
                [float(done_value and self.episode_out_of_bounds)], device=self.device
            ),
            "episode_safety_takeover": torch.tensor(
                [float(done_value and self.episode_safety_takeover)], device=self.device
            ),
            "coverage_target_reached": torch.zeros(1, device=self.device),
            "coverage_target_steps": torch.zeros(1, device=self.device),
            # Extra diagnostics are allowed beyond REQUIRED_STEP_INFO.
            "observation_completed": observation_completed.float().reshape(1),
            "target_distance_m": new_distance.reshape(1),
            "target_yaw_error_rad": yaw_error.abs().reshape(1),
            "body_tilt_rad": tilt.reshape(1),
            "command_body": command.detach().clone(),
            "actual_speed_mps": torch.tensor(
                [float(telemetry["speed_mps"])], device=self.device
            ),
            "obstacle_clearance_m": self.last_clearance_m.reshape(1).clone(),
            "position_m": position.detach().clone(),
            "episode_steps": torch.tensor(
                [float(self.step_count)], device=self.device
            ),
        }

        if self.collect_diagnostics:
            # Capture BEFORE auto-reset: last_telemetry/probe state will describe
            # the next episode once step() returns on a terminal transition.
            reference = getattr(self.probe, "target_position", None)
            info["diagnostic_snapshot"] = {
                "episode_step": self.step_count,
                "position_m": list(telemetry["position_m"]),
                "orientation_wxyz": list(telemetry["orientation_wxyz"]),
                "velocity_world_mps": list(telemetry["linear_velocity_mps"]),
                "actual_speed_mps": float(telemetry["speed_mps"]),
                "rpy_rad": [float(roll), float(pitch), float(telemetry["yaw_rad"])],
                "target_position_m": self.target_position.detach().cpu().tolist(),
                "target_distance_m": float(new_distance),
                "target_yaw_error_rad": float(yaw_error),
                "body_tilt_rad": float(tilt),
                "navigation_reached": bool(navigation_reached),
                "observation_completed": bool(observation_completed),
                "action_navigation": bounded_navigation.reshape(4).detach().cpu().tolist(),
                "command_body": command.detach().cpu().tolist(),
                "applied_command_world": applied_command.reshape(4).detach().cpu().tolist(),
                "controller_reference_position_m": (
                    reference.reshape(3).detach().cpu().tolist() if reference is not None else None
                ),
                "depth": dict(self.last_depth_summary),
                "max_contact_force_n": float(telemetry["max_contact_force_n"]),
                "reward": float(reward.item()),
                "reward_components": {key: float(value.item()) for key, value in components.items()},
                "done": done_value,
                "success": success,
                "collision": collision,
                "out_of_bounds": out_of_bounds,
                "stall": stall,
                "timeout": timeout,
                "safety_takeover": safety_takeover,
            }

        if done_value:
            observation, critic_state = self.reset()
        else:
            observation, critic_state = self._observation(), self._critic_state()
        return observation, critic_state, reward, done, info

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.probe.close()

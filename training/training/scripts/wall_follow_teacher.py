"""Local wall-following teacher used only to generate training demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from tensordict.nn import TensorDictModuleBase

from utils import vec_to_world


PHASE_SIDE_SELECTION = 0
PHASE_WALL_FOLLOW = 1
PHASE_CORNER = 2
PHASE_LEAVE = 3
PHASE_NAMES = {
    PHASE_SIDE_SELECTION: "side_selection",
    PHASE_WALL_FOLLOW: "wall_follow",
    PHASE_CORNER: "corner",
    PHASE_LEAVE: "leave",
}


def _cfg_get(cfg, key: str, default):
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)
    except AttributeError:
        return getattr(cfg, key, default)


def _normalize_xy(vector: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = vector.norm(dim=-1, keepdim=True)
    fallback = torch.zeros_like(vector)
    fallback[..., 0] = 1.0
    return torch.where(norm > eps, vector / norm.clamp_min(eps), fallback)


@dataclass
class LineFit:
    centroid: torch.Tensor
    tangent: torch.Tensor
    normal: torch.Tensor
    distance: torch.Tensor
    span: torch.Tensor
    confidence: torch.Tensor
    inlier_count: int
    inliers: torch.Tensor


class WallFollowTeacher:
    """Stateful local teacher for large static-wall recovery."""

    def __init__(
        self,
        cfg,
        sensor_cfg,
        vo_cfg,
        num_envs: int,
        teacher_env_mask: torch.Tensor,
        action_limit: float,
        dt: float,
        device,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.teacher_env_mask = teacher_env_mask.to(self.device).reshape(-1).bool()
        if self.teacher_env_mask.numel() != self.num_envs:
            raise ValueError("teacher_env_mask must contain one entry per environment.")
        self.teacher_env_indices = self.teacher_env_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
        self.action_limit = max(float(action_limit), 1e-6)
        self.dt = max(float(dt), 1e-6)

        self.lidar_range = float(_cfg_get(sensor_cfg, "lidar_range", 4.0))
        self.horizontal_resolution = float(_cfg_get(sensor_cfg, "lidar_hres", 10.0))
        self.horizontal_beams = int(round(360.0 / self.horizontal_resolution))
        self.vertical_beams = int(_cfg_get(sensor_cfg, "lidar_vbeams", 4))
        vertical_fov = list(_cfg_get(sensor_cfg, "lidar_vfov", [-10.0, 20.0]))
        self.vertical_angles = torch.linspace(
            float(vertical_fov[0]),
            float(vertical_fov[1]),
            self.vertical_beams,
            device=self.device,
        )
        self.ray_directions = self._make_ray_directions()
        requested_elevations = list(_cfg_get(cfg, "fit_elevation_indices", [1, 2]))
        valid_elevations = [index for index in requested_elevations if 0 <= int(index) < self.vertical_beams]
        if not valid_elevations:
            valid_elevations = [int(torch.argmin(self.vertical_angles.abs()).item())]
        self.fit_elevation_indices = torch.as_tensor(valid_elevations, dtype=torch.long, device=self.device)

        self.trigger_steps = max(int(_cfg_get(cfg, "trigger_steps", 20)), 1)
        self.trigger_progress_eps = float(_cfg_get(cfg, "trigger_progress_eps", 0.005))
        self.trigger_max_speed = max(float(_cfg_get(cfg, "trigger_max_speed", 0.5)), 0.0)
        self.trigger_distance = min(float(_cfg_get(cfg, "trigger_distance", 1.8)), self.lidar_range)
        self.emergency_trigger_distance = min(
            float(_cfg_get(cfg, "emergency_trigger_distance", 0.9)),
            self.trigger_distance,
        )
        self.emergency_trigger_steps = max(
            int(_cfg_get(cfg, "emergency_trigger_steps", 5)),
            1,
        )
        self.trigger_front_angle = float(_cfg_get(cfg, "trigger_front_angle_deg", 70.0))
        self.trigger_normal_alignment = float(_cfg_get(cfg, "trigger_normal_alignment", 0.25))
        self.cooldown_steps = max(int(_cfg_get(cfg, "cooldown_steps", 60)), 0)

        self.fit_min_distance = max(float(_cfg_get(cfg, "fit_min_distance", 0.3)), 0.0)
        self.fit_max_distance = min(float(_cfg_get(cfg, "fit_max_distance", self.lidar_range)), self.lidar_range)
        self.fit_ground_min_z = float(_cfg_get(cfg, "fit_ground_min_z", -0.35))
        self.fit_residual_threshold = max(float(_cfg_get(cfg, "fit_residual_threshold", 0.15)), 1e-3)
        self.fit_min_pair_separation = max(float(_cfg_get(cfg, "fit_min_pair_separation", 0.25)), 1e-3)
        self.fit_min_inliers = max(int(_cfg_get(cfg, "fit_min_inliers", 4)), 2)
        self.fit_min_span = max(float(_cfg_get(cfg, "fit_min_span", 0.6)), 1e-3)
        self.fit_max_hypotheses = max(int(_cfg_get(cfg, "fit_max_hypotheses", 128)), 1)
        self.fit_min_confidence = min(max(float(_cfg_get(cfg, "fit_min_confidence", 0.25)), 0.0), 1.0)
        self.segment_max_range_jump = max(
            float(_cfg_get(cfg, "segment_max_range_jump", 0.65)),
            1e-3,
        )
        self.segment_max_point_gap = max(
            float(_cfg_get(cfg, "segment_max_point_gap", 0.85)),
            1e-3,
        )
        self.segment_min_points = max(
            int(_cfg_get(cfg, "segment_min_points", self.fit_min_inliers)),
            2,
        )
        self.tracking_continuity_weight = max(float(_cfg_get(cfg, "tracking_continuity_weight", 0.8)), 0.0)
        self.normal_smoothing = min(max(float(_cfg_get(cfg, "normal_smoothing", 0.65)), 0.0), 0.99)
        self.corner_angle = float(_cfg_get(cfg, "corner_angle_deg", 35.0))
        self.max_missing_fit_steps = max(int(_cfg_get(cfg, "max_missing_fit_steps", 20)), 1)

        self.parallel_speed = max(float(_cfg_get(cfg, "parallel_speed", 1.0)), 0.0)
        self.reference_clearance = max(float(_cfg_get(cfg, "reference_clearance", 1.0)), 0.05)
        self.clearance_gain = max(float(_cfg_get(cfg, "clearance_gain", 1.2)), 0.0)
        self.max_toward_wall_speed = max(float(_cfg_get(cfg, "max_toward_wall_speed", 0.2)), 0.0)
        self.max_away_wall_speed = max(float(_cfg_get(cfg, "max_away_wall_speed", 0.8)), 0.0)
        self.height_gain = max(float(_cfg_get(cfg, "height_gain", 0.6)), 0.0)
        self.max_vertical_speed = max(float(_cfg_get(cfg, "max_vertical_speed", 0.5)), 0.0)
        self.action_smoothing = min(max(float(_cfg_get(cfg, "action_smoothing", 0.40)), 0.0), 0.99)
        self.side_goal_weight = float(_cfg_get(cfg, "side_goal_weight", 1.0))
        self.side_clearance_weight = float(_cfg_get(cfg, "side_clearance_weight", 0.8))
        self.side_velocity_weight = float(_cfg_get(cfg, "side_velocity_weight", 0.25))
        self.side_clearance_angle = float(_cfg_get(cfg, "side_clearance_angle_deg", 35.0))
        self.leave_goal_speed = max(float(_cfg_get(cfg, "leave_goal_speed", 1.2)), 0.0)
        self.leave_goal_blend = min(
            max(float(_cfg_get(cfg, "leave_goal_blend", 0.75)), 0.0),
            1.0,
        )

        self.min_follow_steps = max(int(_cfg_get(cfg, "min_follow_steps", 25)), 0)
        self.min_follow_distance = max(float(_cfg_get(cfg, "min_follow_distance", 0.5)), 0.0)
        self.min_goal_progress = max(float(_cfg_get(cfg, "min_goal_progress", 0.2)), 0.0)
        self.leave_requires_goal_progress = bool(
            _cfg_get(cfg, "leave_requires_goal_progress", True)
        )
        self.goal_clear_angle = float(_cfg_get(cfg, "goal_clear_angle_deg", 18.0))
        self.goal_clear_steps = max(int(_cfg_get(cfg, "goal_clear_steps", 8)), 1)
        self.max_follow_steps = max(int(_cfg_get(cfg, "max_follow_steps", 500)), 1)
        self.max_side_switches = max(int(_cfg_get(cfg, "max_side_switches", 1)), 0)

        self.dynamic_width_resolution = max(float(_cfg_get(cfg, "dynamic_width_resolution", 0.25)), 1e-3)
        self.dynamic_spanning_height = max(float(_cfg_get(cfg, "dynamic_spanning_height", 4.0)), 1e-3)
        self.dynamic_point_margin = max(float(_cfg_get(cfg, "dynamic_point_margin", 0.25)), 0.0)
        self.dynamic_risk_limit = min(max(float(_cfg_get(cfg, "dynamic_risk_limit", 0.15)), 0.0), 1.0)
        self.static_sweep_horizon = max(
            float(_cfg_get(cfg, "static_sweep_horizon", 1.0)),
            self.dt,
        )
        self.static_sweep_steps = max(int(_cfg_get(cfg, "static_sweep_steps", 10)), 1)
        self.static_safety_radius = max(
            float(_cfg_get(cfg, "static_safety_radius", 0.40)),
            0.0,
        )
        self.static_confidence_margin = max(
            float(_cfg_get(cfg, "static_confidence_margin", 0.30)),
            1e-3,
        )
        self.static_away_speeds = tuple(
            max(float(speed), 0.0)
            for speed in _cfg_get(cfg, "static_away_speeds", [0.35, 0.70])
        )
        self.demo_min_confidence = min(max(float(_cfg_get(cfg, "demo_min_confidence", 0.2)), 0.0), 1.0)
        self.demo_min_action_speed = max(
            float(_cfg_get(cfg, "demo_min_action_speed", 0.35)),
            0.0,
        )
        self.action_epsilon = min(max(float(_cfg_get(cfg, "action_epsilon", 1e-4)), 1e-7), 0.1)
        self.vo_horizon = max(float(_cfg_get(vo_cfg, "horizon", 2.1)), 1e-6)
        self.vo_tau = max(float(_cfg_get(vo_cfg, "tau", 0.75)), 1e-6)
        self.vo_xy_margin = max(float(_cfg_get(vo_cfg, "xy_margin", 0.3)), 0.0)
        self.vo_z_margin = max(float(_cfg_get(vo_cfg, "z_margin", 0.3)), 0.0)

        self.active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.blocked_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.cooldown = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.follow_age = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.missing_fit_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clear_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.side_switches = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.handedness = torch.ones(self.num_envs, device=self.device)
        self.wall_normal = torch.zeros(self.num_envs, 2, device=self.device)
        self.wall_tangent = torch.zeros(self.num_envs, 2, device=self.device)
        self.wall_distance = torch.full((self.num_envs,), self.lidar_range, device=self.device)
        self.last_fit_confidence = torch.zeros(self.num_envs, device=self.device)
        self.previous_teacher_action = torch.zeros(self.num_envs, 3, device=self.device)
        self.hit_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.follow_distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self.previous_episode_length = torch.full((self.num_envs,), -1.0, device=self.device)

    def _make_ray_directions(self) -> torch.Tensor:
        horizontal = torch.arange(
            -180.0,
            180.0,
            self.horizontal_resolution,
            device=self.device,
        )
        pitch, yaw = torch.meshgrid(self.vertical_angles, horizontal, indexing="xy")
        pitch = torch.deg2rad(pitch.reshape(-1)) + torch.pi / 2.0
        yaw = torch.deg2rad(yaw.reshape(-1))
        x = torch.sin(pitch) * torch.cos(yaw)
        y = torch.sin(pitch) * torch.sin(yaw)
        z = torch.cos(pitch)
        return -torch.stack([x, y, z], dim=-1).reshape(
            self.horizontal_beams,
            self.vertical_beams,
            3,
        )

    def _reset(self, mask: torch.Tensor, goal_distance: torch.Tensor):
        if not mask.any():
            return
        self.active[mask] = False
        self.blocked_counter[mask] = 0
        self.cooldown[mask] = 0
        self.follow_age[mask] = 0
        self.missing_fit_steps[mask] = 0
        self.clear_counter[mask] = 0
        self.side_switches[mask] = 0
        self.handedness[mask] = 1.0
        self.wall_normal[mask] = 0.0
        self.wall_tangent[mask] = 0.0
        self.wall_distance[mask] = self.lidar_range
        self.last_fit_confidence[mask] = 0.0
        self.previous_teacher_action[mask] = 0.0
        self.hit_goal_distance[mask] = goal_distance[mask]
        self.follow_distance[mask] = 0.0
        self.previous_goal_distance[mask] = goal_distance[mask]

    def _dynamic_geometry(self, dynamic_obstacle: torch.Tensor):
        direction = dynamic_obstacle[..., :3]
        horizontal_distance = dynamic_obstacle[..., 3]
        vertical_distance = dynamic_obstacle[..., 4]
        velocity = dynamic_obstacle[..., 5:8]
        width_code = dynamic_obstacle[..., 8]
        height_code = dynamic_obstacle[..., 9]
        valid = horizontal_distance > 1e-4
        direction_xy = _normalize_xy(direction[..., :2])
        position = torch.cat(
            [direction_xy * horizontal_distance.unsqueeze(-1), vertical_distance.unsqueeze(-1)],
            dim=-1,
        )
        width = (width_code + 1.0) * self.dynamic_width_resolution
        height = torch.where(
            height_code > 1e-4,
            height_code,
            torch.full_like(height_code, self.dynamic_spanning_height),
        )
        return position, velocity, width, height, valid

    def _reconstruct_static_point_grid(
        self,
        lidar: torch.Tensor,
        dynamic_obstacle: torch.Tensor,
    ):
        scan = lidar.squeeze(0)
        distances = (self.lidar_range - scan).clamp(0.0, self.lidar_range)
        points = distances.unsqueeze(-1) * self.ray_directions
        hit = distances < (self.lidar_range - 1e-3)

        elevation_mask = torch.zeros_like(hit)
        elevation_mask[:, self.fit_elevation_indices] = True
        horizontal_distance = points[..., :2].norm(dim=-1)
        fit_valid = (
            hit
            & elevation_mask
            & (horizontal_distance >= self.fit_min_distance)
            & (horizontal_distance <= self.fit_max_distance)
            & (points[..., 2] >= self.fit_ground_min_z)
        )
        safety_valid = hit.clone()

        flat_points = points.reshape(-1, 3)
        fit_valid_flat = fit_valid.reshape(-1)
        safety_valid_flat = safety_valid.reshape(-1)
        if safety_valid_flat.any():
            dyn_position, _, dyn_width, dyn_height, dyn_valid = self._dynamic_geometry(dynamic_obstacle)
            if dyn_valid.any():
                point_delta = flat_points.unsqueeze(1) - dyn_position.unsqueeze(0)
                inside_xy = point_delta[..., :2].norm(dim=-1) <= (
                    dyn_width.unsqueeze(0) / 2.0 + self.dynamic_point_margin
                )
                inside_z = point_delta[..., 2].abs() <= (
                    dyn_height.unsqueeze(0) / 2.0 + self.dynamic_point_margin
                )
                dynamic_hit = (inside_xy & inside_z & dyn_valid.unsqueeze(0)).any(dim=1)
                fit_valid_flat = fit_valid_flat & (~dynamic_hit)
                safety_valid_flat = safety_valid_flat & (~dynamic_hit)
        return (
            points,
            fit_valid_flat.reshape_as(fit_valid),
            safety_valid_flat.reshape_as(safety_valid),
        )

    def _reconstruct_static_points(
        self,
        lidar: torch.Tensor,
        dynamic_obstacle: torch.Tensor,
    ) -> torch.Tensor:
        points, fit_valid, _ = self._reconstruct_static_point_grid(lidar, dynamic_obstacle)
        return points.reshape(-1, 3)[fit_valid.reshape(-1)]

    def segment_static_points(
        self,
        points: torch.Tensor,
        valid: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Split a horizontal LiDAR ring into contiguous static-surface segments."""
        if points.ndim != 3 or valid.shape != points.shape[:2]:
            raise ValueError("points and valid must have shapes [azimuth, elevation, 3] and [azimuth, elevation].")

        horizontal_distance = points[..., :2].norm(dim=-1)
        masked_distance = torch.where(
            valid,
            horizontal_distance,
            torch.full_like(horizontal_distance, float("inf")),
        )
        representative_range, representative_elevation = masked_distance.min(dim=1)
        beam_indices = torch.arange(points.shape[0], device=points.device)
        representative_xy = points[beam_indices, representative_elevation, :2]
        active = valid.any(dim=1)
        adjacent_connected = (
            active[:-1]
            & active[1:]
            & (
                torch.abs(representative_range[1:] - representative_range[:-1])
                <= self.segment_max_range_jump
            )
            & (
                (representative_xy[1:] - representative_xy[:-1]).norm(dim=-1)
                <= self.segment_max_point_gap
            )
        )
        active_flags = active.detach().cpu().tolist()
        connected_flags = adjacent_connected.detach().cpu().tolist()

        index_groups: List[List[int]] = []
        current_group: List[int] = []
        for beam_index in range(points.shape[0]):
            if not active_flags[beam_index]:
                if current_group:
                    index_groups.append(current_group)
                    current_group = []
                continue
            if current_group and not connected_flags[beam_index - 1]:
                index_groups.append(current_group)
                current_group = []
            current_group.append(beam_index)
        if current_group:
            index_groups.append(current_group)

        wrap_connected = False
        if points.shape[0] > 1 and active_flags[0] and active_flags[-1]:
            wrap_connected = bool(
                (
                    torch.abs(representative_range[0] - representative_range[-1])
                    <= self.segment_max_range_jump
                )
                .logical_and(
                    (representative_xy[0] - representative_xy[-1]).norm()
                    <= self.segment_max_point_gap
                )
                .item()
            )
        if (
            len(index_groups) > 1
            and index_groups[0][0] == 0
            and index_groups[-1][-1] == points.shape[0] - 1
            and wrap_connected
        ):
            index_groups[0] = index_groups[-1] + index_groups[0]
            index_groups.pop()

        segments = []
        for indices in index_groups:
            index_tensor = torch.as_tensor(indices, dtype=torch.long, device=points.device)
            segment_points = points[index_tensor][valid[index_tensor]]
            if segment_points.shape[0] >= self.segment_min_points:
                segments.append(segment_points)
        return segments

    def _reconstruct_static_segments(
        self,
        lidar: torch.Tensor,
        dynamic_obstacle: torch.Tensor,
    ) -> List[torch.Tensor]:
        points, fit_valid, _ = self._reconstruct_static_point_grid(lidar, dynamic_obstacle)
        return self.segment_static_points(points, fit_valid)

    def _fit_one_line(self, points: torch.Tensor) -> Optional[LineFit]:
        point_count = int(points.shape[0])
        if point_count < self.fit_min_inliers:
            return None

        pairs = torch.triu_indices(
            point_count,
            point_count,
            offset=1,
            device=points.device,
        ).transpose(0, 1)
        pair_delta = points[pairs[:, 1]] - points[pairs[:, 0]]
        pair_length = pair_delta.norm(dim=-1)
        pairs = pairs[pair_length >= self.fit_min_pair_separation]
        if pairs.shape[0] == 0:
            return None
        if pairs.shape[0] > self.fit_max_hypotheses:
            stride = max(int(pairs.shape[0] // self.fit_max_hypotheses), 1)
            pairs = pairs[::stride][: self.fit_max_hypotheses]

        origins = points[pairs[:, 0]]
        tangents = _normalize_xy(points[pairs[:, 1]] - origins)
        normals = torch.stack([-tangents[:, 1], tangents[:, 0]], dim=-1)
        residual = torch.abs(
            (points.unsqueeze(0) - origins.unsqueeze(1))
            .mul(normals.unsqueeze(1))
            .sum(dim=-1)
        )
        inlier_matrix = residual <= self.fit_residual_threshold
        counts = inlier_matrix.sum(dim=1)
        best_index = int(torch.argmax(counts).item())
        inliers = inlier_matrix[best_index]
        if int(inliers.sum().item()) < self.fit_min_inliers:
            return None

        inlier_points = points[inliers]
        centroid = inlier_points.mean(dim=0)
        centered = inlier_points - centroid
        covariance = centered.transpose(0, 1) @ centered / max(int(inlier_points.shape[0]), 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        tangent = _normalize_xy(eigenvectors[:, -1])
        normal = torch.stack([-tangent[1], tangent[0]])
        refined_residual = torch.abs((points - centroid).mul(normal).sum(dim=-1))
        refined_inliers = refined_residual <= self.fit_residual_threshold
        inlier_points = points[refined_inliers]
        inlier_count = int(refined_inliers.sum().item())
        if inlier_count < self.fit_min_inliers:
            return None

        projection = (inlier_points - centroid).mul(tangent).sum(dim=-1)
        span = projection.max() - projection.min()
        mean_residual = refined_residual[refined_inliers].mean()
        count_score = torch.clamp(
            torch.as_tensor(inlier_count / float(2 * self.fit_min_inliers), device=points.device),
            max=1.0,
        )
        span_score = torch.clamp(span / (2.0 * self.fit_min_span), max=1.0)
        residual_score = torch.exp(-mean_residual / self.fit_residual_threshold)
        confidence = count_score * span_score * residual_score
        distance = torch.abs(centroid.mul(normal).sum())
        if float(span.item()) < self.fit_min_span:
            return None
        return LineFit(
            centroid=centroid,
            tangent=tangent,
            normal=normal,
            distance=distance,
            span=span,
            confidence=confidence.clamp(0.0, 1.0),
            inlier_count=inlier_count,
            inliers=refined_inliers,
        )

    def fit_lines_xy(self, points: torch.Tensor, max_lines: int = 2) -> List[LineFit]:
        """Fit up to two dominant lines; exposed for geometry tests."""
        remaining = points
        fits: List[LineFit] = []
        for _ in range(max(int(max_lines), 1)):
            fit = self._fit_one_line(remaining)
            if fit is None:
                break
            fits.append(fit)
            remaining = remaining[~fit.inliers]
            if remaining.shape[0] < self.fit_min_inliers:
                break
        return fits

    def fit_segmented_lines_xy(
        self,
        point_segments: List[torch.Tensor],
        max_lines_per_segment: int = 2,
    ) -> List[LineFit]:
        """Fit surfaces independently so disconnected clutter cannot form one line."""
        fits: List[LineFit] = []
        for segment in point_segments:
            if segment.shape[0] < self.fit_min_inliers:
                continue
            fits.extend(
                self.fit_lines_xy(
                    segment[..., :2],
                    max_lines=max_lines_per_segment,
                )
            )
        return fits

    @staticmethod
    def _normal_toward_wall(line: LineFit) -> torch.Tensor:
        normal = line.normal
        return torch.where(
            normal.dot(line.centroid) >= 0.0,
            normal,
            -normal,
        )

    def _front_points(self, points: torch.Tensor, goal_direction: torch.Tensor) -> torch.Tensor:
        if points.shape[0] == 0:
            return points[:, :2]
        points_xy = points[:, :2]
        distance = points_xy.norm(dim=-1)
        direction = points_xy / distance.unsqueeze(-1).clamp_min(1e-6)
        cosine = direction @ goal_direction
        mask = (
            (cosine >= torch.cos(torch.deg2rad(torch.as_tensor(self.trigger_front_angle, device=points.device))))
            & (distance <= self.trigger_distance)
        )
        return points_xy[mask]

    def _select_initial_line(
        self,
        lines: List[LineFit],
        goal_direction: torch.Tensor,
    ) -> Optional[LineFit]:
        candidates = []
        for line in lines:
            normal = self._normal_toward_wall(line)
            alignment = torch.abs(normal.dot(goal_direction))
            if (
                float(line.confidence.item()) >= self.fit_min_confidence
                and float(line.distance.item()) <= self.trigger_distance
                and float(alignment.item()) >= self.trigger_normal_alignment
            ):
                score = line.confidence + 0.01 * line.inlier_count
                candidates.append((float(score.item()), line))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _select_tracking_line(self, lines: List[LineFit], previous_normal: torch.Tensor) -> Optional[LineFit]:
        candidates = []
        for line in lines:
            if float(line.confidence.item()) < self.fit_min_confidence:
                continue
            normal = self._normal_toward_wall(line)
            continuity = torch.abs(normal.dot(previous_normal))
            score = line.confidence + self.tracking_continuity_weight * continuity
            candidates.append((float(score.item()), line))
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def _directional_clearance(self, points: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        if points.shape[0] == 0:
            return torch.as_tensor(self.lidar_range, device=self.device)
        points_xy = points[:, :2]
        distance = points_xy.norm(dim=-1)
        unit = points_xy / distance.unsqueeze(-1).clamp_min(1e-6)
        cosine_limit = torch.cos(torch.deg2rad(torch.as_tensor(self.side_clearance_angle, device=points.device)))
        in_sector = (unit @ direction) >= cosine_limit
        if not in_sector.any():
            return torch.as_tensor(self.lidar_range, device=points.device)
        return distance[in_sector].min().clamp(max=self.lidar_range)

    def _choose_tangent(
        self,
        normal_toward: torch.Tensor,
        goal_direction: torch.Tensor,
        velocity: torch.Tensor,
        points: torch.Tensor,
    ):
        base = torch.stack([-normal_toward[1], normal_toward[0]])
        velocity_xy = velocity[:2]
        velocity_direction = torch.where(
            velocity_xy.norm() > 1e-6,
            velocity_xy / velocity_xy.norm().clamp_min(1e-6),
            torch.zeros_like(velocity_xy),
        )
        tangents = torch.stack([base, -base], dim=0)
        scores = []
        for tangent in tangents:
            clearance = self._directional_clearance(points, tangent) / self.lidar_range
            score = (
                self.side_goal_weight * tangent.dot(goal_direction)
                + self.side_clearance_weight * clearance
                + self.side_velocity_weight * tangent.dot(velocity_direction)
            )
            scores.append(score)
        selected = tangents[int(torch.argmax(torch.stack(scores)).item())]
        handedness = torch.sign(selected[0] * normal_toward[1] - selected[1] * normal_toward[0])
        if handedness == 0:
            handedness = torch.as_tensor(1.0, device=self.device)
        return selected, handedness

    def _tangent_for_handedness(self, normal_toward: torch.Tensor, handedness: torch.Tensor) -> torch.Tensor:
        negative = torch.stack([-normal_toward[1], normal_toward[0]])
        return torch.where(handedness < 0.0, negative, -negative)

    def _goal_direction_clear(
        self,
        points: torch.Tensor,
        goal_direction: torch.Tensor,
        goal_distance: torch.Tensor,
    ) -> bool:
        if points.shape[0] == 0:
            return True
        points_xy = points[:, :2]
        distance = points_xy.norm(dim=-1)
        unit = points_xy / distance.unsqueeze(-1).clamp_min(1e-6)
        cosine_limit = torch.cos(torch.deg2rad(torch.as_tensor(self.goal_clear_angle, device=points.device)))
        blocking_distance = min(float(goal_distance.item()), self.lidar_range) - 0.1
        blocked = ((unit @ goal_direction) >= cosine_limit) & (distance < max(blocking_distance, 0.3))
        return not bool(blocked.any().item())

    def _vo_risk(self, action: torch.Tensor, dynamic_obstacle: torch.Tensor) -> torch.Tensor:
        position, obstacle_velocity, width, height, valid = self._dynamic_geometry(dynamic_obstacle)
        if not valid.any():
            return torch.zeros((), device=self.device)

        scale_xy = width / 2.0 + self.vo_xy_margin
        scale_z = height / 2.0 + self.vo_z_margin
        scale = torch.stack([scale_xy, scale_xy, scale_z], dim=-1).clamp_min(1e-6)
        relative_position = position / scale
        relative_velocity = (obstacle_velocity - action.unsqueeze(0)) / scale
        a = relative_velocity.square().sum(dim=-1)
        b = 2.0 * (relative_position * relative_velocity).sum(dim=-1)
        c = relative_position.square().sum(dim=-1) - 1.0
        discriminant = b.square() - 4.0 * a * c
        risk = torch.zeros_like(a)
        risk = torch.where(c <= 0.0, torch.ones_like(risk), risk)
        valid_ttc = valid & (c > 0.0) & (a > 1e-8) & (b < 0.0) & (discriminant >= 0.0)
        ttc = (-b - torch.sqrt(discriminant.clamp_min(0.0))) / (2.0 * a.clamp_min(1e-8))
        valid_ttc = valid_ttc & (ttc > 0.0) & (ttc <= self.vo_horizon)
        risk = torch.where(valid_ttc, torch.exp(-ttc / self.vo_tau), risk)
        risk = torch.where(valid, risk, torch.zeros_like(risk))
        return risk.max()

    def _static_sweep_clearance(
        self,
        action: torch.Tensor,
        static_points: torch.Tensor,
    ) -> torch.Tensor:
        if static_points.shape[0] == 0:
            return torch.as_tensor(self.lidar_range, device=action.device)
        sample_times = torch.linspace(
            self.dt,
            self.static_sweep_horizon,
            self.static_sweep_steps,
            device=action.device,
            dtype=action.dtype,
        )
        relative_points = (
            static_points.unsqueeze(0)
            - sample_times[:, None, None] * action.reshape(1, 1, 3)
        )
        return relative_points.norm(dim=-1).min().clamp(max=self.lidar_range)

    def _limit_horizontal_action(self, action: torch.Tensor) -> torch.Tensor:
        limited = action.clone()
        horizontal_norm = limited[:2].norm()
        if horizontal_norm > self.action_limit:
            limited[:2] *= self.action_limit / horizontal_norm
        return limited.clamp(-self.action_limit, self.action_limit)

    def _select_dynamic_safe_action(
        self,
        teacher_action: torch.Tensor,
        policy_action: torch.Tensor,
        dynamic_obstacle: torch.Tensor,
        static_points: Optional[torch.Tensor] = None,
        wall_normal: Optional[torch.Tensor] = None,
    ):
        candidates = [self._limit_horizontal_action(teacher_action)]
        demonstrable = [True]
        if wall_normal is not None:
            for away_speed in self.static_away_speeds:
                candidate = teacher_action.clone()
                candidate[:2] -= away_speed * wall_normal
                candidates.append(self._limit_horizontal_action(candidate))
                demonstrable.append(True)
        for factor in (0.75, 0.5, 0.25):
            candidate = teacher_action.clone()
            candidate[:2] *= factor
            candidates.append(self._limit_horizontal_action(candidate))
            demonstrable.append(True)
        stop = teacher_action.clone()
        stop[:2] = 0.0
        candidates.extend(
            [
                self._limit_horizontal_action(stop),
                policy_action.clamp(-self.action_limit, self.action_limit),
            ]
        )
        demonstrable.extend([False, False])
        candidate_tensor = torch.stack(candidates, dim=0)
        risks = torch.stack(
            [self._vo_risk(candidate, dynamic_obstacle) for candidate in candidate_tensor],
            dim=0,
        )
        if static_points is None:
            static_clearances = torch.full_like(risks, float("inf"))
            static_safety_radius = 0.0
        else:
            static_clearances = torch.stack(
                [
                    self._static_sweep_clearance(candidate, static_points)
                    for candidate in candidate_tensor
                ],
                dim=0,
            )
            static_safety_radius = self.static_safety_radius
        safe = (risks <= self.dynamic_risk_limit) & (
            static_clearances >= static_safety_radius
        )
        if safe.any():
            selected_index = int(safe.nonzero(as_tuple=False)[0].item())
            selected_safe = True
        else:
            # A teacher action must never replace the policy with a command that
            # fails either safety check. The policy fallback is excluded from
            # the demonstration buffer.
            selected_index = len(candidates) - 1
            selected_safe = False
        selected_is_teacher = (
            demonstrable[selected_index]
            and float(candidate_tensor[selected_index, :2].norm().item())
            >= self.demo_min_action_speed
        )
        return candidate_tensor[selected_index], risks[selected_index], selected_safe, selected_is_teacher

    def _teacher_velocity(self, env_index: int, state: torch.Tensor) -> torch.Tensor:
        distance_error = self.wall_distance[env_index] - self.reference_clearance
        normal_speed = (self.clearance_gain * distance_error).clamp(
            min=-self.max_away_wall_speed,
            max=self.max_toward_wall_speed,
        )
        horizontal = (
            self.parallel_speed * self.wall_tangent[env_index]
            + normal_speed * self.wall_normal[env_index]
        )
        if int(self.clear_counter[env_index].item()) > 0:
            clear_fraction = min(
                float(self.clear_counter[env_index].item()) / float(self.goal_clear_steps),
                1.0,
            )
            blend = self.leave_goal_blend * clear_fraction
            goal_velocity = self.leave_goal_speed * _normalize_xy(state[:2])
            horizontal = (1.0 - blend) * horizontal + blend * goal_velocity
        horizontal_norm = horizontal.norm()
        if horizontal_norm > self.action_limit:
            horizontal = horizontal * (self.action_limit / horizontal_norm)
        vertical = (self.height_gain * state[4]).clamp(
            min=-self.max_vertical_speed,
            max=self.max_vertical_speed,
        )
        action = torch.cat([horizontal, vertical.unsqueeze(0)], dim=0)
        action = (
            self.action_smoothing * self.previous_teacher_action[env_index]
            + (1.0 - self.action_smoothing) * action
        )
        return action.clamp(-self.action_limit, self.action_limit)

    @torch.no_grad()
    def act(self, tensordict) -> Dict[str, torch.Tensor]:
        observation = tensordict["agents", "observation"]
        states = observation["state"]
        lidar = observation["lidar"]
        dynamic_obstacles = observation["dynamic_obstacle"].squeeze(1)
        policy_action_normalized = tensordict["agents", "action_normalized"].reshape(self.num_envs, -1)
        policy_action = (2.0 * policy_action_normalized - 1.0) * self.action_limit
        episode_length = tensordict["stats", "episode_len"].reshape(-1).float()

        goal_distance = torch.sqrt(states[:, 3].square() + states[:, 4].square())
        reset = (self.previous_episode_length < 0.0) | (episode_length < self.previous_episode_length)
        self._reset(reset & self.teacher_env_mask, goal_distance)
        goal_progress = self.previous_goal_distance - goal_distance
        self.cooldown = (self.cooldown - 1).clamp_min(0)

        teacher_action = policy_action.clone()
        teacher_action_normalized = policy_action_normalized.clone()
        active_output = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        demo_valid = torch.zeros_like(active_output)
        commit_pending = torch.zeros_like(active_output)
        discard_pending = reset & self.teacher_env_mask
        confidence = torch.zeros(self.num_envs, device=self.device)
        phase = torch.full((self.num_envs,), PHASE_WALL_FOLLOW, dtype=torch.long, device=self.device)

        metrics = {
            "teacher_steps": len(self.teacher_env_indices),
            "active_steps": 0,
            "fit_attempts": 0,
            "fit_successes": 0,
            "triggers": 0,
            "dynamic_rejections": 0,
            "static_rejections": 0,
            "safety_fallbacks": 0,
            "recoveries": 0,
            "side_switches": 0,
            "demo_candidates": 0,
            "teacher_speed_sum": 0.0,
            "static_clearance_sum": 0.0,
            "recovery_goal_progress_sum": 0.0,
            "recovery_follow_steps_sum": 0.0,
        }

        for env_index in self.teacher_env_indices:
            state = states[env_index]
            goal_direction = _normalize_xy(state[:2])
            point_grid, fit_valid, safety_valid = self._reconstruct_static_point_grid(
                lidar[env_index],
                dynamic_obstacles[env_index],
            )
            point_segments = self.segment_static_points(point_grid, fit_valid)
            fit_points = point_grid.reshape(-1, 3)[fit_valid.reshape(-1)]
            safety_points = point_grid.reshape(-1, 3)[safety_valid.reshape(-1)]

            entered_now = False
            corner_now = False
            fresh_fit = False
            selected_line: Optional[LineFit] = None

            if not bool(self.active[env_index].item()):
                front_segments = []
                for segment in point_segments:
                    front_points = self._front_points(segment, goal_direction)
                    if front_points.shape[0] >= self.fit_min_inliers:
                        front_segments.append(front_points)
                if front_segments:
                    metrics["fit_attempts"] += 1
                    selected_line = self._select_initial_line(
                        self.fit_segmented_lines_xy(front_segments),
                        goal_direction,
                    )
                    if selected_line is not None:
                        metrics["fit_successes"] += 1
                low_motion_blocked = (
                    selected_line is not None
                    and float(goal_progress[env_index].item()) <= self.trigger_progress_eps
                    and float(state[5:8].norm().item()) <= self.trigger_max_speed
                    and int(self.cooldown[env_index].item()) == 0
                )
                close_wall_blocked = (
                    selected_line is not None
                    and float(selected_line.distance.item()) <= self.emergency_trigger_distance
                    and int(self.cooldown[env_index].item()) == 0
                )
                blocked = low_motion_blocked or close_wall_blocked
                if blocked:
                    self.blocked_counter[env_index] += 1
                else:
                    self.blocked_counter[env_index] = 0

                required_trigger_steps = (
                    self.emergency_trigger_steps if close_wall_blocked else self.trigger_steps
                )
                if int(self.blocked_counter[env_index].item()) >= required_trigger_steps:
                    normal = self._normal_toward_wall(selected_line)
                    tangent, handedness = self._choose_tangent(
                        normal,
                        goal_direction,
                        state[5:8],
                        fit_points,
                    )
                    self.active[env_index] = True
                    self.blocked_counter[env_index] = 0
                    self.follow_age[env_index] = 0
                    self.missing_fit_steps[env_index] = 0
                    self.clear_counter[env_index] = 0
                    self.side_switches[env_index] = 0
                    self.handedness[env_index] = handedness
                    self.wall_normal[env_index] = normal
                    self.wall_tangent[env_index] = tangent
                    self.wall_distance[env_index] = selected_line.distance
                    self.last_fit_confidence[env_index] = selected_line.confidence
                    # Do not blend the first recovery command with a possibly
                    # wall-directed policy velocity.
                    self.previous_teacher_action[env_index] = 0.0
                    self.hit_goal_distance[env_index] = goal_distance[env_index]
                    self.follow_distance[env_index] = 0.0
                    entered_now = True
                    fresh_fit = True
                    metrics["triggers"] += 1

            if bool(self.active[env_index].item()) and not entered_now:
                if point_segments:
                    metrics["fit_attempts"] += 1
                    selected_line = self._select_tracking_line(
                        self.fit_segmented_lines_xy(point_segments),
                        self.wall_normal[env_index],
                    )
                    if selected_line is not None:
                        metrics["fit_successes"] += 1
                        fresh_fit = True
                        new_normal = self._normal_toward_wall(selected_line)
                        if new_normal.dot(self.wall_normal[env_index]) < 0.0:
                            new_normal = -new_normal
                        angle = torch.rad2deg(
                            torch.acos(
                                new_normal.dot(self.wall_normal[env_index]).clamp(-1.0, 1.0)
                            )
                        )
                        corner_now = float(angle.item()) >= self.corner_angle
                        if corner_now:
                            smoothed_normal = new_normal
                        else:
                            smoothed_normal = _normalize_xy(
                                self.normal_smoothing * self.wall_normal[env_index]
                                + (1.0 - self.normal_smoothing) * new_normal
                            )
                        self.wall_normal[env_index] = smoothed_normal
                        self.wall_tangent[env_index] = self._tangent_for_handedness(
                            smoothed_normal,
                            self.handedness[env_index],
                        )
                        if corner_now:
                            previous_parallel_speed = self.previous_teacher_action[env_index, :2].dot(
                                self.wall_tangent[env_index]
                            ).clamp_min(0.0)
                            self.previous_teacher_action[env_index, :2] = (
                                previous_parallel_speed * self.wall_tangent[env_index]
                            )
                        self.wall_distance[env_index] = selected_line.distance
                        self.last_fit_confidence[env_index] = selected_line.confidence
                        self.missing_fit_steps[env_index] = 0
                    else:
                        self.missing_fit_steps[env_index] += 1
                else:
                    self.missing_fit_steps[env_index] += 1

            if not bool(self.active[env_index].item()):
                continue

            self.follow_age[env_index] += 1
            self.follow_distance[env_index] += state[5:7].norm() * self.dt
            goal_is_clear = self._goal_direction_clear(
                fit_points,
                goal_direction,
                goal_distance[env_index],
            )
            if goal_is_clear:
                self.clear_counter[env_index] += 1
            else:
                self.clear_counter[env_index] = 0

            progress_recovered = (
                float((self.hit_goal_distance[env_index] - goal_distance[env_index]).item())
                >= self.min_goal_progress
            )
            recovered = (
                int(self.follow_age[env_index].item()) >= self.min_follow_steps
                and float(self.follow_distance[env_index].item()) >= self.min_follow_distance
                and int(self.clear_counter[env_index].item()) >= self.goal_clear_steps
                and (progress_recovered or not self.leave_requires_goal_progress)
            )
            if recovered:
                self.active[env_index] = False
                self.cooldown[env_index] = self.cooldown_steps
                commit_pending[env_index] = True
                metrics["recoveries"] += 1
                metrics["recovery_goal_progress_sum"] += float(
                    (self.hit_goal_distance[env_index] - goal_distance[env_index]).item()
                )
                metrics["recovery_follow_steps_sum"] += float(
                    self.follow_age[env_index].item()
                )
                continue

            if int(self.follow_age[env_index].item()) >= self.max_follow_steps:
                if int(self.side_switches[env_index].item()) < self.max_side_switches:
                    self.side_switches[env_index] += 1
                    self.handedness[env_index] *= -1.0
                    self.wall_tangent[env_index] *= -1.0
                    self.follow_age[env_index] = 0
                    self.clear_counter[env_index] = 0
                    discard_pending[env_index] = True
                    metrics["side_switches"] += 1
                else:
                    self.active[env_index] = False
                    self.cooldown[env_index] = self.cooldown_steps
                    discard_pending[env_index] = True
                    continue

            if (
                int(self.missing_fit_steps[env_index].item()) > self.max_missing_fit_steps
                and int(self.clear_counter[env_index].item()) == 0
            ):
                self.active[env_index] = False
                self.cooldown[env_index] = self.cooldown_steps
                discard_pending[env_index] = True
                continue

            raw_teacher_action = self._teacher_velocity(env_index, state)
            base_dynamic_risk = self._vo_risk(
                raw_teacher_action,
                dynamic_obstacles[env_index],
            )
            base_static_clearance = self._static_sweep_clearance(
                raw_teacher_action,
                safety_points,
            )
            selected_action, dynamic_risk, selected_safe, selected_is_teacher = (
                self._select_dynamic_safe_action(
                    raw_teacher_action,
                    policy_action[env_index],
                    dynamic_obstacles[env_index],
                    static_points=safety_points,
                    wall_normal=self.wall_normal[env_index],
                )
            )
            selected_static_clearance = self._static_sweep_clearance(
                selected_action,
                safety_points,
            )
            if float(base_dynamic_risk.item()) > self.dynamic_risk_limit:
                metrics["dynamic_rejections"] += 1
            if float(base_static_clearance.item()) < self.static_safety_radius:
                metrics["static_rejections"] += 1
            if not selected_is_teacher:
                metrics["safety_fallbacks"] += 1
            self.previous_teacher_action[env_index] = selected_action
            teacher_action[env_index] = selected_action
            teacher_action_normalized[env_index] = (
                0.5 * (selected_action / self.action_limit + 1.0)
            ).clamp(self.action_epsilon, 1.0 - self.action_epsilon)
            active_output[env_index] = True

            if entered_now:
                phase[env_index] = PHASE_SIDE_SELECTION
            elif corner_now:
                phase[env_index] = PHASE_CORNER
            elif int(self.clear_counter[env_index].item()) > 0:
                phase[env_index] = PHASE_LEAVE
            else:
                phase[env_index] = PHASE_WALL_FOLLOW

            fit_confidence = self.last_fit_confidence[env_index]
            if not fresh_fit:
                fit_confidence = fit_confidence * torch.exp(
                    -self.missing_fit_steps[env_index].float() / self.max_missing_fit_steps
                )
            dynamic_confidence = 1.0 - dynamic_risk / max(self.dynamic_risk_limit, 1e-6)
            static_confidence = (
                (selected_static_clearance - self.static_safety_radius)
                / self.static_confidence_margin
            ).clamp(0.0, 1.0)
            confidence[env_index] = (
                fit_confidence
                * dynamic_confidence.clamp(0.0, 1.0)
                * static_confidence
            ).clamp(0.0, 1.0)
            demo_valid[env_index] = (
                selected_safe
                and selected_is_teacher
                and float(confidence[env_index].item()) >= self.demo_min_confidence
            )
            metrics["active_steps"] += 1
            metrics["demo_candidates"] += int(demo_valid[env_index].item())
            metrics["teacher_speed_sum"] += float(selected_action.norm().item())
            metrics["static_clearance_sum"] += float(selected_static_clearance.item())

        self.previous_goal_distance = goal_distance.clone()
        self.previous_episode_length = episode_length.clone()
        return {
            "teacher_action_g": teacher_action,
            "teacher_action_normalized": teacher_action_normalized,
            "active": active_output,
            "demo_valid": demo_valid,
            "commit_pending": commit_pending,
            "discard_pending": discard_pending,
            "confidence": confidence,
            "phase": phase,
            "metrics": metrics,
        }


class TeacherRolloutPolicy(TensorDictModuleBase):
    """Collector policy that overrides only fixed demonstration environments."""

    in_keys = []
    out_keys = [("agents", "action")]

    def __init__(self, base_policy, teacher: WallFollowTeacher, demonstration_buffer):
        super().__init__()
        # Register the PPO policy so TorchRL can synchronize its collector-side copy.
        self.base_policy = base_policy
        self.teacher = teacher
        self.demonstration_buffer = demonstration_buffer
        self._pending_demonstrations = [[] for _ in range(teacher.num_envs)]
        self._metric_totals: Dict[str, float] = {}

    def _append_pending(self, env_index: int, observation, result):
        record = {
            "observation": {
                key: observation[key][env_index].detach().to("cpu")
                for key in self.demonstration_buffer.OBSERVATION_KEYS
            },
            "action": result["teacher_action_normalized"][env_index].detach().to("cpu"),
            "confidence": result["confidence"][env_index].detach().float().to("cpu"),
            "phase": result["phase"][env_index].detach().long().to("cpu"),
        }
        self._pending_demonstrations[env_index].append(record)

    def _commit_pending(self, env_index: int) -> int:
        records = self._pending_demonstrations[env_index]
        if not records:
            return 0
        observation = {
            key: torch.stack([record["observation"][key] for record in records], dim=0)
            for key in self.demonstration_buffer.OBSERVATION_KEYS
        }
        action = torch.stack([record["action"] for record in records], dim=0)
        confidence = torch.stack([record["confidence"] for record in records], dim=0)
        phase = torch.stack([record["phase"] for record in records], dim=0)
        mask = torch.ones(len(records), dtype=torch.bool)
        written = self.demonstration_buffer.add(
            observation,
            action,
            confidence,
            phase,
            mask,
        )
        self._pending_demonstrations[env_index] = []
        return written

    def forward(self, tensordict):
        tensordict = self.base_policy(tensordict)
        result = self.teacher.act(tensordict)
        active = result["active"]
        if active.any():
            direction = tensordict["agents", "observation", "direction"]
            teacher_action_world = vec_to_world(result["teacher_action_g"], direction)
            action = tensordict["agents", "action"]
            active_shape = (active.shape[0],) + (1,) * (action.dim() - 1)
            tensordict["agents", "action"] = torch.where(
                active.reshape(active_shape),
                teacher_action_world,
                action,
            )

        written = 0
        discarded = 0
        for env_index in result["discard_pending"].nonzero(as_tuple=False).flatten().cpu().tolist():
            discarded += len(self._pending_demonstrations[env_index])
            self._pending_demonstrations[env_index] = []
        for env_index in result["commit_pending"].nonzero(as_tuple=False).flatten().cpu().tolist():
            written += self._commit_pending(env_index)
        observation = tensordict["agents", "observation"]
        for env_index in result["demo_valid"].nonzero(as_tuple=False).flatten().cpu().tolist():
            self._append_pending(env_index, observation, result)

        metrics = dict(result["metrics"])
        metrics["demo_written"] = written
        metrics["demo_discarded"] = discarded
        for key, value in metrics.items():
            self._metric_totals[key] = self._metric_totals.get(key, 0.0) + float(value)
        return tensordict

    def pop_metrics(self) -> Dict[str, float]:
        totals = self._metric_totals
        self._metric_totals = {}
        teacher_steps = max(totals.get("teacher_steps", 0.0), 1.0)
        fit_attempts = max(totals.get("fit_attempts", 0.0), 1.0)
        active_steps = max(totals.get("active_steps", 0.0), 1.0)
        triggers = max(totals.get("triggers", 0.0), 1.0)
        recoveries = max(totals.get("recoveries", 0.0), 1.0)
        metrics = {
            "teacher/active_rate": totals.get("active_steps", 0.0) / teacher_steps,
            "teacher/fit_success_rate": totals.get("fit_successes", 0.0) / fit_attempts,
            "teacher/dynamic_rejection_rate": totals.get("dynamic_rejections", 0.0) / active_steps,
            "teacher/static_rejection_rate": totals.get("static_rejections", 0.0) / active_steps,
            "teacher/safety_fallback_rate": totals.get("safety_fallbacks", 0.0) / active_steps,
            "teacher/triggers": totals.get("triggers", 0.0),
            "teacher/recoveries": totals.get("recoveries", 0.0),
            "teacher/recovery_rate": totals.get("recoveries", 0.0) / triggers,
            "teacher/side_switches": totals.get("side_switches", 0.0),
            "teacher/demo_written": totals.get("demo_written", 0.0),
            "teacher/demo_discarded": totals.get("demo_discarded", 0.0),
            "teacher/demo_buffer_size": float(len(self.demonstration_buffer)),
            "teacher/pending_demo_size": float(
                sum(len(records) for records in self._pending_demonstrations)
            ),
            "teacher/action_speed_mean": totals.get("teacher_speed_sum", 0.0) / active_steps,
            "teacher/static_clearance_mean": (
                totals.get("static_clearance_sum", 0.0) / active_steps
            ),
            "teacher/recovery_goal_progress_mean": (
                totals.get("recovery_goal_progress_sum", 0.0) / recoveries
            ),
            "teacher/recovery_follow_steps_mean": (
                totals.get("recovery_follow_steps_sum", 0.0) / recoveries
            ),
        }
        for phase, count in self.demonstration_buffer.phase_counts().items():
            phase_name = PHASE_NAMES.get(phase, str(phase))
            metrics[f"teacher/buffer_phase_{phase_name}"] = float(count)
        return metrics

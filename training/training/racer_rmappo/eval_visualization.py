"""Evaluation-only traces and Isaac 2023.1 debug overlays (no USD colliders).

Geometry/recording helpers use the standard library. Omniverse imports are
delayed until a GUI viewer is explicitly enabled after SimulationApp startup.
The frustum is geometric, NOT an occlusion-aware explored-space estimate.
"""

from __future__ import annotations

from collections import deque
import copy
import itertools
import json
import math
from pathlib import Path


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def rotate_wxyz(quaternion, vector):
    norm = math.sqrt(sum(x * x for x in quaternion))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("invalid body quaternion")
    w, x, y, z = (v / norm for v in quaternion)
    vx, vy, vz = vector
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx)


def box_edges(lower, upper):
    corners = list(itertools.product(*zip(lower, upper)))
    return [(corners[i], corners[j]) for i in range(8) for j in range(i + 1, 8)
            if (i ^ j) in (1, 2, 4)]


def goal_rings(center, radius):
    edges = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        points = []
        for i in range(33):
            p = list(center)
            p[a] += radius * math.cos(2 * math.pi * i / 32)
            p[b] += radius * math.sin(2 * math.pi * i / 32)
            points.append(tuple(p))
        edges.extend(zip(points[:-1], points[1:]))
    return edges


def frustum_edges(position, quaternion, camera, display_range_m=3.0):
    """Full body roll/pitch/yaw, mounting offset and configured pinhole intrinsics."""
    def unit(vector):
        norm = math.sqrt(sum(x * x for x in vector))
        return tuple(x / norm for x in vector)

    forward = unit(camera["forward_axis_body"])
    up = unit(camera["up_axis_body"])
    left = (up[1]*forward[2] - up[2]*forward[1],
            up[2]*forward[0] - up[0]*forward[2],
            up[0]*forward[1] - up[1]*forward[0])
    origin = add(position, rotate_wxyz(quaternion, camera["mount_position_body_m"]))
    distance = min(float(display_range_m), float(camera["max_depth_m"]))
    if distance <= 0 or not math.isfinite(distance):
        raise ValueError("display range must be finite and positive")
    corners = []
    for u, v in ((-0.5, -0.5), (camera["width"]-0.5, -0.5),
                 (camera["width"]-0.5, camera["height"]-0.5), (-0.5, camera["height"]-0.5)):
        horizontal = -(u - camera["cx"]) / camera["fx"]
        vertical = -(v - camera["cy"]) / camera["fy"]
        ray = tuple(distance * (f + horizontal*l + vertical*h)
                    for f, l, h in zip(forward, left, up))
        corners.append(add(origin, rotate_wxyz(quaternion, ray)))
    return [(origin, point) for point in corners] + [
        (corners[i], corners[(i + 1) % 4]) for i in range(4)]


class EvaluationTrace:
    """One-agent terminal-safe JSONL recorder; optional GUI is independent of PPO."""

    def __init__(self, path, isaac_cfg, checkpoint, scenario):
        self.path = Path(path).expanduser().resolve()
        self.cfg = copy.deepcopy(isaac_cfg)
        self.dt = 1.0 / float(isaac_cfg["control"]["control_hz"])
        self.speed_limit = float(isaac_cfg["control"]["max_speed_mps"])
        self.episode = 0
        self.overspeed_steps = 0
        self.navigation_steps = 0
        self.peak = None
        self.viewer = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("w", encoding="utf-8", buffering=1)
        self._write({"type": "metadata", "schema_version": 1,
                     "checkpoint": str(checkpoint), "scenario": scenario,
                     "isaac_config": self.cfg, "coverage_available": False,
                     "note": "Geometric frustum only; no voxel map/frontier in this backend."})

    def _write(self, row):
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")

    def enable_viewer(self, probe, view="overview", draw_every=5, frustum_range=3.0):
        self.viewer = IsaacEvaluationViewer(self.cfg, probe, view, draw_every, frustum_range)

    def __call__(self, step, info):
        snapshot = info.get("diagnostic_snapshot")
        if snapshot is None:
            raise RuntimeError("backend.collect_diagnostics must be enabled before evaluation")
        row = copy.deepcopy(snapshot)
        row.update(type="step", step=step, episode=self.episode,
                   sim_time_s=step * self.dt,
                   episode_time_s=snapshot["episode_step"] * self.dt)
        # sim_time_s remains the historical nominal clock. physics_time_s and
        # timing_step are measured from PhysX callbacks, including reset sync.
        row["nominal_sim_time_s"] = row["sim_time_s"]
        row["overspeed"] = row["actual_speed_mps"] > self.speed_limit + 0.05
        self.overspeed_steps += int(row["overspeed"])
        self.navigation_steps += int(row["navigation_reached"])
        if self.peak is None or row["actual_speed_mps"] > self.peak["actual_speed_mps"]:
            self.peak = row
        self._write(row)
        if self.viewer is not None:
            self.viewer.update(row)
        if row["done"]:
            causes = [key for key in ("success", "collision", "out_of_bounds", "stall", "timeout")
                      if row[key]]
            print(f"[eval episode] id={self.episode} end={','.join(causes)} "
                  f"distance={row['target_distance_m']:.3f} "
                  f"speed={row['actual_speed_mps']:.3f} z={row['position_m'][2]:.3f}", flush=True)
            self.episode += 1

    def summary(self):
        return {"trace_path": str(self.path), "coverage_available": False,
                "overspeed_threshold_mps": self.speed_limit + 0.05,
                "overspeed_control_steps": self.overspeed_steps,
                "navigation_reached_control_steps": self.navigation_steps,
                "peak_speed_transition": self.peak}

    def close(self):
        try:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
        finally:
            if not self.stream.closed:
                self.stream.close()


class IsaacEvaluationViewer:
    """Debug Draw overlays and omni.ui HUD; no new objects in sensor geometry."""

    def __init__(self, cfg, probe, view, draw_every, frustum_range):
        from omni.isaac.core.utils.extensions import enable_extension
        enable_extension("omni.isaac.debug_draw")
        from omni.isaac.debug_draw import _debug_draw
        from omni.isaac.core.utils.viewports import set_camera_view
        import omni.ui as ui

        self.cfg, self.probe = cfg, probe
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.set_camera_view = set_camera_view
        self.view, self.interval = view, max(1, int(draw_every))
        self.frustum_range = frustum_range
        self.trail = deque(maxlen=4000)
        self.last_position = None
        self.last_episode = None
        self.last_end = "none"
        self.window = ui.Window("RACER evaluation diagnostics", width=485, height=430)
        with self.window.frame:
            self.label = ui.Label("Waiting for first evaluation step", word_wrap=True)
        spawn = cfg["scene"]["spawn_position_m"]
        target = cfg["navigation_backend"]["fixed_target_position_m"]
        center = [(x + y) / 2 for x, y in zip(spawn, target)]
        if view == "top":
            self.set_camera_view(eye=add(center, (0.0, -0.01, 22.0)), target=center)
        else:
            self.set_camera_view(eye=add(center, (11.0, -15.0, 11.0)), target=center)

    def update(self, row):
        position = tuple(row["position_m"])
        if self.last_episode != row["episode"]:
            self.last_position = tuple(self.cfg["scene"]["spawn_position_m"])
            self.last_episode = row["episode"]
        failure = row["done"] and not row["success"]
        color = (1.0, 0.15, 0.1, 1.0) if failure else (0.0, 0.85, 1.0, 1.0)
        self.trail.append((self.last_position, position, color))
        self.last_position = position
        if row["done"]:
            self.last_end = ",".join(k for k in ("success", "collision", "out_of_bounds", "stall", "timeout")
                                     if row[k])
        if row["step"] != 1 and row["step"] % self.interval and not row["done"]:
            return

        self.draw.clear_lines()
        starts, ends, colors, widths = [], [], [], []
        def append(edges, color, width=2.0):
            for start, end in edges:
                starts.append(tuple(start)); ends.append(tuple(end))
                colors.append(color); widths.append(width)

        scene = self.cfg["scene"]
        append(box_edges(scene["flight_bounds_min_m"], scene["flight_bounds_max_m"]),
               (0.6, 0.6, 0.6, 0.6), 1.0)
        nav = self.cfg["navigation_backend"]
        append(goal_rings(row["target_position_m"], nav["goal_position_tolerance_m"]),
               (0.2, 1.0, 0.2, 1.0))
        for start, end, trail_color in self.trail:
            append([(start, end)], trail_color)
        append(frustum_edges(position, row["orientation_wxyz"], self.cfg["camera"], self.frustum_range),
               (1.0, 0.75, 0.0, 0.75), 1.0)
        append([(position, add(position, row["velocity_world_mps"]))], (1.0, 0.15, 1.0, 1.0), 4.0)
        append([(position, add(position, row["applied_command_world"][:3]))], (0.2, 1.0, 0.2, 1.0), 3.0)
        self.draw.draw_lines(starts, ends, colors, widths)
        if self.view == "follow":
            self.set_camera_view(eye=add(position, (-5.0, -7.0, 5.0)), target=position)

        self.label.text = (
            f"Episode {row['episode']}  step {row['episode_step']}  t={row['episode_time_s']:.2f}s\n"
            f"Position: {tuple(round(v, 3) for v in position)}\n"
            f"Speed actual: {row['actual_speed_mps']:.3f} m/s "
            f"{'OVERSPEED' if row['overspeed'] else ''}\n"
            f"Body velocity/yaw command: {tuple(round(v, 3) for v in row['command_body'])}\n"
            f"Distance: {row['target_distance_m']:.3f} / {nav['goal_position_tolerance_m']:.2f} m\n"
            f"Yaw error: {row['target_yaw_error_rad']:.3f} rad  tilt: {row['body_tilt_rad']:.3f} rad\n"
            f"At position: {row['navigation_reached']}  pose: {row.get('navigation_pose_reached', False)}\n"
            f"Map observation completed: {row['observation_completed']}\n"
            f"Reward: {row['reward']:.4f}  Last end: {self.last_end}\n"
            "Cyan: trajectory; green sphere: goal tolerance\n"
            "Magenta: actual velocity; green vector: filtered command (1s scale)\n"
            f"Yellow: geometric frustum truncated to {self.frustum_range:g}m\n"
            "Explored voxels / coverage: NOT IMPLEMENTED\n"
            "At episode end, HUD/trace show terminal state; drone may already reset."
        )
        # Only queue drawing data. The next centrally scheduled camera render
        # presents it; GUI callbacks must not tick Kit or the physical world.

    def close(self):
        try:
            self.draw.clear_lines()
        finally:
            self.window.destroy()
            self.label = None

"""First-stage single-drone Isaac Sim physics probe.

This module deliberately does not import Omniverse modules at import time.
The caller must create ``SimulationApp`` first, then construct
``IsaacSingleDroneProbe``. It is not yet the PPO backend: mapping, RACER
candidates and rewards are introduced in later P2/P3 stages.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor

from .d455m_sensor import (
    FixedBodyD455MSensor,
    depth_to_normalized_inverse,
    resize_inverse_depth_for_actor,
    summarize_depth,
    validate_d455m_config,
)


def _require_vector(
    mapping: Mapping[str, Any], key: str, length: int
) -> list[float]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of length {length}")
    result = [float(item) for item in value]
    if len(result) != length:
        raise ValueError(f"{key} must contain {length} values, got {len(result)}")
    return result


def validate_probe_config(cfg: Mapping[str, Any]) -> None:
    """Validate the parts of the P1 probe configuration used at runtime."""
    required = {
        "app",
        "sim",
        "scene",
        "drone",
        "control",
        "collision",
        "camera",
        "navigation_backend",
        "probe",
        "acceptance",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Isaac probe configuration is missing: {sorted(missing)}")

    sim = cfg["sim"]
    if float(sim["physics_dt"]) <= 0.0:
        raise ValueError("sim.physics_dt must be positive")
    if float(sim.get("rendering_dt", sim["physics_dt"])) <= 0.0:
        raise ValueError("sim.rendering_dt must be positive")
    if not str(sim["device"]).startswith("cuda"):
        raise ValueError("the high-fidelity probe requires a CUDA device")

    scene = cfg["scene"]
    size = _require_vector(scene, "size_m", 3)
    lower = _require_vector(scene, "flight_bounds_min_m", 3)
    upper = _require_vector(scene, "flight_bounds_max_m", 3)
    spawn = _require_vector(scene, "spawn_position_m", 3)
    if any(item <= 0.0 for item in size):
        raise ValueError("scene.size_m must be positive")
    if any(lo >= hi for lo, hi in zip(lower, upper)):
        raise ValueError("each flight bound minimum must be smaller than its maximum")
    if any(not lo < value < hi for value, lo, hi in zip(spawn, lower, upper)):
        raise ValueError("scene.spawn_position_m must be strictly inside flight bounds")

    names: set[str] = set()
    for obstacle in scene.get("obstacles", []):
        name = str(obstacle["name"])
        if not name or name in names:
            raise ValueError(f"obstacle names must be non-empty and unique: {name!r}")
        names.add(name)
        _require_vector(obstacle, "position_m", 3)
        obstacle_size = _require_vector(obstacle, "size_m", 3)
        if any(item <= 0.0 for item in obstacle_size):
            raise ValueError(f"obstacle {name} size must be positive")

    control = cfg["control"]
    if str(control.get("command_frame")) != "yaw_local":
        raise ValueError("control.command_frame must be 'yaw_local'")
    control_hz = float(control["control_hz"])
    physics_steps_per_action = int(control["physics_steps_per_action"])
    if control_hz <= 0.0:
        raise ValueError("control.control_hz must be positive")
    if physics_steps_per_action < 1:
        raise ValueError("control.physics_steps_per_action must be at least one")
    configured_control_dt = 1.0 / control_hz
    simulated_control_dt = float(sim["physics_dt"]) * physics_steps_per_action
    if not math.isclose(
        configured_control_dt,
        simulated_control_dt,
        rel_tol=1e-6,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "control_hz must match sim.physics_dt * physics_steps_per_action: "
            f"expected {configured_control_dt:.9f}s, got {simulated_control_dt:.9f}s"
        )
    if float(control["max_speed_mps"]) <= 0.0:
        raise ValueError("control.max_speed_mps must be positive")
    if float(control["max_yaw_rate_rps"]) <= 0.0:
        raise ValueError("control.max_yaw_rate_rps must be positive")
    if float(control["max_acceleration_mps2"]) <= 0.0:
        raise ValueError("control.max_acceleration_mps2 must be positive")
    if float(control["max_yaw_acceleration_rps2"]) <= 0.0:
        raise ValueError("control.max_yaw_acceleration_rps2 must be positive")
    if int(control["random_command_interval_steps"]) < 1:
        raise ValueError("random_command_interval_steps must be at least one")

    validate_d455m_config(cfg["camera"])
    camera_probe = cfg["camera"].get("probe", {})
    if int(camera_probe.get("pose_sync_control_steps", 1)) < 1:
        raise ValueError("camera.probe.pose_sync_control_steps must be at least one")
    if float(cfg["acceptance"]["camera_pose_sync_yaw_abs_error_rad"]) <= 0.0:
        raise ValueError(
            "acceptance.camera_pose_sync_yaw_abs_error_rad must be positive"
        )

    navigation = cfg["navigation_backend"]
    target = _require_vector(navigation, "fixed_target_position_m", 3)
    if any(not lo < value < hi for value, lo, hi in zip(target, lower, upper)):
        raise ValueError("navigation_backend fixed target must be inside flight bounds")
    for key in (
        "goal_position_tolerance_m",
        "goal_yaw_tolerance_rad",
        "goal_tilt_tolerance_rad",
        "episode_seconds",
    ):
        if float(navigation[key]) <= 0.0:
            raise ValueError(f"navigation_backend.{key} must be positive")
    if int(navigation.get("reset_pose_sync_control_steps", 1)) < 1:
        raise ValueError(
            "navigation_backend.reset_pose_sync_control_steps must be at least one"
        )
    lifecycle = navigation.get("lifecycle_probe", {})
    if float(lifecycle.get("reset_center_inverse_depth_tolerance", 0.0)) <= 0.0:
        raise ValueError(
            "navigation_backend.lifecycle_probe."
            "reset_center_inverse_depth_tolerance must be positive"
        )


def limit_velocity_command(
    command: Tensor,
    max_speed_mps: float,
    max_yaw_rate_rps: float,
) -> Tensor:
    """Limit ``[vx, vy, vz, yaw_rate]`` without changing its direction."""
    if command.shape[-1] != 4:
        raise ValueError(f"command must end in four values, got {tuple(command.shape)}")
    if max_speed_mps <= 0.0 or max_yaw_rate_rps <= 0.0:
        raise ValueError("command limits must be positive")

    result = command.clone()
    velocity = result[..., :3]
    speed = velocity.norm(dim=-1, keepdim=True)
    scale = (float(max_speed_mps) / speed.clamp_min(1e-9)).clamp(max=1.0)
    result[..., :3] = velocity * scale
    result[..., 3] = result[..., 3].clamp(
        -float(max_yaw_rate_rps), float(max_yaw_rate_rps)
    )
    return result


def yaw_local_velocity_to_world(velocity: Tensor, yaw: Tensor) -> Tensor:
    """Rotate yaw-local ``[forward, left, up]`` velocity into world axes."""
    if velocity.shape[-1] != 3:
        raise ValueError("velocity must end in three values")
    yaw = torch.as_tensor(yaw, dtype=velocity.dtype, device=velocity.device)
    target_shape = velocity.shape[:-1]
    if not target_shape and yaw.numel() == 1:
        yaw = yaw.reshape(())
    else:
        yaw = torch.broadcast_to(yaw, target_shape)
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    world_x = cosine * velocity[..., 0] - sine * velocity[..., 1]
    world_y = sine * velocity[..., 0] + cosine * velocity[..., 1]
    return torch.stack((world_x, world_y, velocity[..., 2]), dim=-1)


def limit_vector_change(previous: Tensor, desired: Tensor, max_change: float) -> Tensor:
    """Limit the norm of a vector change while preserving its direction."""
    if previous.shape != desired.shape:
        raise ValueError("previous and desired vectors must have identical shapes")
    if max_change <= 0.0:
        raise ValueError("max_change must be positive")
    delta = desired - previous
    delta_norm = delta.norm(dim=-1, keepdim=True)
    scale = (float(max_change) / delta_norm.clamp_min(1e-9)).clamp(max=1.0)
    return previous + delta * scale


def limit_velocity_for_braking_distance(
    position: Tensor,
    velocity: Tensor,
    lower: Tensor,
    upper: Tensor,
    braking_acceleration_mps2: float,
) -> Tensor:
    """Limit outward velocity so it can stop before an axis-aligned bound."""
    if not (position.shape == velocity.shape == lower.shape == upper.shape):
        raise ValueError("position, velocity and bounds must have identical shapes")
    if braking_acceleration_mps2 <= 0.0:
        raise ValueError("braking_acceleration_mps2 must be positive")
    acceleration = float(braking_acceleration_mps2)
    positive_limit = torch.sqrt(
        2.0 * acceleration * (upper - position).clamp_min(0.0)
    )
    negative_limit = torch.sqrt(
        2.0 * acceleration * (position - lower).clamp_min(0.0)
    )
    return torch.minimum(torch.maximum(velocity, -negative_limit), positive_limit)


def quaternion_yaw_wxyz(quaternion: Tensor) -> Tensor:
    """Return yaw from a normalized ``[w, x, y, z]`` quaternion."""
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in four values")
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


class IsaacSingleDroneProbe:
    """Minimal physical scene used to validate P1 before training."""

    def __init__(self, cfg: Mapping[str, Any], render: bool = False) -> None:
        validate_probe_config(cfg)
        self.cfg = dict(cfg)
        self.render = bool(render)
        self.render_on_step = self.render

        # These imports must remain after SimulationApp is constructed.
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.version import get_version
        from omni_drones.controllers import LeePositionController
        from omni_drones.robots.drone import MultirotorBase
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

        sim_cfg = cfg["sim"]
        self.isaac_version = ".".join(str(item) for item in get_version())
        self.physics_dt = float(sim_cfg["physics_dt"])
        self.device = torch.device(str(sim_cfg["device"]))
        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.physics_dt,
            rendering_dt=float(sim_cfg.get("rendering_dt", self.physics_dt)),
            sim_params=dict(sim_cfg.get("params", {})),
            backend="torch",
            device=str(self.device),
        )

        scene_cfg = cfg["scene"]
        stage = get_current_stage()

        def spawn_static_cuboid(
            prim_path: str,
            position: Sequence[float],
            size: Sequence[float],
            color: Sequence[float] | None = None,
        ) -> None:
            """Create a local USD collider without requiring Isaac Orbit."""
            cube = UsdGeom.Cube.Define(stage, prim_path)
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(
                Gf.Vec3d(*(float(item) for item in position))
            )
            cube.AddScaleOp().Set(Gf.Vec3d(*(float(item) for item in size)))
            if color is not None:
                cube.CreateDisplayColorAttr().Set(
                    [Gf.Vec3f(*(float(item) for item in color))]
                )

            prim = cube.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr().Set(0.02)
            physx_collision.CreateRestOffsetAttr().Set(0.0)

        # Local USD primitives avoid both a remote Nucleus asset dependency and
        # the optional ``omni.isaac.orbit`` extension used by older OmniDrones.
        spawn_static_cuboid(
            "/World/defaultGroundPlane",
            position=(0.0, 0.0, -0.05),
            size=(
                float(scene_cfg["size_m"][0]),
                float(scene_cfg["size_m"][1]),
                0.10,
            ),
            color=scene_cfg["ground_color"],
        )

        light = UsdLux.DistantLight.Define(stage, "/World/P1ProbeLight")
        light.CreateIntensityAttr(3000.0)
        light.CreateColorAttr(Gf.Vec3f(0.75, 0.75, 0.75))

        self.obstacles = list(scene_cfg.get("obstacles", []))
        UsdGeom.Xform.Define(stage, "/World/P1Obstacles")
        for obstacle in self.obstacles:
            spawn_static_cuboid(
                f"/World/P1Obstacles/{obstacle['name']}",
                position=obstacle["position_m"],
                size=obstacle["size_m"],
            )

        drone_model = str(cfg["drone"]["model"])
        if drone_model not in MultirotorBase.REGISTRY:
            available = sorted(MultirotorBase.REGISTRY)
            raise ValueError(f"unknown drone model {drone_model!r}; available: {available}")
        drone_class = MultirotorBase.REGISTRY[drone_model]
        self.drone = drone_class(cfg=drone_class.cfg_cls(force_sensor=False))
        self.spawn_position = torch.tensor(
            scene_cfg["spawn_position_m"], dtype=torch.float32, device=self.device
        ).reshape(1, 1, 3)
        self.spawn_orientation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
        ).reshape(1, 1, 4)
        self.drone_root_path = f"/World/envs/env_0/{self.drone.name}_0"
        self.drone.spawn(
            translations=self.spawn_position.reshape(-1, 3),
            prim_paths=[self.drone_root_path],
        )
        self.depth_camera = FixedBodyD455MSensor(
            cfg["camera"],
            parent_prim_path=f"{self.drone_root_path}/base_link",
            simulation_context=self.sim,
        )
        self._closed = False

        try:
            self.sim.reset()
            self.drone.initialize(track_contact_forces=True)
            self.depth_camera.initialize()
        except BaseException:
            # Constructor assignment in the caller has not completed yet, so
            # it cannot call close() for us when camera/PhysX setup fails.
            try:
                self.close()
            except Exception:
                pass
            raise
        self.controller = LeePositionController(
            g=9.81, uav_params=self.drone.params
        ).to(self.device)
        from .physics_timing import PhysicsTimingAudit
        self.timing = PhysicsTimingAudit(
            self.physics_dt, cfg["control"]["physics_steps_per_action"]
        )
        self.sim.add_physics_callback("racer_physics_audit", self.timing.on_physics_step)
        self.depth_camera.render_frame = self.render_frame
        self.last_rotor_action = None
        self.env_ids = torch.tensor([0], dtype=torch.long, device=self.device)
        self.zero_velocities = torch.zeros(1, 1, 6, device=self.device)

        self.bounds_min = torch.tensor(
            scene_cfg["flight_bounds_min_m"], dtype=torch.float32, device=self.device
        )
        self.bounds_max = torch.tensor(
            scene_cfg["flight_bounds_max_m"], dtype=torch.float32, device=self.device
        )
        control = cfg["control"]
        self.control_hz = float(control["control_hz"])
        self.control_dt = 1.0 / self.control_hz
        self.physics_steps_per_action = int(
            control["physics_steps_per_action"]
        )
        self.max_speed = float(control["max_speed_mps"])
        self.max_yaw_rate = float(control["max_yaw_rate_rps"])
        self.max_acceleration = float(control["max_acceleration_mps2"])
        self.max_yaw_acceleration = float(
            control["max_yaw_acceleration_rps2"]
        )
        self.reference_margin = float(control["reference_boundary_margin_m"])
        self.contact_threshold = float(
            cfg["collision"]["contact_force_threshold_n"]
        )
        self.target_position = self.spawn_position.reshape(1, 3).clone()
        self.target_yaw = torch.zeros(1, device=self.device)
        self.last_world_velocity_command = torch.zeros(3, device=self.device)
        self.last_yaw_rate_command = torch.zeros(1, device=self.device)
        self._initial_reset_available = False
        self.reset()
        # The constructor leaves the articulation in the exact state required
        # by a new backend.  Let that backend consume this state once instead
        # of immediately issuing a second tensor teleport.  Isaac Sim 2023.1
        # can reject that duplicate startup teleport and leave the first RTX
        # camera frame stale even though tensor telemetry looks correct.
        self._initial_reset_available = True

    def close(self) -> None:
        """Release PhysX views before the owning SimulationApp shuts down."""
        if self._closed:
            return
        self._closed = True

        sim = self.sim
        drone = self.drone
        depth_camera = self.depth_camera
        camera_close_error: Exception | None = None
        if depth_camera is not None:
            try:
                depth_camera.close()
            except Exception as error:
                # Continue releasing PhysX/SimulationContext resources. The
                # caller still receives the renderer cleanup error afterwards.
                camera_close_error = error
        self.depth_camera = None
        try:
            # OmniDrones keeps a process-wide strong reference to every robot.
            # Remove it before Kit unloads the PhysX plugins.
            from omni_drones.robots import RobotBase

            RobotBase._robots.pop(drone.name, None)
        except (AttributeError, ImportError):
            pass

        self.controller = None
        self.drone = None
        del drone
        gc.collect()
        if sim is not None:
            try:
                sim.stop()
            finally:
                clear_callbacks = getattr(sim, "clear_all_callbacks", None)
                if callable(clear_callbacks):
                    clear_callbacks()
                clear_instance = getattr(sim, "clear_instance", None)
                if callable(clear_instance):
                    clear_instance()
        self.sim = None
        if camera_close_error is not None:
            raise camera_close_error

    def _flush(self) -> None:
        physics_view = getattr(self.sim, "_physics_sim_view", None)
        if physics_view is not None:
            physics_view.flush()

    def render_frame(self) -> None:
        """Use Kit's render-only path; fail closed if the installed build steps physics."""
        self.timing.render_only(self.sim.render)

    def reset(self) -> Dict[str, Any]:
        self._initial_reset_available = False
        self.drone._reset_idx(self.env_ids, train=False)
        self.drone.set_world_poses(self.spawn_position, self.spawn_orientation)
        self.drone.set_velocities(self.zero_velocities)
        self.target_position.copy_(self.spawn_position.reshape(1, 3))
        self.target_yaw.zero_()
        self.last_world_velocity_command.zero_()
        self.last_yaw_rate_command.zero_()
        self._flush()
        return self.telemetry()

    def consume_initial_reset_telemetry(self) -> Dict[str, Any] | None:
        """Return the constructor-reset state exactly once.

        This narrow hand-off avoids a redundant Direct-GPU root-pose write at
        backend startup.  All episode resets still execute :meth:`reset`.
        """
        if not self._initial_reset_available:
            return None
        self._initial_reset_available = False
        return self.telemetry()

    def set_test_state(self, position: Tensor, velocity: Tensor | None = None) -> None:
        position = position.to(device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        self.drone.set_world_poses(position, self.spawn_orientation)
        if velocity is None:
            velocities = self.zero_velocities
        else:
            velocity = velocity.to(device=self.device, dtype=torch.float32)
            if velocity.numel() == 3:
                velocities = torch.cat(
                    (velocity.reshape(1, 1, 3), torch.zeros(1, 1, 3, device=self.device)),
                    dim=-1,
                )
            elif velocity.numel() == 6:
                velocities = velocity.reshape(1, 1, 6)
            else:
                raise ValueError("test velocity must have three or six values")
        self.drone.set_velocities(velocities)
        self.target_position.copy_(position.reshape(1, 3))
        self._flush()

    def set_test_pose(self, position: Tensor, yaw_rad: float) -> None:
        """Teleport the drone while preserving the fixed camera/body transform."""
        half_yaw = 0.5 * float(yaw_rad)
        orientation = torch.tensor(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            dtype=torch.float32,
            device=self.device,
        ).reshape(1, 1, 4)
        position = position.to(device=self.device, dtype=torch.float32).reshape(1, 1, 3)
        self.drone.set_world_poses(position, orientation)
        self.drone.set_velocities(self.zero_velocities)
        self.target_position.copy_(position.reshape(1, 3))
        self.target_yaw.fill_(float(yaw_rad))
        self.last_world_velocity_command.zero_()
        self.last_yaw_rate_command.zero_()
        self._flush()

    def synchronize_pose_to_renderer(self, control_steps: int = 1) -> Dict[str, Any]:
        """Publish a tensor-written root pose to Fabric and settle motion.

        Isaac Sim 2023.1 can expose a new root pose through PhysX tensor reads
        while a fixed child camera still renders its previous transform. One
        or more zero-command physics/control steps propagate the articulation
        transform to Fabric. Velocities are cleared afterwards so an episode
        starts from a settled state; the sub-millimetre position change is
        retained and reported rather than hidden by a second teleport.
        """
        if int(control_steps) < 1:
            raise ValueError("control_steps must be at least one")
        latched_collision = False
        latched_out_of_bounds = False
        finite = True
        max_contact_force = 0.0
        telemetry = self.telemetry()
        sync_start = self.timing.physics_steps
        for _ in range(int(control_steps)):
            telemetry, _ = self.step((0.0, 0.0, 0.0, 0.0))
            latched_collision |= bool(telemetry["collision"])
            latched_out_of_bounds |= bool(telemetry["out_of_bounds"])
            finite &= bool(telemetry["finite"])
            max_contact_force = max(
                max_contact_force, float(telemetry["max_contact_force_n"])
            )

        self.timing.reset_sync_steps += self.timing.physics_steps - sync_start

        self.drone.set_velocities(self.zero_velocities)
        self.last_world_velocity_command.zero_()
        self.last_yaw_rate_command.zero_()
        self._flush()
        settled = self.telemetry()
        settled["collision"] = latched_collision or bool(settled["collision"])
        settled["out_of_bounds"] = latched_out_of_bounds or bool(
            settled["out_of_bounds"]
        )
        settled["finite"] = finite and bool(settled["finite"])
        settled["max_contact_force_n"] = max(
            max_contact_force, float(settled["max_contact_force_n"])
        )
        return settled

    def _state(self) -> Tensor:
        return self.drone.get_state(check_nan=False, env_frame=False)

    def telemetry(self) -> Dict[str, Any]:
        state = self._state()
        position = state[..., :3].reshape(-1, 3)[0]
        orientation = state[..., 3:7].reshape(-1, 4)[0]
        velocity = state[..., 7:13].reshape(-1, 6)[0]
        yaw = quaternion_yaw_wxyz(orientation)
        up_z = state[..., 18].reshape(-1)[0]
        contact_forces = self.drone.base_link.get_net_contact_forces(clone=True)
        max_contact_force = contact_forces.norm(dim=-1).max()
        out_of_bounds = torch.logical_or(
            position < self.bounds_min, position > self.bounds_max
        ).any()
        finite = torch.isfinite(state).all() & torch.isfinite(contact_forces).all()
        return {
            "position_m": position.detach().cpu().tolist(),
            "orientation_wxyz": orientation.detach().cpu().tolist(),
            "linear_velocity_mps": velocity[:3].detach().cpu().tolist(),
            "angular_velocity_rps": velocity[3:].detach().cpu().tolist(),
            "speed_mps": float(velocity[:3].norm().item()),
            "yaw_rad": float(yaw.item()),
            "up_z": float(up_z.item()),
            "max_contact_force_n": float(max_contact_force.item()),
            "collision": bool((max_contact_force > self.contact_threshold).item()),
            "out_of_bounds": bool(out_of_bounds.item()),
            "finite": bool(finite.item()),
        }

    def step(self, command: Tensor | Sequence[float]) -> tuple[Dict[str, Any], Tensor]:
        timing_before = self.timing.snapshot()
        raw = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        command_limited = limit_velocity_command(
            raw, self.max_speed, self.max_yaw_rate
        ).reshape(4)

        reference_min = self.bounds_min + self.reference_margin
        reference_max = self.bounds_max - self.reference_margin
        desired_yaw_rate = command_limited[3].reshape(1)
        command_world_velocity = self.last_world_velocity_command.clone()
        yaw_rate_command = self.last_yaw_rate_command.clone()
        max_contact_force = torch.zeros((), device=self.device)
        collision_any = torch.zeros((), dtype=torch.bool, device=self.device)
        out_of_bounds_any = torch.zeros((), dtype=torch.bool, device=self.device)
        finite_all = torch.ones((), dtype=torch.bool, device=self.device)
        collision_physics_substeps = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        out_of_bounds_physics_substeps = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        root_state = self._state()[..., :13].reshape(-1, 13)

        # Hold one policy action for a fixed number of physics steps. The Lee
        # controller and reference filters still update at the physics rate.
        # Safety events are latched across all substeps so a transient contact
        # cannot disappear before the policy receives the next observation.
        for physics_step in range(self.physics_steps_per_action):
            current_yaw = quaternion_yaw_wxyz(root_state[..., 3:7])
            desired_world_velocity = yaw_local_velocity_to_world(
                command_limited[:3].reshape(1, 3), current_yaw
            ).reshape(3)

            # A policy may change direction instantaneously, while a real
            # multirotor cannot. Rate-limit its reference at the physics rate.
            command_world_velocity = limit_vector_change(
                self.last_world_velocity_command,
                desired_world_velocity,
                self.max_acceleration * self.physics_dt,
            )
            current_position = root_state[0, :3]
            command_world_velocity = limit_velocity_for_braking_distance(
                current_position,
                command_world_velocity,
                reference_min,
                reference_max,
                self.max_acceleration,
            )
            self.last_world_velocity_command.copy_(command_world_velocity.detach())

            yaw_rate_delta = (
                desired_yaw_rate - self.last_yaw_rate_command
            ).clamp(
                -self.max_yaw_acceleration * self.physics_dt,
                self.max_yaw_acceleration * self.physics_dt,
            )
            yaw_rate_command = self.last_yaw_rate_command + yaw_rate_delta
            self.last_yaw_rate_command.copy_(yaw_rate_command.detach())

            self.target_position.add_(command_world_velocity * self.physics_dt)
            self.target_position.clamp_(reference_min, reference_max)
            self.target_yaw.add_(yaw_rate_command * self.physics_dt)
            self.target_yaw.copy_(
                torch.atan2(torch.sin(self.target_yaw), torch.cos(self.target_yaw))
            )

            rotor_action = self.controller(
                root_state,
                target_pos=self.target_position,
                target_vel=command_world_velocity,
                target_yaw=self.target_yaw,
            )
            self.timing.controller_updates += 1
            self.last_rotor_action = rotor_action.detach().clone()
            self.drone.apply_action(rotor_action)
            self.timing.force_applications += 1
            # GUI and headless MUST take the same one-physics-step path.
            # step(render=True) may advance a full rendering interval in Kit.
            self.sim.step(render=False)

            substep_state = self._state()
            substep_position = substep_state[..., :3].reshape(-1, 3)[0]
            substep_contact_forces = (
                self.drone.base_link.get_net_contact_forces(clone=True)
            )
            substep_max_contact = substep_contact_forces.norm(dim=-1).max()
            substep_collision = substep_max_contact > self.contact_threshold
            substep_out_of_bounds = torch.logical_or(
                substep_position < self.bounds_min,
                substep_position > self.bounds_max,
            ).any()
            substep_finite = torch.isfinite(substep_state).all() & torch.isfinite(
                substep_contact_forces
            ).all()

            max_contact_force = torch.maximum(
                max_contact_force, substep_max_contact
            )
            collision_any |= substep_collision
            out_of_bounds_any |= substep_out_of_bounds
            finite_all &= substep_finite
            collision_physics_substeps += substep_collision.to(torch.long)
            out_of_bounds_physics_substeps += substep_out_of_bounds.to(
                torch.long
            )
            root_state = substep_state[..., :13].reshape(-1, 13)

        applied_command = torch.cat(
            (command_world_velocity, yaw_rate_command), dim=0
        )
        telemetry = self.telemetry()
        telemetry["timing_step"] = self.timing.verify_action(timing_before)
        if self.render_on_step:
            self.render_frame()
        telemetry["physics_time_s"] = self.timing.physics_time_s
        telemetry["physics_step_count"] = self.timing.physics_steps
        telemetry["rotor_action"] = self.last_rotor_action.reshape(-1).cpu().tolist()
        telemetry["rotor_thrust_n"] = self.drone.thrusts[..., 2].reshape(-1).detach().cpu().tolist()
        telemetry["controller_reference_position_m"] = self.target_position.reshape(3).detach().cpu().tolist()
        telemetry["velocity_command_limited"] = bool((raw.reshape(4)[:3].norm() > self.max_speed).item())
        telemetry["yaw_rate_command_limited"] = bool((raw.reshape(4)[3].abs() > self.max_yaw_rate).item())
        telemetry["limited_command_body"] = command_limited.detach().cpu().tolist()
        telemetry["max_contact_force_n"] = float(max_contact_force.item())
        telemetry["collision"] = bool(collision_any.item())
        telemetry["out_of_bounds"] = bool(out_of_bounds_any.item())
        telemetry["finite"] = bool(finite_all.item())
        telemetry["collision_physics_substeps"] = int(
            collision_physics_substeps.item()
        )
        telemetry["out_of_bounds_physics_substeps"] = int(
            out_of_bounds_physics_substeps.item()
        )
        return telemetry, applied_command.detach().clone()

    def _progress(self, name: str, step: int, steps: int, telemetry: Mapping[str, Any]) -> None:
        interval = max(int(self.cfg["probe"]["print_interval_steps"]), 1)
        if step == 0 or step + 1 == steps or (step + 1) % interval == 0:
            print(
                f"[{name}] step={step + 1}/{steps} "
                f"pos={[round(v, 3) for v in telemetry['position_m']]} "
                f"speed={telemetry['speed_mps']:.3f} "
                f"contact={telemetry['max_contact_force_n']:.3f}",
                flush=True,
            )

    def run_hover(self, steps: int | None = None) -> Dict[str, Any]:
        steps = int(steps or self.cfg["probe"]["hover_steps"])
        self.reset()
        max_position_error = 0.0
        max_speed = 0.0
        min_up_z = 1.0
        min_altitude = math.inf
        collision_steps = 0
        collision_physics_substeps = 0
        out_of_bounds_steps = 0
        out_of_bounds_physics_substeps = 0
        finite = True
        last = self.telemetry()
        wall_start = time.perf_counter()

        for step in range(steps):
            last, _ = self.step((0.0, 0.0, 0.0, 0.0))
            position = torch.tensor(last["position_m"])
            spawn = self.spawn_position.detach().cpu().reshape(3)
            max_position_error = max(
                max_position_error, float((position - spawn).norm().item())
            )
            max_speed = max(max_speed, float(last["speed_mps"]))
            min_up_z = min(min_up_z, float(last["up_z"]))
            min_altitude = min(min_altitude, float(last["position_m"][2]))
            collision_steps += int(last["collision"])
            collision_physics_substeps += int(
                last["collision_physics_substeps"]
            )
            out_of_bounds_steps += int(last["out_of_bounds"])
            out_of_bounds_physics_substeps += int(
                last["out_of_bounds_physics_substeps"]
            )
            finite = finite and bool(last["finite"])
            self._progress("hover", step, steps, last)
        wall_time = max(time.perf_counter() - wall_start, 1e-9)

        final_position = torch.tensor(last["position_m"])
        final_error = float(
            (final_position - self.spawn_position.detach().cpu().reshape(3)).norm().item()
        )
        acceptance = self.cfg["acceptance"]
        passed = (
            finite
            and collision_steps == 0
            and out_of_bounds_steps == 0
            and max_position_error <= float(acceptance["hover_max_position_error_m"])
            and final_error <= float(acceptance["hover_final_position_error_m"])
            and min_up_z >= float(acceptance["hover_min_up_z"])
        )
        return {
            "passed": passed,
            "steps": steps,
            "duration_s": steps * self.control_dt,
            "wall_time_s": wall_time,
            "control_steps_per_second": steps / wall_time,
            "physics_steps_per_second": (
                steps * self.physics_steps_per_action / wall_time
            ),
            "sim_steps_per_second": (
                steps * self.physics_steps_per_action / wall_time
            ),
            "real_time_factor": (steps * self.control_dt) / wall_time,
            "max_position_error_m": max_position_error,
            "final_position_error_m": final_error,
            "max_speed_mps": max_speed,
            "min_altitude_m": min_altitude,
            "min_up_z": min_up_z,
            "collision_steps": collision_steps,
            "collision_physics_substeps": collision_physics_substeps,
            "out_of_bounds_steps": out_of_bounds_steps,
            "out_of_bounds_physics_substeps": out_of_bounds_physics_substeps,
            "finite": finite,
            "final_telemetry": last,
        }

    def run_reset(self, trials: int | None = None) -> Dict[str, Any]:
        trials = int(trials or self.cfg["probe"]["reset_trials"])
        max_position_error = 0.0
        max_speed = 0.0
        all_finite = True
        for trial in range(trials):
            offset = torch.tensor(
                [1.0 + 0.2 * trial, -0.8, 0.4], device=self.device
            )
            self.set_test_state(
                self.spawn_position.reshape(3) + offset,
                torch.tensor([0.8, -0.4, 0.2], device=self.device),
            )
            telemetry = self.reset()
            error = torch.tensor(telemetry["position_m"]) - self.spawn_position.detach().cpu().reshape(3)
            max_position_error = max(max_position_error, float(error.norm().item()))
            max_speed = max(max_speed, float(telemetry["speed_mps"]))
            all_finite = all_finite and bool(telemetry["finite"])

        acceptance = self.cfg["acceptance"]
        passed = (
            all_finite
            and max_position_error <= float(acceptance["reset_max_position_error_m"])
            and max_speed <= float(acceptance["reset_max_speed_mps"])
        )
        return {
            "passed": passed,
            "trials": trials,
            "max_position_error_m": max_position_error,
            "max_speed_mps": max_speed,
            "finite": all_finite,
        }

    def run_random(self, steps: int | None = None) -> Dict[str, Any]:
        steps = int(steps or self.cfg["probe"]["random_steps"])
        self.reset()
        warmup_steps = int(self.cfg["probe"]["warmup_steps"])
        for _ in range(warmup_steps):
            self.step((0.0, 0.0, 0.0, 0.0))

        interval = int(self.cfg["control"]["random_command_interval_steps"])
        overspeed = float(self.cfg["control"]["random_command_overspeed_factor"])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(self.cfg["probe"]["seed"]))
        raw = torch.zeros(4, device=self.device)
        limited = raw.clone()
        requested_max_speed = 0.0
        limited_max_speed = 0.0
        requested_max_yaw_rate = 0.0
        limited_max_yaw_rate = 0.0
        actual_max_speed = 0.0
        actual_overspeed_steps = 0
        squared_tracking_error = 0.0
        collision_steps = 0
        collision_physics_substeps = 0
        out_of_bounds_steps = 0
        out_of_bounds_physics_substeps = 0
        finite = True
        last = self.telemetry()
        wall_start = time.perf_counter()

        for step in range(steps):
            if step % interval == 0:
                raw = torch.rand(4, generator=generator, device=self.device) * 2.0 - 1.0
                raw[:3] *= self.max_speed * overspeed
                raw[3] *= self.max_yaw_rate * overspeed
            last, limited = self.step(raw)
            actual_velocity = torch.tensor(
                last["linear_velocity_mps"], device=self.device
            )
            requested_max_speed = max(requested_max_speed, float(raw[:3].norm().item()))
            limited_max_speed = max(limited_max_speed, float(limited[:3].norm().item()))
            requested_max_yaw_rate = max(requested_max_yaw_rate, float(raw[3].abs().item()))
            limited_max_yaw_rate = max(limited_max_yaw_rate, float(limited[3].abs().item()))
            actual_max_speed = max(actual_max_speed, float(actual_velocity.norm().item()))
            actual_overspeed_steps += int(last["speed_mps"] > self.max_speed)
            squared_tracking_error += float(
                (actual_velocity - self.last_world_velocity_command).square().sum().item()
            )
            collision_steps += int(last["collision"])
            collision_physics_substeps += int(
                last["collision_physics_substeps"]
            )
            out_of_bounds_steps += int(last["out_of_bounds"])
            out_of_bounds_physics_substeps += int(
                last["out_of_bounds_physics_substeps"]
            )
            finite = finite and bool(last["finite"])
            self._progress("random", step, steps, last)
        wall_time = max(time.perf_counter() - wall_start, 1e-9)

        velocity_rmse = math.sqrt(squared_tracking_error / max(steps, 1))
        acceptance = self.cfg["acceptance"]
        limiter_ok = (
            limited_max_speed <= self.max_speed + 1e-5
            and limited_max_yaw_rate <= self.max_yaw_rate + 1e-5
        )
        passed = (
            finite
            and limiter_ok
            and out_of_bounds_steps == 0
            and velocity_rmse <= float(acceptance["random_max_velocity_rmse_mps"])
            and actual_max_speed <= float(acceptance["random_max_actual_speed_mps"])
        )
        return {
            "passed": passed,
            "steps": steps,
            "wall_time_s": wall_time,
            "control_steps_per_second": steps / wall_time,
            "physics_steps_per_second": (
                steps * self.physics_steps_per_action / wall_time
            ),
            "sim_steps_per_second": (
                steps * self.physics_steps_per_action / wall_time
            ),
            "real_time_factor": (steps * self.control_dt) / wall_time,
            "warmup_steps": warmup_steps,
            "requested_max_speed_mps": requested_max_speed,
            "limited_max_speed_mps": limited_max_speed,
            "requested_max_yaw_rate_rps": requested_max_yaw_rate,
            "limited_max_yaw_rate_rps": limited_max_yaw_rate,
            "actual_max_speed_mps": actual_max_speed,
            "actual_speed_limit_mps": self.max_speed,
            "actual_overspeed_steps": actual_overspeed_steps,
            "actual_overspeed_fraction": actual_overspeed_steps / max(steps, 1),
            "velocity_tracking_rmse_mps": velocity_rmse,
            "collision_steps": collision_steps,
            "collision_physics_substeps": collision_physics_substeps,
            "out_of_bounds_steps": out_of_bounds_steps,
            "out_of_bounds_physics_substeps": out_of_bounds_physics_substeps,
            "finite": finite,
            "limiter_ok": limiter_ok,
            "final_telemetry": last,
        }

    def run_contact(self) -> Dict[str, Any]:
        if not self.obstacles:
            return {"passed": False, "error": "contact probe requires one obstacle"}
        self.reset()
        contact_obstacle = self.obstacles[0]
        obstacle_position = torch.tensor(
            contact_obstacle["position_m"], device=self.device
        )
        self.set_test_state(obstacle_position)

        steps = int(self.cfg["collision"]["intentional_contact_steps"])
        max_force = 0.0
        collision_steps = 0
        collision_physics_substeps = 0
        finite = True
        for _ in range(steps):
            telemetry, _ = self.step((0.0, 0.0, 0.0, 0.0))
            max_force = max(max_force, float(telemetry["max_contact_force_n"]))
            collision_steps += int(telemetry["collision"])
            collision_physics_substeps += int(
                telemetry["collision_physics_substeps"]
            )
            finite = finite and bool(telemetry["finite"])

        self.reset()
        clear_steps = int(self.cfg["collision"]["clear_contact_steps"])
        final_clear_force = math.inf
        for _ in range(clear_steps):
            telemetry, _ = self.step((0.0, 0.0, 0.0, 0.0))
            final_clear_force = float(telemetry["max_contact_force_n"])
            finite = finite and bool(telemetry["finite"])

        detected = max_force > self.contact_threshold and collision_steps > 0
        cleared = final_clear_force <= self.contact_threshold
        return {
            "passed": finite and detected and cleared,
            "obstacle": str(contact_obstacle["name"]),
            "intentional_contact_steps": steps,
            "collision_steps": collision_steps,
            "collision_physics_substeps": collision_physics_substeps,
            "max_contact_force_n": max_force,
            "contact_threshold_n": self.contact_threshold,
            "final_clear_force_n": final_clear_force,
            "detected": detected,
            "cleared_after_reset": cleared,
            "finite": finite,
        }

    def _obstacle_by_name(self, name: str) -> Mapping[str, Any]:
        matches = [item for item in self.obstacles if str(item["name"]) == name]
        if len(matches) != 1:
            raise ValueError(f"camera probe obstacle {name!r} was not found exactly once")
        return matches[0]

    def _expected_camera_wall_depth(
        self,
        obstacle: Mapping[str, Any],
        axis: int,
        body_position_axis_m: float | None = None,
    ) -> float:
        camera_cfg = self.cfg["camera"]
        obstacle_near_face = float(obstacle["position_m"][axis]) - 0.5 * float(
            obstacle["size_m"][axis]
        )
        camera_forward_offset = float(camera_cfg["mount_position_body_m"][0])
        if body_position_axis_m is None:
            body_position_axis_m = float(
                self.spawn_position.reshape(3)[axis].item()
            )
        return obstacle_near_face - body_position_axis_m - camera_forward_offset

    def run_camera(self) -> Dict[str, Any]:
        """Validate metric depth, 20 m clipping and fixed-body yaw behavior."""
        camera_cfg = self.cfg["camera"]
        probe_cfg = camera_cfg["probe"]
        acceptance = self.cfg["acceptance"]
        self.reset()

        front_obstacle = self._obstacle_by_name(str(probe_cfg["front_obstacle"]))
        front_capture_start = time.perf_counter()
        front_depth = self.depth_camera.capture()
        front_capture_wall_time = time.perf_counter() - front_capture_start
        front_plane = summarize_depth(
            front_depth["distance_to_image_plane"],
            float(camera_cfg["min_depth_m"]),
            float(camera_cfg["max_depth_m"]),
            int(probe_cfg["center_patch_px"]),
        )
        front_radial = summarize_depth(
            front_depth["distance_to_camera"],
            float(camera_cfg["min_depth_m"]),
            float(camera_cfg["max_depth_m"]),
            int(probe_cfg["center_patch_px"]),
        )
        front_expected = self._expected_camera_wall_depth(front_obstacle, axis=0)

        yaw_rad = float(probe_cfg["yaw_test_rad"])
        self.set_test_pose(self.spawn_position.reshape(3), yaw_rad)
        # Tensor-API pose writes are immediately visible to PhysX queries, but
        # Isaac Sim 2023.1 with flatcache/Fabric can keep the render camera on
        # its previous transform until physics advances. A zero-command
        # control step preserves the requested hover pose while publishing the
        # new rigid-body transform to the renderer. Merely rendering more
        # frames does not fix this stale-transform failure mode.
        pose_sync_control_steps = int(probe_cfg.get("pose_sync_control_steps", 1))
        pose_sync_start = time.perf_counter()
        synchronized_telemetry = self.synchronize_pose_to_renderer(
            pose_sync_control_steps
        )
        pose_sync_wall_time = time.perf_counter() - pose_sync_start
        actual_yaw_rad = float(synchronized_telemetry["yaw_rad"])
        pose_sync_yaw_error = abs(
            math.atan2(
                math.sin(actual_yaw_rad - yaw_rad),
                math.cos(actual_yaw_rad - yaw_rad),
            )
        )
        yaw_obstacle = self._obstacle_by_name(str(probe_cfg["yaw_obstacle"]))
        yaw_capture_start = time.perf_counter()
        yaw_depth = self.depth_camera.capture()
        yaw_capture_wall_time = time.perf_counter() - yaw_capture_start
        yaw_plane = summarize_depth(
            yaw_depth["distance_to_image_plane"],
            float(camera_cfg["min_depth_m"]),
            float(camera_cfg["max_depth_m"]),
            int(probe_cfg["center_patch_px"]),
        )
        yaw_expected = self._expected_camera_wall_depth(
            yaw_obstacle,
            axis=1,
            body_position_axis_m=float(synchronized_telemetry["position_m"][1]),
        )

        normalized, valid = depth_to_normalized_inverse(
            front_depth["distance_to_image_plane"],
            float(camera_cfg["min_depth_m"]),
            float(camera_cfg["max_depth_m"]),
        )
        actor_width, actor_height = (int(v) for v in camera_cfg["actor_resize"])
        actor_depth = resize_inverse_depth_for_actor(
            normalized,
            output_width=actor_width,
            output_height=actor_height,
        )

        tolerance = float(acceptance["camera_center_depth_abs_error_m"])
        pose_sync_yaw_tolerance = float(
            acceptance["camera_pose_sync_yaw_abs_error_rad"]
        )
        minimum_valid = float(acceptance["camera_min_range_valid_fraction"])
        front_median = front_plane["center_median_m"]
        yaw_median = yaw_plane["center_median_m"]
        front_error_value = (
            abs(float(front_median) - front_expected)
            if front_median is not None
            else None
        )
        yaw_error_value = (
            abs(float(yaw_median) - yaw_expected)
            if yaw_median is not None
            else None
        )
        front_error = front_error_value if front_error_value is not None else math.inf
        yaw_error = yaw_error_value if yaw_error_value is not None else math.inf
        correct_shape = front_plane["shape"] == [
            int(camera_cfg["height"]),
            int(camera_cfg["width"]),
        ]
        depth_within_clipping_range = all(
            summary["valid_max_m"] is None
            or float(summary["valid_max_m"])
            <= float(camera_cfg["max_depth_m"]) + 1e-3
            for summary in (front_plane, front_radial, yaw_plane)
        )
        passed = (
            correct_shape
            and front_plane["finite"]
            and front_radial["finite"]
            and yaw_plane["finite"]
            and float(front_plane["range_valid_fraction"]) >= minimum_valid
            and float(yaw_plane["range_valid_fraction"]) >= minimum_valid
            and front_error <= tolerance
            and yaw_error <= tolerance
            and bool(synchronized_telemetry["finite"])
            and not bool(synchronized_telemetry["collision"])
            and not bool(synchronized_telemetry["out_of_bounds"])
            and pose_sync_yaw_error <= pose_sync_yaw_tolerance
            and depth_within_clipping_range
            and tuple(actor_depth.shape) == (actor_height, actor_width)
            and bool(torch.isfinite(actor_depth).all().item())
            and float(actor_depth.min().item()) >= 0.0
            and float(actor_depth.max().item()) <= 1.0
        )
        self.reset()
        render_frames_per_capture = int(camera_cfg["warmup_render_frames"])
        total_capture_wall_time = front_capture_wall_time + yaw_capture_wall_time
        return {
            "passed": passed,
            "camera_prim_path": self.depth_camera.prim_path,
            "fixed_to_body": bool(camera_cfg["fixed_to_body"]),
            "resolution": [int(camera_cfg["width"]), int(camera_cfg["height"])],
            "actor_depth_shape": list(actor_depth.shape),
            "metric_depth_valid_pixels": int(valid.sum().item()),
            "inverse_depth_min": float(actor_depth.min().item()),
            "inverse_depth_max": float(actor_depth.max().item()),
            "depth_within_clipping_range": depth_within_clipping_range,
            "render_frames_per_capture": render_frames_per_capture,
            "capture_wall_time_s": total_capture_wall_time,
            "render_frames_per_second": (
                2 * render_frames_per_capture / max(total_capture_wall_time, 1e-9)
            ),
            "front": {
                "obstacle": str(front_obstacle["name"]),
                "expected_center_depth_m": front_expected,
                "center_depth_abs_error_m": front_error_value,
                "capture_wall_time_s": front_capture_wall_time,
                "distance_to_image_plane": front_plane,
                "distance_to_camera": front_radial,
            },
            "yaw_follow": {
                "yaw_test_rad": yaw_rad,
                "actual_body_yaw_rad": actual_yaw_rad,
                "body_yaw_abs_error_rad": pose_sync_yaw_error,
                "body_yaw_tolerance_rad": pose_sync_yaw_tolerance,
                "pose_sync_control_steps": pose_sync_control_steps,
                "pose_sync_wall_time_s": pose_sync_wall_time,
                "synchronized_position_m": synchronized_telemetry["position_m"],
                "synchronized_speed_mps": synchronized_telemetry["speed_mps"],
                "synchronized_finite": synchronized_telemetry["finite"],
                "synchronized_collision": synchronized_telemetry["collision"],
                "synchronized_out_of_bounds": synchronized_telemetry[
                    "out_of_bounds"
                ],
                "obstacle": str(yaw_obstacle["name"]),
                "expected_center_depth_m": yaw_expected,
                "center_depth_abs_error_m": yaw_error_value,
                "capture_wall_time_s": yaw_capture_wall_time,
                "distance_to_image_plane": yaw_plane,
            },
        }

    def run(self, probe: str, steps: int | None = None) -> Dict[str, Any]:
        if probe == "hover":
            probes = {"hover": self.run_hover(steps)}
        elif probe == "reset":
            probes = {"reset": self.run_reset()}
        elif probe == "random":
            probes = {"random": self.run_random(steps)}
        elif probe == "contact":
            probes = {"contact": self.run_contact()}
        elif probe == "camera":
            probes = {"camera": self.run_camera()}
        elif probe == "all":
            probes = {
                "hover": self.run_hover(steps),
                "reset": self.run_reset(),
                "random": self.run_random(steps),
                "contact": self.run_contact(),
                "camera": self.run_camera(),
            }
        else:
            raise ValueError(f"unknown probe: {probe}")
        return {
            "schema_version": 1,
            "passed": all(bool(result.get("passed", False)) for result in probes.values()),
            "probe": probe,
            "device": str(self.device),
            "isaac_version": self.isaac_version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(self.device) if torch.cuda.is_available() else None,
            "physics_dt_s": self.physics_dt,
            "physics_hz": 1.0 / self.physics_dt,
            "control_dt_s": self.control_dt,
            "control_hz": self.control_hz,
            "physics_steps_per_action": self.physics_steps_per_action,
            "physics_timing": self.timing.report(),
            "probes": probes,
        }

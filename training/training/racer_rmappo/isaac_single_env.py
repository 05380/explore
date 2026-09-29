"""First-stage single-drone Isaac Sim physics probe.

This module deliberately does not import Omniverse modules at import time.
The caller must create ``SimulationApp`` first, then construct
``IsaacSingleDroneProbe``.  It is not yet the PPO backend: depth sensing,
mapping, RACER candidates and rewards are introduced in later P1/P3 stages.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor


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
        "probe",
        "acceptance",
    }
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Isaac probe configuration is missing: {sorted(missing)}")

    sim = cfg["sim"]
    if float(sim["physics_dt"]) <= 0.0:
        raise ValueError("sim.physics_dt must be positive")
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
    if float(control["max_speed_mps"]) <= 0.0:
        raise ValueError("control.max_speed_mps must be positive")
    if float(control["max_yaw_rate_rps"]) <= 0.0:
        raise ValueError("control.max_yaw_rate_rps must be positive")
    if int(control["random_command_interval_steps"]) < 1:
        raise ValueError("random_command_interval_steps must be at least one")


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

        # These imports must remain after SimulationApp is constructed.
        import omni.isaac.orbit.sim as sim_utils
        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.version import get_version
        from omni_drones.controllers import LeePositionController
        from omni_drones.robots.drone import MultirotorBase

        sim_cfg = cfg["sim"]
        self.isaac_version = ".".join(str(item) for item in get_version())
        self.dt = float(sim_cfg["physics_dt"])
        self.device = torch.device(str(sim_cfg["device"]))
        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self.dt,
            rendering_dt=float(sim_cfg.get("rendering_dt", self.dt)),
            sim_params=dict(sim_cfg.get("params", {})),
            backend="torch",
            device=str(self.device),
        )

        scene_cfg = cfg["scene"]
        # A local static cuboid avoids a runtime dependency on the remote
        # Nucleus ground-plane USD during headless diagnostics.
        ground_cfg = sim_utils.CuboidCfg(
            size=(
                float(scene_cfg["size_m"][0]),
                float(scene_cfg["size_m"][1]),
                0.10,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.02,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(
                    float(item) for item in scene_cfg["ground_color"]
                )
            ),
        )
        ground_cfg.func(
            "/World/defaultGroundPlane",
            ground_cfg,
            translation=(0.0, 0.0, -0.05),
        )

        light_cfg = sim_utils.DistantLightCfg(
            intensity=3000.0, color=(0.75, 0.75, 0.75)
        )
        light_cfg.func("/World/P1ProbeLight", light_cfg)

        self.obstacles = list(scene_cfg.get("obstacles", []))
        for obstacle in self.obstacles:
            obstacle_cfg = sim_utils.CuboidCfg(
                size=tuple(float(item) for item in obstacle["size_m"]),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.02,
                    rest_offset=0.0,
                ),
            )
            obstacle_cfg.func(
                f"/World/P1Obstacles/{obstacle['name']}",
                obstacle_cfg,
                translation=tuple(float(item) for item in obstacle["position_m"]),
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
        self.drone.spawn(translations=self.spawn_position.reshape(-1, 3))

        self.sim.reset()
        self.drone.initialize(track_contact_forces=True)
        self.controller = LeePositionController(
            g=9.81, uav_params=self.drone.params
        ).to(self.device)
        self.env_ids = torch.tensor([0], dtype=torch.long, device=self.device)
        self.zero_velocities = torch.zeros(1, 1, 6, device=self.device)

        self.bounds_min = torch.tensor(
            scene_cfg["flight_bounds_min_m"], dtype=torch.float32, device=self.device
        )
        self.bounds_max = torch.tensor(
            scene_cfg["flight_bounds_max_m"], dtype=torch.float32, device=self.device
        )
        control = cfg["control"]
        self.max_speed = float(control["max_speed_mps"])
        self.max_yaw_rate = float(control["max_yaw_rate_rps"])
        self.reference_margin = float(control["reference_boundary_margin_m"])
        self.contact_threshold = float(
            cfg["collision"]["contact_force_threshold_n"]
        )
        self.target_position = self.spawn_position.reshape(1, 3).clone()
        self.target_yaw = torch.zeros(1, device=self.device)
        self.last_world_velocity_command = torch.zeros(3, device=self.device)
        self.reset()

    def _flush(self) -> None:
        physics_view = getattr(self.sim, "_physics_sim_view", None)
        if physics_view is not None:
            physics_view.flush()

    def reset(self) -> Dict[str, Any]:
        self.drone._reset_idx(self.env_ids, train=False)
        self.drone.set_world_poses(self.spawn_position, self.spawn_orientation)
        self.drone.set_velocities(self.zero_velocities)
        self.target_position.copy_(self.spawn_position.reshape(1, 3))
        self.target_yaw.zero_()
        self.last_world_velocity_command.zero_()
        self._flush()
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
        raw = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        command_limited = limit_velocity_command(
            raw, self.max_speed, self.max_yaw_rate
        ).reshape(4)

        root_state = self._state()[..., :13].reshape(-1, 13)
        current_yaw = quaternion_yaw_wxyz(root_state[..., 3:7])
        command_world_velocity = yaw_local_velocity_to_world(
            command_limited[:3].reshape(1, 3), current_yaw
        )
        self.last_world_velocity_command = command_world_velocity.reshape(3).detach().clone()

        reference_min = self.bounds_min + self.reference_margin
        reference_max = self.bounds_max - self.reference_margin
        self.target_position.add_(command_world_velocity * self.dt)
        self.target_position.clamp_(reference_min, reference_max)
        self.target_yaw.add_(command_limited[3] * self.dt)
        self.target_yaw.copy_(
            torch.atan2(torch.sin(self.target_yaw), torch.cos(self.target_yaw))
        )

        rotor_action = self.controller(
            root_state,
            target_pos=self.target_position,
            target_vel=command_world_velocity,
            target_yaw=self.target_yaw,
        )
        self.drone.apply_action(rotor_action)
        self.sim.step(render=self.render)
        return self.telemetry(), command_limited.detach().clone()

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
        out_of_bounds_steps = 0
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
            out_of_bounds_steps += int(last["out_of_bounds"])
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
            "duration_s": steps * self.dt,
            "wall_time_s": wall_time,
            "sim_steps_per_second": steps / wall_time,
            "real_time_factor": (steps * self.dt) / wall_time,
            "max_position_error_m": max_position_error,
            "final_position_error_m": final_error,
            "max_speed_mps": max_speed,
            "min_altitude_m": min_altitude,
            "min_up_z": min_up_z,
            "collision_steps": collision_steps,
            "out_of_bounds_steps": out_of_bounds_steps,
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
        squared_tracking_error = 0.0
        collision_steps = 0
        out_of_bounds_steps = 0
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
            squared_tracking_error += float(
                (actual_velocity - self.last_world_velocity_command).square().sum().item()
            )
            collision_steps += int(last["collision"])
            out_of_bounds_steps += int(last["out_of_bounds"])
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
            "sim_steps_per_second": steps / wall_time,
            "real_time_factor": (steps * self.dt) / wall_time,
            "warmup_steps": warmup_steps,
            "requested_max_speed_mps": requested_max_speed,
            "limited_max_speed_mps": limited_max_speed,
            "requested_max_yaw_rate_rps": requested_max_yaw_rate,
            "limited_max_yaw_rate_rps": limited_max_yaw_rate,
            "actual_max_speed_mps": actual_max_speed,
            "velocity_tracking_rmse_mps": velocity_rmse,
            "collision_steps": collision_steps,
            "out_of_bounds_steps": out_of_bounds_steps,
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
        finite = True
        for _ in range(steps):
            telemetry, _ = self.step((0.0, 0.0, 0.0, 0.0))
            max_force = max(max_force, float(telemetry["max_contact_force_n"]))
            collision_steps += int(telemetry["collision"])
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
            "max_contact_force_n": max_force,
            "contact_threshold_n": self.contact_threshold,
            "final_clear_force_n": final_clear_force,
            "detected": detected,
            "cleared_after_reset": cleared,
            "finite": finite,
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
        elif probe == "all":
            probes = {
                "hover": self.run_hover(steps),
                "reset": self.run_reset(),
                "random": self.run_random(steps),
                "contact": self.run_contact(),
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
            "physics_dt_s": self.dt,
            "probes": probes,
        }

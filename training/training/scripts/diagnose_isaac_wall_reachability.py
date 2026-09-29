#!/usr/bin/env python3
"""Prove that the P3 wall-behind-goal scene is physically traversable."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = SCRIPT_DIR.parent
TRAINING_ROOT = TRAINING_PACKAGE.parent
OMNIDRONES_SOURCE = TRAINING_ROOT / "third_party" / "OmniDrones"
for source_root in (TRAINING_PACKAGE, OMNIDRONES_SOURCE):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

DEFAULT_ISAAC_CONFIG = TRAINING_ROOT / "configs" / "isaac_single.yaml"
DEFAULT_OUTPUT = (
    TRAINING_ROOT
    / "runs"
    / "isaac_single_wall"
    / "reachability_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-config", type=Path, default=DEFAULT_ISAAC_CONFIG)
    parser.add_argument("--rmappo-config", type=Path, default=None)
    parser.add_argument("--scenario", default="wall_avoidance")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    isaac_config_path = args.isaac_config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    base_isaac_cfg = yaml.safe_load(
        isaac_config_path.read_text(encoding="utf-8")
    )
    if not isinstance(base_isaac_cfg, dict):
        raise ValueError("Isaac configuration root must be a mapping")

    # Do not import the racer_rmappo package before SimulationApp: its public
    # package imports PyTorch, while Isaac owns the locked runtime stack.
    app_cfg = base_isaac_cfg["app"]
    headless = (
        bool(app_cfg["headless"]) if args.headless is None else bool(args.headless)
    )
    simulation_app = None
    backend = None
    probe = None
    observation = None
    critic_state = None
    action = None
    reward = None
    done = None
    info = None
    final_info = None
    backend_close_failed = False
    wall_start = time.perf_counter()
    try:
        from omni.isaac.kit import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": headless,
                "multi_gpu": bool(app_cfg.get("multi_gpu", False)),
                "anti_aliasing": int(app_cfg.get("anti_aliasing", 0)),
                "fast_shutdown": bool(app_cfg.get("fast_shutdown", False)),
            }
        )

        import torch

        from racer_rmappo.config import apply_curriculum_stage, load_config
        from racer_rmappo.isaac_adapter import (
            validate_backend_shapes,
            validate_step_info,
        )
        from racer_rmappo.isaac_single_backend import IsaacSingleNavigationBackend
        from racer_rmappo.isaac_single_env import (
            IsaacSingleDroneProbe,
            validate_probe_config,
        )
        from racer_rmappo.isaac_scenarios import apply_navigation_scenario
        from racer_rmappo.rule_navigation import waypoint_navigation_action

        isaac_cfg = apply_navigation_scenario(base_isaac_cfg, args.scenario)
        scenario_cfg = isaac_cfg["navigation_backend"]["scenarios"][
            args.scenario
        ]
        validate_probe_config(isaac_cfg)
        rmappo_cfg = apply_curriculum_stage(
            load_config(args.rmappo_config), "single_agent_wall_avoidance"
        )
        rmappo_cfg["training"]["backend"] = "isaac"
        rmappo_cfg["training"]["num_parallel_swarms"] = 1
        rmappo_cfg["training"]["device"] = str(isaac_cfg["sim"]["device"])

        waypoints = [
            [float(value) for value in waypoint]
            for waypoint in scenario_cfg["validation_waypoints_m"]
        ]
        target = [
            float(value)
            for value in isaac_cfg["navigation_backend"][
                "fixed_target_position_m"
            ]
        ]
        print(
            "P3_WALL_REACHABILITY_START "
            f"scenario={args.scenario} headless={headless} "
            f"waypoints={waypoints} target={target}",
            flush=True,
        )

        probe = IsaacSingleDroneProbe(isaac_cfg, render=not headless)
        backend = IsaacSingleNavigationBackend(rmappo_cfg, isaac_cfg, probe)
        observation, critic_state = validate_backend_shapes(backend)
        previous_position = torch.tensor(
            backend.last_telemetry["position_m"], dtype=torch.float32
        )
        minimum_clearance = float(backend.last_clearance_m.item())
        path_length = 0.0
        reward_sum = 0.0
        waypoint_index = 0
        finite = True
        max_steps = int(scenario_cfg["validation_max_steps"])
        tolerance = float(scenario_cfg["waypoint_position_tolerance_m"])
        episode_steps = 0

        for step in range(max_steps):
            position = torch.tensor(
                backend.last_telemetry["position_m"], dtype=torch.float32
            )
            while waypoint_index < len(waypoints):
                waypoint = torch.tensor(waypoints[waypoint_index])
                if float((position - waypoint).norm()) > tolerance:
                    break
                waypoint_index += 1
                print(
                    f"[wall] waypoint_reached={waypoint_index}/{len(waypoints)} "
                    f"position={[round(float(v), 3) for v in position]}",
                    flush=True,
                )

            final_leg = waypoint_index >= len(waypoints)
            destination = target if final_leg else waypoints[waypoint_index]
            action = waypoint_navigation_action(
                backend.last_telemetry,
                destination,
                backend.target_cfg,
                backend.action_limits,
                position_gain=float(scenario_cfg["validation_position_gain"]),
                yaw_gain=float(scenario_cfg["validation_yaw_gain"]),
                max_cruise_speed_mps=float(
                    scenario_cfg["validation_cruise_speed_mps"]
                ),
                final_yaw_rad=(
                    float(isaac_cfg["navigation_backend"]["fixed_target_yaw_rad"])
                    if final_leg
                    else None
                ),
                device=backend.device,
            )
            observation, critic_state, reward, done, info = backend.step(action)
            validate_step_info(info, backend.num_envs)
            current_position = info["position_m"].detach().cpu()
            path_length += float((current_position - previous_position).norm())
            previous_position = current_position
            reward_sum += float(reward.item())
            minimum_clearance = min(
                minimum_clearance, float(info["obstacle_clearance_m"].item())
            )
            finite &= bool(torch.isfinite(reward).all().item())
            finite &= bool(torch.isfinite(critic_state).all().item())
            finite &= all(
                bool(torch.isfinite(value).all().item())
                for value in observation.values()
            )
            episode_steps = step + 1
            if (step + 1) % 25 == 0 or bool(done.item()):
                print(
                    f"[wall] step={step + 1}/{max_steps} "
                    f"waypoint={waypoint_index}/{len(waypoints)} "
                    f"target_distance={float(info['target_distance_m'].item()):.3f} "
                    f"clearance={float(info['obstacle_clearance_m'].item()):.3f}",
                    flush=True,
                )
            if bool(done.item()):
                final_info = info
                break

        success = bool(
            final_info is not None
            and float(final_info["episode_success"].item()) == 1.0
        )
        collision = bool(
            final_info is not None
            and float(final_info["episode_collision"].item()) > 0.0
        )
        out_of_bounds = bool(
            final_info is not None
            and float(final_info["episode_out_of_bounds"].item()) > 0.0
        )
        timeout = bool(
            final_info is None
            or float(final_info["episode_timeout"].item()) > 0.0
        )
        stall = bool(
            final_info is not None
            and float(final_info["episode_stall"].item()) > 0.0
        )
        wall_observed = minimum_clearance <= float(
            scenario_cfg["validation_wall_visible_clearance_m"]
        )
        passed = bool(
            success
            and not collision
            and not out_of_bounds
            and not timeout
            and not stall
            and waypoint_index == len(waypoints)
            and wall_observed
            and finite
        )
        result = {
            "schema_version": 1,
            "passed": passed,
            "scenario": args.scenario,
            "headless": headless,
            "success": success,
            "collision": collision,
            "out_of_bounds": out_of_bounds,
            "timeout": timeout,
            "stall": stall,
            "finite": finite,
            "steps": episode_steps,
            "waypoint_count": len(waypoints),
            "waypoints_reached": waypoint_index,
            "target_position_m": target,
            "path_length_m": path_length,
            "minimum_camera_clearance_m": minimum_clearance,
            "wall_observed": wall_observed,
            "reward_sum": reward_sum,
            "wall_time_s": max(time.perf_counter() - wall_start, 1e-9),
            "isaac_config_path": str(isaac_config_path),
            "rmappo_config_path": rmappo_cfg.get("_config_path"),
            "direct_gpu_log_check_required": True,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(f"P3_WALL_REACHABILITY_REPORT={output_path}", flush=True)
        print(
            "P3_WALL_REACHABILITY_RESULT=" + ("PASS" if passed else "FAIL"),
            flush=True,
        )
        if not passed:
            raise SystemExit(2)
    except SystemExit:
        raise
    except BaseException as error:
        failure = {
            "passed": False,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "scenario": args.scenario,
            "isaac_config_path": str(isaac_config_path),
            "headless": headless,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("P3_WALL_REACHABILITY_EXCEPTION", flush=True)
        print(failure["traceback"], file=sys.stderr, end="", flush=True)
        print(f"P3_WALL_REACHABILITY_REPORT={output_path}", flush=True)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
                print("P3_WALL_REACHABILITY_ENV_CLOSED", flush=True)
            except BaseException:
                backend_close_failed = True
                print("P3_WALL_REACHABILITY_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            backend = None
            probe = None
        elif probe is not None:
            try:
                probe.close()
            except BaseException:
                backend_close_failed = True
                print("P3_WALL_REACHABILITY_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            probe = None
        observation = None
        critic_state = None
        action = None
        reward = None
        done = None
        info = None
        final_info = None
        gc.collect()
        if simulation_app is not None:
            from racer_rmappo.isaac_runtime import (
                active_exception_exit_code,
                close_simulation_app_safely,
            )

            shutdown_exit_code = max(
                active_exception_exit_code(), int(backend_close_failed)
            )
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except (NameError, RuntimeError):
                pass
            close_simulation_app_safely(
                simulation_app,
                hard_exit_after_shutdown=bool(
                    app_cfg.get("hard_exit_after_shutdown", False)
                ),
                process_exit_code=shutdown_exit_code,
            )


if __name__ == "__main__":
    main()

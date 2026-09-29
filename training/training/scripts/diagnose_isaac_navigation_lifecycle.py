#!/usr/bin/env python3
"""Validate repeated goal completion, fresh camera reset and episode lifecycle."""

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
    / "isaac_single_backend"
    / "lifecycle_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-config", type=Path, default=DEFAULT_ISAAC_CONFIG)
    parser.add_argument("--rmappo-config", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps-per-episode", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def wrapped_abs_error(actual: float, expected: float) -> float:
    return abs(math.atan2(math.sin(actual - expected), math.cos(actual - expected)))


def main() -> None:
    args = parse_args()
    isaac_config_path = args.isaac_config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    isaac_cfg = yaml.safe_load(isaac_config_path.read_text(encoding="utf-8"))
    if not isinstance(isaac_cfg, dict):
        raise ValueError("Isaac configuration root must be a mapping")
    lifecycle_cfg = isaac_cfg["navigation_backend"]["lifecycle_probe"]
    episodes = int(
        lifecycle_cfg["episodes"] if args.episodes is None else args.episodes
    )
    max_steps = int(
        lifecycle_cfg["max_steps_per_episode"]
        if args.max_steps_per_episode is None
        else args.max_steps_per_episode
    )
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max steps must be positive")
    app_cfg = isaac_cfg["app"]
    headless = (
        bool(app_cfg["headless"]) if args.headless is None else bool(args.headless)
    )

    simulation_app = None
    backend = None
    probe = None
    wall_start = time.perf_counter()
    try:
        from omni.isaac.kit import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": headless,
                "multi_gpu": bool(app_cfg.get("multi_gpu", False)),
                "anti_aliasing": int(app_cfg.get("anti_aliasing", 0)),
                "fast_shutdown": bool(app_cfg.get("fast_shutdown", True)),
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
        from racer_rmappo.rule_navigation import proportional_navigation_action

        validate_probe_config(isaac_cfg)
        rmappo_cfg = apply_curriculum_stage(
            load_config(args.rmappo_config), "single_agent_sparse_static"
        )
        rmappo_cfg["training"]["num_parallel_swarms"] = 1
        rmappo_cfg["training"]["device"] = str(isaac_cfg["sim"]["device"])

        print(
            "P2_LIFECYCLE_START "
            f"episodes={episodes} max_steps={max_steps} headless={headless}",
            flush=True,
        )
        probe = IsaacSingleDroneProbe(isaac_cfg, render=not headless)
        backend = IsaacSingleNavigationBackend(rmappo_cfg, isaac_cfg, probe)
        observation, critic_state = validate_backend_shapes(backend)
        baseline_depth = observation["depth"].detach().clone()
        spawn = torch.tensor(
            isaac_cfg["scene"]["spawn_position_m"],
            dtype=torch.float32,
            device=backend.device,
        )

        def torch_cuda_memory_mib() -> dict[str, float]:
            if not torch.cuda.is_available():
                return {"allocated": 0.0, "reserved": 0.0}
            torch.cuda.synchronize(backend.device)
            scale = 1024.0 * 1024.0
            return {
                "allocated": torch.cuda.memory_allocated(backend.device) / scale,
                "reserved": torch.cuda.memory_reserved(backend.device) / scale,
            }

        memory_start = torch_cuda_memory_mib()
        memory_samples = [dict(memory_start)]
        episode_reports = []
        total_control_steps = 0
        all_finite = True

        for episode_index in range(episodes):
            reward_sum = 0.0
            minimum_distance = math.inf
            final_info = None
            episode_steps = 0
            for episode_step in range(max_steps):
                action = proportional_navigation_action(
                    observation,
                    backend.target_cfg,
                    backend.action_limits,
                    position_gain=float(lifecycle_cfg["position_gain"]),
                    yaw_gain=float(lifecycle_cfg["yaw_gain"]),
                    max_cruise_speed_mps=float(
                        lifecycle_cfg["max_cruise_speed_mps"]
                    ),
                )
                observation, critic_state, reward, done, info = backend.step(action)
                validate_step_info(info, backend.num_envs)
                episode_steps = episode_step + 1
                total_control_steps += 1
                reward_sum += float(reward.item())
                minimum_distance = min(
                    minimum_distance, float(info["target_distance_m"].item())
                )
                all_finite &= bool(torch.isfinite(reward).all().item())
                all_finite &= bool(torch.isfinite(critic_state).all().item())
                all_finite &= all(
                    bool(torch.isfinite(value).all().item())
                    for value in observation.values()
                )
                if bool(done.item()):
                    final_info = info
                    break

            probe_timeout = final_info is None
            if probe_timeout:
                observation, critic_state = backend.reset()
            reset_telemetry = backend.last_telemetry
            reset_position = torch.tensor(
                reset_telemetry["position_m"],
                dtype=torch.float32,
                device=backend.device,
            )
            reset_position_error = float((reset_position - spawn).norm().item())
            reset_depth_mae = float(
                (observation["depth"] - baseline_depth).abs().mean().item()
            )
            depth_frame_consistency = float(
                (
                    observation["depth"][:, :, 1:]
                    - observation["depth"][:, :, :1]
                )
                .abs()
                .max()
                .item()
            )
            reset_yaw_error = wrapped_abs_error(
                float(reset_telemetry["yaw_rad"]), 0.0
            )
            success = bool(
                final_info is not None
                and float(final_info["episode_success"].item()) == 1.0
            )
            report = {
                "episode": episode_index + 1,
                "success": success,
                "steps": episode_steps,
                "reward_sum": reward_sum,
                "minimum_target_distance_m": minimum_distance,
                "collision": bool(
                    final_info is not None
                    and float(final_info["episode_collision"].item()) > 0.0
                ),
                "out_of_bounds": bool(
                    final_info is not None
                    and float(final_info["episode_out_of_bounds"].item()) > 0.0
                ),
                "timeout": bool(
                    probe_timeout
                    or float(final_info["episode_timeout"].item()) > 0.0
                ),
                "stall": bool(
                    final_info is not None
                    and float(final_info["episode_stall"].item()) > 0.0
                ),
                "reset_position_error_m": reset_position_error,
                "reset_speed_mps": float(reset_telemetry["speed_mps"]),
                "reset_yaw_abs_error_rad": reset_yaw_error,
                "reset_depth_mae": reset_depth_mae,
                "reset_depth_frame_consistency_max_abs": depth_frame_consistency,
                "reset_collision": bool(reset_telemetry["collision"]),
                "reset_out_of_bounds": bool(reset_telemetry["out_of_bounds"]),
                "reset_finite": bool(reset_telemetry["finite"]),
            }
            episode_reports.append(report)
            memory_samples.append(torch_cuda_memory_mib())
            print(
                f"[lifecycle] episode={episode_index + 1}/{episodes} "
                f"success={success} steps={episode_steps} "
                f"reset_pos_error={reset_position_error:.6f} "
                f"reset_depth_mae={reset_depth_mae:.6f}",
                flush=True,
            )

        max_position_error = max(
            item["reset_position_error_m"] for item in episode_reports
        )
        max_reset_speed = max(item["reset_speed_mps"] for item in episode_reports)
        max_reset_yaw_error = max(
            item["reset_yaw_abs_error_rad"] for item in episode_reports
        )
        max_reset_depth_mae = max(
            item["reset_depth_mae"] for item in episode_reports
        )
        max_reset_depth_frame_inconsistency = max(
            item["reset_depth_frame_consistency_max_abs"]
            for item in episode_reports
        )
        reset_fault_count = sum(
            int(
                item["reset_collision"]
                or item["reset_out_of_bounds"]
                or not item["reset_finite"]
            )
            for item in episode_reports
        )
        memory_growth = {
            key: memory_samples[-1][key] - memory_start[key]
            for key in memory_start
        }
        memory_peak_growth = {
            key: max(sample[key] for sample in memory_samples) - memory_start[key]
            for key in memory_start
        }
        success_count = sum(int(item["success"]) for item in episode_reports)
        collision_count = sum(int(item["collision"]) for item in episode_reports)
        out_of_bounds_count = sum(
            int(item["out_of_bounds"]) for item in episode_reports
        )
        timeout_count = sum(int(item["timeout"]) for item in episode_reports)
        stall_count = sum(int(item["stall"]) for item in episode_reports)
        passed = (
            all_finite
            and success_count == episodes
            and collision_count == 0
            and out_of_bounds_count == 0
            and timeout_count == 0
            and stall_count == 0
            and reset_fault_count == 0
            and max_position_error
            <= float(lifecycle_cfg["reset_position_tolerance_m"])
            and max_reset_speed <= float(lifecycle_cfg["reset_speed_tolerance_mps"])
            and max_reset_yaw_error
            <= float(lifecycle_cfg["reset_yaw_tolerance_rad"])
            and max_reset_depth_mae
            <= float(lifecycle_cfg["reset_depth_mae_tolerance"])
            and max_reset_depth_frame_inconsistency <= 1e-6
            and memory_peak_growth["reserved"]
            <= float(lifecycle_cfg["max_torch_cuda_memory_growth_mib"])
        )
        wall_time = max(time.perf_counter() - wall_start, 1e-9)
        result = {
            "schema_version": 1,
            "passed": passed,
            "headless": headless,
            "episodes": episodes,
            "success_count": success_count,
            "collision_episode_count": collision_count,
            "out_of_bounds_episode_count": out_of_bounds_count,
            "timeout_episode_count": timeout_count,
            "stall_episode_count": stall_count,
            "reset_fault_count": reset_fault_count,
            "finite": all_finite,
            "total_control_steps": total_control_steps,
            "control_steps_per_second": total_control_steps / wall_time,
            "wall_time_s": wall_time,
            "max_reset_position_error_m": max_position_error,
            "max_reset_speed_mps": max_reset_speed,
            "max_reset_yaw_abs_error_rad": max_reset_yaw_error,
            "max_reset_depth_mae": max_reset_depth_mae,
            "max_reset_depth_frame_consistency_max_abs": (
                max_reset_depth_frame_inconsistency
            ),
            "torch_cuda_memory_start_mib": memory_start,
            "torch_cuda_memory_final_mib": memory_samples[-1],
            "torch_cuda_memory_growth_mib": memory_growth,
            "torch_cuda_memory_peak_growth_mib": memory_peak_growth,
            "memory_scope_note": (
                "torch allocator only; inspect nvidia-smi separately for full Kit/RTX memory"
            ),
            "direct_gpu_log_check_required": True,
            "episode_reports": episode_reports,
            "isaac_config_path": str(isaac_config_path),
            "rmappo_config_path": rmappo_cfg.get("_config_path"),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(f"P2_LIFECYCLE_REPORT={output_path}", flush=True)
        print(
            "P2_LIFECYCLE_RESULT=" + ("PASS" if passed else "FAIL"),
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
            "isaac_config_path": str(isaac_config_path),
            "headless": headless,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("P2_LIFECYCLE_EXCEPTION", flush=True)
        print(failure["traceback"], file=sys.stderr, end="", flush=True)
        print(f"P2_LIFECYCLE_REPORT={output_path}", flush=True)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
                print("P2_LIFECYCLE_ENV_CLOSED", flush=True)
            except BaseException:
                print("P2_LIFECYCLE_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            finally:
                backend = None
                probe = None
                gc.collect()
        elif probe is not None:
            try:
                probe.close()
            finally:
                probe = None
                gc.collect()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()

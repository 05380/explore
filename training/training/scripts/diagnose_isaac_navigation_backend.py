#!/usr/bin/env python3
"""Exercise the one-drone fixed-target MultiUAVBackend without PPO updates."""

from __future__ import annotations

import argparse
import gc
import json
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
    TRAINING_ROOT / "runs" / "isaac_single_backend" / "contract_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-config", type=Path, default=DEFAULT_ISAAC_CONFIG)
    parser.add_argument("--rmappo-config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--forward-action", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if not -1.0 <= float(args.forward_action) <= 1.0:
        raise ValueError("--forward-action must be in [-1, 1]")
    isaac_config_path = args.isaac_config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    isaac_cfg = yaml.safe_load(isaac_config_path.read_text(encoding="utf-8"))
    if not isinstance(isaac_cfg, dict):
        raise ValueError("Isaac configuration root must be a mapping")
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

        validate_probe_config(isaac_cfg)
        rmappo_cfg = apply_curriculum_stage(
            load_config(args.rmappo_config), "single_agent_sparse_static"
        )
        rmappo_cfg["training"]["num_parallel_swarms"] = 1
        rmappo_cfg["training"]["device"] = str(isaac_cfg["sim"]["device"])

        print(
            "P2_BACKEND_START "
            f"steps={args.steps} headless={headless} config={isaac_config_path}",
            flush=True,
        )
        probe = IsaacSingleDroneProbe(isaac_cfg, render=not headless)
        backend = IsaacSingleNavigationBackend(rmappo_cfg, isaac_cfg, probe)
        validate_backend_shapes(backend)
        observation, critic_state = backend.reset()

        action = torch.zeros(1, 1, 9, device=backend.device)
        action[..., 0] = float(args.forward_action)
        reward_sum = 0.0
        reward_min = float("inf")
        reward_max = -float("inf")
        episode_count = 0
        collision_steps = 0
        out_of_bounds_steps = 0
        stall_steps = 0
        safety_takeover_steps = 0
        finite = True
        final_info = None
        loop_start = time.perf_counter()
        for step in range(args.steps):
            observation, critic_state, reward, done, info = backend.step(action)
            validate_step_info(info, backend.num_envs)
            reward_value = float(reward.mean().item())
            reward_sum += reward_value
            reward_min = min(reward_min, reward_value)
            reward_max = max(reward_max, reward_value)
            episode_count += int(info["episode_finished"].item())
            collision_steps += int(info["collision"].item())
            out_of_bounds_steps += int(info["out_of_bounds"].item())
            stall_steps += int(info["stall"].item())
            safety_takeover_steps += int(info["safety_takeover"].item())
            finite = finite and all(
                bool(torch.isfinite(value).all().item())
                for value in observation.values()
            )
            finite = finite and bool(torch.isfinite(critic_state).all().item())
            finite = finite and bool(torch.isfinite(reward).all().item())
            final_info = info
            if step == 0 or step + 1 == args.steps or (step + 1) % 10 == 0:
                print(
                    f"[backend] step={step + 1}/{args.steps} "
                    f"reward={reward_value:.4f} "
                    f"distance={float(info['target_distance_m'].item()):.3f} "
                    f"speed={float(info['actual_speed_mps'].item()):.3f}",
                    flush=True,
                )
        loop_wall_time = max(time.perf_counter() - loop_start, 1e-9)
        assert final_info is not None

        passed = (
            finite
            and collision_steps == 0
            and out_of_bounds_steps == 0
            and stall_steps == 0
            and safety_takeover_steps == 0
            and tuple(observation["depth"].shape) == (1, 1, 3, 40, 64)
            and tuple(critic_state.shape) == (1, 1, 13)
            and bool((observation["decision_mask"] == 0.0).all().item())
        )
        report = {
            "schema_version": 1,
            "passed": passed,
            "headless": headless,
            "steps": args.steps,
            "forward_action": float(args.forward_action),
            "finite": finite,
            "collision_steps": collision_steps,
            "out_of_bounds_steps": out_of_bounds_steps,
            "stall_steps": stall_steps,
            "safety_takeover_steps": safety_takeover_steps,
            "episode_count": episode_count,
            "reward_mean": reward_sum / args.steps,
            "reward_min": reward_min,
            "reward_max": reward_max,
            "control_steps_per_second": args.steps / loop_wall_time,
            "wall_time_s": time.perf_counter() - wall_start,
            "observation_shapes": {
                key: list(value.shape) for key, value in observation.items()
            },
            "critic_state_shape": list(critic_state.shape),
            "final_target_distance_m": float(
                final_info["target_distance_m"].item()
            ),
            "final_actual_speed_mps": float(
                final_info["actual_speed_mps"].item()
            ),
            "final_command_body": final_info["command_body"].detach().cpu().tolist(),
            "isaac_config_path": str(isaac_config_path),
            "rmappo_config_path": rmappo_cfg.get("_config_path"),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"P2_BACKEND_REPORT={output_path}", flush=True)
        print(
            "P2_BACKEND_RESULT=" + ("PASS" if passed else "FAIL"), flush=True
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
        print("P2_BACKEND_EXCEPTION", flush=True)
        print(failure["traceback"], file=sys.stderr, end="", flush=True)
        print(f"P2_BACKEND_REPORT={output_path}", flush=True)
        raise
    finally:
        if backend is not None:
            try:
                backend.close()
                print("P2_BACKEND_ENV_CLOSED", flush=True)
            except BaseException:
                print("P2_BACKEND_ENV_CLOSE_EXCEPTION", flush=True)
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

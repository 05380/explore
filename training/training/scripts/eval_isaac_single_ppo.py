#!/usr/bin/env python3
"""Evaluate a navigation checkpoint with optional Isaac overlays and traces."""

from __future__ import annotations

import argparse
import gc
import json
import sys
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
    TRAINING_ROOT / "runs" / "isaac_nav_curriculum" / "open_eval.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--isaac-config", type=Path, default=DEFAULT_ISAAC_CONFIG)
    parser.add_argument("--rmappo-config", type=Path, default=None)
    parser.add_argument("--scenario", default="open_target")
    parser.add_argument(
        "--stage",
        default=None,
        help="Compatibility check; when set it must match the selected scenario.",
    )
    parser.add_argument("--steps", type=int, default=3200)
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Print evaluation progress every N control steps; use 0 to disable.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--visualize", action="store_true", help="Isaac Debug Draw + status HUD; requires --no-headless")
    parser.add_argument("--view", choices=("overview", "top", "follow"), default="overview")
    parser.add_argument("--draw-every", type=int, default=5, help="Redraw overlays every N control steps")
    parser.add_argument("--trace", type=Path, default=None, help="Per-step JSONL, also works headless")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative")
    if args.draw_every < 1:
        raise ValueError("--draw-every must be positive")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    isaac_config_path = args.isaac_config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if output_path in (checkpoint_path, isaac_config_path):
        raise ValueError("report path must not overwrite the checkpoint or Isaac config")
    base_isaac_cfg = yaml.safe_load(
        isaac_config_path.read_text(encoding="utf-8")
    )
    if not isinstance(base_isaac_cfg, dict):
        raise ValueError("Isaac configuration root must be a mapping")

    # Keep package/PyTorch imports after SimulationApp is alive.
    app_cfg = base_isaac_cfg["app"]
    headless = (
        bool(app_cfg["headless"]) if args.headless is None else bool(args.headless)
    )
    if args.visualize and headless:
        raise ValueError("--visualize requires --no-headless and a working desktop/display")
    trace_path = args.trace
    if args.visualize and trace_path is None:
        trace_path = output_path.with_suffix(".trace.jsonl")
    if trace_path is not None:
        trace_path = trace_path.expanduser().resolve()
        if trace_path in (output_path, checkpoint_path, isaac_config_path):
            raise ValueError("--trace must differ from report/checkpoint/config paths")
    simulation_app = None
    probe = None
    backend = None
    trainer = None
    result = None
    recorder = None
    backend_close_failed = False
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
        from racer_rmappo.isaac_single_backend import IsaacSingleNavigationBackend
        from racer_rmappo.isaac_single_env import (
            IsaacSingleDroneProbe,
            validate_probe_config,
        )
        from racer_rmappo.isaac_scenarios import apply_navigation_scenario
        from racer_rmappo.trainer import RMAPPOTrainer
        from racer_rmappo.policy_contract import checkpoint_policy

        isaac_cfg = apply_navigation_scenario(base_isaac_cfg, args.scenario)
        scenario_cfg = isaac_cfg["navigation_backend"]["scenarios"][args.scenario]
        stage_name = str(scenario_cfg["rmappo_stage"])
        if args.stage is not None and args.stage != stage_name:
            raise ValueError(
                f"scenario {args.scenario!r} requires stage {stage_name!r}; "
                f"received incompatible --stage {args.stage!r}"
            )
        validate_probe_config(isaac_cfg)
        cfg = apply_curriculum_stage(load_config(args.rmappo_config), stage_name)
        checkpoint_metadata = torch.load(checkpoint_path, map_location="cpu")
        cfg["policy"] = {"version": checkpoint_policy(checkpoint_metadata)}
        # Architecture belongs to the checkpoint, not the current YAML default.
        saved_cfg = checkpoint_metadata.get("config", {})
        if "ppo" in saved_cfg:
            cfg["ppo"]["recurrent_hidden_size"] = saved_cfg["ppo"]["recurrent_hidden_size"]
        del checkpoint_metadata
        cfg["training"]["backend"] = "isaac"
        cfg["training"]["device"] = str(isaac_cfg["sim"]["device"])
        cfg["training"]["num_parallel_swarms"] = 1
        cfg["training"]["isaac_config_path"] = str(isaac_config_path)
        cfg["training"]["isaac_scenario"] = args.scenario
        print(
            "P3_ISAAC_PPO_EVAL_START "
            f"checkpoint={checkpoint_path} scenario={args.scenario} "
            f"stage={stage_name} "
            f"steps={args.steps} headless={headless}",
            flush=True,
        )
        probe = IsaacSingleDroneProbe(isaac_cfg, render=not headless)
        backend = IsaacSingleNavigationBackend(cfg, isaac_cfg, probe)
        backend.collect_diagnostics = trace_path is not None
        trainer = RMAPPOTrainer(cfg, backend=backend)
        trainer.load(checkpoint_path, load_optimizer=False)
        if trace_path is not None:
            from racer_rmappo.eval_visualization import EvaluationTrace
            recorder = EvaluationTrace(trace_path, isaac_cfg, checkpoint_path, args.scenario)
            if args.visualize:
                recorder.enable_viewer(probe, view=args.view, draw_every=args.draw_every)
            print(f"P3_EVAL_TRACE={recorder.path}", flush=True)
        result = trainer.evaluate(
            args.steps,
            progress_interval_steps=args.progress_interval,
            step_callback=recorder,
        )
        if recorder is not None:
            result["diagnostics"] = recorder.summary()
        result["physics_timing"] = probe.timing.report()
        if not result["physics_timing"]["valid"]:
            raise RuntimeError("invalid physics timing; this is not a valid policy evaluation")
        result["run_completed"] = True
        result["policy_version"] = trainer.policy_version
        result["coverage_available"] = False
        result["curriculum_passed"] = (
            result["episode_count"] >= 50
            and result["collision_free_success_rate"] >= 0.90
            and result["out_of_bounds_episode_rate"] == 0.0
        )
        result.update(
            {
                "schema_version": 1,
                "checkpoint": str(checkpoint_path),
                "scenario": args.scenario,
                "rmappo_stage": stage_name,
                "steps": args.steps,
                "headless": headless,
                "isaac_config_path": str(isaac_config_path),
                "rmappo_config_path": cfg.get("_config_path"),
            }
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        print(f"P3_ISAAC_PPO_EVAL_REPORT={output_path}", flush=True)
        print("P3_ISAAC_PPO_EVAL_RUN_RESULT=PASS", flush=True)
        print("P3_ISAAC_PPO_CURRICULUM_RESULT=" + ("PASS" if result["curriculum_passed"] else "NOT_PASSED"), flush=True)
    except BaseException:
        print("P3_ISAAC_PPO_EVAL_RUN_RESULT=FAIL", flush=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "run_completed": False, "curriculum_passed": False,
            "exception": traceback.format_exc(),
            "physics_timing": probe.timing.report() if probe is not None and hasattr(probe, "timing") else None,
        }, indent=2), encoding="utf-8")
        traceback.print_exc()
        raise
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except BaseException:
                backend_close_failed = True
                print("P3_ISAAC_PPO_EVAL_VIEWER_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            recorder = None
        if trainer is not None:
            trainer.backend = None
        trainer = None
        result = None
        if backend is not None:
            try:
                backend.close()
                print("P3_ISAAC_PPO_EVAL_ENV_CLOSED", flush=True)
            except BaseException:
                backend_close_failed = True
                print("P3_ISAAC_PPO_EVAL_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            backend = None
            probe = None
        elif probe is not None:
            try:
                probe.close()
            except BaseException:
                backend_close_failed = True
                print("P3_ISAAC_PPO_EVAL_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            probe = None
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

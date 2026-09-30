#!/usr/bin/env python3
"""Train the shared RMAPPO actor on one P3 Isaac navigation curriculum."""

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
DEFAULT_OUTPUT = TRAINING_ROOT / "runs" / "isaac_nav_curriculum" / "open"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-config", type=Path, default=DEFAULT_ISAAC_CONFIG)
    parser.add_argument("--rmappo-config", type=Path, default=None)
    parser.add_argument("--scenario", default="open_target")
    parser.add_argument(
        "--stage",
        default=None,
        help="Compatibility check; when set it must match the selected scenario.",
    )
    parser.add_argument("--total-steps", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=None
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.total_steps < 1:
        raise ValueError("--total-steps must be positive")
    isaac_config_path = args.isaac_config.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
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
    simulation_app = None
    probe = None
    backend = None
    trainer = None
    final_checkpoint = None
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
        cfg["training"]["backend"] = "isaac"
        cfg["training"]["device"] = str(isaac_cfg["sim"]["device"])
        cfg["training"]["num_parallel_swarms"] = 1
        cfg["training"]["output_dir"] = str(output_dir)
        cfg["training"]["isaac_config_path"] = str(isaac_config_path)
        cfg["training"]["isaac_scenario"] = args.scenario
        if int(cfg["experiment"]["num_agents"]) != 1:
            raise ValueError("the first Isaac PPO stage requires exactly one agent")

        print(
            "P3_ISAAC_PPO_TRAIN_START "
            f"scenario={args.scenario} stage={stage_name} "
            f"steps={args.total_steps} device={cfg['training']['device']} "
            f"headless={headless} output={output_dir}",
            flush=True,
        )
        probe = IsaacSingleDroneProbe(isaac_cfg, render=not headless)
        backend = IsaacSingleNavigationBackend(cfg, isaac_cfg, probe)
        trainer = RMAPPOTrainer(cfg, backend=backend)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_isaac_config.json").write_text(
            json.dumps(isaac_cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
        (output_dir / "resolved_rmappo_config.json").write_text(
            json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
        )
        if args.resume is not None:
            resume_path = args.resume.expanduser().resolve()
            trainer.load(resume_path)
            print(f"P3_ISAAC_PPO_RESUME={resume_path}", flush=True)
        final_checkpoint = trainer.train(args.total_steps)
        print(f"P3_ISAAC_PPO_CHECKPOINT={final_checkpoint}", flush=True)
        print("P3_ISAAC_PPO_TRAIN_RESULT=PASS", flush=True)
    except BaseException:
        print("P3_ISAAC_PPO_TRAIN_RESULT=FAIL", flush=True)
        traceback.print_exc()
        raise
    finally:
        if trainer is not None:
            trainer.backend = None
        trainer = None
        if backend is not None:
            try:
                backend.close()
                print("P3_ISAAC_PPO_ENV_CLOSED", flush=True)
            except BaseException:
                backend_close_failed = True
                print("P3_ISAAC_PPO_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            backend = None
            probe = None
        elif probe is not None:
            try:
                probe.close()
            except BaseException:
                backend_close_failed = True
                print("P3_ISAAC_PPO_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            probe = None
        final_checkpoint = None
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

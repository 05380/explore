#!/usr/bin/env python3
"""Run the first-stage Isaac single-drone physics probes without PPO."""

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
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)

DEFAULT_CONFIG = TRAINING_ROOT / "configs" / "isaac_single.yaml"
DEFAULT_OUTPUT = TRAINING_ROOT / "runs" / "isaac_single_probe" / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--probe",
        choices=("all", "hover", "reset", "random", "contact", "camera"),
        default="all",
    )
    parser.add_argument("--steps", type=int, default=None, help="override hover/random steps")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override app.headless from YAML",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")

    app_cfg = cfg["app"]
    headless = bool(app_cfg["headless"]) if args.headless is None else bool(args.headless)

    simulation_app = None
    environment = None
    try:
        # SimulationApp must be created before constructing the environment,
        # because Isaac and OmniDrones modules are imported lazily there.
        from omni.isaac.kit import SimulationApp

        simulation_app = SimulationApp(
            {
                "headless": headless,
                "multi_gpu": bool(app_cfg.get("multi_gpu", False)),
                "anti_aliasing": int(app_cfg.get("anti_aliasing", 0)),
                "fast_shutdown": bool(app_cfg.get("fast_shutdown", False)),
            }
        )

        from racer_rmappo.isaac_single_env import (
            IsaacSingleDroneProbe,
            validate_probe_config,
        )

        validate_probe_config(cfg)
        print(
            f"P1_PROBE_START probe={args.probe} headless={headless} "
            f"config={config_path}",
            flush=True,
        )
        environment = IsaacSingleDroneProbe(cfg, render=not headless)
        report = environment.run(args.probe, args.steps)
        report["config_path"] = str(config_path)
        report["headless"] = headless

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
        resolved_config = output_path.with_name("resolved_config.yaml")
        resolved_config.write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
        )

        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        print(f"P1_PROBE_REPORT={output_path}", flush=True)
        print(
            "P1_PROBE_RESULT=" + ("PASS" if report["passed"] else "FAIL"),
            flush=True,
        )
        if not report["passed"]:
            raise SystemExit(2)
    except SystemExit:
        raise
    except BaseException as error:
        exception_traceback = traceback.format_exc()
        failure_report = {
            "passed": False,
            "probe": args.probe,
            "config_path": str(config_path),
            "headless": headless,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": exception_traceback,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(failure_report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print("P1_PROBE_EXCEPTION", flush=True)
        print(exception_traceback, file=sys.stderr, end="", flush=True)
        print(f"P1_PROBE_REPORT={output_path}", flush=True)
        raise
    finally:
        if environment is not None:
            try:
                environment.close()
                print("P1_PROBE_ENV_CLOSED", flush=True)
            except BaseException:
                print("P1_PROBE_ENV_CLOSE_EXCEPTION", flush=True)
                traceback.print_exc()
            finally:
                environment = None
                gc.collect()
        if simulation_app is not None:
            from racer_rmappo.isaac_runtime import (
                active_exception_exit_code,
                close_simulation_app_safely,
            )

            close_simulation_app_safely(
                simulation_app,
                hard_exit_after_shutdown=bool(
                    app_cfg.get("hard_exit_after_shutdown", False)
                ),
                process_exit_code=active_exception_exit_code(),
            )


if __name__ == "__main__":
    main()

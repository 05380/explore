#!/usr/bin/env python3
"""Native hover/command parity probe; compare mode needs no Isaac or Torch."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "training", ROOT / "third_party" / "OmniDrones"):
    sys.path.insert(0, str(source))


def compare_reports(reports):
    if len(reports) != 3:
        raise ValueError("provide headless, GUI, and GUI-overlay reports")
    modes = {r["mode"] for r in reports}
    if modes != {"headless", "gui", "gui_overlay"}:
        raise ValueError("comparison requires three distinct rendering modes")
    if any(r["config"] != reports[0]["config"] for r in reports[1:]):
        raise ValueError("reports must use exactly the same physics/camera configuration")
    reference = next(r for r in reports if r["mode"] == "headless")
    position_error = velocity_error = 0.0
    timing_equal = True
    for report in reports:
        for phase in ("hover", "sequence"):
            rows, baseline = report["phases"][phase], reference["phases"][phase]
            if len(rows) != len(baseline):
                raise ValueError("trajectory lengths differ")
            for row, expected in zip(rows, baseline):
                position_error = max(position_error, math.dist(row["position_m"], expected["position_m"]))
                velocity_error = max(velocity_error, math.dist(row["velocity_world_mps"], expected["velocity_world_mps"]))
                timing_equal &= row["timing_step"]["physics_steps"] == expected["timing_step"]["physics_steps"] == 6
                timing_equal &= math.isclose(row["timing_step"]["physics_time_s"], .05, abs_tol=1e-7)
    passed = (all(r["passed"] for r in reports) and timing_equal
              and position_error <= .05 and velocity_error <= .1)
    return dict(passed=passed, max_position_difference_m=position_error,
                max_velocity_difference_mps=velocity_error, timing_equal=timing_equal)


def run_probe(args, cfg):
    from omni.isaac.kit import SimulationApp
    app = SimulationApp(dict(headless=args.headless, multi_gpu=False,
                             anti_aliasing=0, fast_shutdown=False))
    # Importing the racer_rmappo package imports Torch: Kit must be alive first.
    from racer_rmappo.isaac_runtime import close_simulation_app_safely
    probe = viewer = None
    phases = {}
    exit_code = 0
    report = dict(schema_version=1, config=cfg, passed=False,
                  mode="headless" if args.headless else "gui_overlay" if args.visualize else "gui")
    try:
        from racer_rmappo.isaac_single_env import IsaacSingleDroneProbe, validate_probe_config
        from racer_rmappo.isaac_single_backend import quaternion_rpy_wxyz
        import torch
        validate_probe_config(cfg)
        probe = IsaacSingleDroneProbe(cfg, render=not args.headless)
        probe.render_on_step = False  # Exactly one sensor-owned render per control action.
        if args.visualize:
            from racer_rmappo.eval_visualization import IsaacEvaluationViewer
            viewer = IsaacEvaluationViewer(cfg, probe, "overview", 5, 3.0)
        max_hover_error, final_hover_error = 0.0, 0.0
        all_safe = True
        for phase_id, (phase, steps) in enumerate((("hover", 600), ("sequence", 200))):
            initial = probe.consume_initial_reset_telemetry()
            if initial is None:
                probe.reset()
            probe.synchronize_pose_to_renderer(1)
            probe.depth_camera.capture()
            rows = []
            phases[phase] = rows
            for i in range(steps):
                theta = 2*math.pi*i/200
                command = (0., 0., 0., 0.) if phase == "hover" else (
                    .3*math.sin(theta), .2*math.cos(theta), .1*math.sin(theta), .15*math.sin(theta))
                telemetry, applied = probe.step(command)
                probe.depth_camera.capture(warmup_frames=1)
                rpy = quaternion_rpy_wxyz(torch.tensor(telemetry["orientation_wxyz"]))
                position = telemetry["position_m"]
                goal = cfg["scene"]["spawn_position_m"]
                distance = math.dist(position, goal)
                row = dict(step=i+1, episode=phase_id, episode_step=i+1,
                           episode_time_s=(i+1)*probe.control_dt, position_m=position,
                           orientation_wxyz=telemetry["orientation_wxyz"],
                           velocity_world_mps=telemetry["linear_velocity_mps"],
                           actual_speed_mps=telemetry["speed_mps"],
                           target_position_m=goal, target_distance_m=distance,
                           target_yaw_error_rad=telemetry["yaw_rad"],
                           body_tilt_rad=math.hypot(float(rpy[0]), float(rpy[1])),
                           command_body=list(command), applied_command_world=applied.cpu().tolist(),
                           navigation_reached=distance <= .5, navigation_pose_reached=False,
                           observation_completed=False, reward=0., overspeed=telemetry["speed_mps"]>2.05,
                           done=False, success=False, collision=telemetry["collision"],
                           out_of_bounds=telemetry["out_of_bounds"], stall=False, timeout=False,
                           timing_step=telemetry["timing_step"], physics_time_s=telemetry["physics_time_s"],
                           rotor_action=telemetry["rotor_action"], rotor_thrust_n=telemetry["rotor_thrust_n"])
                row.update(controller_reference_position_m=telemetry["controller_reference_position_m"],
                           velocity_command_limited=telemetry["velocity_command_limited"],
                           yaw_rate_command_limited=telemetry["yaw_rate_command_limited"],
                           limited_command_body=telemetry["limited_command_body"])
                rows.append(row)
                all_safe &= telemetry["finite"] and not telemetry["collision"] and not telemetry["out_of_bounds"]
                if phase == "hover":
                    max_hover_error = max(max_hover_error, distance)
                    final_hover_error = distance
                if viewer:
                    viewer.update(row)
                if (i+1) % 100 == 0:
                    print(f"[timing {phase}] step={i+1}/{steps} z={position[2]:.4f} physics={row['timing_step']}", flush=True)
        timing = probe.timing.report()
        passed = (all_safe and timing["valid"]
                  and max_hover_error <= cfg["acceptance"]["hover_max_position_error_m"]
                  and final_hover_error <= cfg["acceptance"]["hover_final_position_error_m"])
        report.update(phases=phases, physics_timing=timing, passed=passed,
                      hover_max_position_error_m=max_hover_error,
                      hover_final_position_error_m=final_hover_error)
        exit_code = 0 if passed else 2
    except BaseException:
        exit_code = 1
        report["exception"] = traceback.format_exc()
        report["phases"] = phases
        if probe is not None:
            report["physics_timing"] = probe.timing.report()
        traceback.print_exc()
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"TIMING_REPORT={args.output.resolve()}", flush=True)
        print("TIMING_RESULT=" + ("PASS" if report["passed"] else "FAIL"), flush=True)
        try:
            if viewer:
                viewer.close()
            if probe:
                probe.close()
        except BaseException:
            exit_code = 1
            traceback.print_exc()
        viewer = probe = None
        gc.collect()
        close_simulation_app_safely(app, hard_exit_after_shutdown=bool(
            cfg["app"].get("hard_exit_after_shutdown", False)), process_exit_code=exit_code)
    return exit_code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "isaac_single.yaml")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--compare", type=Path, nargs=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.compare:
        report = compare_reports([json.loads(p.read_text()) for p in args.compare])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 2
    if args.visualize and args.headless:
        parser.error("--visualize requires --no-headless")
    import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    return run_probe(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())

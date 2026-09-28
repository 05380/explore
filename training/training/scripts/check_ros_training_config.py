#!/usr/bin/env python3
"""Fail-fast comparison of the RMAPPO YAML and ROS launch contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = SCRIPT_DIR.parent
REPOSITORY = SCRIPT_DIR.parents[2]
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.config import load_config


DEFAULT_LAUNCH = (
    REPOSITORY
    / "swarm_exploration"
    / "exploration_manager"
    / "launch"
    / "single_drone_rl_d455m.xml"
)
DEFAULT_CAMERA_PROFILE = (
    REPOSITORY
    / "uav_simulator"
    / "local_sensing"
    / "params"
    / "camera_d455m_640x360.yaml"
)
DEFAULT_SWARM_LAUNCH = (
    REPOSITORY
    / "swarm_exploration"
    / "exploration_manager"
    / "launch"
    / "swarm_exploration_rl_d455m_16.launch"
)


def _named_values(elements, attribute: str) -> Dict[str, str]:
    return {
        element.attrib["name"]: element.attrib[attribute]
        for element in elements
        if "name" in element.attrib and attribute in element.attrib
    }


def check_contract(
    cfg: Mapping[str, Any], launch_path: str | Path = DEFAULT_LAUNCH,
    camera_profile_path: str | Path = DEFAULT_CAMERA_PROFILE,
    swarm_launch_path: str | Path = DEFAULT_SWARM_LAUNCH,
) -> Dict[str, Any]:
    launch_path = Path(launch_path).expanduser().resolve()
    camera_profile_path = Path(camera_profile_path).expanduser().resolve()
    swarm_launch_path = Path(swarm_launch_path).expanduser().resolve()
    root = ET.parse(launch_path).getroot()
    with camera_profile_path.open("r", encoding="utf-8") as stream:
        camera_profile = yaml.safe_load(stream)
    root_args = _named_values(root.findall("./arg"), "default")
    planner = next(
        element
        for element in root.findall("./include")
        if "single_drone_planner.xml" in element.attrib.get("file", "")
    )
    planner_args = _named_values(planner.findall("./arg"), "value")
    bridge = next(
        element
        for element in root.findall("./node")
        if element.attrib.get("type") == "rl_velocity_bridge"
    )
    bridge_params = _named_values(bridge.findall("./param"), "value")

    world = cfg["world"]
    camera = cfg["camera"]
    limits = cfg["action"]["physical_limits"]
    selection = cfg["action"]["viewpoint_selection"]
    candidate = cfg["actor_observation"]["racer_candidates"]
    training = cfg["training"]
    sx, sy, sz = (float(value) for value in world["size_m"])
    resolution = float(world["voxel_resolution_m"])

    expected = {
        "root.map_size_x": sx,
        "root.map_size_y": sy,
        "root.map_size_z": sz,
        "root.fx": float(camera["fx"]),
        "root.fy": float(camera["fy"]),
        "root.cx": float(camera["cx"]),
        "root.cy": float(camera["cy"]),
        "planner.map_resolution": resolution,
        "planner.obstacles_inflation": float(world["obstacle_inflation_m"]),
        "planner.sensor_min_range": float(camera["min_depth_m"]),
        "planner.sensor_max_range": float(camera["max_depth_m"]),
        "planner.fov_top_angle": math.radians(float(camera["vertical_fov_deg"]) * 0.5),
        "planner.fov_left_angle": math.radians(float(camera["horizontal_fov_deg"]) * 0.5),
        "planner.fov_right_angle": math.radians(float(camera["horizontal_fov_deg"]) * 0.5),
        "planner.box_min_x": -sx * 0.5,
        "planner.box_min_y": -sy * 0.5,
        "planner.box_min_z": 0.0,
        "planner.box_max_x": sx * 0.5,
        "planner.box_max_y": sy * 0.5,
        "planner.box_max_z": sz,
        "planner.rl_target_reached_dist": float(training["smoke_goal_distance_m"]),
        "planner.rl_target_reached_yaw": float(training["smoke_goal_yaw_rad"]),
        "planner.rl_target_reached_tilt": float(camera["observation_tilt_tolerance_rad"]),
        "planner.rl_max_offset_xy": float(selection["max_position_offset_m"][0]),
        "planner.rl_max_offset_z": float(selection["max_position_offset_m"][2]),
        "planner.rl_max_yaw_offset": float(selection["max_yaw_offset_rad"]),
        "planner.rl_candidate_num": float(candidate["max_candidates"]),
        "bridge.control_rate": float(cfg["experiment"]["control_hz"]),
        "bridge.max_forward_speed": float(limits["forward_mps"]),
        "bridge.max_backward_speed": float(limits["backward_mps"]),
        "bridge.max_lateral_speed": float(limits["lateral_mps"]),
        "bridge.max_vertical_speed": float(limits["vertical_mps"]),
        "bridge.max_speed_norm": float(limits["speed_norm_mps"]),
        "bridge.max_yaw_rate": float(limits["yaw_rate_rps"]),
        "bridge.min_x": -sx * 0.5 + resolution,
        "bridge.max_x": sx * 0.5 - resolution,
        "bridge.min_y": -sy * 0.5 + resolution,
        "bridge.max_y": sy * 0.5 - resolution,
        "bridge.min_z": float(world["min_flight_z_m"]),
        "bridge.max_z": float(world["max_flight_z_m"]),
        "sensor.cam_width": float(camera["width"]),
        "sensor.cam_height": float(camera["height"]),
        "sensor.cam_fx": float(camera["fx"]),
        "sensor.cam_fy": float(camera["fy"]),
        "sensor.cam_cx": float(camera["cx"]),
        "sensor.cam_cy": float(camera["cy"]),
    }
    actual_sources = {
        "root": root_args,
        "planner": planner_args,
        "bridge": bridge_params,
        "sensor": camera_profile,
    }
    mismatches = []
    for qualified_name, expected_value in expected.items():
        source_name, field = qualified_name.split(".", 1)
        raw = actual_sources[source_name].get(field)
        if raw is None:
            mismatches.append(
                {"field": qualified_name, "expected": expected_value, "actual": "missing"}
            )
            continue
        try:
            actual_value = float(raw)
        except ValueError:
            mismatches.append(
                {"field": qualified_name, "expected": expected_value, "actual": raw}
            )
            continue
        if not math.isclose(actual_value, expected_value, rel_tol=1e-5, abs_tol=1e-5):
            mismatches.append(
                {"field": qualified_name, "expected": expected_value, "actual": actual_value}
            )
    swarm_root = ET.parse(swarm_launch_path).getroot()
    swarm_args = _named_values(swarm_root.findall("./arg"), "default")
    if int(swarm_args.get("drone_num", -1)) != int(cfg["experiment"]["num_agents"]):
        mismatches.append({
            "field": "swarm.drone_num_default",
            "expected": int(cfg["experiment"]["num_agents"]),
            "actual": swarm_args.get("drone_num", "missing"),
        })
    agent_includes = [
        element for element in swarm_root.findall("./include")
        if "single_drone_rl_d455m.xml" in element.attrib.get("file", "")
    ]
    if len(agent_includes) != 16:
        mismatches.append({"field": "swarm.agent_include_count", "expected": 16,
                           "actual": len(agent_includes)})
    for expected_id, include in enumerate(agent_includes, 1):
        args = _named_values(include.findall("./arg"), "value")
        expected_condition = (
            f"$(eval {expected_id} <= int(arg('drone_num')) <= 16)"
        )
        if args.get("drone_id") != str(expected_id) or args.get("drone_num") != "$(arg drone_num)" \
                or include.attrib.get("if") != expected_condition:
            mismatches.append({
                "field": f"swarm.agent_{expected_id}_guard",
                "expected": expected_condition,
                "actual": include.attrib.get("if", "missing"),
            })
    return {
        "valid": not mismatches,
        "launch": str(launch_path),
        "camera_profile": str(camera_profile_path),
        "swarm_launch": str(swarm_launch_path),
        "camera_fixed_to_body": bool(camera.get("fixed_to_body", False)),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--camera-profile", type=Path, default=DEFAULT_CAMERA_PROFILE)
    parser.add_argument("--swarm-launch", type=Path, default=DEFAULT_SWARM_LAUNCH)
    args = parser.parse_args()
    report = check_contract(
        load_config(args.config), args.launch, args.camera_profile, args.swarm_launch
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

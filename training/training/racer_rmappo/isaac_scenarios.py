"""Scenario selection and validation for the single-drone Isaac backend.

The base Isaac configuration remains the P2 wall-front regression case.  A
scenario is copied into the active navigation fields before constructing the
backend, so diagnostics, training and evaluation use exactly the same target
without mutating the caller's configuration.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Sequence


def _vector(value: Any, name: str, length: int = 3) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {length} numbers")
    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


def _segment_intersects_box(
    start: Sequence[float],
    end: Sequence[float],
    center: Sequence[float],
    size: Sequence[float],
) -> bool:
    """Return whether a finite segment intersects an axis-aligned box."""
    interval_min = 0.0
    interval_max = 1.0
    for start_axis, end_axis, center_axis, size_axis in zip(
        start, end, center, size
    ):
        lower = float(center_axis) - 0.5 * float(size_axis)
        upper = float(center_axis) + 0.5 * float(size_axis)
        delta = float(end_axis) - float(start_axis)
        if abs(delta) <= 1e-9:
            if float(start_axis) < lower or float(start_axis) > upper:
                return False
            continue
        first = (lower - float(start_axis)) / delta
        second = (upper - float(start_axis)) / delta
        interval_min = max(interval_min, min(first, second))
        interval_max = min(interval_max, max(first, second))
        if interval_min > interval_max:
            return False
    return True


def validate_navigation_scenario(
    isaac_cfg: Mapping[str, Any], scenario_name: str
) -> None:
    navigation = isaac_cfg["navigation_backend"]
    scenarios = navigation.get("scenarios", {})
    if scenario_name not in scenarios:
        names = ", ".join(sorted(str(name) for name in scenarios))
        raise ValueError(
            f"unknown Isaac navigation scenario {scenario_name!r}; "
            f"expected one of: {names}"
        )
    scenario = scenarios[scenario_name]
    target = _vector(
        scenario["fixed_target_position_m"],
        f"navigation_backend.scenarios.{scenario_name}.fixed_target_position_m",
    )
    lower = _vector(isaac_cfg["scene"]["flight_bounds_min_m"], "flight bounds min")
    upper = _vector(isaac_cfg["scene"]["flight_bounds_max_m"], "flight bounds max")
    if any(not lo < value < hi for value, lo, hi in zip(target, lower, upper)):
        raise ValueError(f"scenario {scenario_name!r} target is outside flight bounds")
    yaw = float(scenario["fixed_target_yaw_rad"])
    if not math.isfinite(yaw):
        raise ValueError(f"scenario {scenario_name!r} target yaw must be finite")
    if float(scenario.get("episode_seconds", navigation["episode_seconds"])) <= 0.0:
        raise ValueError(f"scenario {scenario_name!r} episode_seconds must be positive")

    waypoints = scenario.get("validation_waypoints_m", [])
    for index, waypoint in enumerate(waypoints):
        point = _vector(
            waypoint,
            f"navigation_backend.scenarios.{scenario_name}."
            f"validation_waypoints_m[{index}]",
        )
        if any(not lo < value < hi for value, lo, hi in zip(point, lower, upper)):
            raise ValueError(
                f"scenario {scenario_name!r} waypoint {index} is outside flight bounds"
            )
    for key in (
        "waypoint_position_tolerance_m",
        "validation_position_gain",
        "validation_yaw_gain",
        "validation_cruise_speed_mps",
        "validation_wall_visible_clearance_m",
    ):
        if float(scenario[key]) <= 0.0:
            raise ValueError(f"scenario {scenario_name!r} {key} must be positive")
    if int(scenario["validation_max_steps"]) < 1:
        raise ValueError(
            f"scenario {scenario_name!r} validation_max_steps must be positive"
        )
    obstacle = next(
        (
            item
            for item in isaac_cfg["scene"].get("obstacles", [])
            if str(item["name"]) == str(scenario["blocking_obstacle"])
        ),
        None,
    )
    if obstacle is None:
        raise ValueError(
            f"scenario {scenario_name!r} blocking_obstacle is not in the scene"
        )
    spawn = _vector(isaac_cfg["scene"]["spawn_position_m"], "spawn position")
    obstacle_center = _vector(obstacle["position_m"], "blocking obstacle position")
    obstacle_size = _vector(obstacle["size_m"], "blocking obstacle size")
    if not _segment_intersects_box(spawn, target, obstacle_center, obstacle_size):
        raise ValueError(
            f"scenario {scenario_name!r} direct path does not cross its blocking obstacle"
        )
    validation_path = [spawn] + [
        _vector(waypoint, "validation waypoint") for waypoint in waypoints
    ] + [target]
    if any(
        _segment_intersects_box(start, end, obstacle_center, obstacle_size)
        for start, end in zip(validation_path[:-1], validation_path[1:])
    ):
        raise ValueError(
            f"scenario {scenario_name!r} validation waypoint path intersects the obstacle"
        )


def apply_navigation_scenario(
    isaac_cfg: Mapping[str, Any], scenario_name: str
) -> Dict[str, Any]:
    """Return a deep-copied Isaac config with one navigation scenario active."""
    validate_navigation_scenario(isaac_cfg, scenario_name)
    result = copy.deepcopy(dict(isaac_cfg))
    navigation = result["navigation_backend"]
    scenario = navigation["scenarios"][scenario_name]
    for key in (
        "fixed_target_position_m",
        "fixed_target_yaw_rad",
        "episode_seconds",
        "terminate_on_stall",
    ):
        if key in scenario:
            navigation[key] = copy.deepcopy(scenario[key])
    navigation["active_scenario"] = scenario_name
    return result


def active_scenario(isaac_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    navigation = isaac_cfg["navigation_backend"]
    name = navigation.get("active_scenario")
    if name is None:
        raise ValueError("no Isaac navigation scenario has been activated")
    return navigation["scenarios"][name]

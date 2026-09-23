"""Configuration loading and validation for the RACER RMAPPO trainer."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "swarm_exploration"
    / "exploration_manager"
    / "config"
    / "rmappo_d455m_16.yaml"
)


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    validate_config(cfg)
    cfg["_config_path"] = str(config_path)
    return cfg


def _positive(mapping: Mapping[str, Any], key: str) -> float:
    value = float(mapping[key])
    if value <= 0.0:
        raise ValueError(f"{key} must be positive, got {value}")
    return value


def validate_config(cfg: Mapping[str, Any]) -> None:
    required = ("experiment", "world", "camera", "actor_observation", "action", "reward", "ppo", "training", "curriculum")
    missing = [name for name in required if name not in cfg]
    if missing:
        raise ValueError(f"missing configuration sections: {missing}")

    experiment = cfg["experiment"]
    if int(experiment["num_agents"]) < 1:
        raise ValueError("experiment.num_agents must be >= 1")
    _positive(experiment, "control_hz")

    world = cfg["world"]
    size = world["size_m"]
    if len(size) != 3 or any(float(v) <= 0.0 for v in size):
        raise ValueError("world.size_m must contain three positive values")
    _positive(world, "voxel_resolution_m")
    _positive(world, "obstacle_inflation_m")

    camera = cfg["camera"]
    if float(camera["min_depth_m"]) >= float(camera["max_depth_m"]):
        raise ValueError("camera.min_depth_m must be smaller than max_depth_m")
    if list(cfg["actor_observation"]["depth"]["resize"]) != [64, 40]:
        raise ValueError("deployment contract currently requires depth.resize=[64, 40]")

    limits = cfg["action"]["physical_limits"]
    for key in ("speed_norm_mps", "forward_mps", "backward_mps", "lateral_mps", "vertical_mps", "yaw_rate_rps"):
        _positive(limits, key)
    if float(limits["forward_mps"]) > float(limits["speed_norm_mps"]):
        raise ValueError("forward_mps cannot exceed speed_norm_mps")

    ppo = cfg["ppo"]
    if int(ppo["rollout_steps"]) < int(ppo["recurrent_sequence_length"]):
        raise ValueError("ppo.rollout_steps must be >= recurrent_sequence_length")
    if int(ppo["rollout_steps"]) % int(ppo["recurrent_sequence_length"]) != 0:
        raise ValueError("ppo.rollout_steps must be divisible by recurrent_sequence_length")

    near_obstacle = cfg["reward"]["near_obstacle"]
    _positive(near_obstacle, "safe_clearance_m")
    if bool(near_obstacle.get("speed_adaptive", False)):
        _positive(near_obstacle, "braking_deceleration_mps2")
        _positive(near_obstacle, "max_safe_clearance_m")


def apply_curriculum_stage(cfg: Mapping[str, Any], stage: str | int) -> Dict[str, Any]:
    result = copy.deepcopy(dict(cfg))
    stages = result["curriculum"]
    if isinstance(stage, str) and not stage.isdigit():
        matches = [item for item in stages if item["name"] == stage]
        if not matches:
            names = ", ".join(item["name"] for item in stages)
            raise ValueError(f"unknown curriculum stage '{stage}', expected one of: {names}")
        selected = matches[0]
    else:
        index = int(stage)
        if index < 0 or index >= len(stages):
            raise ValueError(f"curriculum stage index out of range: {index}")
        selected = stages[index]
    result["experiment"]["num_agents"] = int(selected["agents"])
    result["world"]["size_m"] = [float(v) for v in selected["world_size_m"]]
    result["training"]["curriculum_stage"] = selected["name"]
    return result

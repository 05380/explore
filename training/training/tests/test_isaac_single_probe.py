from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

TRAINING_PACKAGE = Path(__file__).resolve().parents[1]
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.isaac_single_env import (
    limit_vector_change,
    limit_velocity_for_braking_distance,
    limit_velocity_command,
    validate_probe_config,
    yaw_local_velocity_to_world,
)


def test_velocity_command_limiter_preserves_direction_and_clamps_yaw():
    command = torch.tensor([3.0, 4.0, 0.0, 2.5])
    original = command.clone()
    limited = limit_velocity_command(command, 2.0, 1.0)

    assert torch.equal(command, original)
    assert limited[:3].norm().item() == pytest.approx(2.0)
    assert limited[:2].tolist() == pytest.approx([1.2, 1.6])
    assert limited[3].item() == pytest.approx(1.0)


def test_velocity_command_limiter_keeps_commands_already_in_range():
    command = torch.tensor([[0.5, -0.4, 0.1, -0.3]])
    assert torch.allclose(limit_velocity_command(command, 2.0, 1.0), command)


def test_yaw_local_velocity_uses_forward_left_up_convention():
    velocity = torch.tensor([[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]])
    yaw = torch.tensor([torch.pi / 2.0, torch.pi / 2.0])
    world = yaw_local_velocity_to_world(velocity, yaw)
    assert world[0].tolist() == pytest.approx([0.0, 1.0, 0.2], abs=1e-6)
    assert world[1].tolist() == pytest.approx([-1.0, 0.0, -0.1], abs=1e-6)


def test_vector_change_limiter_applies_acceleration_bound():
    previous = torch.tensor([0.0, 0.0, 0.0])
    desired = torch.tensor([3.0, 4.0, 0.0])
    limited = limit_vector_change(previous, desired, max_change=0.5)
    assert limited.tolist() == pytest.approx([0.3, 0.4, 0.0])


def test_braking_distance_limiter_reduces_only_outward_components():
    position = torch.tensor([0.9, -0.9, 0.0])
    velocity = torch.tensor([2.0, -2.0, 0.5])
    lower = torch.tensor([-1.0, -1.0, -1.0])
    upper = torch.tensor([1.0, 1.0, 1.0])
    limited = limit_velocity_for_braking_distance(
        position, velocity, lower, upper, braking_acceleration_mps2=2.0
    )
    assert limited.tolist() == pytest.approx(
        [0.6324555, -0.6324555, 0.5], abs=1e-6
    )


def test_shipped_probe_configuration_is_valid():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "isaac_single.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    validate_probe_config(cfg)
    assert cfg["control"]["max_speed_mps"] == 2.0
    assert cfg["control"]["max_acceleration_mps2"] == 2.0
    assert cfg["control"]["control_hz"] == 20.0
    assert cfg["control"]["physics_steps_per_action"] == 6
    assert (
        cfg["sim"]["physics_dt"]
        * cfg["control"]["physics_steps_per_action"]
    ) == pytest.approx(1.0 / cfg["control"]["control_hz"])
    assert cfg["control"]["command_frame"] == "yaw_local"
    assert cfg["scene"]["size_m"] == [30.0, 30.0, 5.0]
    assert cfg["app"]["multi_gpu"] is False


def test_probe_scene_does_not_require_optional_orbit_extension():
    source_path = TRAINING_PACKAGE / "racer_rmappo" / "isaac_single_env.py"
    source = source_path.read_text(encoding="utf-8")
    assert "import omni.isaac.orbit" not in source

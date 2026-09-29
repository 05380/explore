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
from racer_rmappo.d455m_sensor import (
    depth_to_normalized_inverse,
    resize_inverse_depth_for_actor,
    summarize_depth,
    validate_d455m_config,
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
    validate_d455m_config(cfg["camera"])
    assert cfg["camera"]["fixed_to_body"] is True
    assert cfg["camera"]["max_depth_m"] == 20.0
    assert cfg["camera"]["actor_resize"] == [64, 40]
    assert cfg["navigation_backend"]["fixed_target_position_m"] == [2.5, 0.0, 1.5]
    assert cfg["navigation_backend"]["goal_position_tolerance_m"] == 0.5


def test_probe_scene_does_not_require_optional_orbit_extension():
    source_path = TRAINING_PACKAGE / "racer_rmappo" / "isaac_single_env.py"
    source = source_path.read_text(encoding="utf-8")
    assert "import omni.isaac.orbit" not in source


def test_d455m_inverse_depth_matches_smoke_contract_and_masks_invalid_values():
    depth = torch.tensor([[0.5, 0.9, 2.0, 20.0, 21.0, float("inf"), float("nan")]])
    inverse, valid = depth_to_normalized_inverse(depth, 0.9, 20.0)
    assert valid.tolist() == [[False, True, True, True, False, False, False]]
    assert inverse[0, 0].item() == 0.0
    assert inverse[0, 1].item() == pytest.approx(1.0)
    expected_two_m = ((1.0 / 2.0) - (1.0 / 20.0)) / (
        (1.0 / 0.9) - (1.0 / 20.0)
    )
    assert inverse[0, 2].item() == pytest.approx(expected_two_m)
    assert inverse[0, 3].item() == pytest.approx(0.0, abs=1e-7)
    assert torch.isfinite(inverse).all()


def test_d455m_actor_resize_preserves_thin_close_obstacle():
    inverse = torch.zeros(360, 640)
    inverse[123, 456] = 0.8
    resized = resize_inverse_depth_for_actor(inverse, 64, 40)
    assert resized.shape == (40, 64)
    assert resized.max().item() == pytest.approx(0.8)


def test_d455m_depth_summary_uses_center_patch_and_range_mask():
    depth = torch.full((9, 11), float("inf"))
    depth[3:6, 4:7] = 3.25
    summary = summarize_depth(depth, 0.9, 20.0, center_patch_px=3)
    assert summary["shape"] == [9, 11]
    assert summary["center_valid_fraction"] == pytest.approx(1.0)
    assert summary["center_median_m"] == pytest.approx(3.25)
    assert summary["valid_min_m"] == pytest.approx(3.25)
    assert summary["valid_max_m"] == pytest.approx(3.25)

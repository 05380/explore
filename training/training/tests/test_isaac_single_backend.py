from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")

TRAINING_PACKAGE = Path(__file__).resolve().parents[1]
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.config import apply_curriculum_stage, load_config
from racer_rmappo.isaac_adapter import validate_backend_shapes, validate_step_info
from racer_rmappo.isaac_single_backend import (
    IsaacSingleNavigationBackend,
    scale_navigation_action,
    world_to_yaw_local,
)


def load_isaac_config():
    path = Path(__file__).resolve().parents[2] / "configs" / "isaac_single.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class FakeDepthCamera:
    def __init__(self, cfg):
        self.cfg = cfg
        self.plane_depth_m = 4.0
        self.radial_depth_m = 4.0

    def capture(self, warmup_frames=None):
        shape = (int(self.cfg["height"]), int(self.cfg["width"]))
        return {
            "distance_to_image_plane": torch.full(shape, self.plane_depth_m),
            "distance_to_camera": torch.full(shape, self.radial_depth_m),
        }


class FakePhysicalProbe:
    def __init__(self, isaac_cfg):
        self.device = torch.device("cpu")
        self.dt = 1.0 / float(isaac_cfg["control"]["control_hz"])
        self.depth_camera = FakeDepthCamera(isaac_cfg["camera"])
        self.position = torch.tensor(isaac_cfg["scene"]["spawn_position_m"]).float()
        self.velocity = torch.zeros(3)
        self.yaw = 0.0
        self.closed = False

    def _telemetry(self):
        half = 0.5 * self.yaw
        return {
            "position_m": self.position.tolist(),
            "orientation_wxyz": [math.cos(half), 0.0, 0.0, math.sin(half)],
            "linear_velocity_mps": self.velocity.tolist(),
            "angular_velocity_rps": [0.0, 0.0, 0.0],
            "speed_mps": float(self.velocity.norm()),
            "yaw_rad": self.yaw,
            "up_z": 1.0,
            "max_contact_force_n": 0.0,
            "collision": False,
            "out_of_bounds": False,
            "finite": True,
        }

    def reset(self):
        self.position = torch.tensor([0.0, 0.0, 1.5])
        self.velocity.zero_()
        self.yaw = 0.0
        return self._telemetry()

    def step(self, command):
        command = torch.as_tensor(command).float()
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        self.velocity = torch.tensor(
            [
                cosine * float(command[0]) - sine * float(command[1]),
                sine * float(command[0]) + cosine * float(command[1]),
                float(command[2]),
            ]
        )
        self.position += self.velocity * self.dt
        self.yaw = math.atan2(
            math.sin(self.yaw + float(command[3]) * self.dt),
            math.cos(self.yaw + float(command[3]) * self.dt),
        )
        return self._telemetry(), command

    def close(self):
        self.closed = True


def make_backend():
    cfg = apply_curriculum_stage(load_config(), "single_agent_sparse_static")
    cfg["training"]["num_parallel_swarms"] = 1
    isaac_cfg = load_isaac_config()
    probe = FakePhysicalProbe(isaac_cfg)
    return IsaacSingleNavigationBackend(cfg, isaac_cfg, probe), probe


def test_navigation_action_scaling_is_asymmetric_and_norm_limited():
    cfg = load_config()
    limits = cfg["action"]["physical_limits"]
    action = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]]
    )
    scaled = scale_navigation_action(action, limits)
    assert scaled[:, :3].norm(dim=-1).max() <= 2.0 + 1e-6
    assert scaled[0, 3].item() == pytest.approx(1.0)
    assert scaled[1, 0].item() == pytest.approx(-0.5)
    assert scaled[1, 3].item() == pytest.approx(-1.0)


def test_world_to_yaw_local_uses_forward_left_up_convention():
    local = world_to_yaw_local(
        torch.tensor([0.0, 1.0, 0.2]), torch.tensor(math.pi / 2.0)
    )
    assert local.tolist() == pytest.approx([1.0, 0.0, 0.2], abs=1e-6)


def test_backend_clearance_uses_validated_axial_depth_not_radial_annotator():
    backend, probe = make_backend()
    probe.depth_camera.plane_depth_m = 4.0
    probe.depth_camera.radial_depth_m = 1.25
    backend.reset()
    assert backend.last_clearance_m.item() == pytest.approx(4.0)


def test_single_navigation_backend_matches_rmappo_contract():
    backend, _ = make_backend()
    validate_backend_shapes(backend)
    observation, critic = backend.reset()
    assert observation["depth"].shape == (1, 1, 3, 40, 64)
    assert observation["ego"].shape == (1, 1, 11)
    assert observation["target"].shape == (1, 1, 7)
    assert observation["neighbors"].shape == (1, 1, 5, 8)
    assert observation["candidates"].shape == (1, 1, 16, 9)
    assert observation["candidates"][0, 0, 0, -1].item() == 1.0
    assert observation["candidates"][0, 0, 1:, -1].sum().item() == 0.0
    assert observation["decision_mask"].item() == 0.0
    assert critic.shape == (1, 1, 13)

    action = torch.zeros(1, 1, 9)
    action[..., 0] = 0.25
    next_observation, next_critic, reward, done, info = backend.step(action)
    validate_step_info(info, 1)
    assert next_observation["depth"].shape == (1, 1, 3, 40, 64)
    assert next_critic.shape == (1, 1, 13)
    assert reward.shape == (1, 1)
    assert done.shape == (1,)
    assert torch.isfinite(reward).all()
    assert info["command_body"][0].item() == pytest.approx(0.5)


def test_navigation_reached_and_observation_completed_are_separate_events():
    backend, probe = make_backend()
    backend.reset()
    backend.target_position.copy_(probe.position)
    backend.target_yaw.fill_(math.pi)

    action = torch.zeros(1, 1, 9)
    _, _, _, done, info = backend.step(action)
    assert done.item() is False
    assert info["navigation_reached"].item() == 1.0
    assert info["observation_completed"].item() == 0.0

    probe.yaw = math.pi
    backend.last_telemetry = probe._telemetry()
    _, _, _, done, info = backend.step(action)
    assert done.item() is True
    assert info["navigation_reached"].item() == 1.0
    assert info["observation_completed"].item() == 1.0
    assert info["episode_success"].item() == 1.0

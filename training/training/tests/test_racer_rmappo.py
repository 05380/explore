from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

TRAINING_PACKAGE = Path(__file__).resolve().parents[1]
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.config import apply_curriculum_stage, load_config
from racer_rmappo.isaac_adapter import validate_backend_shapes
from racer_rmappo.model import CentralizedCritic, SharedRecurrentActor
from racer_rmappo.reward import RewardComposer
from racer_rmappo.smoke_env import ContractSmokeEnv
from racer_rmappo.storage import generalized_advantage_estimate


def small_config():
    cfg = apply_curriculum_stage(load_config(), "four_agent_dense_static")
    cfg["training"]["num_parallel_swarms"] = 2
    cfg["training"]["smoke_max_obstacles"] = 3
    cfg["training"]["smoke_episode_steps"] = 8
    cfg["world"]["voxel_resolution_m"] = 2.0
    return cfg


def test_contract_parameters_are_synchronized():
    cfg = load_config()
    assert cfg["world"]["voxel_resolution_m"] == 0.5
    assert cfg["world"]["obstacle_inflation_m"] == 0.5
    assert cfg["camera"]["max_depth_m"] == 20.0
    assert cfg["action"]["physical_limits"]["speed_norm_mps"] == 2.0


def test_actor_critic_shapes():
    cfg = small_config()
    env = ContractSmokeEnv(cfg, "cpu")
    observation, state = env.reset()
    actor = SharedRecurrentActor(hidden_size=32)
    critic = CentralizedCritic(hidden_size=32)
    hidden = actor.initial_hidden(env.num_envs, env.num_agents, "cpu")
    action, log_prob, entropy, next_hidden = actor.step(observation, hidden)
    value = critic(state)
    assert action.shape == (2, 4, 4)
    assert log_prob.shape == entropy.shape == value.shape == (2, 4)
    assert next_hidden.shape == (2, 4, 32)
    assert torch.all(action.abs() <= 1.0)


def test_smoke_backend_action_speed_and_shapes():
    cfg = small_config()
    env = ContractSmokeEnv(cfg, "cpu")
    validate_backend_shapes(env)
    action = torch.ones(env.num_envs, env.num_agents, 4)
    _, _, reward, done, info = env.step(action)
    assert env.velocities.norm(dim=-1).max() <= 2.0 + 1e-5
    assert reward.shape == (2, 4)
    assert done.shape == (2,)
    assert set(info["reward_components"]) >= {"collision", "near_obstacle", "target_progress"}


def test_voxel_reward_is_capped():
    cfg = load_config()
    composer = RewardComposer(cfg["reward"])
    shape = (1, 1)
    signals = {
        "target_progress_m": torch.zeros(shape),
        "goal_reached": torch.zeros(shape, dtype=torch.bool),
        "local_new_voxels": torch.full(shape, 100000.0),
        "team_unique_new_voxels": torch.full(shape, 100000.0),
        "duplicate_voxels": torch.full(shape, 100000.0),
        "obstacle_clearance_m": torch.full(shape, 20.0),
        "speed_mps": torch.zeros(shape),
        "nearest_drone_m": torch.full(shape, 30.0),
        "action_delta_l2": torch.zeros(shape),
        "vertical_action_l2": torch.zeros(shape),
        "stall": torch.zeros(shape, dtype=torch.bool),
        "collision": torch.zeros(shape, dtype=torch.bool),
        "out_of_bounds": torch.zeros(shape, dtype=torch.bool),
    }
    _, components = composer(signals)
    assert components["local_new_voxels"].item() == pytest.approx(1.0)
    assert components["team_new_voxels"].item() == pytest.approx(1.0)
    assert components["duplicate_voxels"].item() == pytest.approx(-0.25)


def test_gae_stops_at_terminal_transition():
    rewards = torch.tensor([[[1.0]], [[10.0]]])
    values = torch.zeros_like(rewards)
    dones = torch.tensor([[[True]], [[False]]])
    advantage, returns = generalized_advantage_estimate(
        rewards, values, dones, torch.zeros(1, 1), gamma=0.99, gae_lambda=0.95
    )
    assert advantage[0].item() == pytest.approx(1.0)
    assert returns.shape == rewards.shape

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("yaml")

TRAINING_PACKAGE = Path(__file__).resolve().parents[1]
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))
SCRIPTS_PACKAGE = TRAINING_PACKAGE / "scripts"
if str(SCRIPTS_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PACKAGE))

from racer_rmappo.config import apply_curriculum_stage, load_config
from racer_rmappo.isaac_adapter import validate_backend_shapes, validate_step_info
from racer_rmappo.isaac_scenarios import apply_navigation_scenario
from racer_rmappo.model import CentralizedCritic, SharedRecurrentActor
from racer_rmappo.reward import RewardComposer
from racer_rmappo.smoke_env import ContractSmokeEnv
from racer_rmappo.storage import generalized_advantage_estimate
from check_ros_training_config import check_contract


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
    assert cfg["camera"]["fixed_to_body"] is True
    assert cfg["action"]["physical_limits"]["speed_norm_mps"] == 2.0
    assert cfg["actor_observation"]["racer_candidates"]["max_candidates"] == 16
    assert cfg["actor_observation"]["racer_candidates"]["visible_gain_normalizer"] == 20000.0
    assert cfg["action"]["viewpoint_selection"]["max_position_offset_m"] == [1.0, 1.0, 0.5]
    assert check_contract(cfg)["valid"] is True


def test_navigation_curriculum_and_isaac_scenarios_are_explicit():
    isaac_path = Path(__file__).resolve().parents[2] / "configs" / "isaac_single.yaml"
    import yaml

    base = yaml.safe_load(isaac_path.read_text(encoding="utf-8"))
    assert base["navigation_backend"]["fixed_target_position_m"] == [2.5, 0.0, 1.5]
    expected = {
        "open_target": ("single_agent_open_target", [2.0, -2.0, 1.8]),
        "wall_edge": ("single_agent_wall_edge", [5.0, -3.5, 1.8]),
        "wall_avoidance": ("single_agent_wall_avoidance", [6.0, 0.0, 1.5]),
    }
    for scenario_name, (stage_name, target) in expected.items():
        cfg = apply_curriculum_stage(load_config(), stage_name)
        assert cfg["experiment"]["num_agents"] == 1
        assert cfg["training"]["curriculum_stage"] == stage_name
        active = apply_navigation_scenario(base, scenario_name)
        assert active["navigation_backend"]["fixed_target_position_m"] == target
        assert active["navigation_backend"]["active_scenario"] == scenario_name
        assert (
            active["navigation_backend"]["scenarios"][scenario_name][
                "rmappo_stage"
            ]
            == stage_name
        )

    invalid = copy.deepcopy(base)
    invalid["navigation_backend"]["scenarios"]["open_target"][
        "fixed_target_position_m"
    ] = [6.0, 0.0, 1.5]
    with pytest.raises(ValueError, match="direct path violates"):
        apply_navigation_scenario(invalid, "open_target")


def test_actor_critic_shapes():
    cfg = small_config()
    env = ContractSmokeEnv(cfg, "cpu")
    observation, state = env.reset()
    actor = SharedRecurrentActor(hidden_size=32)
    critic = CentralizedCritic(hidden_size=32)
    hidden = actor.initial_hidden(env.num_envs, env.num_agents, "cpu")
    action, log_prob, entropy, next_hidden = actor.step(observation, hidden)
    value = critic(state)
    assert action.shape == (2, 4, 9)
    assert log_prob.shape == entropy.shape == value.shape == (2, 4)
    assert next_hidden.shape == (2, 4, 32)
    assert torch.all(action[..., :4].abs() <= 1.0)
    assert torch.all(action[..., 5:].abs() <= 1.0)
    assert observation["candidates"].shape == (2, 4, 16, 9)
    assert observation["decision_mask"].shape == (2, 4, 1)
    chosen = action[..., 4].round().long().unsqueeze(-1)
    chosen_valid = torch.gather(observation["candidates"][..., -1], -1, chosen)
    assert torch.all(chosen_valid == 1.0)


def test_viewpoint_decision_is_event_driven():
    cfg = small_config()
    env = ContractSmokeEnv(cfg, "cpu")
    observation, _ = env.reset()
    assert torch.all(observation["decision_mask"] == 1.0)
    action = torch.zeros(env.num_envs, env.num_agents, 9)
    action[..., 4] = 0.0
    next_observation, _, _, _, info = env.step(action)
    assert info["viewpoint_decisions"].sum() == env.num_envs * env.num_agents
    # A new decision is raised once per target reached in this step.
    assert next_observation["decision_mask"].sum() == info["goal_reached"].sum()


def test_smoke_backend_action_speed_and_shapes():
    cfg = small_config()
    env = ContractSmokeEnv(cfg, "cpu")
    validate_backend_shapes(env)
    action = torch.zeros(env.num_envs, env.num_agents, 9)
    action[..., :4] = 1.0
    _, _, reward, done, info = env.step(action)
    validate_step_info(info, env.num_envs)
    assert env.velocities.norm(dim=-1).max() <= 2.0 + 1e-5
    assert reward.shape == (2, 4)
    assert done.shape == (2,)
    assert set(info["reward_components"]) >= {"collision", "near_obstacle", "target_progress"}


def test_body_fixed_camera_frustum_updates_seen_voxels():
    cfg = apply_curriculum_stage(load_config(), "single_agent_sparse_static")
    cfg["training"]["num_parallel_swarms"] = 1
    cfg["training"]["smoke_max_obstacles"] = 1
    cfg["world"]["voxel_resolution_m"] = 1.0
    env = ContractSmokeEnv(cfg, "cpu")
    env.positions[0, 0] = torch.tensor([0.0, 0.0, 1.5])
    env.yaw[0, 0] = 0.0
    env.obstacle_position[0, 0] = torch.tensor([5.0, 0.0])
    env.obstacle_radius[0, 0] = 1.0
    env.obstacle_height[0, 0] = 3.0

    forward_depth = env._render_inverse_depth()
    forward_voxels = env._visible_voxel_indices(forward_depth)[0][0]
    assert forward_depth.sum() > 0.0
    assert forward_voxels.numel() > 1
    env._update_seen(forward_depth)
    coverage_after_forward = env._coverage().item()
    assert coverage_after_forward > 0.0

    env.yaw[0, 0] = torch.pi
    backward_depth = env._render_inverse_depth()
    backward_voxels = env._visible_voxel_indices(backward_depth)[0][0]
    assert backward_depth.sum() == 0.0
    assert backward_voxels.numel() == 0


def test_viewpoint_requires_position_and_body_yaw():
    cfg = apply_curriculum_stage(load_config(), "single_agent_sparse_static")
    cfg["training"]["num_parallel_swarms"] = 1
    cfg["training"]["smoke_max_obstacles"] = 1
    cfg["training"]["coverage_success_threshold"] = 1.0
    env = ContractSmokeEnv(cfg, "cpu")
    env.task_decision.zero_()
    env.targets.copy_(env.positions)
    env.target_yaw.copy_(env.yaw + torch.pi)
    env.obstacle_position[0, 0] = torch.tensor([10.0, 10.0])
    env.obstacle_radius[0, 0] = 0.5
    env.obstacle_height[0, 0] = 2.0
    action = torch.zeros(1, 1, 9)

    observation, _, _, _, info = env.step(action)
    assert info["navigation_reached"].item() == 1.0
    assert info["goal_reached"].item() == 0.0
    assert observation["decision_mask"].item() == 0.0

    env.yaw.copy_(env.target_yaw)
    observation, _, _, _, info = env.step(action)
    assert info["goal_reached"].item() == 1.0
    assert observation["decision_mask"].item() == 1.0


def test_episode_metrics_separate_collision_causes():
    cfg = apply_curriculum_stage(load_config(), "single_agent_sparse_static")
    cfg["training"]["num_parallel_swarms"] = 1
    cfg["training"]["smoke_max_obstacles"] = 1
    env = ContractSmokeEnv(cfg, "cpu")
    env.obstacle_position[0, 0] = env.positions[0, 0, :2]
    env.obstacle_radius[0, 0] = 1.0
    env.obstacle_height[0, 0] = 3.0
    action = torch.zeros(1, 1, 9)
    _, _, _, done, info = env.step(action)
    assert done.item() is True
    assert info["episode_finished"].item() == 1.0
    assert info["episode_collision"].item() == 1.0
    assert info["episode_obstacle_collision"].item() == 1.0
    assert info["episode_inter_drone_collision"].item() == 0.0


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

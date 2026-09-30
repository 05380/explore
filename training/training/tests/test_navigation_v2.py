"""Pure-navigation contracts and rule/timing regressions; no Isaac required."""
from __future__ import annotations
import copy
from dataclasses import replace
import math
from pathlib import Path
import sys
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from racer_rmappo.config import load_config, apply_curriculum_stage
from racer_rmappo.model import NavigationRecurrentActor, SharedRecurrentActor
from racer_rmappo.physics_timing import PhysicsTimingAudit
from racer_rmappo.policy_contract import checkpoint_policy, policy_spec, NAVIGATION, LEGACY
from racer_rmappo.rule_goals import GoalCandidate, RuleGoalSelector, goal_events
from racer_rmappo.smoke_env import ContractSmokeEnv
from racer_rmappo.trainer import RMAPPOTrainer


def candidate(name="a", **kw):
    base = GoalCandidate(name, (2., 0., 1.5), 0., 10., 2., True, True, True, True)
    return replace(base, **kw)


@pytest.mark.parametrize("field", ["known_free", "inflated_safe", "connected", "in_task_region"])
def test_selector_filters_unsafe_candidates(field):
    rule = RuleGoalSelector()
    assert rule.select([candidate(**{field: False})], (0,0,1.5), 0, 0) is None
    assert rule.status == "waiting"


def test_selector_score_stable_id_lock_cooldown_and_empty():
    rule = RuleGoalSelector()
    a, b = candidate("a"), candidate("b")
    assert rule.select([b, a], (0,0,1.5), 0, 0).goal_id == "a"
    assert rule.select([replace(b, estimated_gain=1000)], (0,0,1.5), 0, 1).goal_id == "a"
    assert rule.select([a, b], (0,0,1.5), 0, 2, "stall").goal_id == "b"
    assert rule.select([a], (0,0,1.5), 0, 3, "completed") is None
    assert rule.select([a], (0,0,1.5), 0, 12).goal_id == "a"
    assert rule.select([], (0,0,1.5), 0, 13, "invalid") is None


def test_selector_uses_path_yaw_and_eight_metre_limit():
    rule = RuleGoalSelector()
    assert rule.select([candidate("long", path_length_m=20), candidate("turn", yaw_rad=math.pi),
                        candidate("far", position_m=(9,0,1.5), estimated_gain=10000),
                        candidate("near")], (0,0,1.5), 0, 0).goal_id == "near"
    rule.reset()
    assert rule.select([candidate(inflation_m=.49)], (0,0,1.5), 0, 0) is None
    assert rule.select([candidate(position_m=(0,0,1.5), path_length_m=0)], (0,0,1.5), 0, 0)


def test_navigation_pose_is_not_map_observation():
    assert not goal_events(.1, 1., 0.)["navigation_pose_reached"]
    events = goal_events(.1, 0., 0., new_frame_processed=True)
    assert events["navigation_pose_reached"] and not events["observation_completed"]
    assert goal_events(.1, 0., 0., new_frame_fused=True)["observation_completed"]
    assert goal_events(5., 1., 0., frontier_covered=True)["observation_completed"]


def test_actual_physics_callbacks_and_render_guard():
    audit = PhysicsTimingAudit(1/120, 6)
    before = audit.snapshot()
    for _ in range(6):
        audit.controller_updates += 1
        audit.force_applications += 1
        audit.on_physics_step(1/120)
    assert audit.verify_action(before)["physics_time_s"] == pytest.approx(.05)
    audit.render_only(lambda: None)
    assert audit.report()["valid"]
    with pytest.raises(RuntimeError, match="Render"):
        audit.render_only(lambda: audit.on_physics_step(1/120))
    assert not audit.report()["valid"]


def test_missing_or_extra_physics_steps_rejected():
    for steps in (0, 5, 7, 11):
        audit = PhysicsTimingAudit(1/120, 6)
        before = audit.snapshot()
        audit.controller_updates = audit.force_applications = 6
        for _ in range(steps):
            audit.on_physics_step(1/120)
        with pytest.raises(RuntimeError, match="timing"):
            audit.verify_action(before)


def small_cfg(tmp_path, version=NAVIGATION):
    cfg = apply_curriculum_stage(load_config(), "single_agent_sparse_static")
    cfg["policy"] = {"version": version}
    cfg["training"].update(device="cpu", num_parallel_swarms=1, smoke_max_obstacles=2,
                           smoke_episode_steps=4, output_dir=str(tmp_path), backend="smoke")
    cfg["world"]["voxel_resolution_m"] = 2.
    cfg["ppo"].update(recurrent_hidden_size=32, rollout_steps=8, recurrent_sequence_length=4,
                      epochs=1, minibatches=1)
    return cfg


def test_v2_forward_logprob_and_recurrent_sequence(tmp_path):
    cfg = small_cfg(tmp_path)
    env = ContractSmokeEnv(cfg, "cpu")
    obs, _ = env.reset()
    assert set(obs) == {"depth", "ego", "target", "neighbors"}
    actor = NavigationRecurrentActor(hidden_size=32)
    assert not any("candidate" in n or "residual" in n for n, _ in actor.named_parameters())
    hidden = actor.initial_hidden(1, 1, "cpu")
    action, lp, entropy, after = actor.step(obs, hidden)
    assert action.shape == (1,1,4)
    check_lp, _, _ = actor.evaluate_actions(obs, hidden, action)
    assert torch.allclose(check_lp, lp, atol=1e-4)
    assert torch.isfinite(lp).all() and torch.isfinite(entropy).all()
    sequence = {k: v.unsqueeze(0).expand(2, *v.shape) for k,v in obs.items()}
    actions = action.unsqueeze(0).expand(2, *action.shape)
    logs, _ = actor.evaluate_sequence(sequence, torch.ones_like(hidden), actions,
                                     torch.ones(2,1,1,dtype=torch.bool))
    assert torch.allclose(logs[0], logs[1])
    _, _, _, _, info = env.step(action)
    assert info["reward_components"]["viewpoint_gain_prior"].item() == 0
    with pytest.raises(ValueError):
        env.step(torch.zeros(1,1,9))


def test_checkpoint_roundtrip_version_and_short_ppo_update(tmp_path):
    trainer = RMAPPOTrainer(small_cfg(tmp_path / "new"))
    path = trainer.train(8)
    saved = torch.load(path, map_location="cpu")
    assert saved["policy_spec"] == policy_spec(NAVIGATION)
    clone = RMAPPOTrainer(small_cfg(tmp_path / "clone"))
    clone.load(path)
    for k,v in trainer.actor.state_dict().items():
        assert torch.equal(v, clone.actor.state_dict()[k])
    assert clone.evaluate(4)["episode_count"] >= 1
    legacy = RMAPPOTrainer(small_cfg(tmp_path / "legacy", LEGACY))
    old = legacy.save("old.pt")
    raw = torch.load(old)
    raw.pop("policy_spec")
    torch.save(raw, old)
    assert checkpoint_policy(raw) == LEGACY
    legacy.load(old, load_optimizer=False)
    assert legacy.evaluate(4)["episode_count"] >= 1
    with pytest.raises(ValueError, match="checkpoint"):
        clone.load(old)
    with pytest.raises(ValueError, match="evaluation-only"):
        legacy.load(old)


@pytest.mark.parametrize("render", [False, True])
def test_probe_step_has_one_controller_and_force_per_physics_tick(render):
    from types import SimpleNamespace
    from racer_rmappo.isaac_single_env import IsaacSingleDroneProbe
    probe = IsaacSingleDroneProbe.__new__(IsaacSingleDroneProbe)
    probe.device = torch.device("cpu")
    probe.physics_dt = 1/120
    probe.physics_steps_per_action = 6
    probe.timing = PhysicsTimingAudit(probe.physics_dt, 6)
    probe.max_speed, probe.max_yaw_rate = 2., 1.
    probe.max_acceleration = probe.max_yaw_acceleration = 2.
    probe.reference_margin, probe.contact_threshold = 1., .1
    probe.bounds_min = torch.tensor([-10., -10., .5])
    probe.bounds_max = torch.tensor([10., 10., 4.5])
    probe.target_position = torch.tensor([[0.,0.,1.5]])
    probe.target_yaw = torch.zeros(1)
    probe.last_world_velocity_command = torch.zeros(3)
    probe.last_yaw_rate_command = torch.zeros(1)
    probe.render_on_step = render
    state = torch.zeros(1,1,19)
    state[..., 2], state[..., 3], state[..., 18] = 1.5, 1., 1.
    calls = []
    def physics_step(render):
        calls.append(render)
        probe.timing.on_physics_step(probe.physics_dt)
    probe.sim = SimpleNamespace(step=physics_step, render=lambda: None)
    probe.drone = SimpleNamespace(get_state=lambda **kw: state,
        base_link=SimpleNamespace(get_net_contact_forces=lambda **kw: torch.zeros(1,1,3)),
        apply_action=lambda action: None, thrusts=torch.zeros(1,1,4,3))
    probe.controller = lambda *args, **kwargs: torch.zeros(1,4)
    telemetry, _ = probe.step((.5,0.,0.,0.))
    assert calls == [False]*6
    assert telemetry["timing_step"]["controller_updates"] == 6
    assert telemetry["timing_step"]["force_applications"] == 6
    assert probe.timing.render_calls == int(render)


def test_compare_three_native_mode_reports_detects_trajectory_drift():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from diagnose_isaac_timing import compare_reports
    row = dict(position_m=[0.,0.,1.5], velocity_world_mps=[0.,0.,0.],
               timing_step=dict(physics_steps=6, physics_time_s=.05))
    report = dict(config={}, passed=True, phases=dict(hover=[row], sequence=[row]))
    reports = [dict(copy.deepcopy(report), mode=mode) for mode in ("headless", "gui", "gui_overlay")]
    assert compare_reports(reports)["passed"]
    reports[1]["phases"]["sequence"][0]["position_m"][2] -= .5
    assert not compare_reports(reports)["passed"]


@pytest.mark.parametrize("agents", [1, 4, 16])
def test_shared_navigation_actor_supports_multiagent_shapes(tmp_path, agents):
    cfg = small_cfg(tmp_path)
    cfg["experiment"]["num_agents"] = agents
    cfg["training"]["num_parallel_swarms"] = 2
    env = ContractSmokeEnv(cfg, "cpu")
    obs, _ = env.reset()
    actor = NavigationRecurrentActor(hidden_size=32)
    output, lp, _, hidden = actor.step(obs, actor.initial_hidden(2, agents, "cpu"))
    assert output.shape == (2,agents,4)
    assert hidden.shape == (2,agents,32)
    assert torch.isfinite(lp).all()

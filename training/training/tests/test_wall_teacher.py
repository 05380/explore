import pathlib
import sys
import unittest
from types import SimpleNamespace

import torch


SCRIPT_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from demonstration_buffer import DemonstrationBuffer
from wall_follow_teacher import TeacherRolloutPolicy, WallFollowTeacher


def make_line_fitter():
    teacher = WallFollowTeacher.__new__(WallFollowTeacher)
    teacher.device = torch.device("cpu")
    teacher.fit_min_inliers = 5
    teacher.fit_residual_threshold = 0.06
    teacher.fit_min_pair_separation = 0.2
    teacher.fit_max_hypotheses = 256
    teacher.fit_min_span = 1.4
    return teacher


class WallGeometryTest(unittest.TestCase):
    def test_long_wall_is_fit_but_compact_face_is_rejected(self):
        teacher = make_line_fitter()
        y = torch.linspace(-2.0, 2.0, 17)
        wall = torch.stack([1.2 + 0.01 * torch.sin(3.0 * y), y], dim=-1)
        wall = torch.cat([wall, torch.tensor([[0.1, 2.7], [-0.4, -2.8]])], dim=0)

        fit = teacher._fit_one_line(wall)
        self.assertIsNotNone(fit)
        self.assertGreater(float(fit.span), 3.5)
        self.assertGreater(float(torch.abs(fit.normal[0])), 0.98)

        compact_y = torch.linspace(-0.4, 0.4, 7)
        compact_face = torch.stack([torch.ones_like(compact_y), compact_y], dim=-1)
        self.assertIsNone(teacher._fit_one_line(compact_face))

    def test_l_shape_produces_two_dominant_boundaries(self):
        teacher = make_line_fitter()
        vertical_y = torch.linspace(-2.2, 1.0, 15)
        horizontal_x = torch.linspace(1.0, 3.4, 12)
        vertical = torch.stack([torch.ones_like(vertical_y), vertical_y], dim=-1)
        horizontal = torch.stack([horizontal_x, torch.ones_like(horizontal_x)], dim=-1)

        fits = teacher.fit_lines_xy(torch.cat([vertical, horizontal], dim=0), max_lines=2)
        self.assertEqual(len(fits), 2)
        alignment = float(torch.abs(fits[0].tangent.dot(fits[1].tangent)))
        self.assertLess(alignment, 0.15)


class StaticSegmentationAndSweepTest(unittest.TestCase):
    def test_safety_mask_keeps_static_hits_outside_fit_elevations(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.lidar_range = 4.0
        teacher.ray_directions = torch.zeros(2, 4, 3)
        teacher.ray_directions[..., 0] = 1.0
        teacher.fit_elevation_indices = torch.tensor([1, 2])
        teacher.fit_min_distance = 0.3
        teacher.fit_max_distance = 4.0
        teacher.fit_ground_min_z = -0.35
        teacher.dynamic_width_resolution = 0.25
        teacher.dynamic_spanning_height = 4.0
        teacher.dynamic_point_margin = 0.25

        lidar = torch.zeros(1, 2, 4)
        lidar[0, 0, 0] = 2.0
        lidar[0, 1, 1] = 2.0
        dynamic_obstacle = torch.zeros(5, 10)

        _, fit_valid, safety_valid = teacher._reconstruct_static_point_grid(
            lidar,
            dynamic_obstacle,
        )

        self.assertEqual(int(fit_valid.sum()), 1)
        self.assertEqual(int(safety_valid.sum()), 2)

    def test_disconnected_returns_are_split_before_line_fitting(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.segment_max_range_jump = 0.65
        teacher.segment_max_point_gap = 0.85
        teacher.segment_min_points = 5

        points = torch.zeros(8, 2, 3)
        valid = torch.zeros(8, 2, dtype=torch.bool)
        for beam_index in range(3):
            points[beam_index, :, 0] = 1.0
            points[beam_index, :, 1] = 0.15 * beam_index
            points[beam_index, 1, 2] = 0.2
            valid[beam_index] = True
        for beam_index in range(3, 6):
            points[beam_index, :, 0] = 3.0
            points[beam_index, :, 1] = 2.0 + 0.15 * (beam_index - 3)
            points[beam_index, 1, 2] = 0.2
            valid[beam_index] = True

        segments = teacher.segment_static_points(points, valid)

        self.assertEqual(len(segments), 2)
        self.assertEqual([int(segment.shape[0]) for segment in segments], [6, 6])

    def test_disconnected_collinear_clusters_do_not_form_one_long_wall(self):
        teacher = make_line_fitter()
        teacher.segment_max_range_jump = 0.65
        teacher.segment_max_point_gap = 0.85
        teacher.segment_min_points = 5

        points = torch.zeros(7, 2, 3)
        valid = torch.zeros(7, 2, dtype=torch.bool)
        for beam_index, y_position in ((0, 0.0), (1, 0.3), (2, 0.6)):
            points[beam_index, :, :2] = torch.tensor([1.0, y_position])
            valid[beam_index] = True
        for beam_index, y_position in ((4, 2.4), (5, 2.7), (6, 3.0)):
            points[beam_index, :, :2] = torch.tensor([1.0, y_position])
            valid[beam_index] = True

        pooled_points = points[valid][:, :2]
        self.assertIsNotNone(teacher._fit_one_line(pooled_points))

        segments = teacher.segment_static_points(points, valid)
        segmented_fits = teacher.fit_segmented_lines_xy(segments)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segmented_fits, [])

    def test_scan_segments_merge_across_azimuth_wraparound(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.segment_max_range_jump = 0.65
        teacher.segment_max_point_gap = 0.85
        teacher.segment_min_points = 5

        points = torch.zeros(6, 2, 3)
        valid = torch.zeros(6, 2, dtype=torch.bool)
        for beam_index, y_position in ((4, -0.2), (5, -0.1), (0, 0.0), (1, 0.1)):
            points[beam_index, :, 0] = 1.0
            points[beam_index, :, 1] = y_position
            points[beam_index, 1, 2] = 0.2
            valid[beam_index] = True

        segments = teacher.segment_static_points(points, valid)

        self.assertEqual(len(segments), 1)
        self.assertEqual(int(segments[0].shape[0]), 8)

    def test_static_sweep_rejects_forward_motion_but_accepts_side_step(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.dt = 0.05
        teacher.static_sweep_horizon = 1.0
        teacher.static_sweep_steps = 10
        teacher.lidar_range = 4.0
        static_points = torch.tensor([[0.8, 0.0, 0.0]])

        forward_clearance = teacher._static_sweep_clearance(
            torch.tensor([1.0, 0.0, 0.0]),
            static_points,
        )
        lateral_clearance = teacher._static_sweep_clearance(
            torch.tensor([0.0, 1.0, 0.0]),
            static_points,
        )

        self.assertLess(float(forward_clearance), 0.1)
        self.assertGreater(float(lateral_clearance), 0.75)

    def test_static_sweep_uses_three_dimensional_point_clearance(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.dt = 0.05
        teacher.static_sweep_horizon = 1.0
        teacher.static_sweep_steps = 10
        teacher.lidar_range = 4.0

        clearance = teacher._static_sweep_clearance(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([[0.2, 0.0, 1.0]]),
        )

        self.assertGreater(float(clearance), 0.95)

    def test_blocked_tangent_selects_outward_teacher_candidate(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.action_limit = 2.0
        teacher.dynamic_risk_limit = 0.0
        teacher.static_safety_radius = 0.40
        teacher.static_away_speeds = (1.0,)
        teacher.demo_min_action_speed = 0.35
        teacher.dt = 0.05
        teacher.static_sweep_horizon = 1.0
        teacher.static_sweep_steps = 10
        teacher.lidar_range = 4.0
        teacher._vo_risk = lambda action, dynamic: torch.tensor(0.0)

        action, _, safe, is_teacher = teacher._select_dynamic_safe_action(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([-0.2, 0.0, 0.0]),
            torch.empty(0),
            static_points=torch.tensor([[0.8, 0.0, 0.0]]),
            wall_normal=torch.tensor([0.0, 1.0]),
        )

        self.assertTrue(safe)
        self.assertTrue(is_teacher)
        self.assertLess(float(action[1]), -0.5)

    def test_safety_stop_is_not_written_as_teacher_demonstration(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.action_limit = 2.0
        teacher.dynamic_risk_limit = 0.0
        teacher.static_safety_radius = 0.40
        teacher.static_away_speeds = ()
        teacher.demo_min_action_speed = 0.35
        teacher.lidar_range = 4.0
        teacher._vo_risk = lambda action, dynamic: torch.where(
            action[:2].norm() > 1e-6,
            torch.tensor(0.5),
            torch.tensor(0.0),
        )

        action, _, safe, is_teacher = teacher._select_dynamic_safe_action(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([-0.2, 0.0, 0.0]),
            torch.empty(0),
        )

        self.assertTrue(safe)
        self.assertFalse(is_teacher)
        self.assertAlmostEqual(float(action[:2].norm()), 0.0, places=6)


class DynamicSafetyFallbackTest(unittest.TestCase):
    def test_vo_check_distinguishes_approaching_and_departing_obstacle(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.dynamic_width_resolution = 0.25
        teacher.dynamic_spanning_height = 4.0
        teacher.vo_xy_margin = 0.3
        teacher.vo_z_margin = 0.3
        teacher.vo_horizon = 2.1
        teacher.vo_tau = 0.75
        dynamic = torch.zeros(1, 10)
        dynamic[0, 0] = 1.0
        dynamic[0, 3] = 2.0
        dynamic[0, 8] = 1.0
        dynamic[0, 9] = 1.0

        dynamic[0, 5] = -1.0
        approaching_risk = teacher._vo_risk(torch.zeros(3), dynamic)
        dynamic[0, 5] = 1.0
        departing_risk = teacher._vo_risk(torch.zeros(3), dynamic)

        self.assertGreater(float(approaching_risk), 0.0)
        self.assertEqual(float(departing_risk), 0.0)

    def test_unsafe_teacher_candidates_fall_back_to_policy(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.action_limit = 2.0
        teacher.dynamic_risk_limit = 0.0
        teacher._vo_risk = lambda action, dynamic: torch.tensor(0.5)
        teacher_action = torch.tensor([0.8, 0.4, 0.0])
        policy_action = torch.tensor([-0.2, 0.3, 0.1])

        action, _, safe, is_teacher = teacher._select_dynamic_safe_action(
            teacher_action,
            policy_action,
            torch.empty(0),
        )
        self.assertTrue(torch.equal(action, policy_action))
        self.assertFalse(safe)
        self.assertFalse(is_teacher)


class TeacherVelocityTest(unittest.TestCase):
    def test_leave_phase_blends_wall_following_toward_goal(self):
        teacher = WallFollowTeacher.__new__(WallFollowTeacher)
        teacher.device = torch.device("cpu")
        teacher.reference_clearance = 1.0
        teacher.clearance_gain = 1.2
        teacher.max_away_wall_speed = 0.8
        teacher.max_toward_wall_speed = 0.2
        teacher.parallel_speed = 1.0
        teacher.leave_goal_speed = 1.2
        teacher.leave_goal_blend = 0.75
        teacher.goal_clear_steps = 8
        teacher.action_limit = 2.0
        teacher.height_gain = 0.0
        teacher.max_vertical_speed = 0.0
        teacher.action_smoothing = 0.0
        teacher.wall_distance = torch.tensor([1.0])
        teacher.wall_normal = torch.tensor([[1.0, 0.0]])
        teacher.wall_tangent = torch.tensor([[0.0, 1.0]])
        teacher.previous_teacher_action = torch.zeros(1, 3)
        teacher.clear_counter = torch.zeros(1, dtype=torch.long)
        state = torch.tensor([1.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])

        follow_action = teacher._teacher_velocity(0, state)
        teacher.clear_counter[0] = teacher.goal_clear_steps
        leave_action = teacher._teacher_velocity(0, state)

        self.assertAlmostEqual(float(follow_action[0]), 0.0, places=5)
        self.assertGreater(float(leave_action[0]), 0.8)
        self.assertLess(float(leave_action[1]), float(follow_action[1]))


class DemonstrationBufferTest(unittest.TestCase):
    def test_masked_ring_buffer_and_sampling(self):
        buffer = DemonstrationBuffer(capacity=4, balanced_sampling=True)
        count = 6
        observation = {
            "state": torch.arange(count * 8, dtype=torch.float32).reshape(count, 8),
            "lidar": torch.zeros(count, 1, 36, 4),
            "direction": torch.zeros(count, 1, 3),
            "dynamic_obstacle": torch.zeros(count, 1, 5, 10),
        }
        action = torch.full((count, 3), 0.5)
        confidence = torch.linspace(0.4, 0.9, count)
        phase = torch.tensor([0, 1, 1, 2, 2, 3])

        written = buffer.add(
            observation,
            action,
            confidence,
            phase,
            torch.tensor([True, False, True, True, False, True]),
        )
        self.assertEqual(written, 4)
        self.assertEqual(len(buffer), 4)

        batch = buffer.sample(8, "cpu")
        self.assertEqual(tuple(batch.batch_size), (8,))
        self.assertEqual(tuple(batch["teacher_action_normalized"].shape), (8, 3))
        self.assertEqual(tuple(batch["agents", "observation", "lidar"].shape), (8, 1, 36, 4))
        self.assertEqual(sum(buffer.phase_counts().values()), 4)

    def test_sqrt_phase_balancing_does_not_equalize_rare_records(self):
        torch.manual_seed(7)
        buffer = DemonstrationBuffer(
            capacity=128,
            balanced_sampling=True,
            phase_balance_exponent=0.5,
        )
        count = 101
        observation = {
            "state": torch.zeros(count, 8),
            "lidar": torch.zeros(count, 1, 36, 4),
            "direction": torch.zeros(count, 1, 3),
            "dynamic_obstacle": torch.zeros(count, 1, 5, 10),
        }
        phase = torch.cat(
            [torch.zeros(100, dtype=torch.long), torch.ones(1, dtype=torch.long)]
        )
        buffer.add(
            observation,
            torch.full((count, 3), 0.5),
            torch.ones(count),
            phase,
            torch.ones(count, dtype=torch.bool),
        )

        sampled = buffer.sample(10000, "cpu")["teacher_phase"]
        rare_fraction = float((sampled == 1).float().mean())
        self.assertGreater(rare_fraction, 0.04)
        self.assertLess(rare_fraction, 0.20)

    def test_pending_segment_is_written_only_when_committed(self):
        buffer = DemonstrationBuffer(capacity=16)
        wrapper = TeacherRolloutPolicy(
            base_policy=torch.nn.Linear(2, 2),
            teacher=SimpleNamespace(num_envs=1),
            demonstration_buffer=buffer,
        )
        self.assertTrue(any(name.startswith("base_policy.") for name, _ in wrapper.named_parameters()))
        observation = {
            "state": torch.zeros(1, 8),
            "lidar": torch.zeros(1, 1, 36, 4),
            "direction": torch.zeros(1, 1, 3),
            "dynamic_obstacle": torch.zeros(1, 1, 5, 10),
        }
        result = {
            "teacher_action_normalized": torch.full((1, 3), 0.5),
            "confidence": torch.ones(1),
            "phase": torch.zeros(1, dtype=torch.long),
        }

        wrapper._append_pending(0, observation, result)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(wrapper._commit_pending(0), 1)
        self.assertEqual(len(buffer), 1)


if __name__ == "__main__":
    unittest.main()

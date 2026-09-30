"""Standard-library tests: can run without Torch, pytest or Isaac installed."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "eval_visualization", Path(__file__).resolve().parents[1] / "racer_rmappo" / "eval_visualization.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EvaluationVisualizationTests(unittest.TestCase):
    def camera(self):
        return dict(forward_axis_body=[1, 0, 0], up_axis_body=[0, 0, 1],
                    mount_position_body_m=[.25, 0, 0], width=640, height=360,
                    fx=577.296, fy=570.840, cx=319.5, cy=179.5, max_depth_m=20)

    def test_frustum_rotates_with_yaw_and_pitch_and_mount(self):
        angle = math.pi / 2
        yaw = [math.cos(angle/2), 0, 0, math.sin(angle/2)]
        edges = MODULE.frustum_edges([1, 2, 3], yaw, self.camera(), 3)
        self.assertEqual(len(edges), 8)
        for x, expected in zip(edges[0][0], [1, 2.25, 3]):
            self.assertAlmostEqual(x, expected)
        center = [sum(edge[1][k] for edge in edges[:4])/4 for k in range(3)]
        for x, expected in zip(center, [1, 5.25, 3]):
            self.assertAlmostEqual(x, expected)
        pitched = MODULE.rotate_wxyz([math.cos(angle/2), 0, math.sin(angle/2), 0], [1, 0, 0])
        self.assertAlmostEqual(pitched[0], 0, places=6)
        self.assertAlmostEqual(pitched[2], -1, places=6)

    def test_bounds_have_twelve_edges_no_diagonals(self):
        edges = MODULE.box_edges([-1, -2, .5], [1, 2, 4.5])
        self.assertEqual(len(edges), 12)
        for a, b in edges:
            self.assertEqual(sum(x != y for x, y in zip(a, b)), 1)

    def test_trace_preserves_terminal_and_next_episode_identity(self):
        cfg = {"control": {"control_hz": 20, "max_speed_mps": 2}}
        with tempfile.TemporaryDirectory() as directory:
            trace = MODULE.EvaluationTrace(Path(directory)/"trace.jsonl", cfg, "model.pt", "open_target")
            snapshot = dict(episode_step=2, position_m=[1, 0, 1.5], actual_speed_mps=3.87,
                            navigation_reached=False, done=True, success=False, collision=False,
                            out_of_bounds=True, stall=False, timeout=False, target_distance_m=.7)
            trace(2, {"diagnostic_snapshot": snapshot})
            snapshot.update(episode_step=1, position_m=[0, 0, 1.5], actual_speed_mps=0,
                            done=False, out_of_bounds=False)
            trace(3, {"diagnostic_snapshot": snapshot})
            summary = trace.summary()
            trace.close()
            trace.close()
            rows = [json.loads(line) for line in trace.path.read_text().splitlines()]
        self.assertFalse(rows[0]["coverage_available"])
        self.assertEqual(rows[1]["episode"], 0)
        self.assertTrue(rows[1]["done"])
        self.assertEqual(rows[1]["position_m"], [1, 0, 1.5])
        self.assertEqual(rows[2]["episode"], 1)
        self.assertEqual(summary["overspeed_control_steps"], 1)
        self.assertAlmostEqual(summary["peak_speed_transition"]["actual_speed_mps"], 3.87)

    def test_viewer_does_not_connect_across_reset(self):
        # Skip native construction but exercise actual trail/HUD update logic.
        from collections import deque
        viewer = MODULE.IsaacEvaluationViewer.__new__(MODULE.IsaacEvaluationViewer)
        viewer.cfg = {"scene": {"spawn_position_m": [0, 0, 1.5]}}
        viewer.last_episode = 0
        viewer.last_position = (8, 8, 1.5)
        viewer.trail = deque(maxlen=10)
        viewer.interval = 100
        viewer.update(dict(episode=1, step=2, position_m=[.1, 0, 1.5], done=False, success=False))
        self.assertEqual(viewer.trail[-1][0], (0, 0, 1.5))
        self.assertEqual(viewer.trail[-1][1], (.1, 0, 1.5))


if __name__ == "__main__":
    unittest.main()

"""RACER-inspired goal selection, not a mapper or a trajectory controller.

Map providers must supply observed-map safety/connectivity certificates. This
module never queries simulator scene truth. Fixed curriculum goals explicitly
use a separate provider and must not be reported as frontier exploration.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple


@dataclass(frozen=True)
class GoalCandidate:
    candidate_id: str
    position_m: Tuple[float, float, float]
    yaw_rad: float
    estimated_gain: float
    path_length_m: float
    in_task_region: bool
    known_free: bool
    inflated_safe: bool
    connected: bool
    inflation_m: float = 0.5
    source: str = "observed_map"


@dataclass(frozen=True)
class LocalGoal:
    goal_id: str
    position_m: Tuple[float, float, float]
    yaw_rad: float
    source: str
    valid: bool = True
    switch_reason: str = "new_task"

    def __post_init__(self):
        if (not self.goal_id or len(self.position_m) != 3
                or not all(math.isfinite(v) for v in (*self.position_m, self.yaw_rad))):
            raise ValueError("goal requires an ID and finite XYZ/yaw")


class RuleGoalSelector:
    EVENTS = {"new_task", "completed", "invalid", "stall", "timeout", "task_reallocated"}
    FAILURES = {"invalid", "stall", "timeout"}

    def __init__(self, max_radius_m=8.0, inflation_m=0.5, cruise_mps=1.0,
                 yaw_rate_rps=1.0, overhead_s=1.0, cooldown_s=10.0):
        values = (max_radius_m, inflation_m, cruise_mps, yaw_rate_rps, overhead_s, cooldown_s)
        if not all(math.isfinite(x) and x > 0 for x in values):
            raise ValueError("selector parameters must be finite and positive")
        self.radius, self.inflation = max_radius_m, inflation_m
        self.cruise, self.yaw_rate = cruise_mps, yaw_rate_rps
        self.overhead, self.cooldown = overhead_s, cooldown_s
        self.current: Optional[LocalGoal] = None
        self.blocked_until = {}
        self.completed = set()
        self.status = "waiting"

    def reset(self):
        self.current = None
        self.blocked_until.clear()
        self.completed.clear()
        self.status = "waiting"

    def select(self, candidates, position_m, yaw_rad, now_s, event=None):
        if event is not None and event not in self.EVENTS:
            raise ValueError(f"unknown goal event: {event}")
        if not all(math.isfinite(v) for v in (*position_m, yaw_rad, now_s)):
            raise ValueError("goal selection pose/time must be finite")
        if self.current is not None and event is None:
            return self.current  # New scores alone do not move an in-flight goal.
        if self.current is not None:
            if event in self.FAILURES:
                self.blocked_until[self.current.goal_id] = now_s + self.cooldown
            elif event == "completed":
                self.completed.add(self.current.goal_id)
        self.current = None
        ranked, ids = [], set()
        for candidate in candidates:
            if candidate.candidate_id in ids:
                raise ValueError("candidate IDs must be unique and stable")
            ids.add(candidate.candidate_id)
            numbers = (*candidate.position_m, candidate.yaw_rad, candidate.estimated_gain,
                       candidate.path_length_m, candidate.inflation_m)
            if (len(candidate.position_m) != 3 or not all(math.isfinite(v) for v in numbers)
                    or candidate.path_length_m < 0 or candidate.estimated_gain < 0):
                continue
            if not (candidate.in_task_region and candidate.known_free
                    and candidate.inflated_safe and candidate.connected
                    and candidate.inflation_m >= self.inflation):
                continue
            if (candidate.candidate_id in self.completed
                    or self.blocked_until.get(candidate.candidate_id, -math.inf) > now_s
                    or math.dist(candidate.position_m, position_m) > self.radius):
                continue
            yaw_delta = abs(math.atan2(math.sin(candidate.yaw_rad-yaw_rad),
                                       math.cos(candidate.yaw_rad-yaw_rad)))
            cost = candidate.path_length_m/self.cruise + yaw_delta/self.yaw_rate + self.overhead
            ranked.append((-candidate.estimated_gain/cost, candidate.candidate_id, candidate))
        if not ranked:
            self.status = "waiting"  # Never implies full exploration completed.
            return None
        best = min(ranked, key=lambda row: row[:2])[2]
        self.current = LocalGoal(best.candidate_id, tuple(best.position_m), best.yaw_rad,
                                 best.source, switch_reason=event or "new_task")
        self.status = "navigating"
        return self.current


class FixedGoalProvider:
    def __init__(self, position_m, yaw_rad, goal_id="curriculum_target"):
        self.goal = LocalGoal(goal_id, tuple(position_m), float(yaw_rad), "fixed_curriculum")

    def reset(self):
        return replace(self.goal, switch_reason="episode_reset")


def goal_events(distance_m, yaw_error_rad, tilt_rad, *, new_frame_processed=False,
                new_frame_fused=False, frontier_covered=False,
                position_tolerance_m=0.5, yaw_tolerance_rad=0.25, tilt_tolerance_rad=0.20):
    reached = distance_m <= position_tolerance_m
    aligned = reached and abs(yaw_error_rad) <= yaw_tolerance_rad and tilt_rad <= tilt_tolerance_rad
    return {"navigation_reached": reached, "navigation_pose_reached": aligned,
            "new_frame_processed": bool(new_frame_processed),
            "observation_completed": bool(frontier_covered or (aligned and new_frame_fused))}

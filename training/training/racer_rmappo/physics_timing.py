"""Actual PhysX callback accounting, independent of Kit/PyTorch imports."""

from __future__ import annotations

import math


class PhysicsTimingAudit:
    def __init__(self, physics_dt, substeps):
        self.dt = float(physics_dt)
        self.substeps = int(substeps)
        self.physics_steps = 0
        self.physics_time_s = 0.0
        self.controller_updates = 0
        self.force_applications = 0
        self.render_calls = 0
        self.render_physics_steps = 0
        self.reset_sync_steps = 0
        self.verified_actions = 0
        self.errors = []

    def on_physics_step(self, dt):
        self.physics_steps += 1
        self.physics_time_s += float(dt)
        if not math.isclose(float(dt), self.dt, rel_tol=1e-5, abs_tol=1e-8):
            self.errors.append(f"unexpected physics dt: {dt}")

    def snapshot(self):
        return (self.physics_steps, self.physics_time_s,
                self.controller_updates, self.force_applications)

    def verify_action(self, before):
        delta = tuple(a - b for a, b in zip(self.snapshot(), before))
        valid = (delta[0] == delta[2] == delta[3] == self.substeps
                 and math.isclose(delta[1], self.substeps * self.dt,
                                  rel_tol=1e-5, abs_tol=1e-8))
        if not valid or self.errors:
            self.errors.append(f"physics/time/controller/force delta: {delta}")
            raise RuntimeError("Physics timing contract violated: " + self.errors[-1])
        self.verified_actions += 1
        return {"physics_steps": delta[0], "physics_time_s": delta[1],
                "controller_updates": delta[2], "force_applications": delta[3]}

    def render_only(self, render):
        before = self.snapshot()
        try:
            render()
        finally:
            self.render_calls += 1
            count = self.physics_steps - before[0]
            self.render_physics_steps += count
            if self.snapshot() != before:
                self.errors.append("render advanced physics/control state")
                raise RuntimeError("Render must not advance physics/control state")

    def report(self):
        return {"valid": not self.errors and self.verified_actions > 0,
                "physics_steps": self.physics_steps,
                "physics_time_s": self.physics_time_s,
                "controller_updates": self.controller_updates,
                "force_applications": self.force_applications,
                "reset_sync_physics_steps": self.reset_sync_steps,
                "render_calls": self.render_calls,
                "render_physics_steps": self.render_physics_steps,
                "verified_actions": self.verified_actions, "errors": list(self.errors)}

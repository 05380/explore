#!/usr/bin/env python3
"""Plot a recorded evaluation without Isaac/ROS/Torch; requires matplotlib."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8") as stream:
        metadata = json.loads(next(stream))
        rows = [json.loads(line) for line in stream if line.strip()]
    rows = [row for row in rows if row.get("type") == "step"]
    if not rows:
        raise ValueError("trace has no completed control steps")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    cfg = metadata["isaac_config"]
    nav, scene = cfg["navigation_backend"], cfg["scene"]
    episodes = sorted({row["episode"] for row in rows})
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    xy, dist, speed, altitude, attitude, reward = axes.flatten()
    for obstacle in scene.get("obstacles", []):
        x, y, z = obstacle["position_m"]
        sx, sy, sz = obstacle["size_m"]
        xy.add_patch(Rectangle((x-sx/2, y-sy/2), sx, sy, color="gray", alpha=.5))
        xy.text(x, y, f"{obstacle['name']}\nz={z-sz/2:g}..{z+sz/2:g}", fontsize=7, ha="center")
    target = nav["fixed_target_position_m"]
    xy.add_patch(Circle(target[:2], nav["goal_position_tolerance_m"], fill=False, color="green"))
    xy.scatter(*target[:2], marker="*", s=100, color="green", label="target (XY projection)")
    xy.scatter(*scene["spawn_position_m"][:2], marker="o", color="black", label="spawn")
    for episode in episodes:
        chunk = [r for r in rows if r["episode"] == episode]
        times = [r["episode_time_s"] for r in chunk]
        label = f"episode {episode}"
        xy.plot([r["position_m"][0] for r in chunk], [r["position_m"][1] for r in chunk], label=label)
        dist.plot(times, [r["target_distance_m"] for r in chunk], label=label)
        speed.plot(times, [r["actual_speed_mps"] for r in chunk], label=label)
        speed.plot(times, [math.sqrt(sum(x*x for x in r["applied_command_world"][:3])) for r in chunk],
                   linestyle=":", color="gray", alpha=.5)
        altitude.plot(times, [r["position_m"][2] for r in chunk])
        attitude.plot(times, [abs(r["target_yaw_error_rad"]) for r in chunk], alpha=.7)
        attitude.plot(times, [r["body_tilt_rad"] for r in chunk], linestyle=":", alpha=.7)
        reward.plot(times, [r["reward"] for r in chunk])
        if chunk[-1]["done"]:
            terminal = chunk[-1]
            causes = ",".join(k for k in ("success", "collision", "out_of_bounds", "stall", "timeout") if terminal[k])
            xy.annotate(causes, terminal["position_m"][:2], fontsize=7)
    dist.axhline(nav["goal_position_tolerance_m"], color="green", linestyle="--")
    speed.axhline(cfg["control"]["max_speed_mps"], color="red", linestyle="--")
    for z in (scene["flight_bounds_min_m"][2], scene["flight_bounds_max_m"][2]):
        altitude.axhline(z, color="red", linestyle="--")
    altitude.axhline(target[2], color="green", linestyle="--")
    attitude.axhline(nav["goal_yaw_tolerance_rad"], color="green", linestyle="--")
    attitude.axhline(nav["goal_tilt_tolerance_rad"], color="orange", linestyle="--")
    xy.set(title="Scene truth + trajectories (NOT explored map)", xlabel="world x [m]", ylabel="world y [m]")
    xy.set_aspect("equal", adjustable="datalim")
    dist.set(title="3D target distance", ylabel="m")
    speed.set(title="Actual speed / dotted filtered command / red limit", ylabel="m/s")
    altitude.set(title="Altitude / green goal / red flight bounds", ylabel="m")
    attitude.set(title="Absolute yaw error (solid) and tilt (dotted)", ylabel="rad")
    reward.set(title="Per-step reward (including terminal events)", ylabel="reward")
    for ax in (dist, speed, altitude, attitude, reward):
        ax.set_xlabel("episode simulation time [s]")
    for ax in axes.flatten():
        ax.grid(True, alpha=.2)
    if len(episodes) <= 10:
        xy.legend(fontsize=7)
    fig.suptitle(f"{metadata['scenario']} | {len(episodes)} recorded episodes (last may be partial)\n"
                 "Depth fusion/frontiers/coverage are not implemented in this navigation backend")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"P3_EVAL_PLOT={args.output.resolve()}")


if __name__ == "__main__":
    main()

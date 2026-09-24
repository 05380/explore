#!/usr/bin/env python3
"""Export the decentralized actor and its preprocessing metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = SCRIPT_DIR.parent
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.config import apply_curriculum_stage, load_config
from racer_rmappo.model import SharedRecurrentActor


class DeploymentActor(nn.Module):
    def __init__(self, actor: SharedRecurrentActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, depth, ego, target, neighbors, candidates, decision_mask, hidden):
        action, _, _, next_hidden = self.actor.step(
            {
                "depth": depth,
                "ego": ego,
                "target": target,
                "neighbors": neighbors,
                "candidates": candidates,
                "decision_mask": decision_mask,
            },
            hidden,
            deterministic=True,
        )
        return action, next_hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    cfg = apply_curriculum_stage(load_config(args.config), "single_agent_sparse_static")
    ppo = cfg["ppo"]
    frame_stack = int(cfg["actor_observation"]["depth"]["frame_stack"])
    actor = SharedRecurrentActor(
        frame_stack=frame_stack, hidden_size=int(ppo["recurrent_hidden_size"])
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint.expanduser(), map_location=args.device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    wrapper = DeploymentActor(actor)
    height = int(cfg["actor_observation"]["depth"]["resize"][1])
    width = int(cfg["actor_observation"]["depth"]["resize"][0])
    neighbors = int(cfg["actor_observation"]["neighbors"]["max_neighbors"])
    candidates = int(cfg["actor_observation"]["racer_candidates"]["max_candidates"])
    examples = (
        torch.zeros(1, 1, frame_stack, height, width, device=args.device),
        torch.zeros(1, 1, 11, device=args.device),
        torch.zeros(1, 1, 7, device=args.device),
        torch.zeros(1, 1, neighbors, 8, device=args.device),
        torch.zeros(1, 1, candidates, 9, device=args.device),
        torch.ones(1, 1, 1, device=args.device),
        torch.zeros(1, 1, int(ppo["recurrent_hidden_size"]), device=args.device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, examples, check_trace=True)
    traced.save(str(args.output))
    metadata = {
        "source_checkpoint": str(args.checkpoint.resolve()),
        "depth": cfg["actor_observation"]["depth"],
        "camera": cfg["camera"],
        "action": cfg["action"],
        "ego_fields": cfg["actor_observation"]["ego_state"]["fields"],
        "target_fields": cfg["actor_observation"]["selected_target"]["fields"],
        "candidate_fields": cfg["actor_observation"]["racer_candidates"]["fields"],
        "neighbor_fields": cfg["actor_observation"]["neighbors"]["fields"],
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"exported actor: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a RACER RMAPPO checkpoint with deterministic actions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = SCRIPT_DIR.parent
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.config import apply_curriculum_stage, load_config
from racer_rmappo.trainer import RMAPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", default="single_agent_sparse_static")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1024)
    args = parser.parse_args()
    cfg = apply_curriculum_stage(load_config(args.config), args.stage)
    if args.device:
        cfg["training"]["device"] = args.device
    if args.num_envs:
        cfg["training"]["num_parallel_swarms"] = args.num_envs
    trainer = RMAPPOTrainer(cfg)
    trainer.load(args.checkpoint, load_optimizer=False)
    print(json.dumps(trainer.evaluate(args.steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

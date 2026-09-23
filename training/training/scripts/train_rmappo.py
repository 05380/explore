#!/usr/bin/env python3
"""Train the shared recurrent RACER local-navigation policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = SCRIPT_DIR.parent
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))

from racer_rmappo.config import apply_curriculum_stage, load_config
from racer_rmappo.trainer import RMAPPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--stage", default="single_agent_sparse_static")
    parser.add_argument("--backend", choices=("smoke", "isaac"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-envs", type=int, default=None, help="parallel swarms, not UAV count")
    parser.add_argument("--total-steps", type=int, default=None, help="agent transitions")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_curriculum_stage(load_config(args.config), args.stage)
    if args.backend:
        cfg["training"]["backend"] = args.backend
    if args.device:
        cfg["training"]["device"] = args.device
    if args.num_envs:
        cfg["training"]["num_parallel_swarms"] = args.num_envs
    if args.output:
        cfg["training"]["output_dir"] = str(args.output.expanduser().resolve())
    trainer = RMAPPOTrainer(cfg)
    if args.resume:
        trainer.load(args.resume)
    path = trainer.train(args.total_steps)
    print(f"final checkpoint: {path}")


if __name__ == "__main__":
    main()

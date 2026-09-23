#!/usr/bin/env python3
"""Read-only environment and configuration diagnostics."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_PACKAGE = SCRIPT_DIR.parent
if str(TRAINING_PACKAGE) not in sys.path:
    sys.path.insert(0, str(TRAINING_PACKAGE))


def module_status(name: str):
    try:
        module = __import__(name)
        return {"available": True, "version": getattr(module, "__version__", "unknown")}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def command_output(command):
    if shutil.which(command[0]) is None:
        return None
    result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    report = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "modules": {
            name: module_status(name)
            for name in ("torch", "yaml", "hydra", "tensordict", "torchrl")
        },
    }
    try:
        from racer_rmappo.config import load_config

        cfg = load_config(args.config)
        report["config"] = {
            "valid": True,
            "path": cfg["_config_path"],
            "agents": cfg["experiment"]["num_agents"],
            "world_size_m": cfg["world"]["size_m"],
            "voxel_resolution_m": cfg["world"]["voxel_resolution_m"],
            "obstacle_inflation_m": cfg["world"]["obstacle_inflation_m"],
            "speed_norm_mps": cfg["action"]["physical_limits"]["speed_norm_mps"],
        }
    except Exception as exc:
        report["config"] = {"valid": False, "error": str(exc)}
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

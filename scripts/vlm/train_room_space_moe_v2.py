#!/usr/bin/env python3
"""Compatibility entrypoint for RoomSpace V2 MoE baseline."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    script_path = Path(__file__).resolve().with_name("train_room_space_grouped_moe_sklearn.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--input-dir",
        "datasets/cadstruct_cubicasa5k_moe_locked",
        "--output-dir",
        "checkpoints/cadstruct_room_space_moe_v2",
    ]

    # Keep this lightweight compatibility wrapper stable for scripts/reproducibility.
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

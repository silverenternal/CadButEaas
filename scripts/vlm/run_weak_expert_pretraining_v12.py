#!/usr/bin/env python3
"""Unified weak-expert v12 runner."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from v5_pipeline_utils import load_json, update_todo_remove, write_json
except ImportError:
    from scripts.vlm.v5_pipeline_utils import load_json, update_todo_remove, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/vlm/weak_expert_pretraining_v12.json")
    parser.add_argument("--output-manifest", default="reports/vlm/weak_expert_pretraining_manifest_v12.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    manifest: dict[str, Any] = {"version": "weak_expert_pretraining_manifest_v12", "runs": [], "blocked": []}

    steps = [
        {
            "id": "WEAK-V12-T1",
            "cmd": ["uv", "run", "python", "scripts/vlm/audit_weak_expert_datasets_v12.py"],
            "produces": ["reports/vlm/weak_expert_dataset_audit_v12.json", "reports/vlm/weak_expert_priority_rank_v12.json"],
        },
        {
            "id": "WEAK-V12-T3",
            "cmd": ["uv", "run", "python", "scripts/vlm/pretrain_room_space_expert_v12.py"],
            "produces": ["checkpoints/room_space_expert_v12/train_summary.json", "reports/vlm/room_space_expert_v12_eval.json"],
        },
    ]

    for step in steps:
        record = {"id": step["id"], "cmd": step["cmd"], "status": "pending"}
        if args.dry_run:
            record["status"] = "dry_run"
            manifest["runs"].append(record)
            continue
        result = subprocess.run(step["cmd"], check=False)
        record["returncode"] = result.returncode
        record["status"] = "completed" if result.returncode == 0 else "failed"
        record["produces"] = step["produces"]
        manifest["runs"].append(record)
        if result.returncode != 0:
            manifest["blocked"].append({"id": step["id"], "reason": f"command failed with code {result.returncode}"})
            break

    write_json(Path(args.output_manifest), manifest)
    if any(run.get("status") == "completed" for run in manifest["runs"]):
        update_todo_remove([run["id"] for run in manifest["runs"] if run.get("status") == "completed"])
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

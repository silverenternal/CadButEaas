#!/usr/bin/env python3
"""Build a source-held-out hard-case curriculum for v13 specialists."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from v5_pipeline_utils import load_jsonl, sample_id, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1")
    parser.add_argument("--output", default="configs/vlm/frontier_specialist_curriculum_v13.json")
    parser.add_argument("--report", default="reports/vlm/frontier_hard_case_engine_v13.json")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    train = load_jsonl(source_dir / "train.jsonl")
    dev = load_jsonl(source_dir / "dev.jsonl")
    locked = load_jsonl(source_dir / "locked_test.jsonl")
    train_ids = {sample_id(row) for row in [*train, *dev] if sample_id(row)}
    locked_ids = {sample_id(row) for row in locked if sample_id(row)}
    report = {
        "version": "frontier_hard_case_engine_v13",
        "source_dir": str(source_dir),
        "split_counts": {"train": len(train), "dev": len(dev), "locked": len(locked)},
        "leakage_check": {"train_dev_locked_overlap": len(train_ids & locked_ids), "passed": len(train_ids & locked_ids) == 0},
        "curriculum": {
            "boundary": {"source_mix": ["floorplancad", "cubicasa5k"], "hard_case_ratio": 0.7, "target_family": "line_like_geometry"},
            "room_space": {"source_mix": ["cubicasa5k", "resplan"], "hard_case_ratio": 0.5, "target_family": "polygon_completion"},
            "symbol_fixture": {"source_mix": ["cubicasa5k"], "hard_case_ratio": 0.6, "target_family": "long_tail_symbols"},
            "text_dimension": {"source_mix": ["cubicasa5k"], "hard_case_ratio": 0.5, "target_family": "ocr_and_linking"},
        },
        "recommendation": "use source-held-out validation for every specialist before fusion refresh",
    }
    curriculum = {
        "version": "frontier_specialist_curriculum_v13",
        "boundary": report["curriculum"]["boundary"],
        "room_space": report["curriculum"]["room_space"],
        "symbol_fixture": report["curriculum"]["symbol_fixture"],
        "text_dimension": report["curriculum"]["text_dimension"],
        "claim_boundary": "Curriculum is an explicit training contract only; it does not consume locked examples for fitting.",
    }
    write_json(args.output, curriculum)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

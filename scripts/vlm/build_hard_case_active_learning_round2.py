#!/usr/bin/env python3
"""Create hard-case active-learning round 2 candidates from current audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPERT_TARGETS = {
    "wall_opening": {
        "paths": [
            "reports/vlm/hard_case_index_v1.jsonl",
            "reports/vlm/wall_opening_floorplancad_gap_cases_v1.jsonl",
        ],
        "target": 100,
    },
    "symbol_fixture": {
        "paths": [
            "reports/vlm/hard_case_index_v1.jsonl",
            "reports/vlm/symbol_fixture_hard_case_round_1.jsonl",
        ],
        "target": 100,
    },
    "text_dimension": {
        "paths": [
            "reports/vlm/hard_case_index_v1.jsonl",
            "reports/vlm/text_dimension_error_cases_v2.jsonl",
        ],
        "target": 100,
    },
    "scene_graph": {
        "paths": [
            "reports/vlm/scene_graph_error_attribution_v1_cases.jsonl",
        ],
        "target": 100,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="datasets/internal_hard_cases_round_2")
    parser.add_argument("--report", default="reports/vlm/hard_case_mining_round_2.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "version": "internal_hard_cases_round_2",
        "output_dir": str(output_dir),
        "experts": {},
        "policy": "Candidate mining only; human review is required before adding any candidate to training.",
    }

    for expert, cfg in EXPERT_TARGETS.items():
        candidates = collect_candidates(expert, cfg["paths"])
        selected = candidates[: int(cfg["target"])]
        expert_path = output_dir / f"{expert}_candidates.jsonl"
        write_jsonl(expert_path, selected)
        label_counts = Counter(str(row.get("gold") or row.get("label") or row.get("reason") or "unknown") for row in selected)
        manifest["experts"][expert] = {
            "target": cfg["target"],
            "candidate_count": len(selected),
            "available_before_cap": len(candidates),
            "status": "ok" if len(selected) >= cfg["target"] else "data_insufficient",
            "output": str(expert_path),
            "top_labels_or_reasons": dict(label_counts.most_common(20)),
            "input_paths": cfg["paths"],
        }
        report_rows.extend(selected)

    manifest["summary"] = {
        "expert_count": len(EXPERT_TARGETS),
        "total_candidates": sum(item["candidate_count"] for item in manifest["experts"].values()),
        "insufficient_experts": [
            expert for expert, item in manifest["experts"].items() if item["status"] != "ok"
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(Path(args.report), report_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def collect_candidates(expert: str, paths: list[str]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for path_text in paths:
        path = Path(path_text)
        if not path.exists():
            continue
        for row in read_jsonl(path):
            row_expert = str(row.get("expert") or row.get("audit_version") or "")
            if path.name == "hard_case_index_v1.jsonl" and row.get("expert") != expert:
                continue
            key = json.dumps([expert, row.get("image"), row.get("node_id"), row.get("id"), row.get("item"), row.get("error_pair"), row.get("reason")], sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            normalized = {
                "expert": expert,
                "source_path": path_text,
                "image": row.get("image"),
                "source_dataset": row.get("source_dataset"),
                "gold": row.get("gold") or row.get("label"),
                "prediction": row.get("prediction"),
                "confidence": row.get("confidence"),
                "reason": row.get("reason") or row.get("error_bucket") or row.get("stage") or row_expert,
                "tags": row.get("tags") or row.get("error_tags") or [],
                "raw": row,
                "review_status": "pending_human_review",
            }
            normalized["priority_score"] = priority_score(normalized)
            out.append(normalized)
    return sorted(out, key=lambda item: (-float(item.get("priority_score") or 0.0), str(item.get("image") or "")))


def priority_score(row: dict[str, Any]) -> float:
    score = 1.0
    confidence = row.get("confidence")
    if isinstance(confidence, (int, float)):
        score += float(confidence)
    if row.get("gold") and row.get("prediction") and row.get("gold") != row.get("prediction"):
        score += 1.0
    if row.get("reason"):
        score += 0.25
    return round(score, 6)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

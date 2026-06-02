#!/usr/bin/env python3
"""Prepare MoE-compatible symbol-policy overlays for downstream smoke evaluation.

P0-84 bridge: detector policies emit page-level rows keyed by row_id, while the
CadStruct MoE fusion/eval stack consumes record contracts with symbol candidates
(or expected_json.symbol_candidates). This script materializes deterministic
policy overlays without changing the default runtime policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE = ROOT / "datasets/public_raster_moe_supervision_v19/locked.jsonl"
DEFAULT_POLICIES = {
    "v28_frozen_detector_baseline": ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_locked_page_predictions_p064_refresh.jsonl",
    "p076_balanced_opt_in": ROOT / "reports/vlm/symbol_balanced_policy_p076_locked_predictions.jsonl",
}
DEFAULT_OUTPUT_DIR = ROOT / "reports/vlm/symbol_policy_moe_overlay_p084"
DEFAULT_SUMMARY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p084_summary.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_policy_moe_overlay_p084.md"
POLICY_REGISTRY = "configs/vlm/symbol_final_policy_registry_p079.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def target_to_expected(item: dict[str, Any], family: str, index: int) -> dict[str, Any] | None:
    bbox = bbox4(item.get("bbox"))
    if bbox is None:
        return None
    label = str(item.get("semantic_type") or item.get("label") or "unknown")
    target_id = str(item.get("target_id") or f"{family}_{index}")
    base = {
        "id": target_id,
        "target_id": target_id,
        "bbox": bbox,
        "confidence": 1.0,
        "source": "offline_raster_target_passthrough_for_p084_overlay",
        "raw_label": item.get("raw_label") or label,
    }
    if family == "boundary":
        boundary_label = "hard_wall" if label == "wall" else label
        return {**base, "semantic_type": boundary_label}
    if family == "space":
        return {**base, "room_type": label}
    if family == "text":
        return {**base, "text_type": label, "text": item.get("text") or ""}
    if family == "symbol":
        return {**base, "symbol_type": label}
    return None


def predicted_symbol_to_candidate(row_id: str, policy_id: str, pred: dict[str, Any], index: int) -> dict[str, Any] | None:
    bbox = bbox4(pred.get("bbox"))
    if bbox is None:
        return None
    label = str(pred.get("label") or pred.get("symbol_type") or "generic_symbol")
    source_policy = str(pred.get("source_policy") or pred.get("policy_id") or policy_id)
    return {
        "id": f"{row_id}_{policy_id}_symbol_{index:05d}",
        "target_id": f"{row_id}_{policy_id}_symbol_{index:05d}",
        "symbol_type": label,
        "bbox": bbox,
        "confidence": safe_score(pred.get("score")),
        "source": "symbol_policy_overlay_p084",
        "metadata": {
            "symbol_policy_id": policy_id,
            "policy_registry": POLICY_REGISTRY,
            "base_detector": "symbol_yolov8s_seg_rect_v28" if policy_id.startswith("v28") else "symbol_yolov8s_seg_rect_v28_plus_optional_policy",
            "candidate_source": source_policy,
            "score": safe_score(pred.get("score")),
            "tile_id": pred.get("tile_id"),
            "label_id": pred.get("label_id"),
        },
    }


def safe_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def expected_json_from_targets(row: dict[str, Any], symbol_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
    semantic_candidates = [
        converted
        for idx, item in enumerate(targets.get("boundary") or [])
        for converted in [target_to_expected(item, "boundary", idx)]
        if converted is not None
    ]
    room_candidates = [
        converted
        for idx, item in enumerate(targets.get("space") or [])
        for converted in [target_to_expected(item, "space", idx)]
        if converted is not None
    ]
    text_candidates = [
        converted
        for idx, item in enumerate(targets.get("text") or [])
        for converted in [target_to_expected(item, "text", idx)]
        if converted is not None
    ]
    return {
        "semantic_candidates": semantic_candidates,
        "room_candidates": room_candidates,
        "symbol_candidates": symbol_candidates,
        "text_candidates": text_candidates,
    }


def load_policy_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or "")
        if not row_id:
            continue
        by_id[row_id] = list(row.get("predicted_symbols") or [])
    return by_id


def build_overlay(
    base_rows: list[dict[str, Any]],
    policy_id: str,
    predictions: dict[str, list[dict[str, Any]]],
    include_missing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overlay_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    candidate_counts: list[int] = []
    missing_row_ids: list[str] = []

    for row in base_rows:
        row_id = str(row.get("id") or row.get("row_id") or "")
        if not row_id:
            counts["base_rows_without_id"] += 1
            continue
        raw_preds = predictions.get(row_id)
        if raw_preds is None:
            missing_row_ids.append(row_id)
            if not include_missing:
                continue
            raw_preds = []
        symbol_candidates = [
            candidate
            for idx, pred in enumerate(raw_preds)
            for candidate in [predicted_symbol_to_candidate(row_id, policy_id, pred, idx)]
            if candidate is not None
        ]
        candidate_counts.append(len(symbol_candidates))
        expected_json = expected_json_from_targets(row, symbol_candidates)
        overlay = dict(row)
        overlay["row_id"] = row_id
        overlay["image_path"] = row.get("image") or row.get("image_path")
        overlay["expected_json"] = expected_json
        overlay["symbol_candidates"] = symbol_candidates
        overlay["symbol_policy_overlay"] = {
            "id": "P0-84-symbol-policy-to-moe-adapter-smoke",
            "symbol_policy_id": policy_id,
            "policy_registry": POLICY_REGISTRY,
            "prediction_count": len(symbol_candidates),
            "default_policy_unchanged": True,
        }
        overlay_rows.append(overlay)

    summary = {
        "policy_id": policy_id,
        "base_rows": len(base_rows),
        "prediction_rows": len(predictions),
        "overlay_rows": len(overlay_rows),
        "matched_rows": len(base_rows) - len(missing_row_ids),
        "missing_prediction_rows": len(missing_row_ids),
        "missing_prediction_row_ids_sample": missing_row_ids[:20],
        "include_missing_rows": include_missing,
        "candidate_count_total": sum(candidate_counts),
        "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
        "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
        "candidate_count_mean": round(sum(candidate_counts) / len(candidate_counts), 6) if candidate_counts else 0.0,
        "base_rows_without_id": counts["base_rows_without_id"],
    }
    return overlay_rows, summary


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# P0-84 Symbol Policy to MoE Overlay Smoke",
        "",
        "## Decision",
        "",
        "A deterministic adapter is now available for the downstream smoke path. It keeps `v28_frozen_detector_baseline` as the default policy and materializes explicit opt-in overlays for comparison, starting with `v28` vs `P0-76`.",
        "",
        "## Inputs",
        "",
        "- Base records: `{}`".format(summary["base_records"]),
        f"- Policy registry: `{POLICY_REGISTRY}`",
        "- Mapping key: base record `id` == detector prediction `row_id`.",
        "- Default row policy: matched comparable rows only; use `--include-missing` to keep base rows without detector predictions.",
        "",
        "## Overlay Outputs",
        "",
        "| Policy | Output | Rows | Matched | Candidates | Mean/page | Missing rows |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["policies"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} | {} | {} |".format(item["policy_id"], item["output"], item["overlay_rows"], item["matched_rows"], item["candidate_count_total"], item["candidate_count_mean"], item["missing_prediction_rows"])
        )
    lines.extend([
        "",
        "## Contract",
        "",
        "Each overlay row preserves the base raster record, adds `row_id`, `image_path`, `symbol_candidates`, `expected_json.symbol_candidates`, and `symbol_policy_overlay`. Non-symbol targets are passed through into `expected_json` so fusion smoke tests can keep non-symbol inputs identical between policies.",
        "",
        "## Next Step",
        "",
        "Run a scene-graph smoke comparison using the generated `v28` and `P0-76` overlays. If the existing fusion script cannot accept an input override, add a thin runner/CLI wrapper rather than modifying the default policy path.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-records", default=str(DEFAULT_BASE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--policy", action="append", default=[], help="policy_id=path; defaults to v28 and p076")
    parser.add_argument("--include-missing", action="store_true", help="Keep base rows without detector predictions as zero-symbol overlays. Default outputs matched comparable rows only.")
    args = parser.parse_args()

    base_path = Path(args.base_records)
    output_dir = Path(args.output_dir)
    policies = dict(DEFAULT_POLICIES)
    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"Invalid --policy spec {spec!r}; expected policy_id=path")
        policy_id, path = spec.split("=", 1)
        policies[policy_id] = Path(path)

    base_rows = load_jsonl(base_path)
    summaries: list[dict[str, Any]] = []
    outputs: dict[str, str] = {}
    for policy_id, prediction_path in policies.items():
        predictions = load_policy_predictions(prediction_path)
        overlay_rows, policy_summary = build_overlay(base_rows, policy_id, predictions, include_missing=args.include_missing)
        output_path = output_dir / f"{policy_id}.jsonl"
        write_jsonl(output_path, overlay_rows)
        policy_summary["prediction_path"] = str(prediction_path.relative_to(ROOT) if prediction_path.is_absolute() and prediction_path.is_relative_to(ROOT) else prediction_path)
        policy_summary["output"] = str(output_path.relative_to(ROOT) if output_path.is_absolute() and output_path.is_relative_to(ROOT) else output_path)
        summaries.append(policy_summary)
        outputs[policy_id] = policy_summary["output"]

    summary = {
        "id": "P0-84-symbol-policy-to-moe-adapter-smoke",
        "base_records": str(base_path.relative_to(ROOT) if base_path.is_absolute() and base_path.is_relative_to(ROOT) else base_path),
        "mapping_key": "base_record.id == detector_prediction.row_id",
        "include_missing_rows": bool(args.include_missing),
        "default_policy_unchanged": True,
        "default_policy": "v28_frozen_detector_baseline",
        "recommended_comparison": ["v28_frozen_detector_baseline", "p076_balanced_opt_in"],
        "outputs": outputs,
        "policies": summaries,
        "next_step": "Run scene-graph fusion smoke on v28 and p076 overlays with identical non-symbol inputs.",
    }
    write_json(Path(args.summary), summary)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

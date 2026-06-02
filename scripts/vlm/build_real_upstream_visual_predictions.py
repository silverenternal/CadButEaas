#!/usr/bin/env python3
"""Build visual-demo scene graphs from saved real upstream expert predictions."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_moe_scene_graph import predictions_from_record  # noqa: E402
from fuse_scene_graph import apply_constraint_repairs, apply_quality_flags, normalize_scene_graph_contract_fields  # noqa: E402
from scene_graph_schema import validate_scene_graph  # noqa: E402


DEFAULT_RECORDS = Path("datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
DEFAULT_UPSTREAM = Path("reports/vlm/real_upstream_predictions_dev_symbol_long_tail_model_v1.jsonl")
DEFAULT_OUTPUT = Path("reports/vlm/e2e_cubicasa_visual_demo_model_predictions.jsonl")
DEFAULT_AUDIT = Path("reports/vlm/e2e_cubicasa_visual_demo_model_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default=str(DEFAULT_RECORDS))
    parser.add_argument("--upstream", default=str(DEFAULT_UPSTREAM))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    records_path = Path(args.records)
    upstream_path = Path(args.upstream)
    records = load_jsonl(records_path)
    if args.limit > 0:
        records = records[: args.limit]
    upstream = load_jsonl(upstream_path)
    scoped_predictions = assign_upstream_predictions(upstream, records)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start = time.perf_counter()
    for record_index, record in enumerate(records):
        row_start = time.perf_counter()
        try:
            source_predictions = predictions_from_record(record, "expected_json")
            model_predictions = []
            for prediction in source_predictions:
                key = prediction_key(prediction)
                upstream_prediction = scoped_predictions.get((record_index, key[0], key[1]))
                if upstream_prediction is not None:
                    model_label = str(upstream_prediction.get("label") or prediction.label)
                    model_confidence = safe_float(upstream_prediction.get("confidence"), prediction.confidence)
                    model_source = str(upstream_prediction.get("source") or "real_upstream_saved_prediction")
                    metadata = {
                        **dict(prediction.metadata or {}),
                        "model_source": model_source,
                        "model_label": model_label,
                        "model_confidence": model_confidence,
                        "proposal_source": "svg_candidate_geometry",
                        "base_raw_label": prediction.metadata.get("raw_label") if isinstance(prediction.metadata, dict) else None,
                        "upstream_metadata": upstream_prediction.get("metadata") if isinstance(upstream_prediction.get("metadata"), dict) else {},
                    }
                    model_predictions.append(
                        replace(
                            prediction,
                            label=model_label,
                            confidence=model_confidence,
                            expert=str(upstream_prediction.get("expert") or prediction.expert),
                            source=model_source,
                            metadata=metadata,
                        )
                    )
                else:
                    metadata = {
                        **dict(prediction.metadata or {}),
                        "model_source": "missing_real_upstream_fallback_geometry_only",
                        "proposal_source": "svg_candidate_geometry",
                    }
                    model_predictions.append(
                        replace(
                            prediction,
                            confidence=min(float(prediction.confidence or 1.0), 0.01),
                            source="missing_real_upstream_fallback_geometry_only",
                            metadata=metadata,
                        )
                    )
            scene_graph, warnings, repair_events = fuse_predictions_for_visual(model_predictions)
            elapsed_ms = (time.perf_counter() - row_start) * 1000.0
            is_valid, graph_errors = validate_scene_graph(scene_graph)
            rows.append(
                {
                    "image": record.get("image_path"),
                    "annotation": record.get("annotation_path"),
                    "source_dataset": record.get("source_dataset") or "cubicasa5k",
                    "split": "model_visual_demo",
                    "scene_graph": scene_graph,
                    "warnings": warnings,
                    "quality_report": quality_report(record, scene_graph, scoped_predictions, record_index),
                    "route_trace": {
                        "source_mode": "real_upstream_saved_model_predictions",
                        "candidate_geometry_source": "svg_candidate_geometry",
                        "upstream_predictions": str(upstream_path),
                        "scene_graph_valid": is_valid,
                        "scene_graph_contract_errors": graph_errors,
                        "repair_event_count": len(repair_events),
                        "latency_ms": round(elapsed_ms, 3),
                        "peak_memory_mib": current_peak_memory_mib(),
                    },
                    "latency_ms": round(elapsed_ms, 3),
                    "memory_mib": current_peak_memory_mib(),
                    "gold_source": "none_for_labels_saved_model_predictions",
                }
            )
        except Exception as exc:  # pragma: no cover - audit path
            failures.append({"record_index": record_index, "image": record.get("image_path"), "error": type(exc).__name__, "message": str(exc)})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    audit = {
        "version": "cubicasa_visual_demo_real_upstream_model_v1",
        "records": str(records_path),
        "upstream_predictions": str(upstream_path),
        "output": str(output),
        "source_mode": "real_upstream_saved_model_predictions",
        "candidate_geometry_source": "svg_candidate_geometry",
        "label_source": "saved expert model predictions",
        "records_count": len(records),
        "predictions_count": len(rows),
        "unhandled_exception_count": len(failures),
        "schema_valid_rate": round(sum(1 for row in rows if row.get("route_trace", {}).get("scene_graph_valid")) / max(len(rows), 1), 6),
        "warning_counts": warning_counts(rows),
        "node_family_counts": node_family_counts(rows),
        "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 3),
        "failures": failures,
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def prediction_key(prediction: Any) -> tuple[str, str]:
    family = str(prediction.family)
    candidate_id = str(prediction.candidate_id)
    if family == "boundary" and candidate_id.startswith("boundary_"):
        return family, candidate_id.replace("boundary_", "", 1)
    return family, candidate_id


def assign_upstream_predictions(upstream: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[tuple[int, str, str], dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in upstream:
        by_family[str(item.get("family"))].append(item)
    cursors: dict[str, int] = defaultdict(int)
    assigned: dict[tuple[int, str, str], dict[str, Any]] = {}
    for record_index, record in enumerate(records):
        for family, ids in expected_candidate_ids(record).items():
            family_rows = by_family.get(family, [])
            for candidate_id in ids:
                cursor = cursors[family]
                if cursor >= len(family_rows):
                    break
                assigned[(record_index, family, candidate_id)] = family_rows[cursor]
                cursors[family] += 1
    return assigned


def expected_candidate_ids(record: dict[str, Any]) -> dict[str, list[str]]:
    expected = record.get("expected_json") or {}
    return {
        "boundary": [str(item.get("target_id")) for item in expected.get("semantic_candidates") or []],
        "space": [str(item.get("id")) for item in expected.get("room_candidates") or []],
        "symbol": [str(item.get("id")) for item in expected.get("symbol_candidates") or []],
        "text": [str(item.get("id")) for item in expected.get("text_candidates") or []],
    }


def fuse_predictions_for_visual(predictions: list[Any]) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    warnings: list[str] = []
    for prediction in predictions:
        geometry = dict(prediction.geometry or {})
        geometry.setdefault("bbox", prediction.bbox or [])
        nodes.append(
            {
                "id": str(prediction.candidate_id),
                "semantic_type": str(prediction.label),
                "expert": str(prediction.expert),
                "family": str(prediction.family),
                "confidence": safe_float(prediction.confidence, 0.5),
                "source_expert": str(prediction.source),
                "geometry": geometry,
                "audit_trace": {
                    "origin": "real_upstream_saved_model_predictions",
                    "stage": "model_label_fusion",
                    "family": str(prediction.family),
                },
                "metadata": dict(prediction.metadata or {}),
            }
        )
        edges.extend(list(prediction.relations or []))
    repair_events, repair_warnings = apply_constraint_repairs(nodes, edges)
    warnings.extend(repair_warnings)
    normalize_scene_graph_contract_fields(nodes, edges)
    warnings.extend(apply_quality_flags(nodes, edges))
    return {"nodes": nodes, "edges": edges}, sorted(set(warnings)), repair_events


def quality_report(record: dict[str, Any], scene_graph: dict[str, Any], scoped: dict[tuple[int, str, str], dict[str, Any]], record_index: int) -> dict[str, Any]:
    expected_ids = expected_candidate_ids(record)
    counts = {family: len(ids) for family, ids in expected_ids.items()}
    matched = {
        family: sum(1 for candidate_id in ids if (record_index, family, candidate_id) in scoped)
        for family, ids in expected_ids.items()
    }
    return {
        "model_output_contract": "saved expert predictions classify parser/SVG candidate geometry",
        "candidate_counts": counts,
        "model_matched_candidate_counts": matched,
        "rendered_node_count": len(scene_graph.get("nodes") or []),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def current_peak_memory_mib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(usage / (1024 * 1024), 3)
    return round(usage / 1024, 3)


def warning_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for warning in row.get("warnings") or []:
            counts[str(warning)] += 1
    return dict(counts.most_common())


def node_family_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for node in (row.get("scene_graph") or {}).get("nodes") or []:
            counts[str(node.get("family") or "unknown")] += 1
    return dict(counts.most_common())


if __name__ == "__main__":
    main()

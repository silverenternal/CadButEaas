"""Unified high-recall detector candidate contract and audit helpers."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any


CANDIDATE_CONTRACT_VERSION = "detector_candidate_contract_v1"


@dataclass(frozen=True)
class DetectorCandidate:
    """Normalized detector candidate emitted before relation/topology compression."""

    candidate_id: str
    row_id: str
    family: str
    route: str
    candidate_type: str
    bbox: list[int]
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)
    source_integrity: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    audit_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["candidate_contract_version"] = CANDIDATE_CONTRACT_VERSION
        return out


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out = json.loads(json.dumps(payload))
    for key in ["raw_label", "base_raw_label", "parser_raw_label", "gold_candidate"]:
        out.pop(key, None)
    return out


def integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_moe",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
    }


def normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        bbox = [int(v) for v in value]
    except (TypeError, ValueError):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox


def normalize_candidate(
    raw: dict[str, Any],
    family: str,
    row_id: str,
    image: str | None = None,
) -> dict[str, Any] | None:
    bbox = normalize_bbox(raw.get("bbox"))
    if bbox is None:
        return None
    candidate_id = raw.get("candidate_id") or raw.get("id")
    if not candidate_id:
        return None
    route = raw.get("route") or raw.get("expert") or {
        "boundary": "wall_opening",
        "space": "room_space",
        "text": "text_dimension",
        "symbol": "symbol_fixture",
    }.get(family, family)
    payload = clean_payload(raw.get("payload"))
    if image and "image" not in payload:
        payload["image"] = image
    route_trace = raw.get("route_trace") if isinstance(raw.get("route_trace"), dict) else {}
    provenance = {
        "input_source": str(raw.get("input_source") or raw.get("source") or route),
        "raw_candidate_id": str(candidate_id),
        "row_id": row_id,
        "family": family,
        "route": route,
        "image": image,
    }
    audit_trace = {
        "stage": "detector_candidate_normalization",
        "route_trace": route_trace,
        "has_payload": bool(payload),
        "has_image": bool(image),
        "source_integrity": integrity(),
    }
    candidate = DetectorCandidate(
        candidate_id=str(candidate_id),
        row_id=row_id,
        family=family,
        route=str(route),
        candidate_type=str(raw.get("candidate_type") or payload.get("symbol_type") or payload.get("room_type") or family),
        bbox=bbox,
        confidence=round(float(raw.get("confidence") or 0.0), 6),
        payload=payload,
        source_integrity=integrity(),
        provenance=provenance,
        audit_trace=audit_trace,
    )
    return candidate.to_dict()


def summarize_confidence(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(float(v) for v in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "mean": round(mean(ordered), 6),
        "max": round(ordered[-1], 6),
    }


def recall_for(
    candidates_by_row: dict[str, list[dict[str, Any]]],
    gold_by_row: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    total = hit = 0
    for row_id, golds in gold_by_row.items():
        candidates = candidates_by_row.get(row_id, [])
        for gold in golds:
            total += 1
            gb = [int(v) for v in gold["bbox"]]
            if any(center_covered(c["bbox"], gb) or bbox_iou(c["bbox"], gb) >= 0.30 for c in candidates):
                hit += 1
    return {"gold": total, "matched": hit, "center_or_iou_recall": round(hit / max(total, 1), 6)}


def center_covered(pred: list[int], gold: list[int], margin: int = 2) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def build_candidate_audit(
    family_rows: dict[str, dict[str, list[dict[str, Any]]]],
    gold: dict[str, dict[str, list[dict[str, Any]]]],
    caps: dict[str, int],
    selected_rows: dict[str, dict[str, list[dict[str, Any]]]],
    source_violations: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter()
    recall_loss: dict[str, Any] = {}
    family_audit: dict[str, Any] = {}

    for family, rows in family_rows.items():
        before = recall_for(rows, gold[family])
        after = recall_for(selected_rows[family], gold[family])
        recall_loss[family] = {
            "cap_per_page": caps[family] or None,
            "before_cap": before,
            "after_cap": after,
            "absolute_recall_loss": round(before["center_or_iou_recall"] - after["center_or_iou_recall"], 6),
        }

        all_candidates = [item for row_items in rows.values() for item in row_items]
        selected_candidates = [item for row_items in selected_rows[family].values() for item in row_items]
        family_audit[family] = {
            "candidate_count_before_cap": len(all_candidates),
            "candidate_count_after_cap": len(selected_candidates),
            "confidence": summarize_confidence([float(item.get("confidence") or 0.0) for item in all_candidates]),
            "selected_confidence": summarize_confidence([float(item.get("confidence") or 0.0) for item in selected_candidates]),
            "route_counts": dict(Counter(str(item.get("route") or "") for item in all_candidates)),
            "candidate_type_counts": dict(Counter(str(item.get("candidate_type") or "") for item in all_candidates)),
            "payload_key_counts": dict(
                Counter(key for item in all_candidates for key in (item.get("payload") or {}).keys())
            ),
            "provenance_source_counts": dict(
                Counter(str((item.get("provenance") or {}).get("input_source") or "") for item in all_candidates)
            ),
            "missing_provenance_count": sum(1 for item in all_candidates if not item.get("provenance")),
            "missing_audit_trace_count": sum(1 for item in all_candidates if not item.get("audit_trace")),
            "selected_count": len(selected_candidates),
            "before_recall": before,
            "after_recall": after,
        }

        counts[f"{family}_before_cap"] = len(all_candidates)
        counts[f"{family}_after_cap"] = len(selected_candidates)

    return {
        "candidate_contract_version": CANDIDATE_CONTRACT_VERSION,
        "candidate_counts": dict(counts),
        "families": family_audit,
        "recall_loss_accounting": recall_loss,
        "source_integrity": {
            "violations": len(source_violations),
            "sample_violations": source_violations[:20],
        },
        "success_criteria": {
            "source_integrity_violations_zero": len(source_violations) == 0,
            "cap_recall_loss_reported_per_family": set(recall_loss) == set(family_rows),
            "all_detector_families_nonzero": all(counts[f"{family}_after_cap"] > 0 for family in family_rows),
        },
    }

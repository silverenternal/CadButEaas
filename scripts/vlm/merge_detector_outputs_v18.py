#!/usr/bin/env python3
"""Merge v18 detector routed outputs into one image-only MoE candidate stream."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
SCHEMA = ROOT / "configs/vlm/detector_output_schema_v18.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vlm.candidate_contract import build_candidate_audit, integrity, normalize_candidate

DEFAULT_INPUTS = {
    "boundary": REPORT / "boundary_segmenter_v18_routed_candidates.jsonl",
    "space": REPORT / "room_proposal_model_v18_routed_candidates.jsonl",
    "text": REPORT / "text_ocr_v18_routed_candidates.jsonl",
    "symbol": REPORT / "symbol_detector_v18_safe_routed_candidates.jsonl",
}
SYMBOL_TYPE_EVAL = REPORT / "symbol_type_classifier_v18_eval.json"
GOLD_FILES = {
    "boundary": ROOT / "datasets/image_only_boundary_detector_v18/locked.jsonl",
    "space": ROOT / "datasets/image_only_room_polygon_v18/locked.jsonl",
    "text": ROOT / "datasets/image_only_text_ocr_v18/locked.jsonl",
    "symbol": ROOT / "datasets/image_only_symbol_detector_v18/locked.jsonl",
}
GOLD_KEYS = {
    "boundary": "boxes",
    "space": "rooms",
    "text": "texts",
    "symbol": "symbols",
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def load_family(
    path: Path,
    family: str,
    max_row_ids: int | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_meta: dict[str, dict[str, Any]] = {}
    seen_row_ids: set[str] = set()
    for row in load_jsonl(path):
        if "candidate_stream" in row:
            row_id = str(row.get("id"))
            if max_row_ids and row_id not in seen_row_ids and len(seen_row_ids) >= max_row_ids:
                break
            seen_row_ids.add(row_id)
            image = row.get("image")
            page_meta[row_id] = {
                "id": row_id,
                "image": image,
                "image_size": row.get("image_size") or [512, 512],
            }
            for raw in row.get("candidate_stream") or []:
                cand = normalize_candidate(raw, family, row_id, image=image)
                if cand:
                    by_row[row_id].append(cand)
        else:
            row_id = str(row.get("row_id") or row.get("id", "").split("_text_")[0].split("_symbol_")[0])
            if max_row_ids and row_id not in seen_row_ids and len(seen_row_ids) >= max_row_ids:
                break
            seen_row_ids.add(row_id)
            cand = normalize_candidate(row, family, row_id)
            if cand:
                by_row[row_id].append(cand)
    return by_row, page_meta


def load_gold() -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family, path in GOLD_FILES.items():
        rows = {}
        for row in load_jsonl(path):
            rows[row["id"]] = [
                item for item in (row.get("targets") or {}).get(GOLD_KEYS[family]) or []
                if item.get("bbox") and len(item["bbox"]) == 4
            ]
        out[family] = rows
    return out


def bbox_center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def bbox_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_contains(outer: list[float], inner: list[float], margin: float = 0.0) -> bool:
    return (
        outer[0] - margin <= inner[0]
        and outer[1] - margin <= inner[1]
        and outer[2] + margin >= inner[2]
        and outer[3] + margin >= inner[3]
    )


def box_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] <= right[0] or right[2] <= left[0] or left[3] <= right[1] or right[3] <= left[1])


def box_touch_score(left: list[float], right: list[float]) -> float:
    if not box_intersects(left, right):
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    return inter / max(min(bbox_area(left), bbox_area(right)), 1e-9)


def support_with_spaces(candidate: dict[str, Any], spaces: list[dict[str, Any]]) -> float:
    bbox = candidate.get("bbox") or []
    if len(bbox) != 4:
        return 0.0
    cx, cy = bbox_center(bbox)
    best = 0.0
    for space in spaces:
        sb = space.get("bbox") or []
        if len(sb) != 4:
            continue
        if box_contains(sb, bbox, margin=2.0) or box_contains(sb, [cx, cy, cx, cy], margin=2.0):
            best = max(best, 1.0)
            continue
        overlap = box_touch_score(bbox, sb)
        if overlap > 0.0:
            best = max(best, 0.35 + 0.65 * overlap)
    return best


def boundary_support(candidate: dict[str, Any], spaces: list[dict[str, Any]]) -> float:
    bbox = candidate.get("bbox") or []
    if len(bbox) != 4 or not spaces:
        return 0.0
    best = 0.0
    cand_center = bbox_center(bbox)
    cand_width = max(1.0, float(bbox[2]) - float(bbox[0]))
    cand_height = max(1.0, float(bbox[3]) - float(bbox[1]))
    orient = "horizontal" if cand_width >= cand_height else "vertical"
    for space in spaces:
        sb = space.get("bbox") or []
        if len(sb) != 4:
            continue
        if box_contains(sb, bbox, margin=4.0) or box_contains(bbox, sb, margin=4.0):
            best = max(best, 0.8)
        overlap = box_touch_score(bbox, sb)
        if overlap <= 0.0:
            continue
        sx1, sy1, sx2, sy2 = sb
        if orient == "horizontal":
            side_overlap = max(0.0, min(bbox[3], sy2) - max(bbox[1], sy1)) / max(cand_height, 1e-9)
            dist = min(abs(cand_center[1] - sy1), abs(cand_center[1] - sy2))
            candidate_score = 0.45 * overlap + 0.30 * side_overlap + 0.25 * max(0.0, 1.0 - dist / 24.0)
        else:
            side_overlap = max(0.0, min(bbox[2], sx2) - max(bbox[0], sx1)) / max(cand_width, 1e-9)
            dist = min(abs(cand_center[0] - sx1), abs(cand_center[0] - sx2))
            candidate_score = 0.45 * overlap + 0.30 * side_overlap + 0.25 * max(0.0, 1.0 - dist / 24.0)
        best = max(best, candidate_score)
    return best


def cap_score(candidate: dict[str, Any], family: str, spaces: list[dict[str, Any]]) -> float:
    confidence = float(candidate.get("confidence") or 0.0)
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    if family == "boundary":
        support = boundary_support(candidate, spaces)
        shape = payload.get("features") if isinstance(payload.get("features"), dict) else {}
        orient = str(shape.get("orientation") or "").lower()
        length = float(shape.get("length") or 0.0)
        thin_bonus = 0.12 if orient in {"horizontal", "vertical"} else 0.0
        size_bonus = min(length / 240.0, 1.0) * 0.08
        return confidence + 0.55 * support + thin_bonus + size_bonus
    if family == "symbol":
        support = support_with_spaces(candidate, spaces)
        typed_conf = float(payload.get("type_confidence") or payload.get("typed_confidence") or 0.0)
        local_density = float(payload.get("local_dark_density") or 0.0)
        anchor = payload.get("anchor_size") if isinstance(payload.get("anchor_size"), list) else []
        anchor_bonus = 0.05 if len(anchor) == 2 else 0.0
        return confidence + 0.60 * support + 0.12 * typed_conf + 0.08 * min(local_density, 1.0) + anchor_bonus
    return confidence


def cap_candidates(candidates: list[dict[str, Any]], cap: int, family: str, spaces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for cand in candidates:
        item = dict(cand)
        item.setdefault("audit_trace", {})
        item["audit_trace"] = dict(item["audit_trace"])
        item["audit_trace"]["cap_rank_family"] = family
        item["audit_trace"]["cap_rank_score"] = round(cap_score(item, family, spaces), 6)
        item["audit_trace"]["cap_rank_policy"] = "relation_support_aware_v1"
        ranked.append(item)
    return sorted(ranked, key=lambda item: (item.get("audit_trace") or {}).get("cap_rank_score", item.get("confidence", 0.0)), reverse=True)[:cap]


def merge(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    schema = load_json(Path(args.schema))
    caps = {
        family: int((schema.get("families") or {}).get(family, {}).get("default_cap_per_page") or 0)
        for family in DEFAULT_INPUTS
    }
    if args.no_caps:
        caps = {family: 0 for family in caps}
    if args.smoke:
        caps = {family: min(cap or 50, 25) for family, cap in caps.items()}

    family_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    page_meta: dict[str, dict[str, Any]] = {}
    input_paths = dict(DEFAULT_INPUTS)
    if args.space_input:
        input_paths["space"] = Path(args.space_input)
    if args.boundary_input:
        input_paths["boundary"] = Path(args.boundary_input)
    if args.text_input:
        input_paths["text"] = Path(args.text_input)
    if args.symbol_input:
        input_paths["symbol"] = Path(args.symbol_input)
    for family, path in input_paths.items():
        rows, meta = load_family(path, family, max_row_ids=5 if args.smoke else None)
        family_rows[family] = rows
        page_meta.update({k: v for k, v in meta.items() if k not in page_meta})

    all_row_ids = sorted(set().union(*(set(rows) for rows in family_rows.values())))
    if args.smoke:
        all_row_ids = all_row_ids[:5]
    merged_rows: list[dict[str, Any]] = []
    before_cap: dict[str, dict[str, list[dict[str, Any]]]] = {family: defaultdict(list) for family in family_rows}
    after_cap: dict[str, dict[str, list[dict[str, Any]]]] = {family: defaultdict(list) for family in family_rows}
    counts = Counter()
    source_violations: list[dict[str, Any]] = []

    for row_id in all_row_ids:
        meta = page_meta.get(row_id, {"id": row_id, "image": None, "image_size": [512, 512]})
        stream: list[dict[str, Any]] = []
        spaces_for_row = list(family_rows.get("space", {}).get(row_id, []))
        for family, rows in family_rows.items():
            candidates = list(rows.get(row_id, []))
            before_cap[family][row_id] = candidates
            selected = cap_candidates(candidates, caps[family], family, spaces_for_row) if caps[family] else candidates
            after_cap[family][row_id] = selected
            counts[f"{family}_before_cap"] += len(candidates)
            counts[f"{family}_after_cap"] += len(selected)
            for cand in selected:
                if cand["source_integrity"] != integrity():
                    source_violations.append({"row_id": row_id, "candidate_id": cand["candidate_id"], "family": family})
                stream.append(cand)
        merged_rows.append({
            "id": row_id,
            "image": meta.get("image"),
            "image_size": meta.get("image_size") or [512, 512],
            "source_integrity": integrity(),
            "route_trace": {
                **integrity(),
                "stage": "detector_adapter_v18_merge",
                "schema_version": schema.get("schema_version"),
                "candidate_contract_version": "detector_candidate_contract_v1",
            },
            "scene_graph": {
                "nodes": [],
                "relations": [],
                "candidate_stream": stream,
                "candidate_contract_version": "detector_candidate_contract_v1",
                "candidate_counts": {
                    family: len(after_cap[family].get(row_id, []))
                    for family in sorted(family_rows)
                },
            },
        })

    gold = load_gold()
    audit = build_candidate_audit(
        family_rows=family_rows,
        gold=gold,
        caps=caps,
        selected_rows=after_cap,
        source_violations=source_violations,
    )
    audit.update(
        {
            "task": "IMG-MOE-V18-P1-008",
            "schema": str(args.schema),
            "rows": len(merged_rows),
            "caps": caps,
            "input_paths": {family: str(path) for family, path in sorted(input_paths.items())},
            "symbol_type_gate": {
                "eval": str(SYMBOL_TYPE_EVAL),
                "default_symbol_stream": str(DEFAULT_INPUTS["symbol"]),
                "typed_labels_adopted": bool(((load_json(SYMBOL_TYPE_EVAL).get("type_label_adoption") or {}).get("adopted"))),
                "policy": "merge consumes safe symbol stream by default; weak typed labels are diagnostic unless classifier locked gate adopts them",
            },
            "families_nonzero": {family: counts[f"{family}_after_cap"] > 0 for family in sorted(family_rows)},
            "success_criteria": {
                "one_command_emits_validated_stream": True,
                **audit.get("success_criteria", {}),
            },
        }
    )
    sweep = {
        "task": "IMG-MOE-V18-P1-008",
        "mode": "cap_sweep_summary",
        "note": "Current sweep reports selected cap recall loss against locked labels; broader threshold sweeps should reuse this schema.",
        "families": audit.get("recall_loss_accounting", {}),
    }
    return merged_rows, audit, sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default=str(SCHEMA))
    parser.add_argument("--boundary-input", default="")
    parser.add_argument("--space-input", default="")
    parser.add_argument("--text-input", default="")
    parser.add_argument("--symbol-input", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--no-caps", action="store_true")
    parser.add_argument("--output", default=str(REPORT / "detector_adapter_v18_routed_candidates.jsonl"))
    parser.add_argument("--audit", default=str(REPORT / "detector_adapter_v18_audit.json"))
    parser.add_argument("--threshold-sweep", default=str(REPORT / "detector_adapter_v18_threshold_sweep.json"))
    args = parser.parse_args()

    args.schema = Path(args.schema)
    args.output = Path(args.output)
    args.audit = Path(args.audit)
    args.threshold_sweep = Path(args.threshold_sweep)
    rows, audit, sweep = merge(args)
    write_jsonl(args.output, rows)
    write_json(args.audit, audit)
    write_json(args.threshold_sweep, sweep)
    print("task IMG-MOE-V18-P1-008")
    print("rows", audit["rows"])
    print("candidate_counts", json.dumps(audit["candidate_counts"], sort_keys=True))
    print("source_integrity_violations", audit["source_integrity"]["violations"])
    print("success", all(audit["success_criteria"].values()))


if __name__ == "__main__":
    main()

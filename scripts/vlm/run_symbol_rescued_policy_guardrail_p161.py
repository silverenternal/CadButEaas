#!/usr/bin/env python3
"""P161 guardrail for rescued symbol policy on a non-identical CubiCasa subset."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TARGETS = ROOT / "datasets/public_raster_moe_supervision_v19/locked.jsonl"
SOURCE_PREDS = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p113_p116_subset_20260516_142010_predictions.jsonl"
TRAINLIKE_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p160_best.jsonl"
OUT_E_OVERLAY = ROOT / "reports/vlm/symbol_policy_guardrail_p161_p155e_style.jsonl"
OUT_160_OVERLAY = ROOT / "reports/vlm/symbol_policy_guardrail_p161_p160_style.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_rescued_policy_guardrail_p161.json"
OUT_MD = ROOT / "reports/vlm/symbol_rescued_policy_guardrail_p161.md"

P155E_POLICY = {
    "thresholds": {"*": 0.02, "appliance": 0.32, "bathtub": 0.32, "column": 0.40, "equipment": 0.24, "generic_symbol": 0.45, "shower": 0.32, "sink": 0.24, "stair": 0.36},
    "max_page_keep": 80,
    "max_label_keep": None,
    "same_label_nms_iou": 0.45,
    "any_label_nms_iou": 0.75,
    "pre_nms_keep": 180,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 64: return "tiny"
    if value <= 256: return "small"
    if value <= 1024: return "medium"
    if value <= 4096: return "large"
    return "xlarge"


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0: return 0.0
    return inter / max(area(a) + area(b) - inter, 1e-9)


def label(item: dict[str, Any]) -> str:
    return str(item.get("label") or item.get("symbol_type") or "generic_symbol")


def score(item: dict[str, Any]) -> float:
    value = item.get("score") if item.get("score") is not None else item.get("confidence")
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def resize_center(box: list[float], width: float, height: float) -> list[float]:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]


def scale_center(box: list[float], sx: float, sy: float) -> list[float]:
    return resize_center(box, (box[2] - box[0]) * sx, (box[3] - box[1]) * sy)


def nms(items: list[dict[str, Any]], same_iou: float, any_iou: float) -> list[dict[str, Any]]:
    kept = []
    for item in sorted(items, key=score, reverse=True):
        box = bbox4(item.get("bbox"))
        if box is None: continue
        suppress = False
        for old in kept:
            old_box = bbox4(old.get("bbox"))
            if old_box is None: continue
            overlap = iou(box, old_box)
            if label(item) == label(old) and overlap >= same_iou:
                suppress = True; break
            if overlap >= any_iou:
                suppress = True; break
        if not suppress:
            kept.append(item)
    return kept


def apply_p155e(raw_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    t = P155E_POLICY["thresholds"]
    selected = [copy.deepcopy(item) for item in raw_items if bbox4(item.get("bbox")) is not None and score(item) >= t.get(label(item), t.get("*", 0.0))]
    selected = sorted(selected, key=score, reverse=True)[: P155E_POLICY["pre_nms_keep"]]
    selected = nms(selected, P155E_POLICY["same_label_nms_iou"], P155E_POLICY["any_label_nms_iou"])
    return sorted(selected, key=score, reverse=True)[: P155E_POLICY["max_page_keep"]]


def apply_p157_p160_geometry(item: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(item)
    box = bbox4(out.get("bbox"))
    if box is None: return out
    lab = label(out)
    b = bucket(box)
    if b == "small":
        box = scale_center(box, 1.20, 1.05)
    elif b == "medium":
        box = scale_center(box, 1.20, 0.95)
    if lab in {"sink", "equipment", "appliance"}:
        box = scale_center(box, 1.05, 1.0)
    if bucket(box) == "tiny" and score(out) >= 0.35:
        box = resize_center(box, 8.0, 8.0)
    out["bbox"] = box
    return out


def to_candidate(row_id: str, item: dict[str, Any], idx: int, policy_id: str) -> dict[str, Any] | None:
    box = bbox4(item.get("bbox"))
    if box is None: return None
    lab = label(item)
    return {
        "id": f"{row_id}_{policy_id}_symbol_{idx:05d}",
        "target_id": f"{row_id}_{policy_id}_symbol_{idx:05d}",
        "symbol_type": lab,
        "bbox": box,
        "confidence": score(item),
        "source": f"symbol_policy_guardrail_{policy_id}",
        "metadata": {"p161_policy_id": policy_id, "source_tile_id": item.get("tile_id"), "label_id": item.get("label_id")},
    }


def build_overlay_rows(target_by_id: dict[str, dict[str, Any]], pred_rows: list[dict[str, Any]], policy_id: str, geometry: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    missing = []
    for pred_row in pred_rows:
        row_id = str(pred_row.get("row_id") or "")
        target = target_by_id.get(row_id)
        if target is None:
            missing.append(row_id); continue
        selected = apply_p155e(list(pred_row.get("predicted_symbols") or []))
        if geometry:
            selected = [apply_p157_p160_geometry(item) for item in selected]
        candidates = [c for idx, item in enumerate(selected) for c in [to_candidate(row_id, item, idx, policy_id)] if c is not None]
        row = copy.deepcopy(target)
        row["row_id"] = row_id
        row["image_path"] = row.get("image")
        row["symbol_candidates"] = candidates
        row["expected_json"] = {"symbol_candidates": [copy.deepcopy(item) for item in candidates]}
        row["symbol_policy_overlay"] = {"policy_id": policy_id, "description": "P161 non-identical subset guardrail materialization", "source_predictions": str(SOURCE_PREDS)}
        rows.append(row)
    return rows, {"rows": len(rows), "missing_target_rows": missing[:20], "missing_target_count": len(missing)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(TARGETS))
    parser.add_argument("--source-predictions", default=str(SOURCE_PREDS))
    parser.add_argument("--trainlike-overlay", default=str(TRAINLIKE_OVERLAY))
    parser.add_argument("--out-p155e-overlay", default=str(OUT_E_OVERLAY))
    parser.add_argument("--out-p160-overlay", default=str(OUT_160_OVERLAY))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()

    targets = load_jsonl(Path(args.targets))
    target_by_id = {str(row.get("id") or row.get("row_id")): row for row in targets}
    pred_rows = load_jsonl(Path(args.source_predictions))
    trainlike_ids = {str(row.get("row_id") or row.get("id")) for row in load_jsonl(Path(args.trainlike_overlay))}
    source_ids = {str(row.get("row_id")) for row in pred_rows}
    overlap = sorted(source_ids & trainlike_ids)

    e_rows, e_summary = build_overlay_rows(target_by_id, pred_rows, "p161_p155e_style", False)
    p160_rows, p160_summary = build_overlay_rows(target_by_id, pred_rows, "p161_p160_style", True)
    write_jsonl(Path(args.out_p155e_overlay), e_rows)
    write_jsonl(Path(args.out_p160_overlay), p160_rows)

    report = {
        "id": "SCI-P2-161-symbol-rescued-policy-guardrail",
        "created_on": "2026-05-17",
        "decision": "materialized_non_identical_subset_ready_for_eval",
        "claim_boundary": "Guardrail applies P160-style postprocess to an available non-identical CubiCasa locked subset with raw v28 predictions. It is still CubiCasa-only and post-hoc; not a full-raster or cross-dataset claim.",
        "source_integrity": "Runtime policy uses raw detector predicted bbox/label/score plus fixed P155E/P157/P160 config. Gold targets are included only for evaluation.",
        "inputs": {"targets": args.targets, "source_predictions": args.source_predictions, "trainlike_overlay": args.trainlike_overlay},
        "row_counts": {"source_prediction_rows": len(pred_rows), "materialized_rows": len(p160_rows), "overlap_with_p160_74row_subset": len(overlap), "overlap_sample": overlap[:20]},
        "policy": {"p155e": P155E_POLICY, "p157_geometry": "small 1.20x/1.05y; medium 1.20x/0.95y; sink/equipment/appliance x1.05", "p160_tiny": "tiny after geometry and score>=0.35 -> 8x8"},
        "materialization": {"p155e_style": e_summary, "p160_style": p160_summary},
        "outputs": {"p155e_style_overlay": args.out_p155e_overlay, "p160_style_overlay": args.out_p160_overlay, "config_json": args.out_json, "report_md": args.out_md},
    }
    write_json(Path(args.out_json), report)
    lines = ["# P161 Symbol Rescued Policy Guardrail", "", "Decision: **materialized_non_identical_subset_ready_for_eval**", "", "## Scope", "", report["claim_boundary"], "", "## Materialization", "", f"- source prediction rows: `{len(pred_rows)}`", f"- materialized rows: `{len(p160_rows)}`", f"- overlap with P160 74-row subset: `{len(overlap)}`", "", "## Artifacts", ""]
    for value in report["outputs"].values(): lines.append(f"- `{value}`")
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(p160_rows), "overlap_with_trainlike": len(overlap), "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

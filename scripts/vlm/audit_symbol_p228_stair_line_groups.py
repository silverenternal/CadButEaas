#!/usr/bin/env python3
"""Audit stair annotation line/group structure for P228 representation rescue."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p228_stair_line_group_audit.json"
OUT_MD = ROOT / "reports/vlm/symbol_p228_stair_line_group_audit.md"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def box(target: dict[str, Any]) -> list[float]:
    return [float(v) for v in target["bbox"]]


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def is_sentinel(b: list[float]) -> bool:
    return abs(b[0]) < 1e-6 and abs(b[1]) < 1e-6 and abs((b[2] - b[0]) - 10.0) < 1e-6 and abs((b[3] - b[1]) - 12.0) < 1e-6


def bucket(b: list[float]) -> str:
    a = area(b)
    if a <= 64:
        return "tiny_le_64"
    if a <= 256:
        return "small_le_256"
    if a <= 1024:
        return "medium_le_1024"
    if a <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def orientation(b: list[float]) -> str:
    w = max(1e-6, b[2] - b[0]); h = max(1e-6, b[3] - b[1])
    if min(w, h) < 1.0:
        return "ultra_thin_horizontal" if w >= h else "ultra_thin_vertical"
    if w / h >= 3.0:
        return "wide"
    if h / w >= 3.0:
        return "tall"
    return "boxy"


def expanded_overlap(a: list[float], b: list[float], margin: float) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def close_enough(a: list[float], b: list[float], margin: float, center_gap: float) -> bool:
    if expanded_overlap(a, b, margin):
        return True
    acx, acy = center(a); bcx, bcy = center(b)
    dist = math.hypot(acx - bcx, acy - bcy)
    if dist > center_gap:
        return False
    # Allow nearby stair components when they align along one axis.
    aw, ah = max(1e-6, a[2] - a[0]), max(1e-6, a[3] - a[1])
    bw, bh = max(1e-6, b[2] - b[0]), max(1e-6, b[3] - b[1])
    x_overlap = min(a[2], b[2]) - max(a[0], b[0])
    y_overlap = min(a[3], b[3]) - max(a[1], b[1])
    return x_overlap >= -margin and abs(acy - bcy) <= max(ah, bh, margin) * 1.5 or y_overlap >= -margin and abs(acx - bcx) <= max(aw, bw, margin) * 1.5


def components(boxes: list[list[float]], margin: float, center_gap: float) -> list[list[int]]:
    n = len(boxes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if close_enough(boxes[i], boxes[j], margin, center_gap):
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def union_box(items: list[list[float]]) -> list[float]:
    return [min(b[0] for b in items), min(b[1] for b in items), max(b[2] for b in items), max(b[3] for b in items)]


def matched_by_base(gbox: list[float], preds: list[dict[str, Any]], threshold: float = 0.30) -> bool:
    return any(str(pred.get("label")) == "stair" and bbox_iou([float(v) for v in pred["bbox"]], gbox) >= threshold for pred in preds)


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows, base_preds, _golds = load_p206g(Path(args.base))
    totals = Counter()
    orientation_counts = Counter()
    bucket_counts = Counter()
    group_size_counts = Counter()
    rows_out: list[dict[str, Any]] = []
    for row in rows:
        rid = row_id(row)
        stairs = [target for target in (row.get("targets") or {}).get("symbol", []) if str(target.get("semantic_type")) == "stair"]
        if not stairs:
            continue
        stair_boxes = [box(target) for target in stairs]
        sentinel = [b for b in stair_boxes if is_sentinel(b)]
        real = [b for b in stair_boxes if not is_sentinel(b)]
        groups = components(real, args.merge_margin, args.center_gap) if real else []
        group_items = []
        for group in groups:
            members = [real[i] for i in group]
            merged = union_box(members)
            group_items.append({
                "size": len(members),
                "bbox": [round(v, 3) for v in merged],
                "area": round(area(merged), 3),
                "member_buckets": dict(Counter(bucket(b) for b in members)),
                "member_orientations": dict(Counter(orientation(b) for b in members)),
                "base_matched_any_member": any(matched_by_base(b, base_preds.get(rid, []), args.match_iou) for b in members),
                "base_matched_merged": matched_by_base(merged, base_preds.get(rid, []), args.match_iou),
            })
            group_size_counts[len(members)] += 1
        row_counts = Counter(bucket(b) for b in real)
        row_orient = Counter(orientation(b) for b in real)
        totals.update({
            "rows_with_stair": 1,
            "stair_targets": len(stair_boxes),
            "sentinel_targets": len(sentinel),
            "real_targets": len(real),
            "ultra_thin_targets": sum(1 for b in real if min(b[2] - b[0], b[3] - b[1]) < args.ultra_thin_side),
            "base_missed_real_targets": sum(1 for b in real if not matched_by_base(b, base_preds.get(rid, []), args.match_iou)),
            "groups": len(groups),
            "multi_member_groups": sum(1 for g in groups if len(g) > 1),
        })
        bucket_counts.update(row_counts)
        orientation_counts.update(row_orient)
        rows_out.append({
            "row_id": rid,
            "stair_targets": len(stair_boxes),
            "sentinel_targets": len(sentinel),
            "real_targets": len(real),
            "bucket_counts": dict(row_counts),
            "orientation_counts": dict(row_orient),
            "groups": len(groups),
            "multi_member_groups": sum(1 for g in groups if len(g) > 1),
            "largest_group": max((len(g) for g in groups), default=0),
            "group_items": group_items[:20],
        })
    report = {
        "id": "P228_stair_line_group_audit",
        "base": rel(Path(args.base)),
        "config": vars(args),
        "totals": dict(totals),
        "bucket_counts": dict(bucket_counts.most_common()),
        "orientation_counts": dict(orientation_counts.most_common()),
        "group_size_counts": dict(sorted(group_size_counts.items())),
        "rows": sorted(rows_out, key=lambda r: (-r["multi_member_groups"], -r["real_targets"], r["row_id"])),
        "interpretation": "Stair labels include sentinel [0,0,10,12] artifacts and mixed object/line components. P228 should drop sentinels and merge connected real stair components into object-level targets before detector training.",
        "claim_boundary": "Offline annotation audit only; no runtime features are introduced.",
    }
    return report


def render(report: dict[str, Any]) -> str:
    t = report["totals"]
    lines = [
        "# P228 Stair Line Group Audit",
        "",
        "## Totals",
        f"- Rows with stair: `{t.get('rows_with_stair', 0)}`",
        f"- Stair targets / real / sentinel: `{t.get('stair_targets', 0)}` / `{t.get('real_targets', 0)}` / `{t.get('sentinel_targets', 0)}`",
        f"- Ultra-thin real targets: `{t.get('ultra_thin_targets', 0)}`",
        f"- Base-missed real targets: `{t.get('base_missed_real_targets', 0)}`",
        f"- Groups / multi-member groups: `{t.get('groups', 0)}` / `{t.get('multi_member_groups', 0)}`",
        "",
        "## Buckets",
        "| Bucket | Count |",
        "|---|---:|",
    ]
    for key, value in report["bucket_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Orientations", "| Orientation | Count |", "|---|---:|"]
    for key, value in report["orientation_counts"].items():
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Rows With Most Merge Candidates", "| Row | Real | Sentinel | Groups | Multi Groups | Largest |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["rows"][:20]:
        lines.append(f"| {row['row_id']} | {row['real_targets']} | {row['sentinel_targets']} | {row['groups']} | {row['multi_member_groups']} | {row['largest_group']} |")
    lines += ["", "## Interpretation", f"- {report['interpretation']}", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    parser.add_argument("--merge-margin", type=float, default=18.0)
    parser.add_argument("--center-gap", type=float, default=160.0)
    parser.add_argument("--match-iou", type=float, default=0.30)
    parser.add_argument("--ultra-thin-side", type=float, default=1.0)
    args = parser.parse_args()
    report = build(args)
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"totals": report["totals"], "bucket_counts": report["bucket_counts"], "orientation_counts": report["orientation_counts"], "outputs": [rel(Path(args.out_json)), rel(Path(args.out_md))]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render deterministic visual hard-case pack for v18 raster-only MoE debugging."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
OUT = REPORT / "visual_hard_cases_v18"
ASSETS = OUT / "assets"

DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_routed_candidates.jsonl"
DEFAULT_TOPOLOGY = REPORT / "topology_relations_v18_candidates.jsonl"
DEFAULT_REFINER = REPORT / "scene_graph_refiner_v18_final_predictions.jsonl"
DEFAULT_REFINER_DATASET = ROOT / "datasets/image_only_scene_graph_refiner_v18/locked.jsonl"
GOLD_FILES = {
    "boundary": ROOT / "datasets/image_only_boundary_detector_v18/locked.jsonl",
    "space": ROOT / "datasets/image_only_room_polygon_v18/locked.jsonl",
    "symbol": ROOT / "datasets/image_only_symbol_detector_v18/locked.jsonl",
    "text": ROOT / "datasets/image_only_text_ocr_v18/locked.jsonl",
}
GOLD_KEYS = {"boundary": "boxes", "space": "rooms", "symbol": "symbols", "text": "texts"}
COLORS = {
    "boundary": (220, 40, 40, 210),
    "space": (20, 145, 85, 150),
    "symbol": (120, 65, 190, 210),
    "text": (30, 30, 30, 230),
    "gold": (255, 185, 0, 230),
    "refiner": (0, 110, 220, 230),
    "relation": (0, 150, 170, 180),
}


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


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def load_gold() -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family, path in GOLD_FILES.items():
        by_row: dict[str, list[dict[str, Any]]] = {}
        for row in load_jsonl(path):
            by_row[row["id"]] = (row.get("targets") or {}).get(GOLD_KEYS[family]) or []
        out[family] = by_row
    return out


def row_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def stream(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    return (row.get("scene_graph") or {}).get("candidate_stream") or []


def relations(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not row:
        return []
    return (row.get("scene_graph") or {}).get("relations") or []


def draw_boxes(img: Image.Image, items: list[dict[str, Any]], color: tuple[int, int, int, int], label_key: str, limit: int) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    for idx, item in enumerate(items[:limit]):
        b = bbox(item.get("bbox"))
        if not b:
            continue
        draw.rectangle(b, outline=color, width=2)
        label = str(item.get(label_key) or item.get("candidate_type") or item.get("family") or idx)[:24]
        draw.text((b[0] + 2, max(0, b[1] - 11)), label, fill=color)


def draw_relations(img: Image.Image, rels: list[dict[str, Any]], candidates: dict[str, dict[str, Any]], limit: int = 120) -> None:
    draw = ImageDraw.Draw(img, "RGBA")
    for rel in sorted(rels, key=lambda r: (-float(r.get("confidence") or 0.0), str(r.get("relation_id"))))[:limit]:
        left = candidates.get(rel.get("source_candidate_id"))
        right = candidates.get(rel.get("target_candidate_id"))
        lb = bbox(left.get("bbox")) if left else None
        rb = bbox(right.get("bbox")) if right else None
        if not lb or not rb:
            continue
        draw.line([center(lb), center(rb)], fill=COLORS["relation"], width=1)


def render_case(row_id: str, adapter: dict[str, Any], topology: dict[str, Any] | None, refiner: dict[str, Any] | None, gold: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, str]:
    image_path = ROOT / str(adapter.get("image"))
    base = Image.open(image_path).convert("RGBA")
    panels: dict[str, Image.Image] = {
        "gold": base.copy(),
        "adapter": base.copy(),
        "topology": base.copy(),
        "refiner": base.copy(),
    }
    for family, items in gold.items():
        draw_boxes(panels["gold"], items.get(row_id, []), COLORS["gold"], "target_id", 160)
    candidates = sorted(stream(adapter), key=lambda c: (str(c.get("family")), -float(c.get("confidence") or 0.0), str(c.get("candidate_id"))))
    for family in ["space", "boundary", "symbol", "text"]:
        draw_boxes(
            panels["adapter"],
            [c for c in candidates if c.get("family") == family],
            COLORS[family],
            "candidate_id",
            {"boundary": 180, "space": 80, "symbol": 100, "text": 60}[family],
        )
    cand_by_id = {c.get("candidate_id"): c for c in stream(adapter)}
    panels["topology"] = panels["adapter"].copy()
    draw_relations(panels["topology"], relations(topology), cand_by_id)
    for family in ["space", "boundary", "symbol", "text"]:
        draw_boxes(
            panels["refiner"],
            [c for c in stream(refiner) if c.get("family") == family],
            COLORS["refiner"] if family != "space" else COLORS["space"],
            "candidate_id",
            {"boundary": 160, "space": 60, "symbol": 80, "text": 40}[family],
        )

    ASSETS.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, panel in panels.items():
        path = ASSETS / f"{row_id}_{name}.png"
        panel.convert("RGB").save(path)
        paths[name] = f"assets/{path.name}"
    return paths


def page_scores(dataset_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_page: dict[str, dict[str, Any]] = defaultdict(lambda: {"counts": Counter(), "positive": Counter(), "typed_wrong": 0})
    for item in dataset_rows:
        row = by_page[str(item.get("row_id"))]
        family = str(item.get("family"))
        row["counts"][family] += 1
        if item.get("label_keep"):
            row["positive"][family] += 1
        if family == "symbol" and item.get("label_keep") and item.get("typed_correct") is False:
            row["typed_wrong"] += 1
    for row in by_page.values():
        row["counts"] = dict(row["counts"])
        row["positive"] = dict(row["positive"])
    return by_page


def choose_cases(adapter_rows: dict[str, dict[str, Any]], scores: dict[str, dict[str, Any]], max_pages: int) -> list[dict[str, Any]]:
    buckets = {
        "ocr_miss": lambda s: s["positive"].get("text", 0),
        "symbol_confusion": lambda s: s.get("typed_wrong", 0),
        "boundary_duplicate_flood": lambda s: s["counts"].get("boundary", 0) - s["positive"].get("boundary", 0),
        "room_topology_gap": lambda s: s["counts"].get("space", 0) - s["positive"].get("space", 0),
        "false_positive_flood": lambda s: sum(s["counts"].values()) - sum(s["positive"].values()),
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_bucket = max(1, max_pages // max(len(buckets), 1))
    for bucket, fn in buckets.items():
        ranked = sorted(scores.items(), key=lambda kv: (fn(kv[1]), kv[0]), reverse=True)
        added = 0
        for row_id, score in ranked:
            if row_id not in adapter_rows or row_id in seen or fn(score) <= 0:
                continue
            selected.append({"row_id": row_id, "bucket": bucket, "bucket_score": fn(score), "score": score})
            seen.add(row_id)
            added += 1
            if added >= per_bucket:
                break
    if len(selected) < max_pages:
        ranked = sorted(scores.items(), key=lambda kv: (sum(kv[1]["counts"].values()) - sum(kv[1]["positive"].values()), kv[0]), reverse=True)
        for row_id, score in ranked:
            if row_id not in adapter_rows or row_id in seen:
                continue
            selected.append({"row_id": row_id, "bucket": "fill_false_positive_flood", "bucket_score": sum(score["counts"].values()) - sum(score["positive"].values()), "score": score})
            seen.add(row_id)
            if len(selected) >= max_pages:
                break
    return selected[:max_pages]


def build(args: argparse.Namespace) -> dict[str, Any]:
    adapter = row_map(Path(args.adapter))
    topology = row_map(Path(args.topology)) if args.include_topology else {}
    refiner = row_map(Path(args.refiner)) if args.include_refiner else {}
    gold = load_gold()
    scores = page_scores(load_jsonl(Path(args.refiner_dataset)))
    max_pages = 8 if args.smoke else args.max_pages
    cases = choose_cases(adapter, scores, max_pages)
    rendered = []
    for case in cases:
        row_id = case["row_id"]
        paths = render_case(row_id, adapter[row_id], topology.get(row_id), refiner.get(row_id), gold)
        rendered.append({**case, "panels": paths})

    bucket_counts = Counter(item["bucket"] for item in rendered)
    manifest = {
        "task": "IMG-MOE-V18-P2-011",
        "rows_rendered": len(rendered),
        "bucket_counts": dict(bucket_counts),
        "cases": rendered,
        "source_reports": {
            "adapter": str(args.adapter),
            "topology": str(args.topology) if args.include_topology else None,
            "refiner": str(args.refiner) if args.include_refiner else None,
            "refiner_dataset": str(args.refiner_dataset),
        },
        "deterministic": True,
    }
    write_json(OUT / "manifest.json", manifest)
    write_html(manifest)
    return manifest


def write_html(manifest: dict[str, Any]) -> None:
    cards = []
    for case in manifest["cases"]:
        panels = case["panels"]
        imgs = "".join(
            f"<figure><img src='{html.escape(src)}'><figcaption>{html.escape(name)}</figcaption></figure>"
            for name, src in panels.items()
        )
        detail = html.escape(json.dumps({k: v for k, v in case.items() if k not in {"panels"}}, ensure_ascii=False, indent=2))
        cards.append(f"<section><h2>{html.escape(case['row_id'])} / {html.escape(case['bucket'])}</h2><div class='grid'>{imgs}</div><pre>{detail}</pre></section>")
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>CadStruct v18 Visual Hard Cases</title>
<style>
body{{font-family:Arial,sans-serif;margin:24px;color:#151515;background:#fff}}
h1{{font-size:24px}} h2{{font-size:18px;margin-top:28px}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;align-items:start}}
figure{{margin:0}} img{{width:100%;border:1px solid #999;background:white}}
figcaption{{font-size:12px;margin-top:4px}} pre{{background:#f5f5f5;padding:10px;overflow:auto;font-size:12px}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr));}}}}
</style>
<h1>CadStruct v18 Visual Hard Cases</h1>
<pre>{html.escape(json.dumps({k: v for k, v in manifest.items() if k != "cases"}, ensure_ascii=False, indent=2))}</pre>
{''.join(cards)}
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--topology", default=str(REPORT / "topology_relations_v18_candidates.jsonl"))
    parser.add_argument("--refiner", default=str(DEFAULT_REFINER))
    parser.add_argument("--refiner-dataset", default=str(DEFAULT_REFINER_DATASET))
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    parser.add_argument("--source", default="adapter")
    parser.add_argument("--include-topology", action="store_true")
    parser.add_argument("--include-refiner", action="store_true")
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({"rows_rendered": manifest["rows_rendered"], "bucket_counts": manifest["bucket_counts"], "index": str(OUT / "index.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

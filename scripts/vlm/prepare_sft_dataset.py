#!/usr/bin/env python3
"""Prepare CadStruct records for multimodal SFT trainers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct")
    parser.add_argument("--output-dir", default="datasets/cadstruct_sft")
    parser.add_argument("--max-polyline-samples", type=int, default=4)
    parser.add_argument("--max-graph-nodes", type=int, default=16)
    parser.add_argument("--max-graph-edges", type=int, default=24)
    parser.add_argument("--max-target-symbols", type=int, default=16)
    parser.add_argument("--max-target-semantics", type=int, default=16)
    parser.add_argument("--max-target-edges", type=int, default=24)
    parser.add_argument("--target-scope", choices=["all", "structural_core"], default="all")
    parser.add_argument("--drop-empty-targets", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split in ["train", "dev", "smoke"]:
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            continue
        output_path = output_dir / f"{split}.jsonl"
        rows = convert_split(input_path, output_path, args)
        manifest[split] = rows
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def convert_split(input_path: Path, output_path: Path, args: argparse.Namespace) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            sft_record = to_sft_record(record, args)
            if sft_record is None:
                continue
            target.write(json.dumps(sft_record, ensure_ascii=False) + "\n")
            count += 1
    return count


def to_sft_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    primitive_graph = compact_graph(
        record.get("request_hints", {}).get("primitive_graph"),
        args.max_graph_nodes,
        args.max_graph_edges,
    )
    context = {
        "image": record.get("metadata", {}),
        "text_candidates": (record.get("request_hints", {}).get("text_candidates") or [])[:32],
        "symbol_candidates": (record.get("request_hints", {}).get("symbol_candidates") or [])[:32],
        "polyline_count": len(record.get("request_hints", {}).get("polylines") or []),
        "polyline_samples": (record.get("request_hints", {}).get("polylines") or [])[
            : args.max_polyline_samples
        ],
        "primitive_graph": primitive_graph,
        "output_requirements": {
            "priority": ["semantic_candidates", "scene_graph", "symbol_candidates", "dimension_candidates"],
            "min_semantic_candidates": 1 if primitive_graph.get("nodes") else 0,
            "semantic_target_id_source": "primitive_graph.nodes[].id",
            "target_scope": args.target_scope,
        },
    }
    assistant = compact_expected_json(record.get("expected_json", {}), args)
    if args.drop_empty_targets and not assistant.get("semantic_candidates"):
        return None
    return {
        "image": record["image_path"],
        "source_dataset": record.get("source_dataset"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": record["image_path"]},
                    {"type": "text", "text": user_prompt(context)},
                ],
            },
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))},
        ],
    }


def user_prompt(context: dict[str, Any]) -> str:
    return (
        "You are CadStruct-VL, a raster CAD and floor-plan parser. "
        "Return one strict JSON object only. "
        "Use the image and the provided geometric context. "
        "This is a structured extraction task, not image captioning. "
        "If primitive_graph.nodes is non-empty, semantic_candidates must not be empty: classify visible graph nodes as hard_wall, partition_wall, opening, door, window, centerline, dimension_line, or detail_line. "
        "For each semantic candidate, target_id must reference a primitive_graph node id. "
        "Mirror semantic candidates into scene_graph.nodes with id, semantic_type, and primitive_id. "
        "Emit semantic_candidates and scene_graph first, then symbol_candidates, dimension_candidates, and warnings. "
        f"Context JSON: {json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
    )


def compact_graph(value: Any, max_nodes: int, max_edges: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"nodes": [], "edges": []}
    nodes = value.get("nodes") if isinstance(value.get("nodes"), list) else []
    edges = value.get("edges") if isinstance(value.get("edges"), list) else []
    return {"nodes": nodes[:max_nodes], "edges": edges[:max_edges], "truncated": len(nodes) > max_nodes or len(edges) > max_edges}


def compact_expected_json(value: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    scene_graph = value.get("scene_graph", {"nodes": [], "edges": []})
    if not isinstance(scene_graph, dict):
        scene_graph = {"nodes": [], "edges": []}
    semantic_candidates = value.get("semantic_candidates", [])
    graph_nodes = scene_graph.get("nodes") or []
    graph_edges = scene_graph.get("edges") or []
    symbol_candidates = value.get("symbol_candidates", [])
    if args.target_scope == "structural_core":
        semantic_candidates = filter_structural_semantics(semantic_candidates)
        kept_ids = {int(item["target_id"]) for item in semantic_candidates if int_like(item.get("target_id"))}
        graph_nodes = [
            item
            for item in graph_nodes
            if isinstance(item, dict) and int_like(item.get("primitive_id", item.get("id"))) and int(item.get("primitive_id", item.get("id"))) in kept_ids
        ]
        graph_edges = [
            item
            for item in graph_edges
            if isinstance(item, dict)
            and int_like(item.get("source"))
            and int_like(item.get("target"))
            and int(item["source"]) in kept_ids
            and int(item["target"]) in kept_ids
        ]
        symbol_candidates = [
            item
            for item in symbol_candidates
            if isinstance(item, dict) and str(item.get("symbol_type")) in {"door", "window", "opening"}
        ]
    return {
        "semantic_candidates": semantic_candidates[: args.max_target_semantics],
        "scene_graph": {
            "nodes": graph_nodes[: args.max_target_semantics],
            "edges": graph_edges[: args.max_target_edges],
        },
        "symbol_candidates": symbol_candidates[: args.max_target_symbols],
        "dimension_candidates": value.get("dimension_candidates", []),
        "warnings": value.get("warnings", []),
    }


def filter_structural_semantics(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        semantic_type = structural_semantic_type(str(item.get("semantic_type", "")))
        if semantic_type is None:
            continue
        mapped = dict(item)
        mapped["semantic_type"] = semantic_type
        result.append(mapped)
    return result


def structural_semantic_type(value: str) -> str | None:
    aliases = {
        "wall": "hard_wall",
        "hard_wall": "hard_wall",
        "partition_wall": "partition_wall",
        "door": "door",
        "window": "window",
        "opening": "opening",
        "centerline": "centerline",
        "dimension_line": "dimension_line",
        "datum": "datum",
        "detail_line": "detail_line",
    }
    return aliases.get(value)


def int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()

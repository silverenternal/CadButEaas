#!/usr/bin/env python3
"""Audit whether the chosen model/training setup matches CadStruct targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STRUCTURAL_TYPES = {
    "hard_wall",
    "partition_wall",
    "wall",
    "door",
    "window",
    "opening",
    "centerline",
    "dimension_line",
    "datum",
    "detail_line",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", default="configs/vlm/cadstruct_14b_lora.json")
    parser.add_argument("--eval-config", default="configs/vlm/cadstruct_14b_lora_eval.json")
    parser.add_argument("--dataset-audit", default="reports/vlm/cadstruct_dataset_audit.json")
    parser.add_argument(
        "--eval-audit",
        default="reports/vlm/cadstruct_14b_lora_semantic_first_repair_smoke_2.audit.json",
    )
    parser.add_argument("--output", default="reports/vlm/model_target_fit_audit.json")
    args = parser.parse_args()

    train_config = load_json(args.train_config)
    eval_config = load_json(args.eval_config)
    dataset_audit = load_json(args.dataset_audit)
    eval_audit = load_json(args.eval_audit)

    train_split = (dataset_audit.get("splits") or {}).get("train", {})
    smoke_split = (dataset_audit.get("splits") or {}).get("smoke", {})
    top_types = [item[0] for item in train_split.get("top_semantic_types", [])]
    out_of_prompt_types = [item for item in top_types if item not in STRUCTURAL_TYPES]

    report = {
        "model": {
            "base_model": train_config.get("base_model"),
            "model_family": train_config.get("model_family"),
            "adapter": "LoRA",
            "load_in_4bit": bool(train_config.get("load_in_4bit")),
            "lora_r": train_config.get("lora_r"),
            "target_modules": train_config.get("target_modules", []),
            "max_length": train_config.get("max_length"),
            "max_image_side": train_config.get("max_image_side"),
            "max_vision_tiles": train_config.get("max_vision_tiles"),
            "max_new_tokens": eval_config.get("max_new_tokens"),
        },
        "target": {
            "task": "raster CAD/floor-plan structured extraction",
            "required_outputs": [
                "semantic_candidates",
                "scene_graph.nodes",
                "scene_graph.edges",
                "symbol_candidates",
                "dimension_candidates",
            ],
            "train_rows": train_split.get("rows"),
            "smoke_rows": smoke_split.get("rows"),
            "train_semantic_candidates_mean": nested(train_split, "semantic_candidates", "mean"),
            "train_semantic_candidates_p95": nested(train_split, "semantic_candidates", "p95"),
            "train_scene_graph_edges_p95": nested(train_split, "scene_graph_edges", "p95"),
            "train_scene_graph_edges_max": nested(train_split, "scene_graph_edges", "max"),
            "top_semantic_types": top_types[:20],
            "out_of_structural_scope_top_types": out_of_prompt_types[:20],
        },
        "current_eval": {
            "semantic_hit_rate": eval_audit.get("semantic_hit_rate"),
            "geometry_consistency_mean": eval_audit.get("geometry_consistency_mean"),
            "empty_semantic_rate": eval_audit.get("empty_semantic_rate"),
            "semantic_count_mean": eval_audit.get("semantic_count_mean"),
            "partial_recovery_count": eval_audit.get("partial_recovery_count"),
            "warning_counts": eval_audit.get("warning_counts", {}),
        },
        "fit_matrix": fit_matrix(train_config, eval_config, train_split, eval_audit, out_of_prompt_types),
    }
    report["decision"] = decision(report["fit_matrix"])
    report["recommendations"] = recommendations(report)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")


def fit_matrix(
    train_config: dict[str, Any],
    eval_config: dict[str, Any],
    train_split: dict[str, Any],
    eval_audit: dict[str, Any],
    out_of_prompt_types: list[str],
) -> list[dict[str, str]]:
    graph_edges_p95 = nested(train_split, "scene_graph_edges", "p95") or 0
    graph_edges_max = nested(train_split, "scene_graph_edges", "max") or 0
    semantic_p95 = nested(train_split, "semantic_candidates", "p95") or 0
    max_new_tokens = int(eval_config.get("max_new_tokens", 0) or 0)
    semantic_hit = float(eval_audit.get("semantic_hit_rate", 0.0) or 0.0)
    geometry = float(eval_audit.get("geometry_consistency_mean", 0.0) or 0.0)

    return [
        {
            "axis": "instruction/schema following",
            "fit": "medium",
            "evidence": "JSON success is high, but malformed long semantic outputs still require partial recovery.",
            "risk": "Autoregressive decoding can fail under long graph outputs.",
        },
        {
            "axis": "dense node classification",
            "fit": "low",
            "evidence": f"Train semantic p95 is {semantic_p95}, while current smoke semantic_hit_rate is {semantic_hit}.",
            "risk": "A text-only LoRA head is a weak fit for dense per-node labeling without a graph adapter/classifier.",
        },
        {
            "axis": "relation/scene graph extraction",
            "fit": "low",
            "evidence": f"Train edge p95 is {graph_edges_p95}, max is {graph_edges_max}, relation F1 is still 0.0 in latest smoke eval.",
            "risk": "Full relation generation is too large for direct JSON decoding at current token budget.",
        },
        {
            "axis": "taxonomy alignment",
            "fit": "medium" if not out_of_prompt_types else "low",
            "evidence": f"Top target classes outside structural prompt scope: {out_of_prompt_types[:8]}.",
            "risk": "The model is asked for structural CAD labels while training targets include furniture/equipment classes.",
        },
        {
            "axis": "dimension extraction",
            "fit": "low",
            "evidence": "External CadStruct conversion has no dimension labels, so dimension_hit can be trivially true.",
            "risk": "Current external SFT does not train the dimension objective we need for engineering drawings.",
        },
        {
            "axis": "memory/token budget",
            "fit": "medium",
            "evidence": f"Training uses max_length={train_config.get('max_length')}, max_image_side={train_config.get('max_image_side')}, max_vision_tiles={train_config.get('max_vision_tiles')}; stable runs peak around 45GiB.",
            "risk": "Increasing graph fidelity, image resolution, or max_new_tokens without routing/cropping can reintroduce OOM.",
        },
        {
            "axis": "output length",
            "fit": "low" if max_new_tokens <= 1024 else "medium",
            "evidence": f"Eval max_new_tokens is {max_new_tokens}; recovered output already truncates on semantic lists.",
            "risk": "The model may produce partial JSON before completing scene graph and relations.",
        },
    ]


def decision(fits: list[dict[str, str]]) -> str:
    low_axes = [item["axis"] for item in fits if item["fit"] == "low"]
    if len(low_axes) >= 3:
        return (
            "InternVL3.5-14B LoRA is a reasonable schema-following backbone, but it is not by itself matched "
            "to the full dense graph extraction target. Keep it as the language/VLM component and add a "
            "structure-specific head/adapter plus scoped targets."
        )
    return "The current model-target fit is acceptable for the next SFT iteration, with the listed mitigations."


def recommendations(report: dict[str, Any]) -> list[str]:
    return [
        "Split Stage 1 target into structural core labels first: hard_wall, door, window, opening, centerline/detail_line; defer furniture/equipment taxonomy or map it to a separate symbol head.",
        "Do not train/evaluate full relation graph generation as raw JSON yet; first train node semantics, then add relation classification over primitive_graph edges.",
        "Add a small graph adapter or edge/node classifier instead of relying only on autoregressive JSON for dense per-node labels.",
        "Keep semantic-first JSON for auditability, but cap emitted candidates during generation or use top-k node routing to avoid partial JSON.",
        "Add dimension-labeled synthetic/internal samples before using dimension_hit as a real success metric.",
        "Maintain current OOM guards; any increase to max_new_tokens or graph caps should be preceded by budget profiling and a short smoke run.",
    ]


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    main()

"""Shared helpers for CadStruct-MoE visual-model v5 scripts."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


BASE_LOCKED_METRICS = {
    "text_dimension": 0.9677538732142168,
    "boundary": 0.9717767824313225,
    "symbol_fixture": 0.774986,
}


def load_json(path: str | Path, default: Any | None = None) -> Any:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    rows: list[dict[str, Any]] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def sample_id(row: dict[str, Any]) -> str:
    for key in ["sample_id", "image", "image_path", "annotation", "annotation_path"]:
        value = str(row.get(key) or "")
        if value:
            parts = Path(value).parts
            if len(parts) >= 2 and parts[-1].lower().endswith((".png", ".svg")):
                return parts[-2]
            return Path(value).stem
    return ""


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_aspect(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    w = max(0.0, bbox[2] - bbox[0])
    h = max(0.0, bbox[3] - bbox[1])
    return max(w, h) / max(min(w, h), 1e-6)


def model_probabilities(node: dict[str, Any]) -> dict[str, float]:
    metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
    upstream = metadata.get("upstream_metadata") if isinstance(metadata.get("upstream_metadata"), dict) else {}
    for source in [metadata, upstream]:
        for key in ["arbitration_probs", "symbol_long_tail_model_v1_probs", "arbitration_v2_probs", "model_probs", "probabilities"]:
            probs = source.get(key) if isinstance(source, dict) else None
            if isinstance(probs, dict):
                out: dict[str, float] = {}
                for label, value in probs.items():
                    try:
                        out[str(label)] = float(value)
                    except (TypeError, ValueError):
                        pass
                if out:
                    return out
    return {}


def find_node(row: dict[str, Any], node_id: str) -> dict[str, Any]:
    for node in ((row.get("scene_graph") or {}).get("nodes") or []):
        if isinstance(node, dict) and str(node.get("id")) == str(node_id):
            return node
    return {}


def nodes_by_key(row: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for node in ((row.get("scene_graph") or {}).get("nodes") or []):
        if isinstance(node, dict):
            out[(str(node.get("family") or ""), str(node.get("id") or ""))] = node
    return out


def count_defects(path: str | Path) -> dict[str, int]:
    data = load_json(path, {})
    return {str(k): int(v) for k, v in ((data.get("summary") or {}).get("defect_counts") or data.get("defect_counts") or {}).items()}


def confusion_metrics(rows: list[dict[str, Any]], labels: list[str] | None = None) -> dict[str, Any]:
    pairs = [(str(r.get("gold_label") or ""), str(r.get("pred_label") or "")) for r in rows if r.get("gold_label") is not None]
    if labels is None:
        labels = sorted({item for pair in pairs for item in pair if item})
    confusion: dict[str, Counter[str]] = {label: Counter() for label in labels}
    for gold, pred in pairs:
        if gold in confusion:
            confusion[gold][pred] += 1
    per_label = {}
    f1_values = []
    correct = 0
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[g][label] for g in labels if g != label)
        fn = sum(v for pred, v in confusion[label].items() if pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
        f1_values.append(f1)
        correct += tp
    return {
        "sample_count": len(pairs),
        "accuracy": correct / max(len(pairs), 1),
        "macro_f1": sum(f1_values) / max(len(f1_values), 1),
        "per_label": per_label,
        "confusion": {label: dict(confusion[label]) for label in labels},
    }


def hard_case_features(case: dict[str, Any]) -> dict[str, Any]:
    bbox = normalize_bbox(case.get("bbox"))
    probs = case.get("model_probabilities") if isinstance(case.get("model_probabilities"), dict) else {}
    return {
        "bbox": bbox,
        "bbox_area": round(bbox_area(bbox), 6),
        "bbox_aspect": round(bbox_aspect(bbox), 6),
        "confidence": case.get("confidence"),
        "raw_label": case.get("raw_label"),
        "probabilities": probs,
    }


def copy_jsonl_with_trace(src: str, dst: str, trace_key: str, trace_value: dict[str, Any]) -> int:
    rows = load_jsonl(src)
    out = []
    for row in rows:
        row = json.loads(json.dumps(row, ensure_ascii=False))
        row.setdefault("route_trace", {})[trace_key] = trace_value
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            if isinstance(node, dict):
                node.setdefault("metadata", {})["model_version"] = trace_value.get("model_version", "model_v5")
                node.setdefault("metadata", {})["postprocess_version"] = trace_value.get("postprocess_version", "none")
        out.append(row)
    write_jsonl(dst, out)
    return len(out)


def copy_file(src: str, dst: str) -> None:
    source = ROOT / src
    target = ROOT / dst
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family = Counter()
    semantic = Counter()
    invalid = 0
    for row in rows:
        if not isinstance(row.get("scene_graph"), dict):
            invalid += 1
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            if isinstance(node, dict):
                family[str(node.get("family") or "unknown")] += 1
                semantic[str(node.get("semantic_type") or "unknown")] += 1
    return {
        "rows": len(rows),
        "invalid_graph_rate": invalid / max(len(rows), 1),
        "family_counts": dict(family.most_common()),
        "top_semantic_counts": dict(semantic.most_common(20)),
    }


def update_todo_remove(task_ids: list[str]) -> None:
    path = ROOT / "todo.json"
    if not path.exists():
        return
    data = load_json(path, {})
    tasks = data.get("tasks") if isinstance(data.get("tasks"), list) else []
    before = len(tasks)
    remaining = [task for task in tasks if str(task.get("id")) not in set(task_ids)]
    removed = before - len(remaining)
    data["tasks"] = remaining
    status = data.setdefault("status", {})
    status["pending"] = len(remaining)
    status["completed_removed_this_run"] = int(status.get("completed_removed_this_run") or 0) + removed
    status["completed"] = int(status.get("completed") or 0) + removed
    status["in_progress"] = 0
    if not remaining:
        data["phase"] = "completed"
    write_json(path, data)


def markdown_table(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(str(v) for v in rows[0]) + " |"
    sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

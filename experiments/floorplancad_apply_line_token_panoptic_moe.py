#!/usr/bin/env python3
"""Apply a panoptic line-token MoE checkpoint to primitive-cache records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.floorplancad_train_line_token_panoptic_moe import (  # noqa: E402
    FAMILY_NAMES,
    label_family,
    make_panoptic_model,
    set_sq_rq_runtime_thresholds,
    validate_checkpoint_abi,
)
from experiments.floorplancad_window_identity_embedding import reciprocal_embedding_tracks  # noqa: E402
from experiments.floorplancad_query_ownership import (  # noqa: E402
    DEFAULT_MIN_QUERY_SCORE,
    calibrated_proposal_score,
    decode_page_global_track_ownership,
    select_global_owners,
)
from experiments.floorplancad_panoptic_protocol import PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD  # noqa: E402
from experiments.floorplancad_panoptic_runtime_config import DEFAULT_RUNTIME_PROFILE  # noqa: E402
from experiments.floorplancad_panoptic_scoring import (  # noqa: E402
    mask_objectness_scores_np as mask_objectness_scores,
    sigmoid_np,
)
from experiments.floorplancad_train_line_token_transformer_moe import IGNORE_LABEL, import_torch, parse_float, parse_int, rel, write_json  # noqa: E402

DEFAULT_MODEL = ROOT / "reports/vlm/floorplancad_line_token_panoptic_moe/panoptic_component_moe.pt"
DEFAULT_CACHE = DEFAULT_RUNTIME_PROFILE.val_path(ROOT)
DEFAULT_OUTPUT = ROOT / "reports/vlm/floorplancad_line_token_panoptic_moe/val_predictions.jsonl"
DEFAULT_REPORT = ROOT / "results/floorplancad_line_token_panoptic_moe_val_apply.json"
DEFAULT_ROUTE_TRACE = ROOT / "results/floorplancad_line_token_panoptic_moe_val_apply_route_trace.json"
DEFAULT_AFFINITY_SCORES = ROOT / "reports/vlm/floorplancad_line_token_ownership_affinity_expert/val_ownership_affinity_scores_full.jsonl"
STUFF_LABELS = frozenset(range(30, 35))
OBJECT_NORMALIZATION_THRESHOLD = PANOPTIC_OBJECT_NORMALIZATION_THRESHOLD


def update_family_admission_counter(counters: Counter, family: str, stage: str) -> None:
    counters[f"family_admission::{family}::{stage}"] += 1


def family_admission_payload(counters: Counter) -> dict[str, dict[str, int]]:
    stages = (
        "candidate",
        "rejected_no_object",
        "rejected_low_object_margin",
        "rejected_low_score",
        "admitted",
        "rejected_empty_mask",
        "proposal",
    )
    payload: dict[str, dict[str, int]] = {}
    for family in FAMILY_NAMES:
        row = {stage: int(counters[f"family_admission::{family}::{stage}"]) for stage in stages}
        if any(row.values()):
            payload[family] = row
    return payload


def parse_family_float_overrides(value: str | None) -> dict[str, float]:
    if value is None or not str(value).strip():
        return {}
    overrides: dict[str, float] = {}
    valid = set(FAMILY_NAMES)
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("family overrides must use family:value entries")
        family, raw_value = item.split(":", 1)
        family = family.strip()
        if family not in valid:
            raise ValueError(f"unknown family override: {family}")
        threshold = float(raw_value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("family min query score overrides must be in [0, 1]")
        overrides[family] = threshold
    return overrides


def min_query_score_for_family(default_score: float, family: str, overrides: dict[str, float] | None) -> float:
    return float((overrides or {}).get(family, default_score))


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_config_sha256(args: argparse.Namespace) -> str:
    excluded = {"model", "cache", "output", "report", "route_trace_output", "device"}
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in sorted(vars(args).items())
        if key not in excluded
    }
    encoded = json.dumps(config, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def decoder_policy_manifest(
    args: argparse.Namespace, *, quality_enabled: bool | None = None
) -> dict[str, Any]:
    """Freeze every prediction-time admission and merge decision in one payload."""
    return {
        "version": "floorplancad_panoptic_decoder_policy_v4_quality_v5_promoted_deployment_score",
        "proposal_score": "foreground_class_probability_times_independent_quality",
        "quality_enabled": quality_enabled,
        "quality_request": getattr(args, "use_quality", None),
        "min_query_score_applies_to": "foreground_class_probability_times_independent_quality",
        "thing_admission": "query_objectness_margin_then_calibrated_score",
        "stuff_admission": "stuff_presence_margin_then_calibrated_score",
        "min_query_score": float(args.min_query_score),
        "family_min_query_score_overrides": parse_family_float_overrides(
            getattr(args, "family_min_query_score_overrides", "")
        ),
        "min_object_margin": float(args.min_object_margin),
        "mask_threshold": float(args.mask_threshold),
        "semantic_stuff": bool(getattr(args, "semantic_stuff", True)),
        "semantic_stuff_min_score": float(getattr(args, "semantic_stuff_min_score", 0.35)),
        "query_admission_policy": args.query_admission_policy,
        "ownership_decoder": args.ownership_decoder,
        "ownership_membership_gate": "mask_threshold_for_all_ownership_decoders",
        "primitive_conflict_policy": args.primitive_conflict_policy,
        "window_merge_policy": args.window_merge_policy,
        "merge_iou_threshold": float(args.merge_iou_threshold),
        "merge_overlap_threshold": float(args.merge_overlap_threshold),
        "merge_center_distance_threshold": float(args.merge_center_distance_threshold),
        "recall_expansion_policy": args.recall_expansion_policy,
    }


def resolve_quality_admission_enabled(
    abi: dict[str, Any],
    requested: bool | None,
) -> bool:
    trained = bool(abi.get("quality_head_trained", False))
    compatible = bool(abi.get("quality_admission_compatible", False))
    promoted = bool(abi.get("quality_admission_promoted", False))
    if requested is None:
        return trained and compatible and promoted
    if requested and (not trained or not compatible):
        raise ValueError("--use-quality requires the all-query calibrated quality objective ABI")
    return bool(requested)


def ownership_membership_threshold_for_decoder(
    decoder: str,
    *,
    ownership_available: bool,
    legacy_ownership_state: bool,
    mask_threshold: float,
) -> float | None:
    if not 0.0 <= float(mask_threshold) <= 1.0:
        raise ValueError("ownership mask threshold must be in [0, 1]")
    if not ownership_available or decoder == "mask_only":
        return None
    if decoder in {"page_global", "mask_guided"}:
        return float(mask_threshold)
    if decoder == "auto" and not legacy_ownership_state:
        return float(mask_threshold)
    return None


def validate_inference_provenance(record: dict[str, Any]) -> dict[str, Any]:
    provenance = record.get("inference_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("inference cache record is missing inference_provenance")
    required = ("split", "source_sha256", "feature_schema_sha256", "gt_free")
    missing = [key for key in required if key not in provenance]
    if missing:
        raise ValueError(f"inference_provenance missing fields: {missing}")
    if provenance.get("gt_free") is not True:
        raise ValueError("inference_provenance.gt_free must be true")
    for key in ("source_sha256", "feature_schema_sha256"):
        value = provenance.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"inference_provenance.{key} must be a SHA-256 hex digest")
    return provenance


def validate_cache_input_protocol(record: dict[str, Any], abi: dict[str, Any]) -> dict[str, Any]:
    """Reject silent v2/v3 input mixing before model inference."""
    cache_input_schema = record.get("input_schema_version")
    has_segments = any(
        isinstance(row.get("segment_features"), list) and bool(row.get("segment_features"))
        for row in (record.get("primitive_rows") or [])
        if isinstance(row, dict)
    )
    input_protocol = abi.get("input_protocol")
    if input_protocol is None:
        if cache_input_schema == "floorplancad_line_json_primitive_cache_v3_segments" or has_segments:
            raise ValueError("legacy checkpoint has no lossless-segment ABI; it cannot be evaluated on a v3 segment cache")
        return {"status": "legacy_v2_diagnostic_only", "cache_input_schema": cache_input_schema, "segment_features": False}
    if cache_input_schema != input_protocol.get("input_schema_version"):
        raise ValueError(
            f"checkpoint/cache input-schema mismatch: checkpoint={input_protocol.get('input_schema_version')} cache={cache_input_schema}"
        )
    if bool(input_protocol.get("segment_features")) != has_segments:
        raise ValueError("checkpoint/cache segment-feature contract mismatch")
    return {"status": "matched", "cache_input_schema": cache_input_schema, "segment_features": has_segments}


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def load_record_id_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    allowed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value:
                allowed.add(value)
    return allowed


def load_affinity_scores(path: Path | None) -> dict[str, dict[tuple[int, int], float]]:
    if path is None or not path.exists():
        return {}
    by_record: dict[str, dict[tuple[int, int], float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id") or "")
            left = row.get("left_primitive_index", row.get("left"))
            right = row.get("right_primitive_index", row.get("right"))
            score = row.get("affinity_score", row.get("same_instance_score", row.get("score")))
            if not record_id or left is None or right is None or score is None:
                continue
            left_id = parse_int(left, -1)
            right_id = parse_int(right, -1)
            if left_id < 0 or right_id < 0 or left_id == right_id:
                continue
            key = (min(left_id, right_id), max(left_id, right_id))
            by_record[record_id][key] = max(float(by_record[record_id].get(key, 0.0)), parse_float(score, 0.0))
    return by_record


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-9)


def update_distribution_counters(counters: Counter, prefix: str, value: float) -> None:
    if not np.isfinite(value):
        return
    counters[f"{prefix}_count"] += 1
    counters[f"{prefix}_sum"] += float(value)
    counters[f"{prefix}_min"] = min(float(counters.get(f"{prefix}_min", value)), float(value))
    counters[f"{prefix}_max"] = max(float(counters.get(f"{prefix}_max", value)), float(value))
    if value > 0.0:
        counters[f"{prefix}_positive"] += 1


def distribution_payload(counters: Counter, prefix: str) -> dict[str, Any]:
    count = int(counters.get(f"{prefix}_count", 0))
    return {
        "count": count,
        "mean": float(counters.get(f"{prefix}_sum", 0.0)) / max(count, 1),
        "min": None if count <= 0 else float(counters.get(f"{prefix}_min", 0.0)),
        "max": None if count <= 0 else float(counters.get(f"{prefix}_max", 0.0)),
        "positive_rate": int(counters.get(f"{prefix}_positive", 0)) / max(count, 1),
    }


def branch_route_trace_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Serialize both capacity-limited and dropless router diagnostics safely."""
    capacity = value.get("capacity")
    assignment_fraction = value.get("assignment_fraction")
    return {
        "enabled": bool(value.get("enabled", False)),
        "capacity": None if capacity is None else int(capacity),
        "overflow_assignments": int(value.get("overflow_assignments", 0)),
        "usage_gate_passed": bool(value.get("usage_gate_passed", False)),
        "assignment_fraction": (
            [float(item) for item in assignment_fraction.detach().cpu().tolist()]
            if assignment_fraction is not None else []
        ),
    }


def cache_arrays(record: dict[str, Any]) -> tuple[list[int], np.ndarray, np.ndarray | None, np.ndarray | None]:
    primitive_ids: list[int] = []
    features: list[list[float]] = []
    segment_rows: list[np.ndarray] = []
    has_segments: bool | None = None
    for item in record.get("primitive_rows", []) if isinstance(record.get("primitive_rows"), list) else []:
        raw_features = item.get("features") if isinstance(item.get("features"), list) else []
        if not raw_features:
            continue
        primitive_id = parse_int(item.get("primitive_id"), -1)
        if primitive_id < 0:
            continue
        primitive_ids.append(primitive_id)
        features.append([parse_float(value, 0.0) for value in raw_features])
        raw_segments = item.get("segment_features")
        row_has_segments = isinstance(raw_segments, list) and bool(raw_segments)
        if has_segments is None:
            has_segments = row_has_segments
        if has_segments != row_has_segments:
            raise ValueError("inference cache must either retain segments for every primitive or for none")
        if row_has_segments:
            segments = np.asarray(raw_segments, dtype=np.float32)
            if segments.ndim != 2 or segments.shape[1] != len(features[-1]) or not np.isfinite(segments).all():
                raise ValueError("segment_features must be finite [segment, feature] arrays")
            segment_rows.append(segments)
    x = np.nan_to_num(np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if not primitive_ids or not has_segments:
        return primitive_ids, x, None, None
    max_segments = max(len(segments) for segments in segment_rows)
    segment_features = np.zeros((len(segment_rows), max_segments, x.shape[1]), dtype=np.float32)
    segment_padding = np.ones((len(segment_rows), max_segments), dtype=bool)
    for index, segments in enumerate(segment_rows):
        segment_features[index, : len(segments)] = segments
        segment_padding[index, : len(segments)] = False
    return primitive_ids, x, segment_features, segment_padding


def assert_gt_free_record(record: dict[str, Any]) -> None:
    forbidden = {
        "gt_masks", "gt_labels", "page_instance_id", "page_instance_ids",
        "matched_target_indices", "instance_ids", "instance_id", "semantic_id",
        "semantic_ids", "target", "targets", "query_labels", "mask_targets",
    }
    present = forbidden.intersection(record)
    primitive_rows = record.get("primitive_rows")
    if isinstance(primitive_rows, list):
        for row in primitive_rows:
            if isinstance(row, dict):
                present.update(forbidden.intersection(row))
    if present:
        raise ValueError(f"GT-free apply contract violated by fields: {sorted(present)}")


def apply_window(record: dict[str, Any], model: Any, pack: dict[str, Any], device: Any, *, use_quality: bool, use_identity: bool = True, use_ownership: bool = True) -> tuple[str, list[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray]:
    torch = pack["torch"]
    assert_gt_free_record(record)
    original_record_id = str(record.get("original_record_id") or record.get("record_id"))
    primitive_ids, x, segment_features, segment_padding = cache_arrays(record)
    if not primitive_ids:
        if hasattr(model, "last_typed_outputs"):
            model.last_typed_outputs = None
        if hasattr(model, "last_branch_router_diagnostics"):
            model.last_branch_router_diagnostics = None
        return original_record_id, [], np.zeros((0, 16), dtype=np.float32), np.zeros((0, IGNORE_LABEL + 1), dtype=np.float32), np.zeros((0, 0), dtype=np.float32), None, None, None, np.zeros((0, IGNORE_LABEL), dtype=np.float32)
    with torch.no_grad():
        xt = torch.from_numpy(x.astype(np.float32)).to(device).unsqueeze(0)
        segment_tensor = None if segment_features is None else torch.from_numpy(segment_features).to(device).unsqueeze(0)
        segment_padding_tensor = None if segment_padding is None else torch.from_numpy(segment_padding).to(device).unsqueeze(0)
        outputs = model(
            xt, segment_features=segment_tensor, segment_padding_mask=segment_padding_tensor,
            return_quality=use_quality, return_identity=use_identity,
        )
        semantic_logits, query_logits, mask_logits = outputs[:3]
        quality_logits = outputs[3] if use_quality else None
        identity_embeddings = outputs[4 if use_quality else 3] if use_identity else None
        ownership_logits = model.last_ownership_logits if use_ownership else None
    quality = None if quality_logits is None else quality_logits.squeeze(0).detach().cpu().numpy()
    identity = None if identity_embeddings is None else identity_embeddings.squeeze(0).detach().cpu().numpy()
    ownership = None if ownership_logits is None else ownership_logits.squeeze(0).detach().cpu().numpy()
    return original_record_id, primitive_ids, x, query_logits.squeeze(0).detach().cpu().numpy(), mask_logits.squeeze(0).detach().cpu().numpy(), quality, identity, ownership, semantic_logits.squeeze(0).detach().cpu().numpy()


def semantic_stuff_instances(
    record_ids: list[list[int]],
    semantic_logits_rows: list[np.ndarray],
    occupied_primitive_ids: set[int],
    *,
    minimum_score: float,
) -> list[dict[str, Any]]:
    """Fuse overlapping semantic windows into one stuff region per stuff class."""
    logits_by_primitive: dict[int, list[np.ndarray]] = defaultdict(list)
    for primitive_ids, logits in zip(record_ids, semantic_logits_rows, strict=True):
        if logits.shape[0] != len(primitive_ids) or logits.shape[1] != IGNORE_LABEL:
            raise ValueError("semantic logits must align with primitive ids and FloorPlanCAD foreground classes")
        for primitive_id, row in zip(primitive_ids, logits, strict=True):
            logits_by_primitive[int(primitive_id)].append(np.asarray(row, dtype=np.float32))
    by_label: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for primitive_id, rows in logits_by_primitive.items():
        if primitive_id in occupied_primitive_ids:
            continue
        mean_logits = np.mean(np.stack(rows, axis=0), axis=0)
        probability = softmax_np(mean_logits[None, :])[0]
        label = int(probability.argmax())
        score = float(probability[label])
        if label in STUFF_LABELS and score >= minimum_score:
            by_label[label].append((primitive_id, score))
    instances = []
    for label, values in sorted(by_label.items()):
        primitive_ids = sorted(primitive_id for primitive_id, _score in values)
        instances.append({
            "label": label,
            "primitive_ids": primitive_ids,
            "score": float(np.mean([score for _primitive_id, score in values])),
            "class_score": float(np.mean([score for _primitive_id, score in values])),
            "quality_score": 1.0,
            "proposal_source": "semantic_stuff_vote",
            "moe_route": "semantic_stuff_head",
            "sparse_expert_weights": {"semantic_stuff_head": 1.0},
            "decoder": "semantic_stuff_vote",
            "merged_fragments": len(values),
        })
    return instances


def endpoint_affinity_expand_indices(
    selected_indices: list[int],
    mask_prob: np.ndarray,
    features: np.ndarray,
    *,
    expansion_threshold: float,
    endpoint_radius: float,
    max_expansion_ratio: float,
) -> tuple[list[int], dict[str, Any]]:
    if not selected_indices or features.size == 0 or features.shape[0] != mask_prob.shape[0]:
        return selected_indices, {"policy": "endpoint_affinity", "candidate_indices": 0, "added_indices": 0}
    selected = sorted(set(int(idx) for idx in selected_indices if 0 <= int(idx) < mask_prob.shape[0]))
    if not selected:
        return selected, {"policy": "endpoint_affinity", "candidate_indices": 0, "added_indices": 0}

    endpoints = features[:, [0, 1, 2, 3]].reshape(-1, 2, 2)
    centers = features[:, [4, 5]]
    seed_points = np.concatenate([endpoints[selected].reshape(-1, 2), centers[selected]], axis=0)
    selected_set = set(selected)
    candidates: list[tuple[float, float, int]] = []
    for idx, prob in enumerate(mask_prob.tolist()):
        if idx in selected_set or float(prob) < expansion_threshold:
            continue
        points = np.concatenate([endpoints[idx], centers[idx : idx + 1]], axis=0)
        distances = np.linalg.norm(points[:, None, :] - seed_points[None, :, :], axis=-1)
        min_distance = float(distances.min()) if distances.size else float("inf")
        if min_distance <= endpoint_radius:
            candidates.append((float(prob), -min_distance, idx))

    max_added = max(0, int(np.ceil(len(selected) * max(float(max_expansion_ratio), 0.0))))
    candidates.sort(reverse=True)
    added = [idx for _prob, _neg_distance, idx in candidates[:max_added]]
    expanded = sorted(selected_set | set(added))
    return expanded, {
        "policy": "endpoint_affinity",
        "candidate_indices": len(candidates),
        "added_indices": len(added),
        "endpoint_radius": endpoint_radius,
        "expansion_threshold": expansion_threshold,
        "max_expansion_ratio": max_expansion_ratio,
    }


def instances_from_windows(
    record_ids: list[list[int]],
    query_logits_rows: list[np.ndarray],
    mask_logits_rows: list[np.ndarray],
    *,
    feature_rows: list[np.ndarray] | None = None,
    quality_logits_rows: list[np.ndarray] | None = None,
    identity_embedding_rows: list[np.ndarray] | None = None,
    ownership_logits_rows: list[np.ndarray] | None = None,
    ownership_membership_threshold: float | None = None,
    query_admission_policy: str,
    min_query_score: float,
    family_min_query_score_overrides: dict[str, float] | None = None,
    min_object_margin: float = float("-inf"),
    mask_threshold: float,
    recall_expansion_policy: str = "none",
    recall_expansion_threshold: float = 0.0,
    recall_expansion_endpoint_radius: float = 0.0,
    recall_expansion_max_ratio: float = 0.0,
    merge_iou_threshold: float,
    merge_overlap_threshold: float,
    max_instances: int,
    window_merge_policy: str = "reciprocal",
    merge_center_distance_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    counters = Counter()
    ownership_quality_logits_rows = quality_logits_rows
    if feature_rows is None:
        feature_rows = [np.empty((len(primitive_ids), 0), dtype=np.float32) for primitive_ids in record_ids]
    if quality_logits_rows is None:
        quality_logits_rows = [np.zeros(query_logits.shape[0], dtype=np.float32) for query_logits in query_logits_rows]
        quality_enabled = False
    else:
        quality_enabled = True
    if ownership_logits_rows is not None:
        if identity_embedding_rows is None:
            raise ValueError("page-global ownership requires predicted identity embeddings")
        admitted_rows = []
        for query_logits, mask_logits, quality_logits in zip(query_logits_rows, mask_logits_rows, quality_logits_rows, strict=True):
            class_probs = softmax_np(query_logits)
            mask_scores = mask_objectness_scores(mask_logits)
            admitted = np.zeros(class_probs.shape[0], dtype=bool)
            for query_index in range(class_probs.shape[0]):
                full_label = int(class_probs[query_index].argmax())
                object_label = int(class_probs[query_index, :IGNORE_LABEL].argmax())
                family = label_family(object_label)
                family_min_query_score = min_query_score_for_family(
                    min_query_score, family, family_min_query_score_overrides
                )
                object_score = float(class_probs[query_index, object_label])
                object_margin = float(query_logits[query_index, object_label] - query_logits[query_index, IGNORE_LABEL])
                quality_score = float(sigmoid_np(np.asarray([quality_logits[query_index]], dtype=np.float32))[0]) if quality_enabled else 1.0
                mask_score = float(mask_scores[query_index])
                admission_score = calibrated_proposal_score(object_score, quality_score) * mask_score
                update_distribution_counters(counters, "object_margin", object_margin)
                update_distribution_counters(counters, "object_score", object_score)
                update_distribution_counters(counters, "quality_score", quality_score)
                update_distribution_counters(counters, "mask_objectness_score", mask_score)
                update_distribution_counters(counters, "calibrated_score", admission_score)
                update_family_admission_counter(counters, family, "candidate")
                if query_admission_policy == "respect_no_object" and full_label == IGNORE_LABEL:
                    counters["rejected_no_object_queries"] += 1
                    update_family_admission_counter(counters, family, "rejected_no_object")
                    continue
                counters["foreground_candidate_queries"] += 1
                if object_margin < min_object_margin:
                    counters["rejected_low_object_margin_queries"] += 1
                    update_family_admission_counter(counters, family, "rejected_low_object_margin")
                    continue
                if admission_score < family_min_query_score:
                    counters["rejected_low_score_queries"] += 1
                    update_family_admission_counter(counters, family, "rejected_low_score")
                    continue
                admitted[query_index] = True
                counters["admitted_object_queries"] += 1
                update_family_admission_counter(counters, family, "admitted")
            counters["queries"] += int(class_probs.shape[0])
            admitted_rows.append(admitted)
        proposals, ownership_diagnostics = decode_page_global_track_ownership(
            record_ids, query_logits_rows, mask_logits_rows, ownership_logits_rows,
            identity_embedding_rows, ownership_quality_logits_rows, admitted_rows, ignore_label=IGNORE_LABEL,
            ownership_membership_threshold=ownership_membership_threshold,
        )
        diagnostics = {
            "query_admission_policy": query_admission_policy,
            "family_min_query_score_overrides": dict(sorted((family_min_query_score_overrides or {}).items())),
            "queries": int(counters["queries"]),
            "foreground_candidate_queries": int(counters["foreground_candidate_queries"]),
            "admitted_object_queries": int(counters["admitted_object_queries"]),
            "rejected_no_object_queries": int(counters["rejected_no_object_queries"]),
            "rejected_low_object_margin_queries": int(counters["rejected_low_object_margin_queries"]),
            "rejected_low_score_queries": int(counters["rejected_low_score_queries"]),
            "rejected_empty_mask_queries": 0,
            "proposal_queries": len(proposals),
            "recall_expanded_queries": 0,
            "recall_expansion_candidates": 0,
            "recall_expansion_added_primitives": 0,
                "query_score_distributions": {
                "object_margin": distribution_payload(counters, "object_margin"),
                "object_score": distribution_payload(counters, "object_score"),
                "quality_score": distribution_payload(counters, "quality_score"),
                "calibrated_score": distribution_payload(counters, "calibrated_score"),
                },
                "family_admission": family_admission_payload(counters),
                "stage_conservation": {
                "query_partition": int(counters["queries"]) == int(counters["rejected_no_object_queries"]) + int(counters["foreground_candidate_queries"]),
                "admission_partition": int(counters["foreground_candidate_queries"]) == int(counters["rejected_low_object_margin_queries"]) + int(counters["rejected_low_score_queries"]) + int(counters["admitted_object_queries"]),
            },
            "ownership_before_mask": True,
            **ownership_diagnostics,
        }
        return proposals[:max_instances], diagnostics
    for window_index, (primitive_ids, features, query_logits, mask_logits, quality_logits) in enumerate(zip(record_ids, feature_rows, query_logits_rows, mask_logits_rows, quality_logits_rows, strict=True)):
        class_probs = softmax_np(query_logits)
        mask_probs = sigmoid_np(mask_logits)
        mask_scores = mask_objectness_scores(mask_logits)
        admitted = np.zeros(class_probs.shape[0], dtype=bool)
        for admission_index in range(class_probs.shape[0]):
            full_label = int(class_probs[admission_index].argmax())
            object_label = int(class_probs[admission_index, :IGNORE_LABEL].argmax())
            family = label_family(object_label)
            family_min_query_score = min_query_score_for_family(
                min_query_score, family, family_min_query_score_overrides
            )
            object_score = float(class_probs[admission_index, object_label])
            object_margin = float(query_logits[admission_index, object_label] - query_logits[admission_index, IGNORE_LABEL])
            quality_score = float(sigmoid_np(np.asarray([quality_logits[admission_index]], dtype=np.float32))[0]) if quality_enabled else 1.0
            admission_score = calibrated_proposal_score(object_score, quality_score) * float(mask_scores[admission_index])
            admitted[admission_index] = (
                (query_admission_policy == "legacy_force_object" or full_label != IGNORE_LABEL)
                and object_margin >= min_object_margin and admission_score >= family_min_query_score
            )
        owners = None
        if ownership_logits_rows is not None:
            owners = select_global_owners(ownership_logits_rows[window_index], admitted)
            counters["null_owned_primitives"] += int(np.sum(owners == class_probs.shape[0]))
        counters["windows"] += 1
        counters["queries"] += int(class_probs.shape[0])
        for qidx in range(class_probs.shape[0]):
            full_label = int(class_probs[qidx].argmax())
            object_label = int(class_probs[qidx, :IGNORE_LABEL].argmax())
            family = label_family(object_label)
            family_min_query_score = min_query_score_for_family(
                min_query_score, family, family_min_query_score_overrides
            )
            no_object_score = float(class_probs[qidx, IGNORE_LABEL])
            object_score = float(class_probs[qidx, object_label])
            object_margin = float(query_logits[qidx, object_label] - query_logits[qidx, IGNORE_LABEL])
            quality_score = float(sigmoid_np(np.asarray([quality_logits[qidx]], dtype=np.float32))[0]) if quality_enabled else 1.0
            mask_score = float(mask_scores[qidx])
            admission_score = calibrated_proposal_score(object_score, quality_score) * mask_score
            mask_max = float(mask_probs[qidx].max()) if mask_probs.shape[1] > 0 else 0.0
            mask_selected_count = int(np.sum(mask_probs[qidx] >= mask_threshold))
            update_distribution_counters(counters, "object_margin", object_margin)
            update_distribution_counters(counters, "object_score", object_score)
            update_distribution_counters(counters, "no_object_score", no_object_score)
            update_distribution_counters(counters, "quality_score", quality_score)
            update_distribution_counters(counters, "mask_objectness_score", mask_score)
            update_distribution_counters(counters, "calibrated_score", admission_score)
            update_distribution_counters(counters, "mask_max_prob", mask_max)
            update_distribution_counters(counters, "mask_selected_count", float(mask_selected_count))
            update_family_admission_counter(counters, family, "candidate")
            if query_admission_policy == "respect_no_object" and full_label == IGNORE_LABEL:
                counters["rejected_no_object_queries"] += 1
                update_family_admission_counter(counters, family, "rejected_no_object")
                continue
            if query_admission_policy == "legacy_force_object":
                label = object_label
                score = object_score
            elif query_admission_policy == "respect_no_object":
                label = full_label
                score = float(class_probs[qidx, label])
            else:
                raise ValueError(f"unknown query admission policy: {query_admission_policy}")
            counters["foreground_candidate_queries"] += 1
            if object_margin < min_object_margin:
                counters["rejected_low_object_margin_queries"] += 1
                update_family_admission_counter(counters, family, "rejected_low_object_margin")
                continue
            if calibrated_proposal_score(score, quality_score) * mask_score < family_min_query_score:
                counters["rejected_low_score_queries"] += 1
                update_family_admission_counter(counters, family, "rejected_low_score")
                continue
            counters["admitted_object_queries"] += 1
            update_family_admission_counter(counters, family, "admitted")
            selected_indices = (
                np.flatnonzero(owners == qidx).astype(int).tolist()
                if owners is not None
                else [idx for idx, value in enumerate(mask_probs[qidx].tolist()) if value >= mask_threshold]
            )
            expansion = {"policy": "off", "candidate_indices": 0, "added_indices": 0}
            if owners is None and recall_expansion_policy == "endpoint_affinity" and selected_indices:
                selected_indices, expansion = endpoint_affinity_expand_indices(
                    selected_indices,
                    mask_probs[qidx],
                    features,
                    expansion_threshold=recall_expansion_threshold,
                    endpoint_radius=recall_expansion_endpoint_radius,
                    max_expansion_ratio=recall_expansion_max_ratio,
                )
                counters["recall_expansion_candidates"] += int(expansion["candidate_indices"])
                counters["recall_expansion_added_primitives"] += int(expansion["added_indices"])
                counters["recall_expanded_queries"] += int(expansion["added_indices"] > 0)
            selected = [primitive_ids[idx] for idx in selected_indices]
            if not selected:
                counters["rejected_empty_mask_queries"] += 1
                update_family_admission_counter(counters, family, "rejected_empty_mask")
                continue
            sparse_weights = {"line_token_component_panoptic": 1.0}
            if int(expansion["added_indices"]) > 0:
                sparse_weights = {"line_token_component_panoptic": 0.8, "endpoint_affinity_expander": 0.2}
            proposals.append(
                {
                    "label": label,
                    "primitive_ids": sorted(set(selected)),
                    "score": calibrated_proposal_score(score, quality_score) * mask_score,
                    "class_score": score,
                    "quality_score": quality_score,
                    "mask_objectness_score": mask_score,
                    "no_object_score": no_object_score,
                    "object_margin": object_margin,
                    "proposal_source": "line_token_panoptic_moe_component_query",
                    "moe_route": "line_token_component_panoptic",
                    "sparse_expert_weights": sparse_weights,
                    "recall_expansion": expansion,
                    "query_admission_policy": query_admission_policy,
                    "query_index": qidx,
                    "objectness_score": 1.0 - no_object_score,
                    "ownership_before_mask": owners is not None,
                    "window_index": window_index,
                    **({"identity_embedding": identity_embedding_rows[window_index][qidx].tolist()} if identity_embedding_rows is not None else {}),
                }
            )
            counters["proposal_queries"] += 1
            update_family_admission_counter(counters, family, "proposal")
    proposals.sort(key=lambda item: item["score"], reverse=True)
    if identity_embedding_rows is not None:
        merged = reciprocal_embedding_tracks(proposals)[:max_instances]
    else:
        merged = merge_window_proposals(
            proposals,
            merge_iou_threshold=merge_iou_threshold,
            merge_overlap_threshold=merge_overlap_threshold,
            max_instances=max_instances,
            merge_policy=window_merge_policy,
            center_distance_threshold=merge_center_distance_threshold,
        )
    diagnostics = {
        "query_admission_policy": query_admission_policy,
        "family_min_query_score_overrides": dict(sorted((family_min_query_score_overrides or {}).items())),
        "queries": int(counters["queries"]),
        "foreground_candidate_queries": int(counters["foreground_candidate_queries"]),
        "admitted_object_queries": int(counters["admitted_object_queries"]),
        "rejected_no_object_queries": int(counters["rejected_no_object_queries"]),
        "rejected_low_object_margin_queries": int(counters["rejected_low_object_margin_queries"]),
        "rejected_low_score_queries": int(counters["rejected_low_score_queries"]),
        "rejected_empty_mask_queries": int(counters["rejected_empty_mask_queries"]),
        "null_owned_primitives": int(counters["null_owned_primitives"]),
        "ownership_before_mask": ownership_logits_rows is not None,
        "proposal_queries": int(counters["proposal_queries"]),
        "merged_instances": len(merged),
        "window_merge_policy": window_merge_policy,
        "merge_center_distance_threshold": merge_center_distance_threshold,
        "merged_fragments": sum(int(item.get("merged_fragments", 1)) for item in merged),
        "multi_window_merged_instances": sum(int(item.get("merged_fragments", 1)) > 1 for item in merged),
        "recall_expansion_policy": recall_expansion_policy,
        "recall_expanded_queries": int(counters["recall_expanded_queries"]),
        "recall_expansion_candidates": int(counters["recall_expansion_candidates"]),
        "recall_expansion_added_primitives": int(counters["recall_expansion_added_primitives"]),
        "admitted_object_query_rate": int(counters["admitted_object_queries"]) / max(int(counters["queries"]), 1),
        "proposal_query_rate": int(counters["proposal_queries"]) / max(int(counters["queries"]), 1),
        "query_score_distributions": {
            "object_margin": distribution_payload(counters, "object_margin"),
            "object_score": distribution_payload(counters, "object_score"),
            "no_object_score": distribution_payload(counters, "no_object_score"),
            "quality_score": distribution_payload(counters, "quality_score"),
            "calibrated_score": distribution_payload(counters, "calibrated_score"),
            "mask_max_prob": distribution_payload(counters, "mask_max_prob"),
            "mask_selected_count": distribution_payload(counters, "mask_selected_count"),
        },
        "family_admission": family_admission_payload(counters),
        "stage_conservation": {
            "query_partition": int(counters["queries"]) == int(counters["rejected_no_object_queries"]) + int(counters["foreground_candidate_queries"]),
            "admission_partition": int(counters["foreground_candidate_queries"]) == int(counters["rejected_low_object_margin_queries"]) + int(counters["rejected_low_score_queries"]) + int(counters["admitted_object_queries"]),
            "proposal_partition": int(counters["admitted_object_queries"]) == int(counters["rejected_empty_mask_queries"]) + int(counters["proposal_queries"]),
        },
    }
    return merged, diagnostics


def proposal_overlap(a: set[int], b: set[int]) -> tuple[float, float]:
    if not a or not b:
        return 0.0, 0.0
    inter = len(a & b)
    if inter <= 0:
        return 0.0, 0.0
    union = len(a | b)
    iou = inter / max(union, 1)
    overlap = inter / max(min(len(a), len(b)), 1)
    return float(iou), float(overlap)


def should_merge(a: set[int], b: set[int], *, merge_iou_threshold: float, merge_overlap_threshold: float) -> bool:
    iou, overlap = proposal_overlap(a, b)
    return iou >= merge_iou_threshold or overlap >= merge_overlap_threshold


def proposal_center_distance(a: set[int], b: set[int]) -> float:
    """Return primitive-index center distance normalized by the joint span."""
    if not a or not b:
        return float("inf")
    joint = a | b
    span = max(max(joint) - min(joint), 1)
    return abs(float(np.mean(list(a))) - float(np.mean(list(b)))) / span


def reciprocal_merge_pairs(
    proposals: list[dict[str, Any]],
    *,
    merge_iou_threshold: float,
    merge_overlap_threshold: float,
    center_distance_threshold: float,
) -> list[tuple[int, int]]:
    best: dict[int, int] = {}
    for left_index, left in enumerate(proposals):
        left_ids = set(left.get("primitive_ids") or [])
        candidates: list[tuple[float, float, float, float, int]] = []
        for right_index, right in enumerate(proposals):
            if left_index == right_index or int(left["label"]) != int(right["label"]):
                continue
            if int(left.get("window_index", -1)) == int(right.get("window_index", -1)):
                continue
            right_ids = set(right.get("primitive_ids") or [])
            iou, overlap = proposal_overlap(left_ids, right_ids)
            center_distance = proposal_center_distance(left_ids, right_ids)
            if not (iou >= merge_iou_threshold or overlap >= merge_overlap_threshold):
                continue
            if center_distance > center_distance_threshold:
                continue
            candidates.append((overlap, iou, -center_distance, float(right.get("score", 0.0)), -right_index))
        if candidates:
            best[left_index] = -max(candidates)[-1]
    return [
        (left, right)
        for left, right in sorted(best.items())
        if left < right and best.get(right) == left
    ]


def reciprocal_identity_tracks(
    proposals: list[dict[str, Any]],
    *,
    merge_iou_threshold: float,
    merge_overlap_threshold: float,
    center_distance_threshold: float,
) -> list[list[int]]:
    """Build deterministic thing tracks from reciprocal adjacent-window matches."""
    thing_indices = [index for index, item in enumerate(proposals) if int(item["label"]) not in range(30, 35)]
    parent = {index: index for index in thing_indices}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    by_window: dict[int, list[int]] = defaultdict(list)
    for index in thing_indices:
        by_window[int(proposals[index].get("window_index", -1))].append(index)
    windows = sorted(window for window in by_window if window >= 0)
    for left_window, right_window in zip(windows, windows[1:]):
        if right_window != left_window + 1:
            continue
        left_indices = by_window[left_window]
        right_indices = by_window[right_window]

        def best_matches(source: list[int], targets: list[int]) -> dict[int, int]:
            matches: dict[int, int] = {}
            for source_index in source:
                source_item = proposals[source_index]
                source_ids = set(source_item.get("primitive_ids") or [])
                candidates: list[tuple[float, float, float, float, int]] = []
                for target_index in targets:
                    target_item = proposals[target_index]
                    if int(source_item["label"]) != int(target_item["label"]):
                        continue
                    target_ids = set(target_item.get("primitive_ids") or [])
                    iou, overlap = proposal_overlap(source_ids, target_ids)
                    center_distance = proposal_center_distance(source_ids, target_ids)
                    if not (iou >= merge_iou_threshold or overlap >= merge_overlap_threshold):
                        continue
                    if center_distance > center_distance_threshold:
                        continue
                    candidates.append((overlap, iou, -center_distance, float(target_item.get("score", 0.0)), -target_index))
                if candidates:
                    matches[source_index] = -max(candidates)[-1]
            return matches

        forward = best_matches(left_indices, right_indices)
        backward = best_matches(right_indices, left_indices)
        for left_index, right_index in sorted(forward.items()):
            if backward.get(right_index) == left_index:
                union(left_index, right_index)

    tracks: dict[int, list[int]] = defaultdict(list)
    for index in thing_indices:
        tracks[find(index)].append(index)
    groups = [sorted(indices) for _root, indices in sorted(tracks.items())]
    for label in range(30, 35):
        stuff = [index for index, item in enumerate(proposals) if int(item["label"]) == label]
        if stuff:
            groups.append(stuff)
    return groups


def merge_window_proposals(
    proposals: list[dict[str, Any]],
    *,
    merge_iou_threshold: float,
    merge_overlap_threshold: float,
    max_instances: int,
    merge_policy: str = "reciprocal",
    center_distance_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge window-local masks into page instances through identity tracks.

    The panoptic expert is trained on windowed primitive caches, while PQ is
    evaluated per original page. A physical component can therefore be emitted
    as several high-confidence local masks. Thing tracks use only reciprocal
    one-to-one assignments across adjacent windows, preventing unconstrained
    same-label closure; stuff classes use their canonical page-level union.
    """
    if merge_policy not in {"reciprocal", "legacy_connected_components", "topology_consistency"}:
        raise ValueError(f"unknown window merge policy: {merge_policy}")
    if merge_policy in {"reciprocal", "topology_consistency"}:
        track_indices = reciprocal_identity_tracks(
            proposals,
            merge_iou_threshold=merge_iou_threshold,
            merge_overlap_threshold=merge_overlap_threshold,
            center_distance_threshold=center_distance_threshold,
        )
        groups = [[proposals[index] for index in indices] for indices in track_indices]
        merged = []
        for fragments in groups:
            ids = sorted({int(value) for item in fragments for value in item.get("primitive_ids", [])})
            if not ids:
                continue
            winner = max(fragments, key=lambda item: float(item.get("score", 0.0)))
            score = float(winner.get("score", 0.0))
            if merge_policy == "topology_consistency":
                fragment_scores = [float(item.get("score", 0.0)) for item in fragments]
                score = (
                    0.80 * score
                    + 0.10 * (sum(fragment_scores) / max(len(fragment_scores), 1))
                    + 0.05 * min(math.log1p(len(ids)) / math.log(128.0), 1.0)
                    + 0.05 * min(len(fragments) / 4.0, 1.0)
                )
            merged.append({
                "label": int(fragments[0]["label"]),
                "primitive_ids": ids,
                "score": score,
                "class_score": float(winner.get("class_score", 0.0)),
                "quality_score": float(winner.get("quality_score", 1.0)),
                "proposal_source": "line_token_panoptic_moe_component_query",
                "moe_route": "line_token_component_panoptic",
                "query_indices": sorted({int(item.get("query_index", -1)) for item in fragments if int(item.get("query_index", -1)) >= 0}),
                "window_indices": sorted({int(item.get("window_index", -1)) for item in fragments if int(item.get("window_index", -1)) >= 0}),
                "merged_fragments": len(fragments),
                "decoder": "topology_consistency_reciprocal_tracks" if merge_policy == "topology_consistency" else "reciprocal_adjacent_window_identity_tracks",
            })
        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:max_instances]

    groups_by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for proposal in proposals:
        label = int(proposal["label"])
        ids = set(proposal.get("primitive_ids") or [])
        if not ids:
            continue
        matching: list[int] = []
        for group_index, group in enumerate(groups_by_label[label]):
            if should_merge(ids, group["id_set"], merge_iou_threshold=merge_iou_threshold, merge_overlap_threshold=merge_overlap_threshold):
                matching.append(group_index)
        if not matching:
            row = dict(proposal)
            row["id_set"] = set(ids)
            row["scores"] = [float(proposal.get("score", 0.0))]
            row["query_indices"] = [int(proposal.get("query_index", -1))]
            row["window_indices"] = [int(proposal.get("window_index", -1))]
            row["merged_fragments"] = 1
            groups_by_label[label].append(row)
            continue

        first = matching[0]
        target = groups_by_label[label][first]
        target["id_set"].update(ids)
        target["scores"].append(float(proposal.get("score", 0.0)))
        target["query_indices"].append(int(proposal.get("query_index", -1)))
        target["window_indices"].append(int(proposal.get("window_index", -1)))
        target["merged_fragments"] += 1
        for group_index in reversed(matching[1:]):
            other = groups_by_label[label].pop(group_index)
            target["id_set"].update(other["id_set"])
            target["scores"].extend(other["scores"])
            target["query_indices"].extend(other["query_indices"])
            target["window_indices"].extend(other["window_indices"])
            target["merged_fragments"] += int(other["merged_fragments"])

    merged: list[dict[str, Any]] = []
    for label, groups in groups_by_label.items():
        for group in groups:
            scores = [float(value) for value in group.pop("scores", [])]
            ids = sorted(int(value) for value in group.pop("id_set", set()))
            if not ids:
                continue
            row = {
                "label": label,
                "primitive_ids": ids,
                "score": max(scores) if scores else float(group.get("score", 0.0)),
                "proposal_source": "line_token_panoptic_moe_component_query",
                "moe_route": "line_token_component_panoptic",
                "query_indices": sorted(set(int(value) for value in group.pop("query_indices", []) if int(value) >= 0)),
                "window_indices": sorted(set(int(value) for value in group.pop("window_indices", []) if int(value) >= 0)),
                "merged_fragments": int(group.get("merged_fragments", 1)),
                "decoder": "overlap_graph_window_merge",
            }
            merged.append(row)
    merged.sort(key=lambda item: item["score"], reverse=True)
    return merged[:max_instances]


def primitive_reuse_diagnostics(instances: list[dict[str, Any]]) -> dict[str, Any]:
    by_primitive: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for instance_index, item in enumerate(instances):
        label = int(item.get("label", IGNORE_LABEL))
        for primitive_id in item.get("primitive_ids", []) if isinstance(item.get("primitive_ids"), list) else []:
            by_primitive[int(primitive_id)].append((instance_index, label))
    reused = {pid: owners for pid, owners in by_primitive.items() if len(owners) > 1}
    cross_label = {
        pid: owners
        for pid, owners in reused.items()
        if len({label for _instance_index, label in owners}) > 1
    }
    return {
        "reused_primitives": len(reused),
        "cross_label_reused_primitives": len(cross_label),
        "max_owners_per_primitive": max((len(owners) for owners in reused.values()), default=0),
    }


def instance_primitive_ids(item: dict[str, Any]) -> list[int]:
    values = item.get("primitive_ids")
    if not isinstance(values, list):
        return []
    return [int(value) for value in values]


def instance_trace(item: dict[str, Any], instance_index: int) -> dict[str, Any]:
    return {
        "instance_index": int(instance_index),
        "label": int(item.get("label", IGNORE_LABEL)),
        "score": float(item.get("score", 0.0)),
        "query_indices": list(item.get("query_indices") or []),
        "window_indices": list(item.get("window_indices") or []),
        "decoder": item.get("decoder"),
        "moe_route": item.get("moe_route"),
        "proposal_source": item.get("proposal_source"),
    }


def geometry_affinity_score(left: np.ndarray | None, right: np.ndarray | None, *, radius: float) -> float:
    if left is None or right is None or left.shape[0] < 6 or right.shape[0] < 6 or radius <= 0.0:
        return 0.0
    left_points = np.asarray([[left[0], left[1]], [left[2], left[3]], [left[4], left[5]]], dtype=np.float32)
    right_points = np.asarray([[right[0], right[1]], [right[2], right[3]], [right[4], right[5]]], dtype=np.float32)
    distances = np.linalg.norm(left_points[:, None, :] - right_points[None, :, :], axis=-1)
    min_distance = float(distances.min()) if distances.size else float("inf")
    if min_distance > radius:
        return 0.0
    return max(0.0, 1.0 - min_distance / max(radius, 1e-9))


def affinity_support(
    primitive_id: int,
    primitive_ids: list[int],
    affinity_edges: dict[tuple[int, int], float],
    *,
    threshold: float,
    primitive_features: dict[int, np.ndarray] | None = None,
    geometry_radius: float = 0.0,
) -> float:
    values = []
    for other_id in primitive_ids:
        if other_id == primitive_id:
            continue
        score = float(affinity_edges.get((min(primitive_id, other_id), max(primitive_id, other_id)), 0.0))
        if score >= threshold:
            values.append(score)
        geometry_score = geometry_affinity_score(
            None if primitive_features is None else primitive_features.get(primitive_id),
            None if primitive_features is None else primitive_features.get(other_id),
            radius=geometry_radius,
        )
        if geometry_score >= threshold:
            values.append(geometry_score)
    return max(values) if values else 0.0


def ownership_supports_for_instance(
    primitive_ids: list[int],
    affinity_edges: dict[tuple[int, int], float],
    *,
    threshold: float,
    primitive_features: dict[int, np.ndarray] | None = None,
    geometry_radius: float = 0.0,
) -> dict[int, float]:
    return {
        primitive_id: affinity_support(
            primitive_id,
            primitive_ids,
            affinity_edges,
            threshold=threshold,
            primitive_features=primitive_features,
            geometry_radius=geometry_radius,
        )
        for primitive_id in primitive_ids
    }


def apply_ownership_membership_gate(
    instances: list[dict[str, Any]],
    *,
    affinity_edges: dict[tuple[int, int], float],
    primitive_features: dict[int, np.ndarray] | None,
    affinity_threshold: float,
    geometry_affinity_radius: float,
    min_supported_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gated: list[dict[str, Any]] = []
    removed = 0
    supported_assignments = 0
    gate_considered = 0
    gate_applied = 0
    dropped_empty = 0
    removal_ledger: list[dict[str, Any]] = []
    for source_index, item in enumerate(instances):
        primitive_ids = instance_primitive_ids(item)
        if len(primitive_ids) <= 1:
            gated.append(item)
            continue
        supports = ownership_supports_for_instance(
            primitive_ids,
            affinity_edges,
            threshold=affinity_threshold,
            primitive_features=primitive_features,
            geometry_radius=geometry_affinity_radius,
        )
        supported = {pid for pid, score in supports.items() if score > 0.0}
        supported_assignments += len(supported)
        gate_considered += len(primitive_ids)
        supported_fraction = len(supported) / max(len(primitive_ids), 1)
        if supported_fraction < min_supported_fraction:
            row = dict(item)
            row["ownership_certification"] = {
                "gate": "not_applied_low_coverage",
                "supported_fraction": float(supported_fraction),
                "supported_primitives": len(supported),
                "primitive_count": len(primitive_ids),
            }
            gated.append(row)
            continue
        kept_ids = [pid for pid in primitive_ids if pid in supported]
        removed_ids = sorted(set(primitive_ids) - set(kept_ids))
        removed += len(removed_ids)
        gate_applied += 1
        source_trace = instance_trace(item, source_index)
        for primitive_id in removed_ids:
            removal_ledger.append(
                {
                    "stage": "ownership_membership_gate",
                    "primitive_id": int(primitive_id),
                    "label": source_trace["label"],
                    "source_instance": source_trace,
                    "support": float(supports.get(primitive_id, 0.0)),
                    "supported_fraction": float(supported_fraction),
                    "reason": "unsupported_by_ownership_affinity_gate",
                }
            )
        if not kept_ids:
            dropped_empty += 1
            continue
        row = dict(item)
        row["primitive_ids"] = sorted(set(kept_ids))
        row["decoder"] = f"{row.get('decoder', 'query_mask')}_ownership_moe_membership_gate"
        weights = dict(row.get("sparse_expert_weights") or {"line_token_component_panoptic": 1.0})
        weights["line_token_component_panoptic"] = min(float(weights.get("line_token_component_panoptic", 1.0)), 0.75)
        weights["ownership_affinity_gate"] = max(float(weights.get("ownership_affinity_gate", 0.0)), 0.25)
        row["sparse_expert_weights"] = weights
        row["moe_route"] = "line_token_component_panoptic+ownership_affinity_gate"
        row["ownership_certification"] = {
            "gate": "applied",
            "supported_fraction": float(supported_fraction),
            "supported_primitives": len(supported),
            "primitive_count": len(primitive_ids),
            "removed_primitives": len(primitive_ids) - len(kept_ids),
            "min_supported_fraction": float(min_supported_fraction),
        }
        gated.append(row)
    return gated, {
        "membership_gate_removed_assignments": removed,
        "membership_gate_supported_assignments": supported_assignments,
        "membership_gate_considered_assignments": gate_considered,
        "membership_gate_applied_instances": gate_applied,
        "membership_gate_dropped_empty_instances": dropped_empty,
        "membership_gate_min_supported_fraction": float(min_supported_fraction),
        "primitive_assignment_removal_ledger": removal_ledger,
    }


def apply_ownership_calibration(
    instances: list[dict[str, Any]],
    *,
    affinity_edges: dict[tuple[int, int], float],
    primitive_features: dict[int, np.ndarray] | None,
    affinity_threshold: float,
    geometry_affinity_radius: float,
    affinity_weight: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    supported_assignments = 0
    considered_assignments = 0
    calibrated_instances = 0
    score_delta_sum = 0.0
    for item in instances:
        primitive_ids = instance_primitive_ids(item)
        if len(primitive_ids) <= 1:
            calibrated.append(item)
            continue
        supports = ownership_supports_for_instance(
            primitive_ids,
            affinity_edges,
            threshold=affinity_threshold,
            primitive_features=primitive_features,
            geometry_radius=geometry_affinity_radius,
        )
        supported = {pid for pid, score in supports.items() if score > 0.0}
        supported_fraction = len(supported) / max(len(primitive_ids), 1)
        mean_support = float(np.mean([score for score in supports.values()])) if supports else 0.0
        considered_assignments += len(primitive_ids)
        supported_assignments += len(supported)
        row = dict(item)
        base_score = float(row.get("score", 0.0))
        # Soft calibration: reward internally consistent ownership evidence, but
        # do not delete primitive memberships. This keeps PQ-sensitive masks
        # intact while exposing ownership as a routed expert signal.
        delta = float(affinity_weight) * (0.5 * supported_fraction + 0.5 * mean_support)
        row["score"] = base_score + delta
        score_delta_sum += delta
        calibrated_instances += 1
        weights = dict(row.get("sparse_expert_weights") or {"line_token_component_panoptic": 1.0})
        weights["ownership_affinity_calibrator"] = max(float(weights.get("ownership_affinity_calibrator", 0.0)), min(0.25, max(delta, 0.05)))
        row["sparse_expert_weights"] = weights
        row["moe_route"] = "line_token_component_panoptic+ownership_affinity_calibrator"
        row["decoder"] = f"{row.get('decoder', 'query_mask')}_ownership_moe_calibrated"
        row["ownership_certification"] = {
            "gate": "soft_calibrated",
            "supported_fraction": float(supported_fraction),
            "mean_support": float(mean_support),
            "supported_primitives": len(supported),
            "primitive_count": len(primitive_ids),
            "score_delta": float(delta),
        }
        calibrated.append(row)
    calibrated.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return calibrated, {
        "calibrated_instances": calibrated_instances,
        "calibration_supported_assignments": supported_assignments,
        "calibration_considered_assignments": considered_assignments,
        "calibration_support_rate": supported_assignments / max(considered_assignments, 1),
        "calibration_score_delta_mean": score_delta_sum / max(calibrated_instances, 1),
    }


def apply_ownership_hybrid_safe(
    instances: list[dict[str, Any]],
    *,
    affinity_edges: dict[tuple[int, int], float],
    primitive_features: dict[int, np.ndarray] | None,
    affinity_threshold: float,
    geometry_affinity_radius: float,
    affinity_weight: float,
    hard_supported_fraction: float = 0.85,
    max_hard_remove_fraction: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fused: list[dict[str, Any]] = []
    removal_ledger: list[dict[str, Any]] = []
    supported_assignments = 0
    considered_assignments = 0
    calibrated_instances = 0
    hard_pruned_instances = 0
    hard_removed = 0
    dropped_empty = 0
    score_delta_sum = 0.0
    for source_index, item in enumerate(instances):
        primitive_ids = instance_primitive_ids(item)
        if len(primitive_ids) <= 1:
            fused.append(item)
            continue
        supports = ownership_supports_for_instance(
            primitive_ids,
            affinity_edges,
            threshold=affinity_threshold,
            primitive_features=primitive_features,
            geometry_radius=geometry_affinity_radius,
        )
        supported = {pid for pid, score in supports.items() if score > 0.0}
        supported_fraction = len(supported) / max(len(primitive_ids), 1)
        mean_support = float(np.mean([score for score in supports.values()])) if supports else 0.0
        supported_assignments += len(supported)
        considered_assignments += len(primitive_ids)
        removed_ids = sorted(set(primitive_ids) - supported)
        remove_fraction = len(removed_ids) / max(len(primitive_ids), 1)
        hard_prune = (
            supported_fraction >= hard_supported_fraction
            and 0 < remove_fraction <= max_hard_remove_fraction
        )
        row = dict(item)
        base_score = float(row.get("score", 0.0))
        delta = float(affinity_weight) * (0.5 * supported_fraction + 0.5 * mean_support)
        row["score"] = base_score + delta
        score_delta_sum += delta
        calibrated_instances += 1
        weights = dict(row.get("sparse_expert_weights") or {"line_token_component_panoptic": 1.0})
        weights["ownership_affinity_hybrid_safe"] = max(float(weights.get("ownership_affinity_hybrid_safe", 0.0)), min(0.25, max(delta, 0.05)))
        row["sparse_expert_weights"] = weights
        row["moe_route"] = "line_token_component_panoptic+ownership_affinity_hybrid_safe"
        if hard_prune:
            kept_ids = [pid for pid in primitive_ids if pid in supported]
            hard_pruned_instances += 1
            hard_removed += len(removed_ids)
            source_trace = instance_trace(item, source_index)
            for primitive_id in removed_ids:
                removal_ledger.append(
                    {
                        "stage": "ownership_hybrid_safe_hard_prune",
                        "primitive_id": int(primitive_id),
                        "label": source_trace["label"],
                        "source_instance": source_trace,
                        "support": float(supports.get(primitive_id, 0.0)),
                        "supported_fraction": float(supported_fraction),
                        "remove_fraction": float(remove_fraction),
                        "reason": "high_confidence_low_fraction_unsupported_prune",
                    }
                )
            if not kept_ids:
                dropped_empty += 1
                continue
            row["primitive_ids"] = sorted(set(kept_ids))
            row["decoder"] = f"{row.get('decoder', 'query_mask')}_ownership_moe_hybrid_safe_pruned"
            row["ownership_certification"] = {
                "gate": "hybrid_safe_hard_pruned",
                "supported_fraction": float(supported_fraction),
                "mean_support": float(mean_support),
                "supported_primitives": len(supported),
                "primitive_count": len(primitive_ids),
                "removed_primitives": len(removed_ids),
                "hard_supported_fraction": float(hard_supported_fraction),
                "max_hard_remove_fraction": float(max_hard_remove_fraction),
                "score_delta": float(delta),
            }
        else:
            row["decoder"] = f"{row.get('decoder', 'query_mask')}_ownership_moe_hybrid_safe_soft"
            row["ownership_certification"] = {
                "gate": "hybrid_safe_soft_calibrated",
                "supported_fraction": float(supported_fraction),
                "mean_support": float(mean_support),
                "supported_primitives": len(supported),
                "primitive_count": len(primitive_ids),
                "would_remove_primitives": len(removed_ids),
                "hard_supported_fraction": float(hard_supported_fraction),
                "max_hard_remove_fraction": float(max_hard_remove_fraction),
                "score_delta": float(delta),
            }
        fused.append(row)
    fused.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return fused, {
        "hybrid_safe_instances": calibrated_instances,
        "hybrid_safe_hard_pruned_instances": hard_pruned_instances,
        "hybrid_safe_hard_removed_assignments": hard_removed,
        "hybrid_safe_dropped_empty_instances": dropped_empty,
        "hybrid_safe_supported_assignments": supported_assignments,
        "hybrid_safe_considered_assignments": considered_assignments,
        "hybrid_safe_support_rate": supported_assignments / max(considered_assignments, 1),
        "hybrid_safe_score_delta_mean": score_delta_sum / max(calibrated_instances, 1),
        "hybrid_safe_hard_supported_fraction": float(hard_supported_fraction),
        "hybrid_safe_max_hard_remove_fraction": float(max_hard_remove_fraction),
        "calibrated_instances": calibrated_instances,
        "calibration_supported_assignments": supported_assignments,
        "calibration_considered_assignments": considered_assignments,
        "calibration_support_rate": supported_assignments / max(considered_assignments, 1),
        "calibration_score_delta_mean": score_delta_sum / max(calibrated_instances, 1),
        "membership_gate_removed_assignments": hard_removed,
        "membership_gate_supported_assignments": supported_assignments,
        "membership_gate_considered_assignments": considered_assignments,
        "membership_gate_applied_instances": hard_pruned_instances,
        "membership_gate_dropped_empty_instances": dropped_empty,
        "primitive_assignment_removal_ledger": removal_ledger,
    }


def primitive_cross_label_reuse_ids(instances: list[dict[str, Any]]) -> set[int]:
    owners: dict[int, set[int]] = defaultdict(set)
    for item in instances:
        label = int(item.get("label", IGNORE_LABEL))
        for primitive_id in instance_primitive_ids(item):
            owners[int(primitive_id)].add(label)
    return {primitive_id for primitive_id, labels in owners.items() if len(labels) > 1}


def apply_ownership_selective_hard_safe(
    instances: list[dict[str, Any]],
    *,
    affinity_edges: dict[tuple[int, int], float],
    primitive_features: dict[int, np.ndarray] | None,
    affinity_threshold: float,
    geometry_affinity_radius: float,
    affinity_weight: float,
    hard_supported_fraction: float = 0.45,
    max_hard_remove_fraction: float = 0.35,
    min_cross_label_remove_fraction: float = 0.60,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fused: list[dict[str, Any]] = []
    removal_ledger: list[dict[str, Any]] = []
    cross_label_ids = primitive_cross_label_reuse_ids(instances)
    supported_assignments = 0
    considered_assignments = 0
    calibrated_instances = 0
    hard_pruned_instances = 0
    hard_removed = 0
    dropped_empty = 0
    score_delta_sum = 0.0
    for source_index, item in enumerate(instances):
        primitive_ids = instance_primitive_ids(item)
        if len(primitive_ids) <= 1:
            fused.append(item)
            continue
        supports = ownership_supports_for_instance(
            primitive_ids,
            affinity_edges,
            threshold=affinity_threshold,
            primitive_features=primitive_features,
            geometry_radius=geometry_affinity_radius,
        )
        supported = {pid for pid, score in supports.items() if score > 0.0}
        supported_fraction = len(supported) / max(len(primitive_ids), 1)
        mean_support = float(np.mean([score for score in supports.values()])) if supports else 0.0
        supported_assignments += len(supported)
        considered_assignments += len(primitive_ids)
        unsupported_ids = sorted(set(primitive_ids) - supported)
        cross_label_unsupported = [pid for pid in unsupported_ids if pid in cross_label_ids]
        remove_fraction = len(unsupported_ids) / max(len(primitive_ids), 1)
        cross_label_remove_fraction = len(cross_label_unsupported) / max(len(unsupported_ids), 1)
        hard_prune = (
            supported_fraction >= hard_supported_fraction
            and 0 < remove_fraction <= max_hard_remove_fraction
            and cross_label_remove_fraction >= min_cross_label_remove_fraction
        )
        row = dict(item)
        base_score = float(row.get("score", 0.0))
        delta = float(affinity_weight) * (0.5 * supported_fraction + 0.5 * mean_support)
        row["score"] = base_score + delta
        score_delta_sum += delta
        calibrated_instances += 1
        weights = dict(row.get("sparse_expert_weights") or {"line_token_component_panoptic": 1.0})
        weights["ownership_affinity_selective_hard_safe"] = max(float(weights.get("ownership_affinity_selective_hard_safe", 0.0)), min(0.25, max(delta, 0.05)))
        row["sparse_expert_weights"] = weights
        row["moe_route"] = "line_token_component_panoptic+ownership_affinity_selective_hard_safe"
        if hard_prune:
            kept_ids = [pid for pid in primitive_ids if pid not in set(cross_label_unsupported)]
            hard_pruned_instances += 1
            hard_removed += len(cross_label_unsupported)
            source_trace = instance_trace(item, source_index)
            for primitive_id in cross_label_unsupported:
                removal_ledger.append(
                    {
                        "stage": "ownership_selective_hard_safe_cross_label_prune",
                        "primitive_id": int(primitive_id),
                        "label": source_trace["label"],
                        "source_instance": source_trace,
                        "support": float(supports.get(primitive_id, 0.0)),
                        "supported_fraction": float(supported_fraction),
                        "remove_fraction": float(remove_fraction),
                        "cross_label_remove_fraction": float(cross_label_remove_fraction),
                        "reason": "unsupported_cross_label_reuse_prune",
                    }
                )
            if not kept_ids:
                dropped_empty += 1
                continue
            row["primitive_ids"] = sorted(set(kept_ids))
            row["decoder"] = f"{row.get('decoder', 'query_mask')}_ownership_moe_selective_hard_safe_pruned"
            row["ownership_certification"] = {
                "gate": "selective_hard_safe_pruned",
                "supported_fraction": float(supported_fraction),
                "mean_support": float(mean_support),
                "supported_primitives": len(supported),
                "primitive_count": len(primitive_ids),
                "removed_primitives": len(cross_label_unsupported),
                "would_remove_primitives": len(unsupported_ids),
                "cross_label_reused_primitives": len([pid for pid in primitive_ids if pid in cross_label_ids]),
                "hard_supported_fraction": float(hard_supported_fraction),
                "max_hard_remove_fraction": float(max_hard_remove_fraction),
                "min_cross_label_remove_fraction": float(min_cross_label_remove_fraction),
                "score_delta": float(delta),
            }
        else:
            row["decoder"] = f"{row.get('decoder', 'query_mask')}_ownership_moe_selective_hard_safe_soft"
            row["ownership_certification"] = {
                "gate": "selective_hard_safe_soft_calibrated",
                "supported_fraction": float(supported_fraction),
                "mean_support": float(mean_support),
                "supported_primitives": len(supported),
                "primitive_count": len(primitive_ids),
                "would_remove_primitives": len(unsupported_ids),
                "cross_label_reused_primitives": len([pid for pid in primitive_ids if pid in cross_label_ids]),
                "hard_supported_fraction": float(hard_supported_fraction),
                "max_hard_remove_fraction": float(max_hard_remove_fraction),
                "min_cross_label_remove_fraction": float(min_cross_label_remove_fraction),
                "score_delta": float(delta),
            }
        fused.append(row)
    fused.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return fused, {
        "selective_hard_safe_instances": calibrated_instances,
        "selective_hard_safe_hard_pruned_instances": hard_pruned_instances,
        "selective_hard_safe_hard_removed_assignments": hard_removed,
        "selective_hard_safe_dropped_empty_instances": dropped_empty,
        "selective_hard_safe_supported_assignments": supported_assignments,
        "selective_hard_safe_considered_assignments": considered_assignments,
        "selective_hard_safe_support_rate": supported_assignments / max(considered_assignments, 1),
        "selective_hard_safe_score_delta_mean": score_delta_sum / max(calibrated_instances, 1),
        "selective_hard_safe_cross_label_reused_primitives": len(cross_label_ids),
        "calibrated_instances": calibrated_instances,
        "calibration_supported_assignments": supported_assignments,
        "calibration_considered_assignments": considered_assignments,
        "calibration_support_rate": supported_assignments / max(considered_assignments, 1),
        "calibration_score_delta_mean": score_delta_sum / max(calibrated_instances, 1),
        "membership_gate_removed_assignments": hard_removed,
        "membership_gate_supported_assignments": supported_assignments,
        "membership_gate_considered_assignments": considered_assignments,
        "membership_gate_applied_instances": hard_pruned_instances,
        "membership_gate_dropped_empty_instances": dropped_empty,
        "primitive_assignment_removal_ledger": removal_ledger,
    }


def enforce_primitive_conflicts(
    instances: list[dict[str, Any]],
    policy: str,
    *,
    affinity_edges: dict[tuple[int, int], float] | None = None,
    primitive_features: dict[int, np.ndarray] | None = None,
    affinity_threshold: float = 0.6,
    affinity_weight: float = 0.25,
    geometry_affinity_radius: float = 0.0025,
    ownership_gate_min_supported_fraction: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if policy == "report_only":
        return instances, {"policy": policy, "removed_primitive_assignments": 0, "dropped_empty_instances": 0, "affinity_supported_assignments": 0}
    ownership_policies = {"ownership_moe_gate", "ownership_moe_calibrated", "ownership_moe_hybrid_safe", "ownership_moe_selective_hard_safe"}
    labelwise_policies = {"label_wise_winner_take_score", "affinity_label_wise_winner_take_score", *ownership_policies}
    affinity_policies = {"affinity_label_wise_winner_take_score", *ownership_policies}
    if policy not in {"winner_take_score", *labelwise_policies}:
        raise ValueError(f"unknown primitive conflict policy: {policy}")

    owners: dict[tuple[int, int | None], tuple[int, float]] = {}
    affinity_edges = affinity_edges or {}
    gate_report = {
        "membership_gate_removed_assignments": 0,
        "membership_gate_supported_assignments": 0,
        "membership_gate_considered_assignments": 0,
        "membership_gate_applied_instances": 0,
        "membership_gate_dropped_empty_instances": 0,
        "membership_gate_min_supported_fraction": float(ownership_gate_min_supported_fraction),
        "calibrated_instances": 0,
        "calibration_supported_assignments": 0,
        "calibration_considered_assignments": 0,
        "calibration_support_rate": 0.0,
        "calibration_score_delta_mean": 0.0,
        "hybrid_safe_instances": 0,
        "hybrid_safe_hard_pruned_instances": 0,
        "hybrid_safe_hard_removed_assignments": 0,
        "hybrid_safe_dropped_empty_instances": 0,
        "hybrid_safe_supported_assignments": 0,
        "hybrid_safe_considered_assignments": 0,
        "hybrid_safe_support_rate": 0.0,
        "hybrid_safe_score_delta_mean": 0.0,
        "selective_hard_safe_instances": 0,
        "selective_hard_safe_hard_pruned_instances": 0,
        "selective_hard_safe_hard_removed_assignments": 0,
        "selective_hard_safe_dropped_empty_instances": 0,
        "selective_hard_safe_supported_assignments": 0,
        "selective_hard_safe_considered_assignments": 0,
        "selective_hard_safe_support_rate": 0.0,
        "selective_hard_safe_score_delta_mean": 0.0,
        "selective_hard_safe_cross_label_reused_primitives": 0,
        "primitive_assignment_removal_ledger": [],
    }
    if policy == "ownership_moe_gate":
        instances, gate_report = apply_ownership_membership_gate(
            instances,
            affinity_edges=affinity_edges,
            primitive_features=primitive_features,
            affinity_threshold=affinity_threshold,
            geometry_affinity_radius=geometry_affinity_radius,
            min_supported_fraction=ownership_gate_min_supported_fraction,
        )
    elif policy == "ownership_moe_calibrated":
        instances, calibration_report = apply_ownership_calibration(
            instances,
            affinity_edges=affinity_edges,
            primitive_features=primitive_features,
            affinity_threshold=affinity_threshold,
            geometry_affinity_radius=geometry_affinity_radius,
            affinity_weight=affinity_weight,
        )
        gate_report.update(calibration_report)
    elif policy == "ownership_moe_hybrid_safe":
        instances, hybrid_report = apply_ownership_hybrid_safe(
            instances,
            affinity_edges=affinity_edges,
            primitive_features=primitive_features,
            affinity_threshold=affinity_threshold,
            geometry_affinity_radius=geometry_affinity_radius,
            affinity_weight=affinity_weight,
        )
        gate_report.update(hybrid_report)
    elif policy == "ownership_moe_selective_hard_safe":
        instances, selective_report = apply_ownership_selective_hard_safe(
            instances,
            affinity_edges=affinity_edges,
            primitive_features=primitive_features,
            affinity_threshold=affinity_threshold,
            geometry_affinity_radius=geometry_affinity_radius,
            affinity_weight=affinity_weight,
        )
        gate_report.update(selective_report)
    affinity_supported = 0
    removed = 0
    for instance_index, item in enumerate(instances):
        base_score = float(item.get("score", 0.0))
        primitive_ids = instance_primitive_ids(item)
        label_key = int(item.get("label", IGNORE_LABEL)) if policy in labelwise_policies else None
        for primitive_id in primitive_ids:
            support = 0.0
            if policy in affinity_policies:
                support = affinity_support(
                    primitive_id,
                    primitive_ids,
                    affinity_edges,
                    threshold=affinity_threshold,
                    primitive_features=primitive_features,
                    geometry_radius=geometry_affinity_radius,
                )
                affinity_supported += int(support > 0.0)
            score = base_score + float(affinity_weight) * support
            owner_key = (primitive_id, label_key)
            previous = owners.get(owner_key)
            if previous is None or score > previous[1] or (score == previous[1] and instance_index < previous[0]):
                owners[owner_key] = (instance_index, score)

    resolved: list[dict[str, Any]] = []
    dropped_empty = 0
    removal_ledger = list(gate_report.get("primitive_assignment_removal_ledger") or [])
    for instance_index, item in enumerate(instances):
        primitive_ids = instance_primitive_ids(item)
        label_key = int(item.get("label", IGNORE_LABEL)) if policy in labelwise_policies else None
        source_trace = instance_trace(item, instance_index)
        kept_ids = [
            primitive_id
            for primitive_id in primitive_ids
            if owners.get((primitive_id, label_key), (-1, 0.0))[0] == instance_index
        ]
        for primitive_id in sorted(set(primitive_ids) - set(kept_ids)):
            winner_index, winner_score = owners.get((primitive_id, label_key), (-1, 0.0))
            winner = instances[winner_index] if 0 <= winner_index < len(instances) else {}
            removal_ledger.append(
                {
                    "stage": "primitive_conflict_resolution",
                    "policy": policy,
                    "primitive_id": int(primitive_id),
                    "label": source_trace["label"],
                    "source_instance": source_trace,
                    "winner_instance": instance_trace(winner, winner_index) if winner else None,
                    "source_score": float(item.get("score", 0.0)),
                    "winner_score": float(winner_score),
                    "owner_key": [int(primitive_id), label_key],
                    "reason": "lost_winner_take_assignment",
                }
            )
        removed += max(len(primitive_ids) - len(kept_ids), 0)
        if not kept_ids:
            dropped_empty += 1
            continue
        row = dict(item)
        row["primitive_ids"] = sorted(set(kept_ids))
        row["decoder"] = f"{row.get('decoder', 'query_mask')}_{policy}"
        if policy in affinity_policies:
            weights = dict(row.get("sparse_expert_weights") or {"line_token_component_panoptic": 1.0})
            if policy == "ownership_moe_gate":
                weights["ownership_affinity_gate"] = max(float(weights.get("ownership_affinity_gate", 0.0)), 0.25)
                row["moe_route"] = "line_token_component_panoptic+ownership_affinity_gate"
                row["decoder"] = f"{row.get('decoder', 'query_mask')}_label_wise_ownership_conflict"
            elif policy == "ownership_moe_calibrated":
                weights["ownership_affinity_calibrator"] = max(float(weights.get("ownership_affinity_calibrator", 0.0)), 0.25)
                row["moe_route"] = "line_token_component_panoptic+ownership_affinity_calibrator"
                row["decoder"] = f"{row.get('decoder', 'query_mask')}_label_wise_calibrated_ownership_conflict"
            elif policy == "ownership_moe_hybrid_safe":
                weights["ownership_affinity_hybrid_safe"] = max(float(weights.get("ownership_affinity_hybrid_safe", 0.0)), 0.25)
                row["moe_route"] = "line_token_component_panoptic+ownership_affinity_hybrid_safe"
                row["decoder"] = f"{row.get('decoder', 'query_mask')}_label_wise_hybrid_safe_ownership_conflict"
            elif policy == "ownership_moe_selective_hard_safe":
                weights["ownership_affinity_selective_hard_safe"] = max(float(weights.get("ownership_affinity_selective_hard_safe", 0.0)), 0.25)
                row["moe_route"] = "line_token_component_panoptic+ownership_affinity_selective_hard_safe"
                row["decoder"] = f"{row.get('decoder', 'query_mask')}_label_wise_selective_hard_safe_ownership_conflict"
            else:
                weights["label_ownership_affinity"] = max(float(weights.get("label_ownership_affinity", 0.0)), 0.25)
                row["moe_route"] = "line_token_component_panoptic+label_ownership_affinity"
            row["sparse_expert_weights"] = weights
        resolved.append(row)
    return resolved, {
        **gate_report,
        "policy": policy,
        "removed_primitive_assignments": removed,
        "dropped_empty_instances": dropped_empty,
        "affinity_supported_assignments": affinity_supported,
        "affinity_edges_available": len(affinity_edges),
        "affinity_threshold": affinity_threshold,
        "affinity_weight": affinity_weight,
        "geometry_affinity_radius": geometry_affinity_radius,
        "primitive_assignment_removal_ledger": removal_ledger,
        "primitive_assignment_removal_ledger_count": len(removal_ledger),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--route-trace-output", type=Path, default=DEFAULT_ROUTE_TRACE)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-windows", type=int, default=0)
    parser.add_argument("--record-id-allowlist", type=Path, default=None)
    parser.add_argument("--query-admission-policy", choices=["respect_no_object", "legacy_force_object"], default="respect_no_object")
    parser.add_argument("--min-query-score", type=float, default=DEFAULT_MIN_QUERY_SCORE)
    parser.add_argument(
        "--family-min-query-score-overrides",
        default="",
        help="Diagnostic-only comma-separated family:threshold overrides, e.g. furniture:0.15.",
    )
    parser.add_argument(
        "--use-quality",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use a trained quality head for proposal ranking. --no-use-quality restores pure class-score ranking for a reversible ablation.",
    )
    parser.add_argument("--min-object-margin", type=float, default=0.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--semantic-stuff", action=argparse.BooleanOptionalAction, default=True,
        help="Decode stuff from page-merged semantic logits after thing ownership/conflict resolution.",
    )
    parser.add_argument("--semantic-stuff-min-score", type=float, default=0.35)
    parser.add_argument("--recall-expansion-policy", choices=["off", "endpoint_affinity"], default="off")
    parser.add_argument("--recall-expansion-threshold", type=float, default=0.35)
    parser.add_argument("--recall-expansion-endpoint-radius", type=float, default=0.0025)
    parser.add_argument("--recall-expansion-max-ratio", type=float, default=0.35)
    parser.add_argument("--merge-iou-threshold", type=float, default=0.25)
    parser.add_argument("--merge-overlap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--window-merge-policy",
        choices=["reciprocal", "legacy_connected_components", "topology_consistency"],
        default="reciprocal",
    )
    parser.add_argument("--merge-center-distance-threshold", type=float, default=0.5)
    parser.add_argument("--max-instances-per-record", type=int, default=256)
    parser.add_argument("--primitive-conflict-policy", choices=["report_only", "winner_take_score", "label_wise_winner_take_score", "affinity_label_wise_winner_take_score", "ownership_moe_gate", "ownership_moe_calibrated", "ownership_moe_hybrid_safe", "ownership_moe_selective_hard_safe"], default="report_only")
    parser.add_argument("--affinity-scores", type=Path, default=None)
    parser.add_argument("--affinity-threshold", type=float, default=0.5660592496395112)
    parser.add_argument("--affinity-weight", type=float, default=0.25)
    parser.add_argument("--geometry-affinity-radius", type=float, default=0.0025)
    parser.add_argument("--ownership-gate-min-supported-fraction", type=float, default=0.35)
    parser.add_argument(
        "--ownership-decoder",
        choices=["auto", "page_global", "mask_guided", "mask_only"],
        default="auto",
        help="Use page-global ownership, mask-guided ownership, or query-mask-only decoding; all ownership modes honor --mask-threshold.",
    )
    parser.add_argument("--model-mode", choices=["eval", "train"], default="eval", help="Diagnostic switch; official inference uses eval.")
    parser.add_argument(
        "--legacy-position-compat",
        action="store_true",
        help="Diagnostic-only: load ABI-incomplete checkpoints with legacy-v1 position and untrained quality multiplier fixed to 1.",
    )
    parser.add_argument("--allow-unbound-cache", action="store_true", help="Diagnostic-only override for caches without inference provenance; never valid for production evaluation.")
    args = parser.parse_args()
    if not 0.0 <= float(args.min_query_score) <= 1.0:
        parser.error("--min-query-score must be in [0, 1]")
    if not 0.0 <= float(args.mask_threshold) <= 1.0:
        parser.error("--mask-threshold must be in [0, 1]")
    return args


def main() -> int:
    args = parse_args()
    missing = [path for path in [args.model, args.cache] if not path.exists()]
    if missing:
        payload = {"schema_version": "floorplancad_line_token_panoptic_moe_apply_v1", "created_utc": utc_now(), "status": "blocked", "blockers": [f"missing input: {rel(path)}" for path in missing]}
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    pack = import_torch()
    torch = pack["torch"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    abi = validate_checkpoint_abi(ckpt, legacy_position_compat=args.legacy_position_compat)
    quality_enabled = resolve_quality_admission_enabled(abi, args.use_quality)
    quality_mask_threshold = float(
        (abi.get("quality_objective_config") or {}).get("mask_threshold", -1.0)
    )
    if quality_enabled and not math.isclose(
        float(args.mask_threshold), quality_mask_threshold, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            "quality admission requires the checkpoint mask threshold protocol: "
            f"requested={float(args.mask_threshold):.6f}, trained={quality_mask_threshold:.6f}"
        )
    if args.ownership_decoder == "page_global" and not bool(abi.get("ownership_enabled", False)):
        raise ValueError("--ownership-decoder page_global requires an ownership-enabled checkpoint")
    model = make_panoptic_model(
        pack["nn"],
        torch,
        len(ckpt["feature_names"]),
        int(ckpt["hidden_dim"]),
        int(ckpt["layers"]),
        int(ckpt["heads"]),
        int(ckpt["num_queries"]),
        int(ckpt.get("num_labels", 36)),
        int(ckpt.get("query_decoder_layers", 1)),
        float(ckpt.get("dropout", 0.1)),
        position_encoding_version=str(abi["position_encoding_version"]),
        identity_dim=int(ckpt.get("identity_dim", 32)),
        geometry_decoder_mode=str(abi.get("geometry_decoder_mode", "legacy_debug")),
        num_stuff_queries=int((abi.get("geometry_config") or {}).get("num_stuff_queries", 32)),
        typed_stuff_slots=bool((abi.get("geometry_config") or {}).get("typed_stuff_slots", False)),
        geometry_local_neighbors=int((abi.get("geometry_config") or {}).get("local_neighbors", 4)),
        geometry_coarse_grid_size=int((abi.get("geometry_config") or {}).get("coarse_grid_size", 4)),
        tensor_ring_rank=int((abi.get("geometry_config") or {}).get("tensor_ring_rank", 0)),
        geometry_attention_tile_size=int((abi.get("geometry_config") or {}).get("geometry_attention_tile_size", 0)),
        sq_rq_enabled=bool(abi.get("sq_rq_enabled", False)),
        sq_rq_gradient_scale=float((abi.get("sq_rq_config") or {}).get("gradient_scale", 0.0)),
        sq_rq_query_confidence_threshold=float((abi.get("sq_rq_config") or {}).get("query_confidence_threshold", 0.6)),
        sq_rq_token_membership_threshold=float((abi.get("sq_rq_config") or {}).get("token_membership_threshold", 0.5)),
        semantic_query_residual_enabled=bool((abi.get("sq_rq_config") or {}).get("semantic_query_residual_enabled", False)),
        ownership_enabled=bool(abi.get("ownership_enabled", False)),
        learned_sparse_router=bool(abi.get("sparse_router_enabled", False)),
        router_num_experts=int((abi.get("sparse_router_config") or {}).get("num_experts", 4)),
        router_top_k=int((abi.get("sparse_router_config") or {}).get("top_k", 2)),
        router_temperature=float((abi.get("sparse_router_config") or {}).get("temperature", 1.0)),
        typed_branch_routers=bool((abi.get("sparse_router_config") or {}).get("typed_branch_routers", False)),
        branch_num_experts=int((abi.get("sparse_router_config") or {}).get("branch_num_experts", 2)),
        branch_top_k=int((abi.get("sparse_router_config") or {}).get("branch_top_k", 1)),
        branch_capacity_factor=float((abi.get("sparse_router_config") or {}).get("branch_capacity_factor", 1.25)),
        branch_dropless=bool((abi.get("sparse_router_config") or {}).get("branch_dropless", False)),
        content_seeded_queries=bool((abi.get("input_protocol") or {}).get("content_seeded_queries", False)),
    ).to(device)
    state_dict = dict(ckpt["state_dict"])
    legacy_zero_initialized_keys: set[str] = set()
    for key, value in model.state_dict().items():
        if (key.startswith(("query_objectness_head.", "stuff_presence_head.")) or key == "ownership_residual_gate") and key not in state_dict:
            state_dict[key] = torch.zeros_like(value)
            legacy_zero_initialized_keys.add(key)
    if abi.get("sparse_router_enabled", False) and abi.get("input_protocol") is not None:
        model.load_state_dict(state_dict, strict=True)
        missing_keys: set[str] = set()
        unexpected_keys: set[str] = set()
    else:
        incompatible = model.load_state_dict(state_dict, strict=False)
        missing_keys = set(incompatible.missing_keys)
        unexpected_keys = set(incompatible.unexpected_keys)
    allowed_legacy_missing = {"query_quality_head.weight", "query_quality_head.bias"} if not abi["quality_head_trained"] else set()
    if abi.get("input_protocol") is None:
        allowed_legacy_missing.update(
            key for key in model.state_dict()
            if key.startswith(("segment_encoder.", "segment_pool_score.", "segment_aggregate_encoder.", "segment_fusion_gate.", "segment_fusion_norm.", "thing_seed_score."))
        )
    if not abi.get("identity_head_trained", False):
        allowed_legacy_missing.update(key for key in model.state_dict() if key.startswith("query_identity_head."))
    if not abi.get("ownership_enabled", False):
        allowed_legacy_missing.update(
            key for key in model.state_dict() if "ownership_head" in key or key == "ownership_residual_gate"
        )
    if missing_keys not in (set(), allowed_legacy_missing) or unexpected_keys:
        raise RuntimeError(
            "checkpoint architecture mismatch: "
            f"missing={sorted(missing_keys)} unexpected={sorted(unexpected_keys)}"
        )
    model.sq_rq_runtime_enabled = bool(abi.get("sq_rq_deployment_enabled", False))
    if abi.get("sq_rq_deployment") is not None:
        set_sq_rq_runtime_thresholds(model, abi["sq_rq_deployment"])
    if args.model_mode == "train":
        model.train()
    else:
        model.eval()
    limit = args.limit_windows if args.limit_windows > 0 else None
    allowed_record_ids = load_record_id_allowlist(args.record_id_allowlist)
    affinity_by_record = load_affinity_scores(args.affinity_scores)

    primitive_ids_by_record: dict[str, list[list[int]]] = {}
    features_by_record: dict[str, list[np.ndarray]] = {}
    primitive_features_by_record: dict[str, dict[int, np.ndarray]] = {}
    query_logits_by_record: dict[str, list[np.ndarray]] = {}
    mask_logits_by_record: dict[str, list[np.ndarray]] = {}
    quality_logits_by_record: dict[str, list[np.ndarray]] = {}
    identity_embeddings_by_record: dict[str, list[np.ndarray]] = {}
    ownership_logits_by_record: dict[str, list[np.ndarray]] = {}
    semantic_logits_by_record: dict[str, list[np.ndarray]] = {}
    route_trace_rows: list[dict[str, Any]] = []
    counters = Counter()
    cache_provenance: dict[str, Any] | None = None
    cache_input_protocol: dict[str, Any] | None = None
    for record in iter_jsonl(args.cache, limit):
        if cache_provenance is None:
            try:
                cache_provenance = validate_inference_provenance(record)
            except ValueError:
                if not args.allow_unbound_cache:
                    raise
                cache_provenance = {"status": "unbound_diagnostic_override", "gt_free": False}
        record_input_protocol = validate_cache_input_protocol(record, abi)
        if cache_input_protocol is None:
            cache_input_protocol = record_input_protocol
        elif record_input_protocol != cache_input_protocol:
            raise ValueError(
                f"inference cache mixes input protocols: first={cache_input_protocol} current={record_input_protocol}"
            )
        original_record_id = str(record.get("original_record_id") or record.get("record_id"))
        if allowed_record_ids is not None and original_record_id not in allowed_record_ids:
            counters["skipped_by_record_allowlist"] += 1
            continue
        use_ownership = bool(abi.get("ownership_enabled", False)) and args.ownership_decoder != "mask_only"
        original_record_id, primitive_ids, features, query_logits, mask_logits, quality_logits, identity_embeddings, ownership_logits, semantic_logits = apply_window(
            record, model, pack, device,
            use_quality=quality_enabled,
            use_identity=bool(abi.get("identity_head_trained", False)),
            use_ownership=use_ownership,
        )
        typed_outputs = getattr(model, "last_typed_outputs", None) or {}
        branch_diagnostics = getattr(model, "last_branch_router_diagnostics", None) or {}
        route_trace = {
            "model_ran": bool(primitive_ids),
            "record_id": original_record_id,
            "semantic_base_logits_shape": list(typed_outputs["semantic_base_logits"].shape) if typed_outputs.get("semantic_base_logits") is not None else None,
            "semantic_query_residual_shape": list(typed_outputs["semantic_query_residual"].shape) if typed_outputs.get("semantic_query_residual") is not None else None,
            "semantic_query_residual_l2": float(typed_outputs["semantic_query_residual"].detach().float().norm().item()) if typed_outputs.get("semantic_query_residual") is not None else 0.0,
            "branch_routes": {
                name: branch_route_trace_payload(value)
                for name, value in branch_diagnostics.items()
            },
        }
        if primitive_ids:
            route_trace_rows.append(route_trace)
        if not primitive_ids:
            continue
        primitive_ids_by_record.setdefault(original_record_id, []).append(primitive_ids)
        features_by_record.setdefault(original_record_id, []).append(features)
        feature_map = primitive_features_by_record.setdefault(original_record_id, {})
        for primitive_id, feature in zip(primitive_ids, features, strict=False):
            feature_map.setdefault(int(primitive_id), np.asarray(feature, dtype=np.float32))
        query_logits_by_record.setdefault(original_record_id, []).append(query_logits)
        mask_logits_by_record.setdefault(original_record_id, []).append(mask_logits)
        if quality_logits is not None:
            quality_logits_by_record.setdefault(original_record_id, []).append(quality_logits)
        if identity_embeddings is not None:
            identity_embeddings_by_record.setdefault(original_record_id, []).append(identity_embeddings)
        if ownership_logits is not None:
            ownership_logits_by_record.setdefault(original_record_id, []).append(ownership_logits)
        semantic_logits_by_record.setdefault(original_record_id, []).append(semantic_logits)
        counters["windows"] += 1
        counters["window_primitives"] += len(primitive_ids)

    def rows() -> Iterable[dict[str, Any]]:
        for record_id in sorted(primitive_ids_by_record):
            ownership_membership_threshold = ownership_membership_threshold_for_decoder(
                args.ownership_decoder,
                ownership_available=ownership_logits_by_record.get(record_id) is not None,
                legacy_ownership_state=bool(legacy_zero_initialized_keys),
                mask_threshold=float(args.mask_threshold),
            )
            instances, query_admission = instances_from_windows(
                primitive_ids_by_record[record_id],
                query_logits_by_record[record_id],
                mask_logits_by_record[record_id],
                feature_rows=features_by_record[record_id],
                quality_logits_rows=quality_logits_by_record.get(record_id),
                identity_embedding_rows=identity_embeddings_by_record.get(record_id),
                ownership_logits_rows=ownership_logits_by_record.get(record_id),
                ownership_membership_threshold=ownership_membership_threshold,
                query_admission_policy=args.query_admission_policy,
                min_query_score=args.min_query_score,
                family_min_query_score_overrides=parse_family_float_overrides(args.family_min_query_score_overrides),
                min_object_margin=args.min_object_margin,
                mask_threshold=args.mask_threshold,
                recall_expansion_policy=args.recall_expansion_policy,
                recall_expansion_threshold=args.recall_expansion_threshold,
                recall_expansion_endpoint_radius=args.recall_expansion_endpoint_radius,
                recall_expansion_max_ratio=args.recall_expansion_max_ratio,
                merge_iou_threshold=args.merge_iou_threshold,
                merge_overlap_threshold=args.merge_overlap_threshold,
                max_instances=args.max_instances_per_record,
                window_merge_policy=args.window_merge_policy,
                merge_center_distance_threshold=args.merge_center_distance_threshold,
            )
            before_reuse = primitive_reuse_diagnostics(instances)
            if ownership_logits_by_record.get(record_id) is not None:
                if before_reuse["reused_primitives"] or before_reuse["cross_label_reused_primitives"]:
                    raise AssertionError(
                        "ownership-before-mask invariant violated after window tracking; "
                        f"diagnostic={before_reuse}"
                    )
                conflict = {
                    "policy": "ownership_before_mask_assert_only",
                    "removed_primitive_assignments": 0,
                    "dropped_empty_instances": 0,
                }
            else:
                instances, conflict = enforce_primitive_conflicts(
                    instances,
                    args.primitive_conflict_policy,
                    affinity_edges=affinity_by_record.get(record_id),
                    primitive_features=primitive_features_by_record.get(record_id),
                    affinity_threshold=args.affinity_threshold,
                    affinity_weight=args.affinity_weight,
                    geometry_affinity_radius=args.geometry_affinity_radius,
                    ownership_gate_min_supported_fraction=args.ownership_gate_min_supported_fraction,
                )
            semantic_stuff = []
            if args.semantic_stuff:
                occupied = {
                    int(primitive_id)
                    for instance in instances
                    for primitive_id in instance.get("primitive_ids", [])
                }
                semantic_stuff = semantic_stuff_instances(
                    primitive_ids_by_record[record_id], semantic_logits_by_record[record_id], occupied,
                    minimum_score=float(args.semantic_stuff_min_score),
                )
                instances.extend(semantic_stuff)
            reuse = primitive_reuse_diagnostics(instances)
            counters["records"] += 1
            counters["pred_instances"] += len(instances)
            counters["queries"] += int(query_admission["queries"])
            counters["foreground_candidate_queries"] += int(query_admission.get("foreground_candidate_queries", 0))
            counters["admitted_object_queries"] += int(query_admission["admitted_object_queries"])
            counters["rejected_no_object_queries"] += int(query_admission["rejected_no_object_queries"])
            counters["rejected_low_object_margin_queries"] += int(query_admission["rejected_low_object_margin_queries"])
            counters["rejected_low_score_queries"] += int(query_admission["rejected_low_score_queries"])
            counters["rejected_empty_mask_queries"] += int(query_admission["rejected_empty_mask_queries"])
            counters["proposal_queries"] += int(query_admission["proposal_queries"])
            for prefix, summary in (query_admission.get("query_score_distributions") or {}).items():
                weight = int(summary.get("count", 0)) if isinstance(summary, dict) else 0
                if weight <= 0:
                    continue
                counters[f"{prefix}_count"] += weight
                counters[f"{prefix}_sum"] += float(summary.get("mean", 0.0)) * weight
                if summary.get("min") is not None:
                    counters[f"{prefix}_min"] = min(float(counters.get(f"{prefix}_min", summary["min"])), float(summary["min"]))
                if summary.get("max") is not None:
                    counters[f"{prefix}_max"] = max(float(counters.get(f"{prefix}_max", summary["max"])), float(summary["max"]))
                counters[f"{prefix}_positive"] += int(round(float(summary.get("positive_rate", 0.0)) * weight))
            counters["recall_expanded_queries"] += int(query_admission["recall_expanded_queries"])
            counters["recall_expansion_candidates"] += int(query_admission["recall_expansion_candidates"])
            counters["recall_expansion_added_primitives"] += int(query_admission["recall_expansion_added_primitives"])
            counters["merged_fragments"] += sum(max(int(item.get("merged_fragments", 1)) - 1, 0) for item in instances)
            counters["pre_conflict_reused_primitives"] += int(before_reuse["reused_primitives"])
            counters["pre_conflict_cross_label_reused_primitives"] += int(before_reuse["cross_label_reused_primitives"])
            counters["reused_primitives"] += int(reuse["reused_primitives"])
            counters["cross_label_reused_primitives"] += int(reuse["cross_label_reused_primitives"])
            counters["max_owners_per_primitive"] = max(counters["max_owners_per_primitive"], int(reuse["max_owners_per_primitive"]))
            counters["removed_primitive_assignments"] += int(conflict["removed_primitive_assignments"])
            counters["dropped_empty_instances"] += int(conflict["dropped_empty_instances"])
            counters["affinity_supported_assignments"] += int(conflict.get("affinity_supported_assignments", 0))
            counters["affinity_edges_available"] += int(conflict.get("affinity_edges_available", 0))
            counters["membership_gate_removed_assignments"] += int(conflict.get("membership_gate_removed_assignments", 0))
            counters["membership_gate_supported_assignments"] += int(conflict.get("membership_gate_supported_assignments", 0))
            counters["membership_gate_considered_assignments"] += int(conflict.get("membership_gate_considered_assignments", 0))
            counters["membership_gate_applied_instances"] += int(conflict.get("membership_gate_applied_instances", 0))
            counters["membership_gate_dropped_empty_instances"] += int(conflict.get("membership_gate_dropped_empty_instances", 0))
            counters["calibrated_instances"] += int(conflict.get("calibrated_instances", 0))
            counters["calibration_supported_assignments"] += int(conflict.get("calibration_supported_assignments", 0))
            counters["calibration_considered_assignments"] += int(conflict.get("calibration_considered_assignments", 0))
            counters["calibration_score_delta_sum"] += float(conflict.get("calibration_score_delta_mean", 0.0)) * int(conflict.get("calibrated_instances", 0))
            ledger = conflict.get("primitive_assignment_removal_ledger") if isinstance(conflict.get("primitive_assignment_removal_ledger"), list) else []
            counters["primitive_assignment_removal_ledger_count"] += len(ledger)
            for entry in ledger:
                if isinstance(entry, dict):
                    counters[f"primitive_assignment_removal_stage:{entry.get('stage') or 'unknown'}"] += 1
            yield {"record_id": record_id, "pred_instances": instances, "prediction_diagnostics": {"query_admission": query_admission, "semantic_stuff_instances": len(semantic_stuff), "before_conflict_resolution": before_reuse, "after_conflict_resolution": reuse, "conflict_resolution": conflict}}

    written = write_jsonl(args.output, rows())
    provenance = {
        "model_sha256": sha256_file(args.model),
        "cache_sha256": sha256_file(args.cache),
        "output_sha256": sha256_file(args.output),
        "code_sha256": sha256_file(Path(__file__)),
        "apply_config_sha256": apply_config_sha256(args),
        "record_allowlist_sha256": sha256_file(args.record_id_allowlist) if args.record_id_allowlist is not None else None,
        "affinity_scores_sha256": sha256_file(args.affinity_scores) if args.affinity_scores is not None else None,
        "cache_record": cache_provenance,
        "cache_input_protocol": cache_input_protocol,
        "diagnostic_unbound_override": bool(args.allow_unbound_cache),
        "legacy_zero_initialized_state_keys": sorted(legacy_zero_initialized_keys),
    }
    policy_manifest = decoder_policy_manifest(args, quality_enabled=quality_enabled)
    policy_bytes = json.dumps(policy_manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    write_json(args.route_trace_output, {
        "schema_version": "floorplancad_panoptic_apply_route_trace_v1",
        "checkpoint_abi": abi,
        "decoder_policy_manifest": policy_manifest,
        "decoder_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "gt_free": True,
        "provenance": provenance,
        "rows": route_trace_rows,
    })
    payload = {
        "schema_version": "floorplancad_line_token_panoptic_moe_apply_v1",
        "created_utc": utc_now(),
        "status": "exported",
        "inputs": {"model": rel(args.model), "cache": rel(args.cache), "record_id_allowlist": rel(args.record_id_allowlist), "affinity_scores": rel(args.affinity_scores)},
        "provenance": {
            **provenance,
            "route_trace_sha256": sha256_file(args.route_trace_output),
        },
        "checkpoint_abi": abi,
        "diagnostic_only": not bool(abi["production_compatible"]),
        "gt_free_apply_contract": {
            "status": "enforced",
            "forbidden_fields": ["gt_masks", "gt_labels", "page_instance_id", "page_instance_ids", "matched_target_indices", "instance_ids", "instance_id", "semantic_id", "semantic_ids", "target", "targets", "query_labels", "mask_targets"],
            "semantic_query_path": "model_forward_query_logits_includes_zero_initialized_mask_weighted_semantic_residual",
        },
        "typed_output_contract": "semantic_panoptic_typed_outputs_v1",
        "quality_policy": (
            "trained_sigmoid" if quality_enabled else "disabled_class_score_only"
        ),
        "output": rel(args.output),
        "route_trace_output": rel(args.route_trace_output),
        "records": written,
        "windows": counters["windows"],
        "skipped_by_record_allowlist": counters["skipped_by_record_allowlist"],
        "window_primitives": counters["window_primitives"],
        "pred_instances": counters["pred_instances"],
        "merged_fragments": counters["merged_fragments"],
        "query_admission_diagnostics": {
            "query_admission_policy": args.query_admission_policy,
            "ownership_decoder": args.ownership_decoder,
            "queries": counters["queries"],
            "foreground_candidate_queries": counters["foreground_candidate_queries"],
            "admitted_object_queries": counters["admitted_object_queries"],
            "rejected_no_object_queries": counters["rejected_no_object_queries"],
            "rejected_low_object_margin_queries": counters["rejected_low_object_margin_queries"],
            "rejected_low_score_queries": counters["rejected_low_score_queries"],
            "rejected_empty_mask_queries": counters["rejected_empty_mask_queries"],
            "proposal_queries": counters["proposal_queries"],
            "admitted_object_query_rate": counters["admitted_object_queries"] / max(counters["queries"], 1),
            "proposal_query_rate": counters["proposal_queries"] / max(counters["queries"], 1),
            "recall_expansion_policy": args.recall_expansion_policy,
            "recall_expanded_queries": counters["recall_expanded_queries"],
            "recall_expansion_candidates": counters["recall_expansion_candidates"],
            "recall_expansion_added_primitives": counters["recall_expansion_added_primitives"],
            "query_score_distributions": {
                "object_margin": distribution_payload(counters, "object_margin"),
                "object_score": distribution_payload(counters, "object_score"),
                "no_object_score": distribution_payload(counters, "no_object_score"),
                "quality_score": distribution_payload(counters, "quality_score"),
                "calibrated_score": distribution_payload(counters, "calibrated_score"),
                "mask_max_prob": distribution_payload(counters, "mask_max_prob"),
                "mask_selected_count": distribution_payload(counters, "mask_selected_count"),
            },
        },
        "primitive_reuse_diagnostics": {
            "pre_conflict_reused_primitives": counters["pre_conflict_reused_primitives"],
            "pre_conflict_cross_label_reused_primitives": counters["pre_conflict_cross_label_reused_primitives"],
            "reused_primitives": counters["reused_primitives"],
            "cross_label_reused_primitives": counters["cross_label_reused_primitives"],
            "max_owners_per_primitive": counters["max_owners_per_primitive"],
            "removed_primitive_assignments": counters["removed_primitive_assignments"],
            "dropped_empty_instances": counters["dropped_empty_instances"],
            "affinity_supported_assignments": counters["affinity_supported_assignments"],
            "affinity_edges_available": counters["affinity_edges_available"],
            "membership_gate_removed_assignments": counters["membership_gate_removed_assignments"],
            "membership_gate_supported_assignments": counters["membership_gate_supported_assignments"],
            "membership_gate_considered_assignments": counters["membership_gate_considered_assignments"],
            "membership_gate_applied_instances": counters["membership_gate_applied_instances"],
            "membership_gate_dropped_empty_instances": counters["membership_gate_dropped_empty_instances"],
            "membership_gate_support_rate": counters["membership_gate_supported_assignments"] / max(counters["membership_gate_considered_assignments"], 1),
            "calibrated_instances": counters["calibrated_instances"],
            "calibration_supported_assignments": counters["calibration_supported_assignments"],
            "calibration_considered_assignments": counters["calibration_considered_assignments"],
            "calibration_support_rate": counters["calibration_supported_assignments"] / max(counters["calibration_considered_assignments"], 1),
            "calibration_score_delta_mean": counters["calibration_score_delta_sum"] / max(counters["calibrated_instances"], 1),
            "primitive_assignment_removal_ledger_count": counters["primitive_assignment_removal_ledger_count"],
            "primitive_assignment_removal_stage_histogram": {
                key.split(":", 1)[1]: int(value)
                for key, value in sorted(counters.items())
                if str(key).startswith("primitive_assignment_removal_stage:")
            },
        },
        "policy": {
            "min_query_score": args.min_query_score,
            "family_min_query_score_overrides": parse_family_float_overrides(args.family_min_query_score_overrides),
            "family_min_query_score_overrides_paper_eligible": not bool(args.family_min_query_score_overrides.strip()),
            "min_object_margin": args.min_object_margin,
            "query_admission_policy": args.query_admission_policy,
            "model_mode": args.model_mode,
            "mask_threshold": args.mask_threshold,
            "recall_expansion_policy": args.recall_expansion_policy,
            "recall_expansion_threshold": args.recall_expansion_threshold,
            "recall_expansion_endpoint_radius": args.recall_expansion_endpoint_radius,
            "recall_expansion_max_ratio": args.recall_expansion_max_ratio,
            "merge_iou_threshold": args.merge_iou_threshold,
            "merge_overlap_threshold": args.merge_overlap_threshold,
            "window_merge_policy": args.window_merge_policy,
            "merge_center_distance_threshold": args.merge_center_distance_threshold,
            "max_instances_per_record": args.max_instances_per_record,
            "primitive_conflict_policy": args.primitive_conflict_policy,
            "affinity_threshold": args.affinity_threshold,
            "affinity_weight": args.affinity_weight,
            "geometry_affinity_radius": args.geometry_affinity_radius,
            "ownership_gate_min_supported_fraction": args.ownership_gate_min_supported_fraction,
            "stitching": "query-mask windows use reciprocal same-label matching with overlap and primitive-center constraints by default; legacy connected-components remains diagnostic-only",
        },
        "claim_boundary": "Panoptic component query export. PQ/RQ/SQ requires primitive-set evaluator before claim.",
        "comparable_for_matrix": False,
    }
    write_json(args.report, payload)
    print(json.dumps({"status": "exported", "report": rel(args.report), "output": rel(args.output), "records": written, "windows": counters["windows"], "pred_instances": counters["pred_instances"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

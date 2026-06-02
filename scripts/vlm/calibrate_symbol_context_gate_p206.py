#!/usr/bin/env python3
"""P206 calibrated gate for P205b high-recall candidates over P202 baseline."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

import sweep_symbol_disagreement_backfill_p165 as p165
import train_symbol_context_verifier_p202 as p202
from fuse_symbol_detector_with_p182_p186 import detector_by_row, load_jsonl, write_json, write_jsonl
from train_symbol_crop_verifier_p200 import LABELS as P200_LABELS

TARGET_LABELS = {"sink", "shower", "equipment", "stair", "appliance", "bathtub"}
WET_LABELS = {"sink", "shower", "bathtub"}
ROOT = Path(__file__).resolve().parents[2]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box)})
    return out


def crop_image(page_image: Path, bbox: list[float], image_size: Any, pad: float = 0.20, crop_size: int = 96) -> Image.Image:
    width, height = 4096, 4096
    if isinstance(image_size, list) and len(image_size) >= 2:
        width, height = int(image_size[0]), int(image_size[1])
    x1, y1, x2, y2 = bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    px = bw * pad
    py = bh * pad
    box = [max(0, int(x1 - px)), max(0, int(y1 - py)), min(width, int(x2 + px)), min(height, int(y2 + py))]
    with Image.open(page_image) as image:
        return image.convert("RGB").crop(tuple(box)).resize((crop_size, crop_size), Image.Resampling.BILINEAR)


def load_p200_model(checkpoint: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location=device)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(P200_LABELS))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def score_crops(rows: list[dict[str, Any]], pred_maps: dict[str, list[dict[str, Any]]], checkpoint: Path, device: torch.device, batch: int) -> dict[tuple[str, int], dict[str, Any]]:
    model = load_p200_model(checkpoint, device)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tasks: list[tuple[str, int, torch.Tensor]] = []
    for row in rows:
        rid = row_id(row)
        page = Path(str(row.get("image") or row.get("image_path") or ""))
        if not page.is_absolute():
            page = ROOT / page
        width, height = 4096, 4096
        image_size = row.get("image_size")
        if isinstance(image_size, list) and len(image_size) >= 2:
            width, height = int(image_size[0]), int(image_size[1])
        page_tasks = [(idx, pred) for idx, pred in enumerate(pred_maps.get(rid, [])) if pred.get("source_policy") == "p205b_detector"]
        if not page_tasks:
            continue
        with Image.open(page) as opened:
            image = opened.convert("RGB")
            for idx, pred in page_tasks:
                x1, y1, x2, y2 = pred["bbox"]
                bw = max(1.0, x2 - x1)
                bh = max(1.0, y2 - y1)
                px = bw * 0.20
                py = bh * 0.20
                box = [max(0, int(x1 - px)), max(0, int(y1 - py)), min(width, int(x2 + px)), min(height, int(y2 + py))]
                crop = image.crop(tuple(box)).resize((96, 96), Image.Resampling.BILINEAR)
                tasks.append((rid, idx, tf(crop)))
    out: dict[tuple[str, int], dict[str, Any]] = {}
    with torch.no_grad():
        for start in range(0, len(tasks), batch):
            chunk = tasks[start:start + batch]
            images = torch.stack([item[2] for item in chunk]).to(device)
            probs = torch.softmax(model(images), dim=1).cpu()
            preds = probs.argmax(dim=1)
            for (rid, idx, _tensor), prob, pred in zip(chunk, probs, preds):
                pred_label = P200_LABELS[int(pred)]
                out[(rid, idx)] = {
                    "p200_pred_label": pred_label,
                    "p200_false_positive_prob": round(float(prob[0]), 6),
                    "p200_real_prob": round(float(1.0 - prob[0]), 6),
                    "p200_pred_prob": round(float(prob[int(pred)]), 6),
                    "p200_probs": {P200_LABELS[i]: round(float(prob[i]), 6) for i in range(len(P200_LABELS))},
                }
    return out


def score_context(rows: list[dict[str, Any]], checkpoint: Path) -> dict[tuple[str, int], float]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = p202.MLP(int(ckpt["dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    examples, _dim = p202.build_examples(rows)
    scored = p202.score_examples(model, examples)
    return {(ex["row_id"], int(ex["candidate_index"])): float(ex["context_score"]) for ex in scored}


def build_candidate_rows(base_rows: list[dict[str, Any]], detector: dict[str, list[dict[str, Any]]], min_detector_score: float, detector_nms: float, labels: set[str]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    merged_rows = []
    pred_maps: dict[str, list[dict[str, Any]]] = {}
    for row in base_rows:
        rid = row_id(row)
        core = p165.normalized(row.get("symbol_candidates") or [], "p202_core")
        additions = []
        for cand in sorted(detector.get(rid, []), key=lambda item: float(item.get("score") or 0.0), reverse=True):
            if cand["label"] not in labels:
                continue
            if float(cand.get("score") or 0.0) < min_detector_score:
                continue
            if any(p165.iou(cand["bbox"], old["bbox"]) >= detector_nms for old in additions):
                continue
            item = copy.deepcopy(cand)
            item["source_policy"] = "p205b_detector"
            additions.append(item)
        preds = core + additions
        pred_maps[rid] = preds
        material = []
        for idx, pred in enumerate(preds):
            raw = copy.deepcopy(pred.get("raw") or {})
            raw["bbox"] = pred["bbox"]
            raw["symbol_type"] = pred["label"]
            raw["confidence"] = float(pred["score"])
            raw["source"] = pred.get("source_policy", "p202_core")
            raw.setdefault("metadata", {})["p206_source_policy"] = pred.get("source_policy")
            material.append(raw)
        new_row = copy.deepcopy(row)
        new_row["symbol_candidates"] = material
        merged_rows.append(new_row)
    return merged_rows, pred_maps


def accept_addition(pred: dict[str, Any], crop: dict[str, Any], context_score: float, policy: dict[str, Any]) -> bool:
    if pred["source_policy"] != "p205b_detector":
        return True
    label = pred["label"]
    detector_score = float(pred.get("score") or 0.0)
    fp_prob = float(crop.get("p200_false_positive_prob", 1.0))
    real_prob = float(crop.get("p200_real_prob", 0.0))
    crop_label = str(crop.get("p200_pred_label") or "")
    if label not in set(policy["labels"]):
        return False
    if detector_score < float(policy["detector_score"]):
        return False
    if fp_prob > float(policy["max_fp_prob"]):
        return False
    if real_prob < float(policy["min_real_prob"]):
        return False
    if context_score < float(policy["min_context_score"]):
        return False
    if policy.get("require_crop_label_match") and crop_label not in {label, "generic_symbol"}:
        return False
    return True


def apply_policy(pred_maps: dict[str, list[dict[str, Any]]], crop_scores: dict[tuple[str, int], dict[str, Any]], context_scores: dict[tuple[str, int], float], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for rid, preds in pred_maps.items():
        kept = []
        add_count = 0
        for idx, pred in enumerate(preds):
            if pred.get("source_policy") == "p205b_detector" and add_count >= int(policy["max_add_per_row"]):
                continue
            crop = crop_scores.get((rid, idx), {})
            context = context_scores.get((rid, idx), 0.0)
            if not accept_addition(pred, crop, context, policy):
                continue
            item = copy.deepcopy(pred)
            raw = copy.deepcopy(item.get("raw") or {})
            raw.setdefault("metadata", {})["p206_gate"] = {"policy": policy["name"], "context_score": round(context, 6), **crop}
            item["raw"] = raw
            kept.append(item)
            if pred.get("source_policy") == "p205b_detector":
                add_count += 1
        out[rid] = sorted(kept, key=lambda item: float(item.get("score") or 0.0), reverse=True)[: int(policy["max_total_per_row"])]
    return out


def candidate_policies(quick: bool = False) -> list[dict[str, Any]]:
    policies = []
    label_sets = [WET_LABELS, TARGET_LABELS]
    detector_scores = [0.01, 0.02] if quick else [0.005, 0.01, 0.02]
    max_fp_probs = [0.25, 0.35, 0.50] if quick else [0.15, 0.25, 0.35, 0.50]
    min_real_probs = [0.50, 0.65, 0.75] if quick else [0.50, 0.65, 0.75, 0.85]
    context_scores = [0.10, 0.20, 0.30] if quick else [0.05, 0.10, 0.15, 0.20, 0.30]
    max_adds = [0, 1, 2, 3] if quick else [0, 1, 2, 3, 5, 8]
    matches = [False, True]
    for labels in label_sets:
        for detector_score in detector_scores:
            for max_fp_prob in max_fp_probs:
                for min_real_prob in min_real_probs:
                    for min_context_score in context_scores:
                        for max_add in max_adds:
                            for match in matches:
                                policies.append({
                                    "name": f"p206_l{len(labels)}_ds{detector_score}_fp{max_fp_prob}_rp{min_real_prob}_cs{min_context_score}_a{max_add}_m{int(match)}",
                                    "labels": sorted(labels),
                                    "detector_score": detector_score,
                                    "max_fp_prob": max_fp_prob,
                                    "min_real_prob": min_real_prob,
                                    "min_context_score": min_context_score,
                                    "max_add_per_row": max_add,
                                    "max_total_per_row": 128,
                                    "require_crop_label_match": match,
                                })
    return policies


def materialize(rows: list[dict[str, Any]], pred_map: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        rid = row_id(row)
        candidates = []
        for idx, pred in enumerate(pred_map.get(rid, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{rid}_p206_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_context_gate_p206"
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = copy.deepcopy(candidates)
        row["symbol_policy_overlay"] = {"policy_id": "p206_context_gate", "policy": policy}
        out.append(row)
    return out


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P206 Recall-Preserving Context Gate",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    b = report["baseline_metrics"]
    m = report["best_metrics"]
    lines.append(f"| `P202_baseline` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines.append(f"| `P206_best` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    lines += ["", "## Best Policy", "", "```json", json.dumps(report["best_policy"], ensure_ascii=False, indent=2), "```", "", "## Added Candidate Audit", "", "```json", json.dumps(report["added_candidate_audit"], ensure_ascii=False, indent=2), "```", "", "## Top Policies", ""]
    for item in report["top_candidates"][:20]:
        x = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{x['f1']:.6f}` P `{x['precision']:.6f}` R `{x['recall']:.6f}` center `{x['center_recall']:.6f}` inflation `{x['prediction_inflation']:.6f}`")
    return "\n".join(lines) + "\n"


def added_audit(pred_map: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = Counter()
    for preds in pred_map.values():
        for pred in preds:
            if pred.get("source_policy") == "p205b_detector":
                counts["total"] += 1
                counts[f"label_{pred['label']}"] += 1
                counts[f"bucket_{pred['bucket']}"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--detector-predictions", default="reports/vlm/symbol_tiled_recall_p205b_30k_p101_predictions.jsonl")
    parser.add_argument("--p200-checkpoint", default="checkpoints/symbol_crop_verifier_p200/model.pt")
    parser.add_argument("--p202-checkpoint", default="checkpoints/symbol_context_verifier_p202/model.pt")
    parser.add_argument("--out-json", default="configs/vlm/symbol_context_gate_p206.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_context_gate_p206_eval.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_context_gate_p206_overlay.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    base_rows = load_jsonl(Path(args.base_overlay))
    detector = detector_by_row(Path(args.detector_predictions))
    merged_rows, pred_maps = build_candidate_rows(base_rows, detector, min_detector_score=(0.01 if args.quick else 0.003), detector_nms=0.55, labels=TARGET_LABELS)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    crop_scores = score_crops(base_rows, pred_maps, Path(args.p200_checkpoint), device, args.batch)
    context_scores = score_context(merged_rows, Path(args.p202_checkpoint))
    golds = {row_id(row): target_symbols(row) for row in base_rows}
    baseline = p165.evaluate(golds, {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "p202") for row in base_rows})
    scored = []
    for policy in candidate_policies(args.quick):
        pred_map = apply_policy(pred_maps, crop_scores, context_scores, policy)
        scored.append({"policy": policy, "metrics": p165.evaluate(golds, pred_map), "added": added_audit(pred_map)})
    scored.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["center_recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best = scored[0]
    best_map = apply_policy(pred_maps, crop_scores, context_scores, best["policy"])
    write_jsonl(Path(args.out_overlay), materialize(base_rows, best_map, best["policy"]))
    report = {
        "id": "P206_recall_preserving_context_gate",
        "claim_boundary": "P101-selected gate over raster-only P205b detector candidates. P200/P202 scores use trained models; gold labels are used only for offline evaluation/policy selection.",
        "inputs": {"base_overlay": args.base_overlay, "detector_predictions": args.detector_predictions, "p200_checkpoint": args.p200_checkpoint, "p202_checkpoint": args.p202_checkpoint},
        "candidate_counts": {"rows": len(base_rows), "detector_rows": len(detector), "crop_scored_additions": len(crop_scores)},
        "baseline_metrics": baseline,
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_baseline": p165.delta(best["metrics"], baseline),
        "added_candidate_audit": added_audit(best_map),
        "decision": "promote_candidate" if best["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_P202",
        "searched_policy_count": len(scored),
        "top_candidates": scored[:50],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "candidate_counts": report["candidate_counts"], "baseline": baseline, "best_metrics": best["metrics"], "delta": report["delta_vs_baseline"], "added": report["added_candidate_audit"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

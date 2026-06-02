#!/usr/bin/env python3
"""P202 lightweight context verifier over symbol candidates.

Gold is used only to supervise/evaluate candidate TP/FP labels offline. Runtime
features are candidate score/label/bucket/geometry and page-level neighbor context.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn

import sweep_symbol_disagreement_backfill_p165 as p165

LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
BUCKETS = ["tiny", "small", "medium", "large", "xlarge"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box)})
    return out


def best_iou_to_gold(pred: dict[str, Any], golds: list[dict[str, Any]]) -> float:
    return max((p165.iou(pred["bbox"], gold["bbox"]) for gold in golds), default=0.0)


def center_hit_to_gold(pred: dict[str, Any], golds: list[dict[str, Any]]) -> bool:
    return any(p165.center_covered(pred["bbox"], gold["bbox"]) for gold in golds)


def feature_vector(pred: dict[str, Any], preds: list[dict[str, Any]], image_size: Any) -> list[float]:
    box = pred["bbox"]
    width, height = 4096.0, 4096.0
    if isinstance(image_size, list) and len(image_size) >= 2:
        width, height = float(image_size[0]), float(image_size[1])
    elif isinstance(image_size, dict):
        width = float(image_size.get("width") or image_size.get("w") or width)
        height = float(image_size.get("height") or image_size.get("h") or height)
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    cx = (box[0] + box[2]) / 2.0 / max(width, 1.0)
    cy = (box[1] + box[3]) / 2.0 / max(height, 1.0)
    area = bw * bh
    same_label = 0
    near = 0
    overlaps = 0
    min_dist = 1e6
    for other in preds:
        if other is pred:
            continue
        dist = p165.center_distance(box, other["bbox"])
        min_dist = min(min_dist, dist)
        if dist <= 80:
            near += 1
        if other["label"] == pred["label"]:
            same_label += 1
        if p165.iou(box, other["bbox"]) > 0.1:
            overlaps += 1
    label_oh = [1.0 if pred["label"] == label else 0.0 for label in LABELS]
    bucket_oh = [1.0 if pred["bucket"] == bucket else 0.0 for bucket in BUCKETS]
    score = float(pred.get("score") or 0.0)
    return [
        score,
        math.log1p(score),
        cx,
        cy,
        bw / max(width, 1.0),
        bh / max(height, 1.0),
        math.log1p(area) / 12.0,
        bw / max(bh, 1.0),
        len(preds) / 100.0,
        same_label / 50.0,
        near / 50.0,
        overlaps / 20.0,
        min(min_dist, 1000.0) / 1000.0,
    ] + label_oh + bucket_oh


def build_examples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    examples = []
    dim = 0
    for row in rows:
        rid = row_id(row)
        preds = p165.normalized(row.get("symbol_candidates") or [], "p202")
        golds = target_symbols(row)
        for idx, pred in enumerate(preds):
            feats = feature_vector(pred, preds, row.get("image_size"))
            label = 1 if best_iou_to_gold(pred, golds) >= 0.30 else 0
            examples.append({"row_id": rid, "candidate_index": idx, "features": feats, "target": label})
            dim = len(feats)
    return examples, dim


class MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Dropout(0.10), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[set[str], set[str], set[str]]:
    ids = [row_id(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    return set(ids[: int(n * 0.70)]), set(ids[int(n * 0.70): int(n * 0.85)]), set(ids[int(n * 0.85):])


def tensorize(examples: list[dict[str, Any]], ids: set[str]) -> tuple[torch.Tensor, torch.Tensor]:
    subset = [ex for ex in examples if ex["row_id"] in ids]
    x = torch.tensor([ex["features"] for ex in subset], dtype=torch.float32)
    y = torch.tensor([ex["target"] for ex in subset], dtype=torch.float32)
    return x, y


def binary_metrics(scores: torch.Tensor, y: torch.Tensor, threshold: float) -> dict[str, Any]:
    pred = scores >= threshold
    true = y >= 0.5
    tp = int((pred & true).sum().item())
    fp = int((pred & ~true).sum().item())
    fn = int((~pred & true).sum().item())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}


def score_examples(model: nn.Module, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = torch.tensor([ex["features"] for ex in examples], dtype=torch.float32)
    with torch.no_grad():
        scores = torch.sigmoid(model(x)).cpu().tolist()
    out = []
    for ex, score in zip(examples, scores):
        item = dict(ex)
        item["context_score"] = round(float(score), 6)
        out.append(item)
    return out


def apply_threshold(rows: list[dict[str, Any]], scored: dict[tuple[str, int], float], threshold: float, protected: set[str]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in rows:
        rid = row_id(row)
        preds = p165.normalized(row.get("symbol_candidates") or [], "p202_apply")
        kept = []
        for idx, pred in enumerate(preds):
            score = float(scored.get((rid, idx), 0.0))
            if pred["label"] not in protected and score < threshold:
                continue
            item = copy.deepcopy(pred)
            raw = copy.deepcopy(item.get("raw") or {})
            raw.setdefault("metadata", {})["p202_context_score"] = round(score, 6)
            item["raw"] = raw
            kept.append(item)
        out[rid] = kept
    return out


def materialize(rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        rid = row_id(row)
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(rid, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{rid}_p202_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_context_verifier_p202"
            item.setdefault("metadata", {})["p202_policy"] = policy
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p202_context_verifier", "description": "P202 candidate context verifier", "policy": policy}
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_sink_shower_specialist_p198_over_p200_best_overlay.jsonl")
    parser.add_argument("--out-json", default="configs/vlm/symbol_context_verifier_p202.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_context_verifier_p202_ablation.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--predictions-jsonl", default="reports/vlm/symbol_context_verifier_p202_predictions.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_context_verifier_p202/model.pt")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=20260519)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = load_jsonl(Path(args.base_overlay))
    examples, dim = build_examples(rows)
    train_ids, val_ids, test_ids = split_rows(rows, args.seed)
    x_train, y_train = tensorize(examples, train_ids)
    x_val, y_val = tensorize(examples, val_ids)
    x_test, y_test = tensorize(examples, test_ids)
    model = MLP(dim)
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best = {"score": -1.0, "epoch": 0, "threshold": 0.5, "metrics": None}
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_scores = torch.sigmoid(model(x_val))
            candidates = []
            for threshold in [i / 100 for i in range(10, 91, 5)]:
                m = binary_metrics(val_scores, y_val, threshold)
                candidates.append((m["f1"], m["precision"], m["recall"], threshold, m))
            candidates.sort(reverse=True)
            score, _p, _r, threshold, metrics = candidates[0]
            item = {"epoch": epoch, "loss": round(float(loss.item()), 6), "val_best_threshold": threshold, "val_metrics": metrics}
            history.append(item)
            if score > best["score"]:
                best = {"score": score, "epoch": epoch, "threshold": threshold, "metrics": metrics}
                torch.save({"model_state": model.state_dict(), "dim": dim, "labels": LABELS, "buckets": BUCKETS, "best": best, "args": vars(args)}, args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        test_scores = torch.sigmoid(model(x_test))
    test_metrics = binary_metrics(test_scores, y_test, float(best["threshold"]))
    scored_examples = score_examples(model, examples)
    write_jsonl(Path(args.predictions_jsonl), scored_examples)
    score_map = {(ex["row_id"], int(ex["candidate_index"])): float(ex["context_score"]) for ex in scored_examples}
    golds = {row_id(row): target_symbols(row) for row in rows}
    baseline = p165.evaluate(golds, {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "baseline") for row in rows})
    policies = []
    for threshold in [i / 100 for i in range(10, 91, 5)]:
        for protected in [[], ["sink", "shower"], ["sink", "shower", "stair"]]:
            policies.append({"name": f"p202_t{threshold}_protect{len(protected)}", "threshold": threshold, "protected_labels": protected})
    scored_policies = []
    for policy in policies:
        pred_map = apply_threshold(rows, score_map, float(policy["threshold"]), set(policy["protected_labels"]))
        metrics = p165.evaluate(golds, pred_map)
        scored_policies.append({"policy": policy, "metrics": metrics})
    scored_policies.sort(key=lambda x: (x["metrics"]["f1"], x["metrics"]["precision"], x["metrics"]["recall"]), reverse=True)
    best_policy = scored_policies[0]
    best_pred_map = apply_threshold(rows, score_map, float(best_policy["policy"]["threshold"]), set(best_policy["policy"]["protected_labels"]))
    write_jsonl(Path(args.out_overlay), materialize(rows, best_pred_map, best_policy["policy"]))
    report = {
        "id": "P202_symbol_context_verifier",
        "claim_boundary": "Candidate context verifier trained with gold-derived TP/FP labels offline. Runtime features use candidate score/label/bucket/geometry and page-level neighbor context only.",
        "inputs": {"base_overlay": args.base_overlay},
        "split_rows": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
        "candidate_counts": {"total": len(examples), "train": int(y_train.numel()), "val": int(y_val.numel()), "test": int(y_test.numel())},
        "best_training": best,
        "test_binary_metrics": test_metrics,
        "baseline_metrics": baseline,
        "best_policy": best_policy["policy"],
        "best_metrics": best_policy["metrics"],
        "delta_vs_baseline": p165.delta(best_policy["metrics"], baseline),
        "top_policies": scored_policies[:30],
        "history_tail": history[-20:],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay, "checkpoint": args.checkpoint, "predictions": args.predictions_jsonl},
    }
    write_json(Path(args.out_json), report)
    lines = ["# P202 Symbol Context Verifier", "", "Decision: **" + ("promote_candidate" if best_policy["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_baseline") + "**", "", "| Variant | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|", f"| `baseline` | {baseline['precision']:.6f} | {baseline['recall']:.6f} | {baseline['f1']:.6f} | {baseline['center_recall']:.6f} | {baseline['prediction_inflation']:.6f} |", f"| `p202_best` | {best_policy['metrics']['precision']:.6f} | {best_policy['metrics']['recall']:.6f} | {best_policy['metrics']['f1']:.6f} | {best_policy['metrics']['center_recall']:.6f} | {best_policy['metrics']['prediction_inflation']:.6f} |", "", "## Best Policy", "", "```json", json.dumps(best_policy["policy"], ensure_ascii=False, indent=2), "```", "", "## Binary Verifier", "", f"- val best: `{best}`", f"- test metrics: `{test_metrics}`"]
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(json.dumps({"baseline": baseline, "best_policy": best_policy["policy"], "best_metrics": best_policy["metrics"], "delta": report["delta_vs_baseline"], "test_binary_metrics": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

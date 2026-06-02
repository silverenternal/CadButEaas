#!/usr/bin/env python3
"""P206d learned ranker/regressor for P205b additions over P202 baseline."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

import sweep_symbol_disagreement_backfill_p165 as p165
from calibrate_symbol_context_gate_p206 import build_candidate_rows, score_context, score_crops, target_symbols
from fuse_symbol_detector_with_p182_p186 import detector_by_row, load_jsonl, write_json, write_jsonl

LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
BUCKETS = ["tiny", "small", "medium", "large", "xlarge"]
ROOT = Path(__file__).resolve().parents[2]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def best_gold(pred: dict[str, Any], golds: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best = None
    best_iou = 0.0
    for gold in golds:
        overlap = p165.iou(pred["bbox"], gold["bbox"])
        if overlap > best_iou:
            best_iou = overlap
            best = gold
    return best, best_iou


def box_target(pred_box: list[float], gold_box: list[float]) -> list[float]:
    px = (pred_box[0] + pred_box[2]) / 2.0
    py = (pred_box[1] + pred_box[3]) / 2.0
    pw = max(1.0, pred_box[2] - pred_box[0])
    ph = max(1.0, pred_box[3] - pred_box[1])
    gx = (gold_box[0] + gold_box[2]) / 2.0
    gy = (gold_box[1] + gold_box[3]) / 2.0
    gw = max(1.0, gold_box[2] - gold_box[0])
    gh = max(1.0, gold_box[3] - gold_box[1])
    return [(gx - px) / pw, (gy - py) / ph, math.log(gw / pw), math.log(gh / ph)]


def apply_delta(box: list[float], delta: list[float], clamp: float = 1.5) -> list[float]:
    dx, dy, dw, dh = [max(-clamp, min(clamp, float(v))) for v in delta]
    px = (box[0] + box[2]) / 2.0
    py = (box[1] + box[3]) / 2.0
    pw = max(1.0, box[2] - box[0])
    ph = max(1.0, box[3] - box[1])
    cx = px + dx * pw
    cy = py + dy * ph
    w = pw * math.exp(dw)
    h = ph * math.exp(dh)
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def feature_vector(pred: dict[str, Any], crop: dict[str, Any], context_score: float, row: dict[str, Any], core: list[dict[str, Any]]) -> list[float]:
    box = pred["bbox"]
    width, height = 4096.0, 4096.0
    if isinstance(row.get("image_size"), list) and len(row["image_size"]) >= 2:
        width, height = float(row["image_size"][0]), float(row["image_size"][1])
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    best_iou_core, best_dist_core = p165.best_overlap_to_core(pred, core)
    label_oh = [1.0 if pred["label"] == label else 0.0 for label in LABELS]
    bucket_oh = [1.0 if pred["bucket"] == bucket else 0.0 for bucket in BUCKETS]
    p200_probs = crop.get("p200_probs") or {}
    return [
        float(pred.get("score") or 0.0),
        math.log1p(float(pred.get("score") or 0.0)),
        context_score,
        float(crop.get("p200_false_positive_prob", 1.0)),
        float(crop.get("p200_real_prob", 0.0)),
        float(crop.get("p200_pred_prob", 0.0)),
        (box[0] + box[2]) / 2.0 / max(width, 1.0),
        (box[1] + box[3]) / 2.0 / max(height, 1.0),
        bw / max(width, 1.0),
        bh / max(height, 1.0),
        bw / max(bh, 1.0),
        math.log1p(bw * bh) / 12.0,
        best_iou_core,
        min(best_dist_core, 1000.0) / 1000.0,
        len(core) / 100.0,
    ] + [float(p200_probs.get(label, 0.0)) for label in ["false_positive"] + LABELS] + label_oh + bucket_oh


class RankReg(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(dim, 96), nn.ReLU(), nn.Dropout(0.10), nn.Linear(96, 48), nn.ReLU())
        self.rank = nn.Linear(48, 1)
        self.reg = nn.Linear(48, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.rank(h).squeeze(-1), self.reg(h)


def split_ids(rows: list[dict[str, Any]], seed: int) -> tuple[set[str], set[str], set[str]]:
    ids = [row_id(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    return set(ids[: int(n * 0.70)]), set(ids[int(n * 0.70): int(n * 0.85)]), set(ids[int(n * 0.85):])


def build_examples(rows: list[dict[str, Any]], pred_maps: dict[str, list[dict[str, Any]]], crop_scores: dict[tuple[str, int], dict[str, Any]], context_scores: dict[tuple[str, int], float]) -> tuple[list[dict[str, Any]], int]:
    rows_by_id = {row_id(row): row for row in rows}
    examples = []
    dim = 0
    for rid, preds in pred_maps.items():
        row = rows_by_id[rid]
        golds = target_symbols(row)
        core = [pred for pred in preds if pred.get("source_policy") != "p205b_detector"]
        for idx, pred in enumerate(preds):
            if pred.get("source_policy") != "p205b_detector":
                continue
            gold, overlap = best_gold(pred, golds)
            target = 1.0 if overlap >= 0.30 else 0.0
            center = 1.0 if gold is not None and p165.center_covered(pred["bbox"], gold["bbox"]) else 0.0
            delta = box_target(pred["bbox"], gold["bbox"]) if gold is not None and center > 0 else [0.0, 0.0, 0.0, 0.0]
            feat = feature_vector(pred, crop_scores.get((rid, idx), {}), context_scores.get((rid, idx), 0.0), row, core)
            dim = len(feat)
            examples.append({"row_id": rid, "candidate_index": idx, "features": feat, "target": target, "center_target": center, "iou": overlap, "delta": delta, "label": pred["label"], "bucket": pred["bucket"]})
    return examples, dim


def tensorize(examples: list[dict[str, Any]], ids: set[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sub = [ex for ex in examples if ex["row_id"] in ids]
    return (
        torch.tensor([ex["features"] for ex in sub], dtype=torch.float32),
        torch.tensor([ex["target"] for ex in sub], dtype=torch.float32),
        torch.tensor([ex["center_target"] for ex in sub], dtype=torch.float32),
        torch.tensor([ex["delta"] for ex in sub], dtype=torch.float32),
    )


def score_examples(model: RankReg, examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = torch.tensor([ex["features"] for ex in examples], dtype=torch.float32)
    with torch.no_grad():
        logits, deltas = model(x)
        scores = torch.sigmoid(logits).cpu().tolist()
        deltas_list = deltas.cpu().tolist()
    out = []
    for ex, score, delta in zip(examples, scores, deltas_list):
        item = dict(ex)
        item["rank_score"] = float(score)
        item["pred_delta"] = [float(v) for v in delta]
        out.append(item)
    return out


def make_pred_map(base_rows: list[dict[str, Any]], pred_maps: dict[str, list[dict[str, Any]]], scored: dict[tuple[str, int], dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in base_rows:
        rid = row_id(row)
        core = [copy.deepcopy(pred) for pred in pred_maps[rid] if pred.get("source_policy") != "p205b_detector"]
        additions = []
        for idx, pred in enumerate(pred_maps[rid]):
            if pred.get("source_policy") != "p205b_detector":
                continue
            sc = scored.get((rid, idx))
            if sc is None or sc["rank_score"] < policy["rank_threshold"]:
                continue
            if pred["label"] not in set(policy["labels"]):
                continue
            item = copy.deepcopy(pred)
            if policy.get("apply_regression"):
                item["bbox"] = apply_delta(item["bbox"], sc["pred_delta"], policy.get("delta_clamp", 1.2))
                item["bucket"] = p165.bucket(item["bbox"])
                raw = copy.deepcopy(item.get("raw") or {})
                raw["bbox"] = item["bbox"]
                raw.setdefault("metadata", {})["p206d_regression"] = {"rank_score": round(sc["rank_score"], 6), "delta": [round(v, 6) for v in sc["pred_delta"]]}
                item["raw"] = raw
            best_iou, best_dist = p165.best_overlap_to_core(item, core)
            if best_iou >= policy["max_iou_to_core"] or best_dist < policy["min_dist_to_core"]:
                continue
            additions.append(item)
        additions.sort(key=lambda pred: scored[(rid, int(pred.get("source_index", 0)))]["rank_score"] if (rid, int(pred.get("source_index", 0))) in scored else float(pred.get("score") or 0.0), reverse=True)
        out[rid] = sorted(core + additions[: policy["max_add_per_row"]], key=lambda item: float(item.get("score") or 0.0), reverse=True)[:128]
    return out


def materialize(rows: list[dict[str, Any]], pred_map: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        new = copy.deepcopy(row)
        rid = row_id(row)
        candidates = []
        for idx, pred in enumerate(pred_map.get(rid, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{rid}_p206d_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_p205b_ranker_regressor_p206d"
            candidates.append(item)
        new["symbol_candidates"] = candidates
        if isinstance(new.get("expected_json"), dict):
            new["expected_json"]["symbol_candidates"] = copy.deepcopy(candidates)
        new["symbol_policy_overlay"] = {"policy_id": "p206d_ranker_regressor", "policy": policy}
        out.append(new)
    return out


def render(report: dict[str, Any]) -> str:
    b = report["baseline_metrics"]
    m = report["best_metrics"]
    lines = ["# P206d Learned Candidate Ranker/Regressor", "", f"Decision: **{report['decision']}**", "", "| Variant | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|", f"| `P202_baseline` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |", f"| `P206d_best` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |", "", "## Best Policy", "", "```json", json.dumps(report["best_policy"], ensure_ascii=False, indent=2), "```", "", "## Training", "", "```json", json.dumps(report["training"], ensure_ascii=False, indent=2), "```", "", "## Top Policies"]
    for item in report["top_candidates"][:20]:
        x = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{x['f1']:.6f}` P `{x['precision']:.6f}` R `{x['recall']:.6f}` center `{x['center_recall']:.6f}` inflation `{x['prediction_inflation']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--detector-predictions", default="reports/vlm/symbol_tiled_recall_p205b_30k_p101_predictions.jsonl")
    parser.add_argument("--p200-checkpoint", default="checkpoints/symbol_crop_verifier_p200/model.pt")
    parser.add_argument("--p202-checkpoint", default="checkpoints/symbol_context_verifier_p202/model.pt")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_p205b_ranker_regressor_p206d/model.pt")
    parser.add_argument("--out-json", default="configs/vlm/symbol_p205b_ranker_regressor_p206d.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_p205b_ranker_regressor_p206d_eval.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_p205b_ranker_regressor_p206d_overlay.jsonl")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=206)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_rows = load_jsonl(Path(args.base_overlay))
    detector = detector_by_row(Path(args.detector_predictions))
    merged_rows, pred_maps = build_candidate_rows(base_rows, detector, min_detector_score=0.01, detector_nms=0.55, labels=set(LABELS))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    crop_scores = score_crops(base_rows, pred_maps, Path(args.p200_checkpoint), device, 256)
    context_scores = score_context(merged_rows, Path(args.p202_checkpoint))
    examples, dim = build_examples(base_rows, pred_maps, crop_scores, context_scores)
    train_ids, val_ids, test_ids = split_ids(base_rows, args.seed)
    x_train, y_train, center_train, delta_train = tensorize(examples, train_ids)
    x_val, y_val, center_val, delta_val = tensorize(examples, val_ids)
    model = RankReg(dim)
    pos = float(y_train.sum().item())
    neg = float(y_train.numel() - pos)
    rank_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)]))
    reg_loss = nn.SmoothL1Loss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = {"score": -1.0, "epoch": 0, "val": None}
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits, deltas = model(x_train)
        loss_rank = rank_loss(logits, y_train)
        reg_mask = center_train[:, None]
        loss_reg = (reg_loss(deltas, delta_train) * reg_mask).sum() / max(float(reg_mask.sum().item()) * 4.0, 1.0)
        loss = loss_rank + 0.25 * loss_reg
        loss.backward()
        opt.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(x_val)
                val_scores = torch.sigmoid(val_logits)
            vals = []
            for threshold in [i / 100 for i in range(5, 96, 5)]:
                pred = val_scores >= threshold
                true = y_val >= 0.5
                tp = int((pred & true).sum().item()); fp = int((pred & ~true).sum().item()); fn = int((~pred & true).sum().item())
                prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1); f1 = 0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
                vals.append((f1, prec, rec, threshold, {"tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6)}))
            vals.sort(reverse=True)
            item = {"epoch": epoch, "loss": round(float(loss.item()), 6), "best_threshold": vals[0][3], "val_metrics": vals[0][4]}
            history.append(item)
            if vals[0][0] > best["score"]:
                best = {"score": vals[0][0], "epoch": epoch, "threshold": vals[0][3], "val": vals[0][4]}
                torch.save({"model_state": model.state_dict(), "dim": dim, "best": best, "labels": LABELS, "buckets": BUCKETS}, args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scored_examples = score_examples(model, examples)
    scored_map = {(ex["row_id"], int(ex["candidate_index"])): ex for ex in scored_examples}
    golds = {row_id(row): target_symbols(row) for row in base_rows}
    baseline = p165.evaluate(golds, {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "p202") for row in base_rows})
    policies = []
    for labels in [{"sink", "shower", "bathtub"}, {"sink", "shower", "equipment", "stair", "appliance"}]:
        for threshold in [0.15, 0.25, 0.35, 0.45, 0.55, float(best.get("threshold", 0.5))]:
            for max_add in [0, 1, 2, 3, 5, 8]:
                for apply_regression in [False, True]:
                    policies.append({"name": f"p206d_l{len(labels)}_t{threshold:.2f}_a{max_add}_r{int(apply_regression)}", "labels": sorted(labels), "rank_threshold": threshold, "max_add_per_row": max_add, "max_iou_to_core": 0.08, "min_dist_to_core": 8, "apply_regression": apply_regression, "delta_clamp": 1.2})
    scored_policies = []
    for policy in policies:
        pm = make_pred_map(base_rows, pred_maps, scored_map, policy)
        scored_policies.append({"policy": policy, "metrics": p165.evaluate(golds, pm)})
    scored_policies.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["center_recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best_policy = scored_policies[0]
    best_map = make_pred_map(base_rows, pred_maps, scored_map, best_policy["policy"])
    write_jsonl(Path(args.out_overlay), materialize(base_rows, best_map, best_policy["policy"]))
    report = {
        "id": "P206d_p205b_ranker_regressor",
        "claim_boundary": "P101-selected learned ranker/regressor; gold labels used only for offline supervision/evaluation. Requires independent validation before paper claim.",
        "candidate_counts": {"examples": len(examples), "positive_iou": int(sum(ex["target"] for ex in examples)), "center_positive": int(sum(ex["center_target"] for ex in examples))},
        "training": {"best": best, "history_tail": history[-10:]},
        "baseline_metrics": baseline,
        "best_policy": best_policy["policy"],
        "best_metrics": best_policy["metrics"],
        "delta_vs_baseline": p165.delta(best_policy["metrics"], baseline),
        "decision": "promote_candidate" if best_policy["metrics"]["f1"] > baseline["f1"] else "no_promotion_keep_P202",
        "top_candidates": scored_policies[:50],
        "outputs": {"checkpoint": args.checkpoint, "json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "candidate_counts": report["candidate_counts"], "baseline": baseline, "best_metrics": best_policy["metrics"], "delta": report["delta_vs_baseline"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

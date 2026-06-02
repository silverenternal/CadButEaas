#!/usr/bin/env python3
"""P206e crop-visual ranker/regressor for P205b additions over P206d/P202."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import sweep_symbol_disagreement_backfill_p165 as p165
from calibrate_symbol_context_gate_p206 import build_candidate_rows, score_context, score_crops, target_symbols
from fuse_symbol_detector_with_p182_p186 import detector_by_row, load_jsonl, write_json, write_jsonl
from train_symbol_p205b_ranker_regressor_p206d import LABELS, BUCKETS, apply_delta, best_gold, box_target, feature_vector, row_id

ROOT = Path(__file__).resolve().parents[2]
TARGET_LABELS = ["appliance", "equipment", "shower", "sink", "stair"]


def image_path(row: dict[str, Any]) -> Path | None:
    for key in ["image_path", "raster_path", "png_path", "image"]:
        value = row.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            return path if path.is_absolute() else ROOT / path
    for container in [row.get("source"), row.get("metadata")]:
        if isinstance(container, dict):
            for key in ["image_path", "raster_path", "png_path"]:
                value = container.get(key)
                if isinstance(value, str) and value:
                    path = Path(value)
                    return path if path.is_absolute() else ROOT / path
    return None


def padded_crop_box(box: list[float], width: int, height: int, pad_ratio: float) -> tuple[int, int, int, int]:
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    pad = max(8.0, pad_ratio * max(bw, bh))
    x1 = max(0, int(math.floor(box[0] - pad)))
    y1 = max(0, int(math.floor(box[1] - pad)))
    x2 = min(width, int(math.ceil(box[2] + pad)))
    y2 = min(height, int(math.ceil(box[3] + pad)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_tensor(row: dict[str, Any], box: list[float], crop_size: int, pad_ratio: float) -> torch.Tensor:
    path = image_path(row)
    if path is None or not path.exists():
        return torch.zeros(1, crop_size, crop_size)
    with Image.open(path) as image:
        gray = image.convert("L")
        crop = gray.crop(padded_crop_box(box, gray.width, gray.height, pad_ratio)).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    data = torch.ByteTensor(torch.ByteStorage.from_buffer(crop.tobytes())).float().view(crop_size, crop_size) / 255.0
    return data.unsqueeze(0)


def split_ids(rows: list[dict[str, Any]], seed: int) -> tuple[set[str], set[str], set[str]]:
    ids = [row_id(row) for row in rows]
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    return set(ids[: int(n * 0.70)]), set(ids[int(n * 0.70): int(n * 0.85)]), set(ids[int(n * 0.85):])


def build_examples(rows: list[dict[str, Any]], pred_maps: dict[str, list[dict[str, Any]]], crop_scores: dict[tuple[str, int], dict[str, Any]], context_scores: dict[tuple[str, int], float], crop_size: int, pad_ratio: float) -> tuple[list[dict[str, Any]], int]:
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
            center = 1.0 if gold is not None and p165.center_covered(pred["bbox"], gold["bbox"]) else 0.0
            delta = box_target(pred["bbox"], gold["bbox"]) if gold is not None and center > 0 else [0.0, 0.0, 0.0, 0.0]
            feat = feature_vector(pred, crop_scores.get((rid, idx), {}), context_scores.get((rid, idx), 0.0), row, core)
            dim = len(feat)
            examples.append({
                "row_id": rid,
                "candidate_index": idx,
                "features": feat,
                "target": 1.0 if overlap >= 0.30 else 0.0,
                "center_target": center,
                "iou": overlap,
                "delta": delta,
                "label": pred["label"],
                "bucket": pred["bucket"],
                "crop": crop_tensor(row, pred["bbox"], crop_size, pad_ratio),
            })
    return examples, dim


class CandidateDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], ids: set[str]) -> None:
        self.examples = [ex for ex in examples if ex["row_id"] in ids]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        return (
            ex["crop"],
            torch.tensor(ex["features"], dtype=torch.float32),
            torch.tensor(float(ex["target"]), dtype=torch.float32),
            torch.tensor(float(ex["center_target"]), dtype=torch.float32),
            torch.tensor(ex["delta"], dtype=torch.float32),
        )


class CropRankReg(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.feature = nn.Sequential(nn.Linear(feature_dim, 64), nn.ReLU(), nn.Dropout(0.10))
        self.body = nn.Sequential(nn.Linear(128, 96), nn.ReLU(), nn.Dropout(0.15), nn.Linear(96, 48), nn.ReLU())
        self.rank = nn.Linear(48, 1)
        self.reg = nn.Linear(48, 4)

    def forward(self, crop: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        visual = self.cnn(crop).flatten(1)
        feat = self.feature(features)
        body = self.body(torch.cat([visual, feat], dim=1))
        return self.rank(body).squeeze(-1), self.reg(body)


def val_metrics(scores: torch.Tensor, y: torch.Tensor) -> tuple[float, dict[str, Any]]:
    best = (-1.0, {})
    for threshold in [i / 100 for i in range(5, 96, 5)]:
        pred = scores >= threshold
        true = y >= 0.5
        tp = int((pred & true).sum().item())
        fp = int((pred & ~true).sum().item())
        fn = int((~pred & true).sum().item())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        if f1 > best[0]:
            best = (f1, {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)})
    return best


def train_model(model: CropRankReg, train_ds: CandidateDataset, val_ds: CandidateDataset, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    labels = torch.tensor([float(ex["target"]) for ex in train_ds.examples])
    pos = float(labels.sum().item())
    neg = float(labels.numel() - pos)
    weights = torch.where(labels > 0.5, torch.tensor(neg / max(pos, 1.0)), torch.tensor(1.0))
    sampler = WeightedRandomSampler(weights.double(), num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.to(device)
    rank_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    reg_loss = nn.SmoothL1Loss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {"score": -1.0, "epoch": 0, "threshold": 0.5, "val": None}
    history = []
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for crop, feat, y, center, delta in train_loader:
            crop, feat, y, center, delta = crop.to(device), feat.to(device), y.to(device), center.to(device), delta.to(device)
            opt.zero_grad(set_to_none=True)
            logits, pred_delta = model(crop, feat)
            loss_rank = rank_loss(logits, y)
            mask = center[:, None]
            loss_reg = (reg_loss(pred_delta, delta) * mask).sum() / max(float(mask.sum().item()) * 4.0, 1.0)
            loss = loss_rank + args.reg_weight * loss_reg
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            all_scores, all_y = [], []
            with torch.no_grad():
                for crop, feat, y, _, _ in val_loader:
                    logits, _ = model(crop.to(device), feat.to(device))
                    all_scores.append(torch.sigmoid(logits).cpu())
                    all_y.append(y)
            _, vm = val_metrics(torch.cat(all_scores), torch.cat(all_y))
            item = {"epoch": epoch, "loss": round(sum(losses) / max(len(losses), 1), 6), "val_metrics": vm}
            history.append(item)
            if vm["f1"] > best["score"]:
                best = {"score": vm["f1"], "epoch": epoch, "threshold": vm["threshold"], "val": vm}
                torch.save({"model_state": model.state_dict(), "feature_dim": model.feature[0].in_features, "best": best, "labels": LABELS, "buckets": BUCKETS, "crop_size": args.crop_size, "pad_ratio": args.pad_ratio}, args.checkpoint)
    return {"best": best, "history_tail": history[-10:]}


def score_examples(model: CropRankReg, examples: list[dict[str, Any]], device: torch.device, batch_size: int) -> list[dict[str, Any]]:
    model.eval().to(device)
    out = []
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            crops = torch.stack([ex["crop"] for ex in batch]).to(device)
            feats = torch.tensor([ex["features"] for ex in batch], dtype=torch.float32, device=device)
            logits, deltas = model(crops, feats)
            scores = torch.sigmoid(logits).cpu().tolist()
            deltas_list = deltas.cpu().tolist()
            for ex, score, delta in zip(batch, scores, deltas_list):
                item = {k: v for k, v in ex.items() if k != "crop"}
                item["rank_score"] = float(score)
                item["pred_delta"] = [float(v) for v in delta]
                out.append(item)
    return out


def make_pred_map(base_rows: list[dict[str, Any]], pred_maps: dict[str, list[dict[str, Any]]], scored: dict[tuple[str, int], dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    labels = set(policy["labels"])
    for row in base_rows:
        rid = row_id(row)
        core = [copy.deepcopy(pred) for pred in pred_maps[rid] if pred.get("source_policy") != "p205b_detector"]
        additions = []
        for idx, pred in enumerate(pred_maps[rid]):
            if pred.get("source_policy") != "p205b_detector":
                continue
            sc = scored.get((rid, idx))
            if sc is None or sc["rank_score"] < policy["rank_threshold"] or pred["label"] not in labels:
                continue
            item = copy.deepcopy(pred)
            if policy.get("apply_regression"):
                item["bbox"] = apply_delta(item["bbox"], sc["pred_delta"], policy.get("delta_clamp", 1.2))
                item["bucket"] = p165.bucket(item["bbox"])
                raw = copy.deepcopy(item.get("raw") or {})
                raw["bbox"] = item["bbox"]
                raw.setdefault("metadata", {})["p206e_regression"] = {"rank_score": round(sc["rank_score"], 6), "delta": [round(v, 6) for v in sc["pred_delta"]]}
                item["raw"] = raw
            best_iou, best_dist = p165.best_overlap_to_core(item, core)
            if best_iou >= policy["max_iou_to_core"] or best_dist < policy["min_dist_to_core"]:
                continue
            item["p206e_candidate_index"] = idx
            additions.append(item)
        additions.sort(key=lambda pred: scored[(rid, int(pred["p206e_candidate_index"]))]["rank_score"], reverse=True)
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
            item["id"] = f"{rid}_p206e_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_p205b_crop_ranker_p206e"
            candidates.append(item)
        new["symbol_candidates"] = candidates
        if isinstance(new.get("expected_json"), dict):
            new["expected_json"]["symbol_candidates"] = copy.deepcopy(candidates)
        new["symbol_policy_overlay"] = {"policy_id": "p206e_crop_ranker", "policy": policy}
        out.append(new)
    return out


def render(report: dict[str, Any]) -> str:
    b = report["baseline_metrics"]
    d = report["p206d_metrics"]
    m = report["best_metrics"]
    lines = [
        "# P206e Crop-Visual Candidate Ranker/Regressor",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `P202_baseline` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |",
        f"| `P206d_current_best` | {d['precision']:.6f} | {d['recall']:.6f} | {d['f1']:.6f} | {d['center_recall']:.6f} | {d['prediction_inflation']:.6f} |",
        f"| `P206e_best` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |",
        "",
        "## Best Policy",
        "",
        "```json",
        json.dumps(report["best_policy"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Training",
        "",
        "```json",
        json.dumps(report["training"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top Policies",
    ]
    for item in report["top_candidates"][:20]:
        x = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{x['f1']:.6f}` P `{x['precision']:.6f}` R `{x['recall']:.6f}` center `{x['center_recall']:.6f}` inflation `{x['prediction_inflation']:.6f}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--p206d-overlay", default="reports/vlm/symbol_p205b_ranker_regressor_p206d_overlay.jsonl")
    parser.add_argument("--detector-predictions", default="reports/vlm/symbol_tiled_recall_p205b_30k_p101_predictions.jsonl")
    parser.add_argument("--p200-checkpoint", default="checkpoints/symbol_crop_verifier_p200/model.pt")
    parser.add_argument("--p202-checkpoint", default="checkpoints/symbol_context_verifier_p202/model.pt")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_p205b_crop_ranker_p206e/model.pt")
    parser.add_argument("--out-json", default="configs/vlm/symbol_p205b_crop_ranker_p206e.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_p205b_crop_ranker_p206e_eval.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_p205b_crop_ranker_p206e_overlay.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--pad-ratio", type=float, default=1.5)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reg-weight", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=2065)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    base_rows = load_jsonl(Path(args.base_overlay))
    p206d_rows = load_jsonl(Path(args.p206d_overlay))
    detector = detector_by_row(Path(args.detector_predictions))
    merged_rows, pred_maps = build_candidate_rows(base_rows, detector, min_detector_score=0.01, detector_nms=0.55, labels=set(LABELS))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    crop_scores = score_crops(base_rows, pred_maps, Path(args.p200_checkpoint), device, 256)
    context_scores = score_context(merged_rows, Path(args.p202_checkpoint))
    examples, dim = build_examples(base_rows, pred_maps, crop_scores, context_scores, args.crop_size, args.pad_ratio)
    train_ids, val_ids, test_ids = split_ids(base_rows, args.seed)
    train_ds = CandidateDataset(examples, train_ids)
    val_ds = CandidateDataset(examples, val_ids)
    test_ds = CandidateDataset(examples, test_ids)
    model = CropRankReg(dim)
    training = train_model(model, train_ds, val_ds, args, device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    scored_examples = score_examples(model, examples, device, args.batch_size)
    scored_map = {(ex["row_id"], int(ex["candidate_index"])): ex for ex in scored_examples}
    golds = {row_id(row): target_symbols(row) for row in base_rows}
    baseline = p165.evaluate(golds, {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "p202") for row in base_rows})
    p206d_metrics = p165.evaluate(golds, {row_id(row): p165.normalized(row.get("symbol_candidates") or [], "p206d") for row in p206d_rows})
    thresholds = sorted(set([0.25, 0.35, 0.45, 0.55, 0.65, float(training["best"].get("threshold", 0.5))]))
    policies = []
    for labels in [{"sink", "shower", "bathtub"}, set(TARGET_LABELS)]:
        for threshold in thresholds:
            for max_add in [0, 1, 2, 3, 5, 8]:
                for apply_regression in [False, True]:
                    policies.append({"name": f"p206e_l{len(labels)}_t{threshold:.2f}_a{max_add}_r{int(apply_regression)}", "labels": sorted(labels), "rank_threshold": threshold, "max_add_per_row": max_add, "max_iou_to_core": 0.08, "min_dist_to_core": 8, "apply_regression": apply_regression, "delta_clamp": 1.2})
    scored_policies = []
    for policy in policies:
        pred_map = make_pred_map(base_rows, pred_maps, scored_map, policy)
        scored_policies.append({"policy": policy, "metrics": p165.evaluate(golds, pred_map)})
    scored_policies.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["center_recall"], -item["metrics"]["prediction_inflation"]), reverse=True)
    best_policy = scored_policies[0]
    best_map = make_pred_map(base_rows, pred_maps, scored_map, best_policy["policy"])
    write_jsonl(Path(args.out_overlay), materialize(base_rows, best_map, best_policy["policy"]))
    report = {
        "id": "P206e_p205b_crop_ranker_regressor",
        "claim_boundary": "P101-selected crop-visual ranker/regressor; gold labels used only for offline supervision/evaluation. Requires independent validation before paper claim.",
        "candidate_counts": {"examples": len(examples), "train": len(train_ds), "val": len(val_ds), "test": len(test_ds), "positive_iou": int(sum(ex["target"] for ex in examples)), "center_positive": int(sum(ex["center_target"] for ex in examples))},
        "training": training,
        "baseline_metrics": baseline,
        "p206d_metrics": p206d_metrics,
        "best_policy": best_policy["policy"],
        "best_metrics": best_policy["metrics"],
        "delta_vs_p202": p165.delta(best_policy["metrics"], baseline),
        "delta_vs_p206d": p165.delta(best_policy["metrics"], p206d_metrics),
        "decision": "promote_candidate" if best_policy["metrics"]["f1"] > p206d_metrics["f1"] else "no_promotion_keep_P206d",
        "top_candidates": scored_policies[:50],
        "outputs": {"checkpoint": args.checkpoint, "json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "candidate_counts": report["candidate_counts"], "baseline": baseline, "p206d": p206d_metrics, "best_metrics": best_policy["metrics"], "delta_vs_p206d": report["delta_vs_p206d"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

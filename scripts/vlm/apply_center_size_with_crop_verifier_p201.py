#!/usr/bin/env python3
"""P201: route high-recall center-size proposals through P200 crop verifier."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

import sweep_symbol_disagreement_backfill_p165 as p165

LABELS = ["false_positive", "appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


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


def crop_box(box: list[float], image_size: tuple[int, int], scale: float, min_size: int) -> tuple[int, int, int, int]:
    width, height = image_size
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    bw, bh = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    side = max(min_size, bw * scale, bh * scale)
    x1 = max(0, int(round(cx - side / 2)))
    y1 = max(0, int(round(cy - side / 2)))
    x2 = min(width, int(round(cx + side / 2)))
    y2 = min(height, int(round(cy + side / 2)))
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


class ProposalCropDataset(Dataset):
    def __init__(self, proposals: list[dict[str, Any]], overlay_by_row: dict[str, dict[str, Any]], crop_size: int, crop_scale: float, transform: transforms.Compose) -> None:
        self.proposals = proposals
        self.overlay_by_row = overlay_by_row
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.transform = transform
        self.image_cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.proposals)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, Any]]:
        prop = self.proposals[idx]
        overlay = self.overlay_by_row[prop["row_id"]]
        image_path = str(overlay.get("image") or overlay.get("image_path"))
        if image_path not in self.image_cache:
            self.image_cache[image_path] = Image.open(image_path).convert("RGB")
        image = self.image_cache[image_path]
        box = prop["bbox"]
        x1, y1, x2, y2 = crop_box(box, image.size, self.crop_scale, self.crop_size)
        crop = image.crop((x1, y1, x2, y2)).resize((self.crop_size, self.crop_size))
        return self.transform(crop), prop


def collate(batch):
    images, props = zip(*batch)
    return torch.stack(images), list(props)


def load_verifier(checkpoint: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def collect_proposals(prediction_rows: list[dict[str, Any]], overlay_by_row: dict[str, dict[str, Any]], topk_input: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in prediction_rows:
        rid = str(row.get("row_id") or row.get("id"))
        if rid not in overlay_by_row:
            continue
        preds = []
        for idx, raw in enumerate(row.get("predicted_symbols") or row.get("symbol_candidates") or []):
            box = p165.bbox4(raw.get("bbox"))
            if box is None:
                continue
            score = float(raw.get("score") if raw.get("score") is not None else raw.get("confidence") or 0.0)
            preds.append({"row_id": rid, "proposal_index": idx, "bbox": box, "raw_label": str(raw.get("label") or raw.get("symbol_type") or "generic_symbol"), "proposal_score": score, "tile_id": raw.get("tile_id"), "proposal_source": raw.get("proposal_source") or "center_size"})
        preds.sort(key=lambda x: x["proposal_score"], reverse=True)
        out.extend(preds[:topk_input])
    return out


def annotate_with_verifier(proposals: list[dict[str, Any]], overlay_by_row: dict[str, dict[str, Any]], checkpoint: Path, device: torch.device, batch_size: int, crop_size: int, crop_scale: float) -> list[dict[str, Any]]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ProposalCropDataset(proposals, overlay_by_row, crop_size, crop_scale, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = load_verifier(checkpoint, device)
    annotated: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, props in loader:
            probs = torch.softmax(model(images.to(device)), dim=1).cpu()
            for prop, prob in zip(props, probs):
                pred_id = int(prob.argmax().item())
                item = dict(prop)
                item["verifier_label"] = LABELS[pred_id]
                item["verifier_label_prob"] = round(float(prob[pred_id]), 6)
                item["verifier_false_positive_prob"] = round(float(prob[0]), 6)
                item["verifier_real_prob"] = round(float(1.0 - prob[0]), 6)
                item["verifier_probs"] = {LABELS[i]: round(float(prob[i]), 6) for i in range(len(LABELS))}
                annotated.append(item)
    return annotated


def predictions_for_policy(annotated: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    real_min = float(policy["real_min"])
    score_min = float(policy["score_min"])
    max_per_page = int(policy["max_per_page"])
    allowed_labels = set(policy.get("allowed_labels") or [])
    for prop in annotated:
        label = str(prop["verifier_label"])
        if label == "false_positive":
            continue
        if allowed_labels and label not in allowed_labels:
            continue
        real_prob = float(prop["verifier_real_prob"])
        if real_prob < real_min:
            continue
        combined = float(prop["proposal_score"]) * real_prob * float(prop["verifier_probs"].get(label, prop["verifier_label_prob"]))
        if combined < score_min:
            continue
        grouped.setdefault(prop["row_id"], []).append({
            "bbox": prop["bbox"],
            "label": label,
            "score": round(combined, 6),
            "raw": prop,
        })
    rows = []
    for rid, preds in grouped.items():
        preds.sort(key=lambda x: x["score"], reverse=True)
        symbols = []
        for idx, pred in enumerate(preds[:max_per_page]):
            symbols.append({
                "bbox": pred["bbox"],
                "label": pred["label"],
                "score": pred["score"],
                "proposal_source": "p201_center_size_plus_p200_crop_verifier",
                "source_index": idx,
                "metadata": {"p201": pred["raw"]},
            })
        rows.append({"row_id": rid, "predicted_symbols": symbols})
    return rows


def eval_detector_rows(det_rows: list[dict[str, Any]], overlay_rows: list[dict[str, Any]]) -> dict[str, Any]:
    det = {str(row["row_id"]): p165.normalized(row.get("predicted_symbols") or [], "p201") for row in det_rows}
    golds = {row_id(row): target_symbols(row) for row in overlay_rows}
    return p165.evaluate(golds, det)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default="reports/vlm/symbol_sink_shower_specialist_p198_over_p200_best_overlay.jsonl")
    parser.add_argument("--center-predictions", default="reports/vlm/symbol_center_size_proposal_v25_predictions.jsonl")
    parser.add_argument("--verifier-checkpoint", default="checkpoints/symbol_crop_verifier_p200/model.pt")
    parser.add_argument("--out-json", default="configs/vlm/symbol_center_size_p201_crop_verified.json")
    parser.add_argument("--out-predictions", default="reports/vlm/symbol_center_size_p201_crop_verified_predictions.jsonl")
    parser.add_argument("--out-annotated", default="reports/vlm/symbol_center_size_p201_annotated_proposals.jsonl")
    parser.add_argument("--topk-input", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--crop-size", type=int, default=160)
    parser.add_argument("--crop-scale", type=float, default=4.0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    overlay_rows = load_jsonl(Path(args.base_overlay))
    overlay_by_row = {row_id(row): row for row in overlay_rows}
    center_rows = load_jsonl(Path(args.center_predictions))
    proposals = collect_proposals(center_rows, overlay_by_row, args.topk_input)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    annotated = annotate_with_verifier(proposals, overlay_by_row, Path(args.verifier_checkpoint), device, args.batch_size, args.crop_size, args.crop_scale)
    policies = []
    label_sets = [[], ["sink", "shower"], ["sink", "shower", "appliance", "equipment", "generic_symbol"]]
    for real_min in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92]:
        for score_min in [0.005, 0.01, 0.02, 0.04, 0.08, 0.12]:
            for max_per_page in [20, 40, 80, 120, 180]:
                for labels in label_sets:
                    policies.append({"name": f"p201_r{real_min}_s{score_min}_m{max_per_page}_l{len(labels)}", "real_min": real_min, "score_min": score_min, "max_per_page": max_per_page, "allowed_labels": labels})
    scored = []
    for policy in policies:
        det_rows = predictions_for_policy(annotated, policy)
        metrics = eval_detector_rows(det_rows, overlay_rows)
        scored.append({"policy": policy, "detector_metrics": metrics, "rows": len(det_rows), "predictions": sum(len(r.get("predicted_symbols") or []) for r in det_rows)})
    scored.sort(key=lambda x: (x["detector_metrics"]["f1"], x["detector_metrics"]["recall"], x["detector_metrics"]["precision"]), reverse=True)
    best = scored[0]
    best_rows = predictions_for_policy(annotated, best["policy"])
    write_jsonl(Path(args.out_predictions), best_rows)
    write_jsonl(Path(args.out_annotated), annotated)
    report = {
        "id": "P201_center_size_proposals_with_crop_verifier",
        "claim_boundary": "P201 routes raster-only center-size proposals through the P200 crop verifier. Policy is selected offline on P101 and requires held-out validation before paper claim.",
        "inputs": {"base_overlay": args.base_overlay, "center_predictions": args.center_predictions, "verifier_checkpoint": args.verifier_checkpoint},
        "proposal_count": len(proposals),
        "annotated_count": len(annotated),
        "best_policy": best["policy"],
        "best_detector_metrics": best["detector_metrics"],
        "best_predictions": {"rows": best["rows"], "predictions": best["predictions"]},
        "top_candidates": [{k: v for k, v in item.items() if k != "policy"} | {"policy": item["policy"]} for item in scored[:30]],
        "outputs": {"json": args.out_json, "predictions": args.out_predictions, "annotated": args.out_annotated},
    }
    write_json(Path(args.out_json), report)
    print(json.dumps({"best_policy": best["policy"], "best_detector_metrics": best["detector_metrics"], "best_predictions": report["best_predictions"], "proposal_count": len(proposals)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

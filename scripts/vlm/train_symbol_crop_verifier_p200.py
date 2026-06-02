#!/usr/bin/env python3
"""Train P200 crop verifier/classifier on candidate crops."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

LABELS = ["false_positive", "appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        row = self.rows[idx]
        image = Image.open(row["image"]).convert("RGB")
        return self.transform(image), torch.tensor(int(row["label_id"]), dtype=torch.long), row


def collate(batch):
    images, labels, rows = zip(*batch)
    return torch.stack(images), torch.stack(labels), list(rows)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    preds = logits.argmax(dim=1)
    total = labels.numel()
    correct = int((preds == labels).sum().item())
    fp_id = 0
    true_real = labels != fp_id
    pred_real = preds != fp_id
    tp_real = int((true_real & pred_real).sum().item())
    pred_real_count = int(pred_real.sum().item())
    true_real_count = int(true_real.sum().item())
    precision = tp_real / max(pred_real_count, 1)
    recall = tp_real / max(true_real_count, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": round(correct / max(total, 1), 6),
        "binary_real_precision": round(precision, 6),
        "binary_real_recall": round(recall, 6),
        "binary_real_f1": round(f1, 6),
        "total": int(total),
        "correct": correct,
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    all_logits = []
    all_labels = []
    out_rows = []
    with torch.no_grad():
        for images, labels, rows in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            for row, prob, pred in zip(rows, probs, preds):
                out = dict(row)
                out["verifier_pred_label"] = LABELS[int(pred)]
                out["verifier_pred_label_id"] = int(pred)
                out["verifier_false_positive_prob"] = round(float(prob[0]), 6)
                out["verifier_real_prob"] = round(float(1.0 - prob[0]), 6)
                out["verifier_probs"] = {LABELS[i]: round(float(prob[i]), 6) for i in range(len(LABELS))}
                out_rows.append(out)
    return metrics_from_logits(torch.cat(all_logits), torch.cat(all_labels)), out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="datasets/symbol_crop_verifier_p200")
    parser.add_argument("--output-dir", default="checkpoints/symbol_crop_verifier_p200")
    parser.add_argument("--report-json", default="reports/vlm/symbol_crop_verifier_p200_train.json")
    parser.add_argument("--predictions-jsonl", default="reports/vlm/symbol_crop_verifier_p200_predictions.jsonl")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "val.jsonl")
    test_rows = read_jsonl(data_dir / "test.jsonl")
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_counts = Counter(row["label_id"] for row in train_rows)
    weights = torch.ones(len(LABELS), dtype=torch.float32)
    for idx in range(len(LABELS)):
        weights[idx] = len(train_rows) / max(train_counts.get(idx, 0), 1)
    weights = weights / weights.mean()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(CropDataset(train_rows, train_tf), batch_size=args.batch, shuffle=True, num_workers=4, collate_fn=collate)
    val_loader = DataLoader(CropDataset(val_rows, eval_tf), batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate)
    test_loader = DataLoader(CropDataset(test_rows, eval_tf), batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate)
    best = {"score": -1.0, "epoch": 0, "metrics": None}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels, _rows in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * labels.numel()
            total += labels.numel()
        val_metrics, _ = evaluate(model, val_loader, device)
        score = val_metrics["binary_real_f1"] + 0.25 * val_metrics["accuracy"]
        item = {"epoch": epoch, "train_loss": round(total_loss / max(total, 1), 6), "val_metrics": val_metrics, "selection_score": round(score, 6)}
        history.append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch, "metrics": val_metrics}
            torch.save({"model_state": model.state_dict(), "labels": LABELS, "args": vars(args), "best": best}, output_dir / "model.pt")
    ckpt = torch.load(output_dir / "model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_metrics, val_preds = evaluate(model, val_loader, device)
    test_metrics, test_preds = evaluate(model, test_loader, device)
    all_rows = train_rows + val_rows + test_rows
    all_loader = DataLoader(CropDataset(all_rows, eval_tf), batch_size=args.batch, shuffle=False, num_workers=4, collate_fn=collate)
    all_metrics, all_preds = evaluate(model, all_loader, device)
    Path(args.predictions_jsonl).parent.mkdir(parents=True, exist_ok=True)
    Path(args.predictions_jsonl).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in all_preds) + "\n")
    report = {
        "id": "P200_symbol_crop_verifier_train",
        "labels": LABELS,
        "best": best,
        "history": history,
        "metrics": {"val": val_metrics, "test": test_metrics, "all": all_metrics},
        "counts": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "outputs": {"model": str(output_dir / "model.pt"), "predictions": args.predictions_jsonl, "report": args.report_json},
        "claim_boundary": "Crop verifier labels are derived from offline gold only for supervised training/evaluation; runtime uses crop pixels and candidate metadata only.",
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"best": best, "metrics": report["metrics"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

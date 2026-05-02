#!/usr/bin/env python3
"""Train CadStruct structural node classifier.

This is the first explicit CadStruct-owned model component. It predicts a
semantic class per primitive node from deterministic geometry features, instead
of asking the VLM to autoregress dense node labels as JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from graph_node_model import (
    DEFAULT_LABELS,
    build_model,
    build_feature_spec,
    class_weight_tensor,
    evaluate_model,
    routing_balance_loss,
    routing_summary,
    save_checkpoint,
    tensorize,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_graph_nodes")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_graph_node_classifier")
    parser.add_argument("--record-key", default="nodes", choices=["nodes", "groups"])
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-type", choices=["mlp", "gated", "tr_mlp", "tr_gated"], default="mlp")
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--tr-rank", type=int, default=4)
    parser.add_argument("--routing-balance-weight", type=float, default=0.0)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--expert-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260429)
    parser.add_argument("--eval-tile-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    label_to_id = {label: index for index, label in enumerate(labels)}
    train_rows = load_nodes(Path(args.dataset_dir) / "train.jsonl", label_to_id, args.record_key)
    dev_rows = load_nodes(Path(args.dataset_dir) / "dev.jsonl", label_to_id, args.record_key)
    if not train_rows:
        raise SystemExit("no training nodes found")
    if not dev_rows:
        raise SystemExit("no dev nodes found")

    feature_spec = build_feature_spec(train_rows, labels)
    train_x, train_y = tensorize(train_rows, feature_spec, label_to_id)
    dev_x, dev_y = tensorize(dev_rows, feature_spec, label_to_id)

    device = torch.device(args.device)
    model = build_model(
        args.model_type,
        train_x.shape[1],
        args.hidden_dim,
        len(labels),
        args.dropout,
        args.experts,
        args.tr_rank,
        args.gate_temperature,
        args.top_k,
        args.expert_dropout,
    ).to(device)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    class_weights = class_weight_tensor(train_y, len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best_dev_macro_f1 = -1.0
    history = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        ce_losses = []
        balance_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            ce_loss = loss_fn(logits, batch_y)
            balance_loss = routing_balance_loss(model, batch_x)
            if balance_loss is not None and args.routing_balance_weight > 0:
                loss = ce_loss + args.routing_balance_weight * balance_loss
                balance_losses.append(float(balance_loss.detach().cpu()))
            else:
                loss = ce_loss
            loss.backward()
            optimizer.step()
            ce_losses.append(float(ce_loss.detach().cpu()))
            losses.append(float(loss.detach().cpu()))
        train_metrics = evaluate_model(model, train_x, train_y, labels, args.eval_tile_size, device)
        dev_metrics = evaluate_model(model, dev_x, dev_y, labels, args.eval_tile_size, device)
        row = {
            "epoch": epoch,
            "loss": round(sum(losses) / len(losses), 6),
            "ce_loss": round(sum(ce_losses) / len(ce_losses), 6),
            "routing_balance_loss": round(sum(balance_losses) / len(balance_losses), 6) if balance_losses else 0.0,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if dev_metrics["macro_f1"] > best_dev_macro_f1:
            best_dev_macro_f1 = dev_metrics["macro_f1"]
            save_checkpoint(output_dir / "model_best.pt", model, feature_spec, args, dev_metrics)

    final_metrics = evaluate_model(model, dev_x, dev_y, labels, args.eval_tile_size, device)
    save_checkpoint(output_dir / "model_final.pt", model, feature_spec, args, final_metrics)
    summary = {
        "ok": True,
        "dataset_dir": args.dataset_dir,
        "record_key": args.record_key,
        "output_dir": str(output_dir),
        "labels": labels,
        "train_nodes": len(train_rows),
        "dev_nodes": len(dev_rows),
        "epochs": args.epochs,
        "seed": args.seed,
        "model_type": args.model_type,
        "experts": args.experts if args.model_type in {"gated", "tr_gated"} else None,
        "tr_rank": args.tr_rank if args.model_type in {"tr_mlp", "tr_gated"} else None,
        "routing_balance_weight": args.routing_balance_weight,
        "gate_temperature": args.gate_temperature,
        "top_k": args.top_k,
        "expert_dropout": args.expert_dropout,
        "eval_tile_size": args.eval_tile_size,
        "feature_dim": int(train_x.shape[1]),
        "feature_names": feature_spec.numeric_features + [f"orientation:{item}" for item in feature_spec.orientations] + [f"primitive_type:{item}" for item in feature_spec.primitive_types],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_dev_macro_f1": round(best_dev_macro_f1, 6),
        "final_dev_metrics": final_metrics,
        "routing_summary": routing_summary(model, dev_x, args.eval_tile_size, device),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 3) if device.type == "cuda" else 0.0,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_nodes(path: Path, label_to_id: dict[str, int], record_key: str = "nodes") -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample = json.loads(line)
            for node in sample.get(record_key) or []:
                label = node.get("label")
                if label not in label_to_id:
                    continue
                rows.append({"features": node.get("features") or {}, "label": label})
    return rows


if __name__ == "__main__":
    main()

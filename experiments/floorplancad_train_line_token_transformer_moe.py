#!/usr/bin/env python3
"""Train a CadStruct-MoE line-token transformer expert.

This consumes the same FloorPlanCAD sampled line-json MoE records used by the
VecFormer comparison boundary. It trains a CadStruct-owned expert with a token
semantic head and an instance-embedding head; downstream apply/export converts
the expert output into primitive-set ``pred_instances``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = ROOT / "reports/vlm/floorplancad_line_json_moe_dataset"
DEFAULT_OUTPUT_DIR = ROOT / "reports/vlm/floorplancad_line_token_transformer_moe"
DEFAULT_MODEL = DEFAULT_OUTPUT_DIR / "line_token_transformer_moe.pt"
DEFAULT_REPORT = ROOT / "results/floorplancad_line_token_transformer_moe_train.json"
IGNORE_LABEL = 35
POSITION_ENCODING_VERSION = "continuous_fourier_logspace_v2"
POSITION_MAX_FREQUENCY_LOG2 = 7.0


BASE_FEATURES = [
    "x1_norm",
    "y1_norm",
    "x2_norm",
    "y2_norm",
    "cx_norm",
    "cy_norm",
    "length_norm",
    "log_length_norm",
    "orientation",
    "horizontal",
    "vertical",
    "stroke_width_norm",
    "stroke_width_raw",
    "color_hash_norm",
    "layer_id_norm",
    "primitive_id_norm",
]


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_jsonl(path: Path, limit: int | None = None) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield json.loads(line)


def parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_supervision(record: dict[str, Any]) -> dict[str, dict[str, int]]:
    items = ((record.get("supervision") or {}).get("line_tokens") or [])
    out: dict[str, dict[str, int]] = {}
    for item in items:
        cid = str(item.get("candidate_id"))
        out[cid] = {
            "semantic_id": parse_int(item.get("semantic_id"), IGNORE_LABEL),
            "instance_id": parse_int(item.get("instance_id"), -1),
            "primitive_id": parse_int(item.get("primitive_id"), -1),
        }
    return out


def token_feature_row(token: dict[str, Any], primitive_count: int) -> list[float]:
    features = token.get("features") if isinstance(token.get("features"), dict) else {}
    primitive_id = parse_int(token.get("primitive_id"), -1)
    row = [parse_float(features.get(name), 0.0) for name in BASE_FEATURES[:12]]
    row.extend([
        parse_float(token.get("stroke_width"), 0.0),
        parse_float(token.get("color_hash"), 0.0) / 4096.0,
        parse_float(token.get("layer_id"), -1.0) / 512.0,
        primitive_id / max(float(primitive_count), 1.0),
    ])
    return row


def contiguous_supervision_window(rows: list[tuple[list[float], int, int, int]], max_tokens: int) -> list[tuple[list[float], int, int, int]]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if len(rows) <= max_tokens:
        return rows
    labeled = [int(row[1] != IGNORE_LABEL) for row in rows]
    current = sum(labeled[:max_tokens])
    best_count = current
    best_start = 0
    for start in range(1, len(rows) - max_tokens + 1):
        current += labeled[start + max_tokens - 1] - labeled[start - 1]
        if current > best_count:
            best_count = current
            best_start = start
    return rows[best_start : best_start + max_tokens]


def record_to_arrays(record: dict[str, Any], max_tokens: int, rng: random.Random, input_level: str = "primitive") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    cached_rows = record.get("primitive_rows") if isinstance(record.get("primitive_rows"), list) else None
    if cached_rows is not None:
        rows = []
        for item in cached_rows:
            features = item.get("features") if isinstance(item.get("features"), list) else []
            if len(features) != len(BASE_FEATURES):
                continue
            rows.append((
                [parse_float(value, 0.0) for value in features],
                max(0, min(IGNORE_LABEL, parse_int(item.get("semantic_id"), IGNORE_LABEL))),
                parse_int(item.get("instance_id"), -1),
                parse_int(item.get("primitive_id"), -1),
            ))
        if not rows:
            return None
        if len(rows) > max_tokens:
            rows = contiguous_supervision_window(rows, max_tokens)
        x = np.asarray([row[0] for row in rows], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.asarray([row[1] for row in rows], dtype=np.int64)
        inst = np.asarray([row[2] for row in rows], dtype=np.int64)
        prim = np.asarray([row[3] for row in rows], dtype=np.int64)
        return x, y, inst, prim

    tokens = record.get("line_tokens") if isinstance(record.get("line_tokens"), list) else []
    supervision = load_supervision(record)
    primitive_count = max((parse_int(tok.get("primitive_id"), -1) for tok in tokens), default=0) + 1
    if input_level == "primitive":
        grouped: dict[int, dict[str, Any]] = {}
        for token in tokens:
            cid = str(token.get("id"))
            sup = supervision.get(cid)
            if sup is None:
                continue
            primitive_id = parse_int(token.get("primitive_id"), sup.get("primitive_id", -1))
            if primitive_id < 0:
                continue
            item = grouped.setdefault(
                primitive_id,
                {
                    "features": [],
                    "labels": Counter(),
                    "instances": Counter(),
                },
            )
            item["features"].append(token_feature_row(token, primitive_count))
            item["labels"][max(0, min(IGNORE_LABEL, sup["semantic_id"]))] += 1
            item["instances"][sup["instance_id"]] += 1
        rows: list[tuple[list[float], int, int, int]] = []
        for primitive_id, item in grouped.items():
            label = int(item["labels"].most_common(1)[0][0])
            instance_id = int(item["instances"].most_common(1)[0][0])
            feat = np.asarray(item["features"], dtype=np.float32)
            rows.append((feat.mean(axis=0).astype(np.float32).tolist(), label, instance_id, primitive_id))
    else:
        rows = []
        for token in tokens:
            cid = str(token.get("id"))
            sup = supervision.get(cid)
            if sup is None:
                continue
            primitive_id = parse_int(token.get("primitive_id"), sup.get("primitive_id", -1))
            label = max(0, min(IGNORE_LABEL, sup["semantic_id"]))
            rows.append((token_feature_row(token, primitive_count), label, sup["instance_id"], primitive_id))
    if not rows:
        return None
    if len(rows) > max_tokens:
        rows = contiguous_supervision_window(rows, max_tokens)
    x = np.asarray([row[0] for row in rows], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.asarray([row[1] for row in rows], dtype=np.int64)
    inst = np.asarray([row[2] for row in rows], dtype=np.int64)
    prim = np.asarray([row[3] for row in rows], dtype=np.int64)
    return x, y, inst, prim


def import_torch() -> dict[str, Any]:
    import torch
    from torch import nn

    return {"torch": torch, "nn": nn}


def sinusoidal_position(x: Any, dim: int, version: str = POSITION_ENCODING_VERSION) -> Any:
    """Encode normalized centers with versioned multiscale Fourier features."""
    torch = sys.modules["torch"]
    bands = dim // 4
    if bands <= 0:
        return x.new_zeros((*x.shape[:-1], 0))
    if version == "continuous_fourier_legacy_v1":
        freqs = torch.arange(bands, device=x.device, dtype=x.dtype).clamp_min(1)
        freqs = torch.pow(2.0, freqs / max(float(bands), 1.0))
    elif version == POSITION_ENCODING_VERSION:
        exponents = torch.linspace(0.0, POSITION_MAX_FREQUENCY_LOG2, bands, device=x.device, dtype=x.dtype)
        freqs = (2.0 * math.pi) * torch.pow(2.0, exponents)
    else:
        raise ValueError(f"unsupported position encoding version: {version}")
    enc = []
    for col in [4, 5]:
        value = x[..., col : col + 1] * freqs
        enc.extend([torch.sin(value), torch.cos(value)])
    return torch.cat(enc, dim=-1)


def make_model(nn: Any, torch: Any, feature_dim: int, hidden_dim: int, layers: int, heads: int, embed_dim: int, num_labels: int = 36) -> Any:
    class LineTokenTransformerMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.semantic_head = nn.Linear(hidden_dim, num_labels)
            self.embedding_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embed_dim))

        def forward(self, x: Any) -> tuple[Any, Any]:
            h = self.input_proj(x)
            pos = sinusoidal_position(x, h.shape[-1])
            if pos.shape[-1] >= h.shape[-1]:
                h = h + pos[..., : h.shape[-1]]
            else:
                h[..., : pos.shape[-1]] = h[..., : pos.shape[-1]] + pos
            h = self.encoder(h)
            logits = self.semantic_head(h)
            emb = self.embedding_head(h)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return logits, emb

    return LineTokenTransformerMoE()


def instance_loss(torch: Any, emb: Any, labels: Any, instances: Any, pair_samples: int, temperature: float) -> Any:
    valid = torch.nonzero((labels != IGNORE_LABEL) & (instances >= 0), as_tuple=False).flatten()
    if valid.numel() < 2:
        return emb.sum() * 0.0
    n = int(valid.numel())
    count = min(pair_samples, n * n)
    left = valid[torch.randint(0, n, (count,), device=emb.device)]
    right = valid[torch.randint(0, n, (count,), device=emb.device)]
    same_bool = (instances[left] == instances[right]) & (labels[left] == labels[right]) & (left != right)
    if bool(same_bool.all()) or not bool(same_bool.any()):
        return emb.sum() * 0.0
    same = same_bool.float()
    logits = (emb[left] * emb[right]).sum(dim=-1) / temperature
    logits = logits.clamp(min=-30.0, max=30.0)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, same)


def semantic_class_weights(torch: Any, path: Path, max_tokens: int, limit: int | None, rng: random.Random, input_level: str) -> Any:
    counts = torch.ones(36, dtype=torch.float32)
    for record in iter_jsonl(path, limit):
        arrays = record_to_arrays(record, max_tokens, rng, input_level)
        if arrays is None:
            continue
        _x, y_np, _inst, _prim = arrays
        for label in y_np.tolist():
            if 0 <= int(label) < IGNORE_LABEL:
                counts[int(label)] += 1.0
    median = counts[:IGNORE_LABEL].median().clamp_min(1.0)
    weights = (median / counts).clamp(min=0.1, max=10.0)
    weights[IGNORE_LABEL] = 0.0
    return weights


def evaluate(model: Any, pack: dict[str, Any], path: Path, device: Any, max_tokens: int, limit: int | None, rng: random.Random, input_level: str) -> dict[str, Any]:
    torch = pack["torch"]
    model.eval()
    counts = Counter()
    correct = Counter()
    loss_total = 0.0
    ce = pack["nn"].CrossEntropyLoss(ignore_index=IGNORE_LABEL)
    with torch.no_grad():
        for record in iter_jsonl(path, limit):
            arrays = record_to_arrays(record, max_tokens, rng, input_level)
            if arrays is None:
                continue
            x_np, y_np, inst_np, _prim_np = arrays
            x = torch.from_numpy(x_np).to(device).unsqueeze(0)
            y = torch.from_numpy(y_np).to(device)
            inst = torch.from_numpy(inst_np).to(device)
            logits, emb = model(x)
            logits = logits.squeeze(0)
            loss = ce(logits, y) + 0.2 * instance_loss(torch, emb.squeeze(0), y, inst, 2048, 0.2)
            pred = logits.argmax(dim=-1)
            mask = y != IGNORE_LABEL
            counts["records"] += 1
            counts["tokens"] += int(y.numel())
            counts["labeled_tokens"] += int(mask.sum().item())
            correct["semantic"] += int(((pred == y) & mask).sum().item())
            loss_total += float(loss.item())
    denom = max(counts["labeled_tokens"], 1)
    return {
        "records": counts["records"],
        "tokens": counts["tokens"],
        "labeled_tokens": counts["labeled_tokens"],
        "semantic_token_accuracy": correct["semantic"] / denom,
        "loss": loss_total / max(counts["records"], 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train", type=Path, default=None)
    parser.add_argument("--val", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--max-tokens-per-record", type=int, default=2048)
    parser.add_argument("--input-level", choices=["primitive", "token"], default="primitive")
    parser.add_argument("--pair-samples", type=int, default=4096)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--instance-loss-weight", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--val-limit-records", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260630)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_path = args.train or args.data_dir / "train_moe_records.jsonl"
    val_path = args.val or args.data_dir / "val_moe_records.jsonl"
    missing = [path for path in [train_path, val_path] if not path.exists()]
    if missing:
        payload = {"schema_version": "floorplancad_line_token_transformer_moe_train_v1", "created_utc": utc_now(), "status": "blocked", "blockers": [f"missing input: {rel(path)}" for path in missing]}
        write_json(args.report, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    pack = import_torch()
    torch = pack["torch"]
    nn = pack["nn"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = make_model(nn, torch, len(BASE_FEATURES), args.hidden_dim, args.layers, args.heads, args.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    weight_limit = min(args.limit_records, 512) if args.limit_records > 0 else 512
    class_weights = None if args.no_class_weights else semantic_class_weights(torch, train_path, args.max_tokens_per_record, weight_limit, random.Random(args.seed + 17), args.input_level).to(device)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, weight=class_weights)
    train_limit = args.limit_records if args.limit_records > 0 else None
    val_limit = args.val_limit_records if args.val_limit_records > 0 else None
    history = []
    best_val = -1.0
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        counters = Counter()
        loss_sum = 0.0
        for record in iter_jsonl(train_path, train_limit):
            arrays = record_to_arrays(record, args.max_tokens_per_record, rng, args.input_level)
            if arrays is None:
                continue
            x_np, y_np, inst_np, _prim_np = arrays
            x = torch.from_numpy(x_np).to(device).unsqueeze(0)
            y = torch.from_numpy(y_np).to(device)
            inst = torch.from_numpy(inst_np).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, emb = model(x)
            logits = logits.squeeze(0)
            emb = emb.squeeze(0)
            semantic = ce(logits, y)
            inst_loss = instance_loss(torch, emb, y, inst, args.pair_samples, 0.2)
            loss = args.semantic_loss_weight * semantic + args.instance_loss_weight * inst_loss
            if not torch.isfinite(loss):
                counters["nonfinite_loss_skipped"] += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            counters["records"] += 1
            counters["tokens"] += int(y.numel())
            counters["labeled_tokens"] += int((y != IGNORE_LABEL).sum().item())
            loss_sum += float(loss.item())
        val = evaluate(model, pack, val_path, device, args.max_tokens_per_record, val_limit, rng, args.input_level)
        row = {
            "epoch": epoch,
            "train_records": counters["records"],
            "train_tokens": counters["tokens"],
            "train_labeled_tokens": counters["labeled_tokens"],
            "train_loss": loss_sum / max(counters["records"], 1),
            "nonfinite_loss_skipped": counters["nonfinite_loss_skipped"],
            "val": val,
        }
        history.append(row)
        if val["semantic_token_accuracy"] > best_val:
            best_val = val["semantic_token_accuracy"]
            torch.save({
                "schema_version": "floorplancad_line_token_transformer_moe_checkpoint_v2",
                "state_dict": model.state_dict(),
                "feature_names": BASE_FEATURES,
                "position_encoding_version": POSITION_ENCODING_VERSION,
                "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
                "long_record_policy": "max_labeled_contiguous_window",
                "hidden_dim": args.hidden_dim,
                "layers": args.layers,
                "heads": args.heads,
                "embed_dim": args.embed_dim,
                "num_labels": 36,
                "ignore_label": IGNORE_LABEL,
                "source": "CadStruct-MoE line_token_component expert",
                "input_level": args.input_level,
                "train_path": rel(train_path),
                "val_path": rel(val_path),
            }, args.model_output)
        write_json(
            args.report,
            {
                "schema_version": "floorplancad_line_token_transformer_moe_train_v1",
                "created_utc": utc_now(),
                "status": "running",
                "model_output": rel(args.model_output),
                "inputs": {"train": rel(train_path), "val": rel(val_path)},
                "architecture": {
                    "family": "line_token_component",
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "heads": args.heads,
                    "embed_dim": args.embed_dim,
                    "feature_count": len(BASE_FEATURES),
                    "position_encoding_version": POSITION_ENCODING_VERSION,
                    "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
                    "long_record_policy": "max_labeled_contiguous_window",
                    "losses": ["semantic_ce", "instance_embedding_bce"],
                    "class_weights": "inverse_frequency_clamped" if class_weights is not None else "disabled",
                    "input_level": args.input_level,
                },
                "training_scope": {
                    "limit_records": args.limit_records,
                    "val_limit_records": args.val_limit_records,
                    "max_tokens_per_record": args.max_tokens_per_record,
                    "paper_metric": args.limit_records == 0,
                },
                "current_epoch": epoch,
                "target_epochs": args.epochs,
                "best_val_semantic_token_accuracy": best_val,
                "latest_epoch": row,
                "history": history,
                "claim_boundary": "Training artifact only; PQ/RQ/SQ requires apply/export through primitive-set evaluator.",
                "comparable_for_matrix": False,
            },
        )
        print(json.dumps(row, ensure_ascii=False))

    payload = {
        "schema_version": "floorplancad_line_token_transformer_moe_train_v1",
        "created_utc": utc_now(),
        "status": "trained",
        "model_output": rel(args.model_output),
        "inputs": {"train": rel(train_path), "val": rel(val_path)},
        "architecture": {
            "family": "line_token_component",
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "embed_dim": args.embed_dim,
            "feature_count": len(BASE_FEATURES),
            "position_encoding_version": POSITION_ENCODING_VERSION,
            "position_max_frequency_log2": POSITION_MAX_FREQUENCY_LOG2,
            "long_record_policy": "max_labeled_contiguous_window",
            "losses": ["semantic_ce", "instance_embedding_bce"],
            "class_weights": "inverse_frequency_clamped" if class_weights is not None else "disabled",
            "input_level": args.input_level,
        },
        "training_scope": {
            "limit_records": args.limit_records,
            "val_limit_records": args.val_limit_records,
            "max_tokens_per_record": args.max_tokens_per_record,
            "paper_metric": args.limit_records == 0,
        },
        "best_val_semantic_token_accuracy": best_val,
        "history": history,
        "claim_boundary": "Training artifact only; PQ/RQ/SQ requires apply/export through primitive-set evaluator.",
        "comparable_for_matrix": False,
    }
    write_json(args.report, payload)
    print(json.dumps({"status": "trained", "report": rel(args.report), "model": rel(args.model_output), "best_val_semantic_token_accuracy": best_val}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

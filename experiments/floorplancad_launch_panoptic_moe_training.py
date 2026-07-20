#!/usr/bin/env python3
"""Launch or stage the bottleneck-aware panoptic MoE long training run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TRAIN = ROOT / "reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v2/train_windowed_primitive_cache.jsonl"
DEFAULT_VAL = ROOT / "reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v2/val_windowed_primitive_cache.jsonl"
DEFAULT_MODEL = ROOT / "reports/vlm/floorplancad_line_token_panoptic_moe_bottleneck_scale_aligned/panoptic_component_moe.pt"
DEFAULT_REPORT = ROOT / "results/floorplancad_line_token_panoptic_moe_bottleneck_scale_aligned_train.json"
DEFAULT_LOG = ROOT / "logs/floorplancad_line_token_panoptic_moe_bottleneck_gpu1.log"
DEFAULT_PLAN = ROOT / "results/floorplancad_panoptic_moe_training_launch_plan.json"
DEFAULT_LEDGER = ROOT / "results/floorplancad_true_moe_pq95_runtime_locked_full433_adapter_bottleneck_ledger.json"


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "invalid_json"}


def gpu_processes() -> list[dict[str, str]]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id,used_memory", "--format=csv,noheader"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"pid": parts[0], "gpu_bus_id": parts[1], "used_memory": parts[2]})
    return rows


def gpu_index_uuid_map() -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return {}
    mapping = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            mapping[parts[0]] = parts[1]
    return mapping


def count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return sum(1 for _line in handle)
    except OSError:
        return None


def panoptic_training_processes() -> list[dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["pgrep", "-af", "floorplancad_train_line_token_panoptic_moe.py"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append({"pid": pid, "cmd": parts[1] if len(parts) > 1 else ""})
    return rows


def existing_training_state(report: Path, model_output: Path) -> dict[str, Any]:
    report_data = load_json(report)
    status = str(report_data.get("status") or "")
    processes = panoptic_training_processes()
    if status in {"trained", "completed", "evaluated"} and model_output.exists():
        state = "already_trained"
    elif processes:
        state = "already_running"
    elif status == "running":
        state = "stale_running_report"
    else:
        state = "not_running"
    return {
        "state": state,
        "report": rel(report),
        "report_exists": report.exists(),
        "report_status": status or None,
        "model": rel(model_output),
        "model_exists": model_output.exists(),
        "processes": processes,
    }


def estimate_model_params(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from experiments.floorplancad_train_line_token_panoptic_moe import make_panoptic_model
        from experiments.floorplancad_train_line_token_transformer_moe import BASE_FEATURES, import_torch

        pack = import_torch()
        torch = pack["torch"]
        model = make_panoptic_model(
            pack["nn"],
            torch,
            len(BASE_FEATURES),
            int(args.hidden_dim),
            int(args.layers),
            int(args.heads),
            int(args.num_queries),
            query_decoder_layers=int(args.query_decoder_layers),
            dropout=float(getattr(args, "dropout", 0.1)),
        )
        params = sum(int(param.numel()) for param in model.parameters())
        return {
            "status": "estimated_from_model",
            "params": params,
            "trainable_params": params,
            "feature_dim": len(BASE_FEATURES),
        }
    except Exception as exc:
        return {
            "status": "estimate_failed",
            "error": str(exc),
        }


def command(args: argparse.Namespace) -> list[str]:
    return [
        ".venv-vlm/bin/python",
        "-u",
        "experiments/floorplancad_train_line_token_panoptic_moe.py",
        "--device",
        "cuda:0",
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--num-queries",
        str(args.num_queries),
        "--query-decoder-layers",
        str(args.query_decoder_layers),
        "--dropout",
        str(args.dropout),
        "--max-tokens-per-record",
        str(args.max_tokens_per_record),
        "--train",
        str(args.train),
        "--val",
        str(args.val),
        "--model-output",
        str(args.model_output),
        "--report",
        str(args.report),
        "--bottleneck-ledger",
        str(args.bottleneck_ledger),
        "--checkpoint-metric",
        args.checkpoint_metric,
        "--lr",
        str(args.lr),
        "--val-limit-records",
        str(args.val_limit_records),
        "--mask-positive-weight",
        str(args.mask_positive_weight),
        "--mask-focal-gamma",
        str(args.mask_focal_gamma),
        "--query-objectness-loss-weight",
        str(args.query_objectness_loss_weight),
        "--query-objectness-positive-weight",
        str(args.query_objectness_positive_weight),
        "--query-objectness-negative-weight",
        str(args.query_objectness_negative_weight),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-index", default="1")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--plan-output", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--bottleneck-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--query-decoder-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-tokens-per-record", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-limit-records", type=int, default=0)
    parser.add_argument("--checkpoint-metric", choices=["neg_loss", "semantic_token_accuracy", "component_proxy_score"], default="component_proxy_score")
    parser.add_argument("--mask-positive-weight", type=float, default=8.0)
    parser.add_argument("--mask-focal-gamma", type=float, default=1.5)
    parser.add_argument("--query-objectness-loss-weight", type=float, default=0.25)
    parser.add_argument("--query-objectness-positive-weight", type=float, default=2.0)
    parser.add_argument("--query-objectness-negative-weight", type=float, default=1.0)
    parser.add_argument("--start-if-free", action="store_true")
    parser.add_argument("--allow-busy-launch", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [path for path in [args.train, args.val, args.bottleneck_ledger] if not path.exists()]
    cmd = command(args)
    train_windows = count_jsonl(args.train)
    val_windows = count_jsonl(args.val)
    model_params = estimate_model_params(args)
    index_to_bus = gpu_index_uuid_map()
    target_bus = index_to_bus.get(str(args.gpu_index))
    processes = gpu_processes()
    existing = existing_training_state(args.report, args.model_output)
    busy = bool(target_bus and any(item.get("gpu_bus_id") == target_bus for item in processes))
    status = "staged_busy" if busy else "staged_ready"
    pid = None
    if missing:
        status = "blocked_missing_inputs"
    elif existing["state"] in {"already_running", "already_trained"}:
        status = existing["state"]
        existing_pids = [int(item["pid"]) for item in existing.get("processes", []) if str(item.get("pid", "")).isdigit()]
        pid = existing_pids[0] if existing_pids else None
    elif args.start_if_free and (not busy or args.allow_busy_launch):
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        log_handle = args.log.open("a", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        pid = proc.pid
        status = "launched"
    payload = {
        "schema_version": "floorplancad_panoptic_moe_training_launch_plan_v1",
        "created_utc": utc_now(),
        "status": status,
        "pid": pid,
        "gpu_index": args.gpu_index,
        "target_gpu_bus_id": target_bus,
        "gpu_processes": processes,
        "busy": busy,
        "existing_training": existing,
        "missing_inputs": [rel(path) for path in missing],
        "command": cmd,
        "shell_command": "CUDA_VISIBLE_DEVICES={} {} >> {} 2>&1 &".format(args.gpu_index, " ".join(cmd), rel(args.log)),
        "inputs": {
            "train": rel(args.train),
            "val": rel(args.val),
            "bottleneck_ledger": rel(args.bottleneck_ledger),
        },
        "outputs": {
            "model": rel(args.model_output),
            "report": rel(args.report),
            "log": rel(args.log),
            "plan": rel(args.plan_output),
        },
        "training_intent": {
            "model_side_goal": "Raise RQ/PQ with a true query-mask component expert inside CadStruct-MoE.",
            "not_a_threshold_search": True,
            "uses_bottleneck_ledger": True,
            "checkpoint_metric": args.checkpoint_metric,
            "mask_positive_weight": args.mask_positive_weight,
            "mask_focal_gamma": args.mask_focal_gamma,
            "query_objectness_loss_weight": args.query_objectness_loss_weight,
            "query_objectness_positive_weight": args.query_objectness_positive_weight,
            "query_objectness_negative_weight": args.query_objectness_negative_weight,
            "budget_alignment": "Default launch is configured as a long 500-epoch CadStruct-MoE panoptic expert run so it is not under-budget relative to the 500-epoch external reference.",
        },
        "dataset_budget": {
            "train_windows": train_windows,
            "val_windows": val_windows,
            "max_tokens_per_record": args.max_tokens_per_record,
            "val_limit_records": args.val_limit_records,
            "full_validation": args.val_limit_records == 0,
        },
        "architecture_budget": {
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "heads": args.heads,
            "num_queries": args.num_queries,
            "query_decoder_layers": args.query_decoder_layers,
            "epochs": args.epochs,
            "model_params": model_params,
        },
    }
    write_json(args.plan_output, payload)
    print(json.dumps({"status": status, "pid": pid, "busy": busy, "plan": rel(args.plan_output), "shell_command": payload["shell_command"]}, ensure_ascii=False, indent=2))
    return 0 if status != "blocked_missing_inputs" else 1


if __name__ == "__main__":
    raise SystemExit(main())

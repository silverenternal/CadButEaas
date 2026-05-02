#!/usr/bin/env python3
"""Run zero-shot VLM benchmarks with isolated server lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_CONFIGS = [
    "configs/vlm/qwen3_vl_8b_smoke.json",
    "configs/vlm/qwen3_vl_32b_paper.json",
    "configs/vlm/internvl3_5_14b_eval.json",
    "configs/vlm/glm4_6v_baseline.json",
    "configs/vlm/kimi_vl_efficiency_baseline.json",
    "configs/vlm/internvl3_5_baseline.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    parser.add_argument("--dataset", default="datasets/cadstruct/smoke.jsonl")
    parser.add_argument("--output-dir", default="reports/vlm/zero_shot_runs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8765)
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-remote-models", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = [Path(item.strip()) for item in args.configs.split(",") if item.strip()]
    summary = {
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "allow_remote_models": args.allow_remote_models,
        "runs": [],
    }
    for index, config_path in enumerate(configs):
        run = run_one(config_path, index, args)
        summary["runs"].append(run)
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(run, ensure_ascii=False), flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def run_one(config_path: Path, index: int, args: argparse.Namespace) -> dict[str, Any]:
    if not config_path.exists():
        return {"config": str(config_path), "status": "missing_config"}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_adapter_path"):
        return {"config": str(config_path), "status": "skipped_adapter_config", "reason": "zero-shot requires no adapter"}
    model_path = str(config.get("model_path") or "")
    local_model = model_is_local(model_path)
    if not local_model and not args.allow_remote_models:
        return {
            "config": str(config_path),
            "model_name": config.get("model_name"),
            "model_path": model_path,
            "status": "skipped_remote_model",
            "reason": "use --allow-remote-models to download/load remote model ids",
        }

    slug = slugify(str(config.get("model_name") or config_path.stem))
    report_path = Path(args.output_dir) / f"{slug}_{Path(args.dataset).stem}.json"
    if args.limit is not None:
        report_path = Path(args.output_dir) / f"{slug}_{Path(args.dataset).stem}_limit{args.limit}.json"
    if args.skip_existing and report_path.exists():
        return {"config": str(config_path), "model_name": config.get("model_name"), "status": "skipped_existing", "report": str(report_path)}

    port = args.base_port + index
    url = f"http://{args.host}:{port}/analyze_raster"
    materialized = materialize_config(config, args.host, port, args.gpu, Path(args.output_dir) / "configs" / config_path.name)
    command = [
        sys.executable,
        "scripts/vlm/server.py",
        "--config",
        str(materialized),
    ]
    eval_command = [
        sys.executable,
        "scripts/vlm/evaluate_backend.py",
        "--dataset",
        args.dataset,
        "--url",
        url,
        "--timeout",
        str(config.get("request_timeout_seconds", 240)),
        "--output",
        str(report_path),
    ]
    if args.limit is not None:
        eval_command.extend(["--limit", str(args.limit)])
    if args.dry_run:
        return {
            "config": str(config_path),
            "model_name": config.get("model_name"),
            "model_path": model_path,
            "status": "dry_run",
            "server_command": command,
            "eval_command": eval_command,
            "report": str(report_path),
        }

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(config.get("cuda_visible_devices") or args.gpu)
    started = time.perf_counter()
    process = subprocess.Popen(command, cwd=Path.cwd(), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait_for_health(args.host, port, args.startup_timeout)
        eval_result = subprocess.run(eval_command, cwd=Path.cwd(), env=env, text=True, capture_output=True, check=False)
        status = "ok" if eval_result.returncode == 0 else "eval_failed"
        return {
            "config": str(config_path),
            "model_name": config.get("model_name"),
            "status": status,
            "report": str(report_path),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "eval_returncode": eval_result.returncode,
            "eval_stdout_tail": tail(eval_result.stdout),
            "eval_stderr_tail": tail(eval_result.stderr),
        }
    except Exception as exc:
        return {
            "config": str(config_path),
            "model_name": config.get("model_name"),
            "status": "failed",
            "error": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "server_output_tail": collect_process_tail(process),
        }
    finally:
        terminate_process(process)


def materialize_config(config: dict[str, Any], host: str, port: int, gpu: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    copy = dict(config)
    copy["host"] = host
    copy["port"] = port
    copy.setdefault("cuda_visible_devices", gpu)
    copy["peft_adapter_path"] = None
    path.write_text(json.dumps(copy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def wait_for_health(host: str, port: int, timeout: float) -> None:
    deadline = time.perf_counter() + timeout
    url = f"http://{host}:{port}/health"
    last_error = None
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("ok"):
                    return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"server did not become healthy on {url}: {last_error}")


def model_is_local(model_path: str) -> bool:
    if not model_path:
        return True
    path = Path(model_path)
    return path.exists() or model_path.startswith("/") or model_path.startswith(".")


def slugify(value: str) -> str:
    keep = []
    for char in value.lower():
        keep.append(char if char.isalnum() else "_")
    return "_".join("".join(keep).split("_")).strip("_")


def terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def collect_process_tail(process: subprocess.Popen[str]) -> str:
    if process.stdout is None:
        return ""
    try:
        return tail(process.stdout.read())
    except Exception:
        return ""


def tail(text: str | None, max_chars: int = 4000) -> str:
    if not text:
        return ""
    return text[-max_chars:]


if __name__ == "__main__":
    main()

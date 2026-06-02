#!/usr/bin/env python3
"""Diagnose the P115 inference environment before retrying symbol eval.

This script intentionally uses only the Python standard library. Heavy imports
such as torch and ultralytics are executed in child processes with timeouts so
the diagnostic can finish even when the inference stack hangs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = ROOT / "configs/vlm/inference_environment_diagnosis_p115.json"
DEFAULT_REPORT = ROOT / "reports/vlm/inference_environment_diagnosis_p115.md"

P113_INPUTS = {
    "data": "datasets/symbol_tile_detector_tiny_sahi_v21",
    "yolo_dir": "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22_locked_full",
    "weights": "runs/segment/runs/segment/runs/vlm/symbol_yolov8s_seg_rect_v28/weights/best.pt",
    "eval_script": "scripts/vlm/eval_symbol_yolo_tile_detector_v22.py",
    "runner": "scripts/vlm/run_full_symbol_eval_p113.sh",
    "status_checker": "scripts/vlm/check_full_symbol_eval_p113.py",
}

PACKAGE_IMPORTS = {
    "torch": "torch",
    "ultralytics": "ultralytics",
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def run_cmd(command: list[str], timeout: float, cwd: Path = Path("/tmp")) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "timeout_seconds": timeout,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "returncode": result.returncode,
            "timed_out": False,
            "stdout_tail": result.stdout.splitlines()[-40:],
            "stderr_tail": result.stderr.splitlines()[-40:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "timeout_seconds": timeout,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "returncode": 124,
            "timed_out": True,
            "stdout_tail": (exc.stdout or "").splitlines()[-40:] if isinstance(exc.stdout, str) else [],
            "stderr_tail": (exc.stderr or "").splitlines()[-40:] if isinstance(exc.stderr, str) else [],
        }


def package_probe(import_name: str, distribution_name: str, python: str, timeout: float) -> dict[str, Any]:
    probe: dict[str, Any] = {
        "import_name": import_name,
        "distribution_name": distribution_name,
    }
    metadata_code = (
        "import json\n"
        "from importlib import metadata\n"
        "from importlib.util import find_spec\n"
        f"import_name = {import_name!r}\n"
        f"distribution_name = {distribution_name!r}\n"
        "spec = find_spec(import_name)\n"
        "try:\n"
        "    version = metadata.version(distribution_name)\n"
        "except metadata.PackageNotFoundError:\n"
        "    version = None\n"
        "print(json.dumps({"
        "'find_spec_found': spec is not None, "
        "'origin': getattr(spec, 'origin', None) if spec is not None else None, "
        "'version_without_import': version"
        "}, ensure_ascii=False))\n"
    )
    metadata_check = run_cmd([python, "-c", metadata_code], min(timeout, 10.0))
    probe["metadata_check"] = metadata_check
    if metadata_check["stdout_tail"]:
        try:
            probe.update(json.loads(metadata_check["stdout_tail"][-1]))
        except json.JSONDecodeError:
            probe["metadata_parse_error"] = True
    else:
        probe["find_spec_found"] = None
        probe["origin"] = None
        probe["version_without_import"] = None

    code = (
        "import importlib, json, time\n"
        f"name = {import_name!r}\n"
        "started = time.perf_counter()\n"
        "mod = importlib.import_module(name)\n"
        "elapsed = time.perf_counter() - started\n"
        "print(json.dumps({"
        "'import_name': name, "
        "'elapsed_seconds': round(elapsed, 3), "
        "'version': getattr(mod, '__version__', None), "
        "'file': getattr(mod, '__file__', None)"
        "}, ensure_ascii=False))\n"
    )
    result = run_cmd([python, "-c", code], timeout)
    probe["import_check"] = result
    parsed = None
    if result["stdout_tail"]:
        try:
            parsed = json.loads(result["stdout_tail"][-1])
        except json.JSONDecodeError:
            parsed = None
    probe["import_result"] = parsed
    if result["timed_out"]:
        probe["status"] = "timeout"
    elif result["returncode"] != 0:
        probe["status"] = "failed"
    else:
        probe["status"] = "passed"
    return probe


def path_artifact(path_value: str) -> dict[str, Any]:
    path = ROOT / path_value
    quoted = shlex.quote(str(path))
    check = run_cmd(
        [
            "bash",
            "-lc",
            (
                f"if test -e {quoted}; then "
                f"kind=$(test -f {quoted} && echo file || (test -d {quoted} && echo dir || echo other)); "
                f"size=$(test -f {quoted} && wc -c < {quoted} || echo 0); "
                "printf 'exists\\n%s\\n%s\\n' \"$kind\" \"$size\"; "
                "else printf 'missing\\nmissing\\n0\\n'; fi"
            ),
        ],
        5.0,
    )
    lines = check.get("stdout_tail") or []
    exists = bool(lines and lines[0] == "exists")
    kind = lines[1] if len(lines) > 1 else "unknown"
    item: dict[str, Any] = {
        "path": path_value,
        "exists": exists,
        "is_file": kind == "file",
        "is_dir": kind == "dir",
        "check": check,
    }
    if exists and kind == "file":
        try:
            item["size_bytes"] = int(lines[2]) if len(lines) > 2 else None
        except ValueError:
            item["size_bytes"] = None
    return item


def latest_p113_status() -> dict[str, Any]:
    log_dir = ROOT / "reports/vlm/full_symbol_eval_p113_logs"
    quoted = shlex.quote(str(log_dir))
    found = run_cmd(["bash", "-lc", f"ls -1t {quoted}/*.status.json 2>/dev/null | head -n 1"], 5.0)
    if found["returncode"] != 0 or not found["stdout_tail"]:
        return {"exists": False, "check": found}
    path = Path(found["stdout_tail"][-1])
    cat = run_cmd(["bash", "-lc", "cat " + shlex.quote(str(path))], 5.0)
    if cat["returncode"] != 0 or not cat["stdout_tail"]:
        return {"exists": True, "path": rel(path), "load_error": "could not read status", "check": cat}
    try:
        data = json.loads("\n".join(cat["stdout_tail"]))
    except Exception as exc:
        return {"exists": True, "path": rel(path), "load_error": str(exc)}
    data["path"] = rel(path)
    return data


def classify_gate(probes: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    blockers: list[str] = []
    recommendations: list[str] = []

    torch_probe = probes["packages"].get("torch") or {}
    ultra_probe = probes["packages"].get("ultralytics") or {}
    if torch_probe.get("status") != "passed":
        blockers.append(f"torch import status is {torch_probe.get('status')}")
        recommendations.append("Do not retry P113 until torch imports within the timeout.")
    if ultra_probe.get("status") != "passed":
        blockers.append(f"ultralytics import status is {ultra_probe.get('status')}")
        recommendations.append("Do not retry P113 until ultralytics imports within the timeout.")

    if not probes["commands"]["nvidia_smi"].get("found"):
        recommendations.append("nvidia-smi is not available in this session; use CPU/device=cpu or run on a GPU-visible remote shell.")
    elif probes["commands"]["nvidia_smi"]["check"].get("returncode") != 0:
        recommendations.append("nvidia-smi exists but failed; inspect driver/container GPU visibility.")

    missing_inputs = [item["path"] for item in probes["p113_inputs"].values() if not item["exists"]]
    if missing_inputs:
        blockers.append("missing P113 inputs: " + ", ".join(missing_inputs))

    gate = "pass_retry_sanity_allowed" if not blockers else "blocked_do_not_retry_p113"
    if not recommendations:
        recommendations.append("Environment probes passed; retry P113 sanity before any subset/full run.")
    return gate, blockers, recommendations


def write_report(report_path: Path, data: dict[str, Any]) -> None:
    packages = data["packages"]
    lines = [
        "# P2-115 Inference Environment Diagnosis",
        "",
        "## Decision",
        "",
        f"- Gate: `{data['gate_decision']}`",
    ]
    for blocker in data["blockers"]:
        lines.append(f"- Blocker: {blocker}")
    if not data["blockers"]:
        lines.append("- Blocker: none")
    lines.extend([
        "",
        "## Python",
        "",
        f"- Executable used for probes: `{data['python']['probe_executable']}`",
        f"- Current diagnostic executable: `{data['python']['diagnostic_executable']}`",
        f"- Platform: `{data['system']['platform']}`",
        f"- CWD: `{data['cwd']}`",
        "",
        "## Import Timing",
        "",
        "| Package | Spec | Version | Status | Import seconds |",
        "|---|---:|---|---|---:|",
    ])
    for name in PACKAGE_IMPORTS:
        probe = packages.get(name, {})
        import_result = probe.get("import_result") or {}
        elapsed = import_result.get("elapsed_seconds")
        elapsed_text = f"{elapsed:.3f}" if isinstance(elapsed, (int, float)) else ""
        lines.append(
            f"| `{name}` | {str(bool(probe.get('find_spec_found'))).lower()} | "
            f"`{probe.get('version_without_import') or import_result.get('version') or ''}` | "
            f"`{probe.get('status')}` | {elapsed_text} |"
        )

    nvidia = data["commands"]["nvidia_smi"]
    lines.extend([
        "",
        "## GPU Command",
        "",
        f"- `nvidia-smi` found: `{nvidia.get('found')}`",
    ])
    if nvidia.get("found"):
        lines.append(f"- `nvidia-smi` return code: `{nvidia['check'].get('returncode')}`")

    lines.extend([
        "",
        "## P113 Inputs",
        "",
        "| Role | Path | Exists | Kind |",
        "|---|---|---:|---|",
    ])
    for role, item in data["p113_inputs"].items():
        kind = "dir" if item["is_dir"] else "file" if item["is_file"] else "missing"
        lines.append(f"| `{role}` | `{item['path']}` | {str(item['exists']).lower()} | {kind} |")

    lines.extend([
        "",
        "## Recommendations",
        "",
    ])
    for recommendation in data["recommendations"]:
        lines.append(f"- {recommendation}")
    lines.extend([
        "",
        "## Claim Boundary",
        "",
        "- This is an environment diagnostic only.",
        "- P114 produced no metrics and must remain non-claim evidence.",
        "- P113 subset/full should not run unless the gate allows a sanity retry first.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--import-timeout", type=float, default=20.0)
    parser.add_argument("--command-timeout", type=float, default=10.0)
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--run-remote-doctor", action="store_true")
    parser.add_argument("--remote-timeout", type=float, default=25.0)
    args = parser.parse_args()

    python = str(Path(args.python))
    probes: dict[str, Any] = {
        "id": "SCI-P2-115-inference-environment-diagnosis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT),
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "env_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "path": os.environ.get("PATH"),
        },
        "python": {
            "diagnostic_executable": sys.executable,
            "diagnostic_version": sys.version,
            "probe_executable": python,
            "probe_executable_exists": None,
        },
        "packages": {},
        "commands": {},
        "p113_inputs": {role: path_artifact(path) for role, path in P113_INPUTS.items()},
        "latest_p113_status": latest_p113_status(),
    }

    probes["python"]["probe_version_check"] = run_cmd([python, "--version"], args.command_timeout)
    probes["python"]["probe_executable_exists"] = probes["python"]["probe_version_check"]["returncode"] == 0

    for import_name, distribution_name in PACKAGE_IMPORTS.items():
        probes["packages"][import_name] = package_probe(import_name, distribution_name, python, args.import_timeout)

    nvidia_path = shutil.which("nvidia-smi")
    probes["commands"]["nvidia_smi"] = {
        "found": nvidia_path is not None,
        "path": nvidia_path,
        "check": run_cmd([nvidia_path or "nvidia-smi"], args.command_timeout) if nvidia_path else None,
    }

    remote_ctl = ROOT / "scripts/remote_ctl.sh"
    probes["commands"]["remote_ctl"] = {
        "exists": remote_ctl.exists(),
        "doctor_ran": bool(args.run_remote_doctor),
        "doctor": run_cmd([str(remote_ctl), "doctor"], args.remote_timeout) if args.run_remote_doctor and remote_ctl.exists() else None,
    }

    gate, blockers, recommendations = classify_gate(probes)
    probes["gate_decision"] = gate
    probes["blockers"] = blockers
    probes["recommendations"] = recommendations

    output = Path(args.output)
    report = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(probes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(report, probes)


if __name__ == "__main__":
    main()

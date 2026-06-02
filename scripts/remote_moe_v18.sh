#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/logs"
REPORT="${ROOT}/reports/vlm/v18_reproducibility_check.json"
LOG="${LOG_DIR}/remote_moe_run.log"
GROUP="preflight"
SMOKE=0

usage() {
  cat <<'EOF'
Usage:
  scripts/remote_moe_v18.sh --smoke
  scripts/remote_moe_v18.sh --group topology-smoke
  scripts/remote_moe_v18.sh --group topology-locked
  scripts/remote_moe_v18.sh --group refiner-locked
  scripts/remote_moe_v18.sh --group visual-smoke
  scripts/remote_moe_v18.sh --group visual-locked
  scripts/remote_moe_v18.sh --group quality-smoke
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      GROUP="quality-smoke"
      shift
      ;;
    --group)
      GROUP="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${LOG_DIR}" "$(dirname "${REPORT}")"

run_cmd() {
  echo "+ $*" | tee -a "${LOG}"
  "$@" 2>&1 | tee -a "${LOG}"
}

write_report() {
  local status="$1"
  local exit_status="$2"
  "${PY}" - "$REPORT" "$GROUP" "$status" "$exit_status" "$LOG" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

path, group, status, exit_status, log_path = sys.argv[1:6]
def capture(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

report = {
    "task": "IMG-MOE-V18-P2-012",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "cwd": os.getcwd(),
    "group": group,
    "status": status,
    "exit_status": int(exit_status),
    "log_path": log_path,
    "python": sys.executable,
    "python_version": capture([sys.executable, "--version"]),
    "git_status_short": capture(["git", "status", "--short"]),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

preflight() {
  run_cmd test -x "${PY}"
  run_cmd "${PY}" -m json.tool "${ROOT}/todo.json" >/dev/null
  run_cmd test -f "${ROOT}/reports/vlm/detector_adapter_v18_routed_candidates.jsonl"
  run_cmd test -f "${ROOT}/reports/vlm/topology_relations_v18_eval.json"
  run_cmd test -f "${ROOT}/reports/vlm/scene_graph_refiner_v18_eval.json"
  run_cmd "${PY}" -m json.tool "${ROOT}/reports/vlm/topology_relations_v18_eval.json" >/dev/null
  run_cmd "${PY}" -m json.tool "${ROOT}/reports/vlm/scene_graph_refiner_v18_eval.json" >/dev/null
}

trap 'code=$?; write_report failed "$code"; exit "$code"' ERR

{
  echo "==== remote_moe_v18 start $(date -Iseconds) ===="
  echo "cwd=${ROOT}"
  echo "group=${GROUP}"
  echo "python=${PY}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
} >> "${LOG}"

cd "${ROOT}"
preflight

case "${GROUP}" in
  preflight)
    ;;
  topology-smoke)
    run_cmd "${PY}" scripts/vlm/build_topology_relations_v18.py --mode oracle-eval --smoke --output /tmp/topology_relations_v18_smoke.jsonl --features-output /tmp/topology_relations_v18_smoke_features.jsonl --eval-output /tmp/topology_relations_v18_smoke_eval.json --warning-audit /tmp/topology_relations_v18_smoke_audit.json --threshold-sweep /tmp/topology_relations_v18_smoke_sweep.json
    ;;
  topology-locked)
    run_cmd "${PY}" scripts/vlm/build_topology_relations_v18.py --mode predicted --locked --cap-sweep
    ;;
  refiner-locked)
    run_cmd "${PY}" scripts/vlm/build_scene_graph_refiner_dataset_v18.py
    run_cmd "${PY}" scripts/vlm/train_scene_graph_refiner_v18.py
    ;;
  visual-smoke)
    run_cmd "${PY}" scripts/vlm/build_visual_hard_case_pack_v18.py --smoke --source adapter --include-topology --include-refiner
    ;;
  visual-locked)
    run_cmd "${PY}" scripts/vlm/build_visual_hard_case_pack_v18.py --locked --max-pages 40 --include-topology --include-refiner
    ;;
  quality-smoke)
    run_cmd "${PY}" scripts/vlm/build_topology_relations_v18.py --mode oracle-eval --smoke --output /tmp/topology_relations_v18_smoke.jsonl --features-output /tmp/topology_relations_v18_smoke_features.jsonl --eval-output /tmp/topology_relations_v18_smoke_eval.json --warning-audit /tmp/topology_relations_v18_smoke_audit.json --threshold-sweep /tmp/topology_relations_v18_smoke_sweep.json
    run_cmd "${PY}" scripts/vlm/build_visual_hard_case_pack_v18.py --smoke --source adapter --include-topology --include-refiner
    ;;
  *)
    echo "Unknown group: ${GROUP}" >&2
    usage >&2
    exit 2
    ;;
esac

write_report passed 0
echo "==== remote_moe_v18 end $(date -Iseconds) ====" >> "${LOG}"

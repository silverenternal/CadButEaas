#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-47.110.35.232}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/hugo/codes/CadButEaas}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$REMOTE_PROJECT/.venv/bin/python}"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_PROJECT/logs/remote_moe_run.log}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/remote_moe.sh [moe-command-and-args...]
  scripts/remote_moe.sh moe [moe-command-and-args...]
  scripts/remote_moe.sh -- <remote shell command>

Defaults:
  host:    47.110.35.232
  project: /home/hugo/codes/CadButEaas
  python:  /home/hugo/codes/CadButEaas/.venv/bin/python
  log:     /home/hugo/codes/CadButEaas/logs/remote_moe_run.log

Examples:
  scripts/remote_moe.sh
  scripts/remote_moe.sh evaluate --limit 128
  scripts/remote_moe.sh moe run-all --limit 768 --max-samples 36
  CUDA_VISIBLE_DEVICES=1 scripts/remote_moe.sh run-all --limit 256
  scripts/remote_moe.sh -- nvidia-smi

Overrides:
  REMOTE_HOST, REMOTE_PROJECT, REMOTE_PYTHON, REMOTE_LOG, CUDA_VISIBLE_DEVICES
USAGE
}

shell_join() {
  local out="" arg
  for arg in "$@"; do
    printf -v arg "%q" "$arg"
    out+="${out:+ }$arg"
  done
  printf '%s' "$out"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
  if [[ "$#" -eq 0 ]]; then
    echo "remote_moe.sh: missing remote shell command after --" >&2
    exit 2
  fi
  REMOTE_RUN_CMD="$(shell_join "$@")"
else
  if [[ "${1:-}" == "moe" ]]; then
    shift
  fi
  if [[ "$#" -eq 0 ]]; then
    set -- run-all
  fi
  REMOTE_RUN_CMD="$(shell_join "$REMOTE_PYTHON" "scripts/vlm/image_only_moe_v17_pipeline.py" "$@")"
fi

REMOTE_SCRIPT=$(cat <<REMOTE
set -euo pipefail

PROJECT_DIR=$(printf '%q' "$REMOTE_PROJECT")
PYTHON_BIN=$(printf '%q' "$REMOTE_PYTHON")
LOG_FILE=$(printf '%q' "$REMOTE_LOG")
RUN_CMD=$(printf '%q' "$REMOTE_RUN_CMD")
CUDA_DEVICES=$(printf '%q' "$CUDA_VISIBLE_DEVICES")

mkdir -p "\$(dirname "\$LOG_FILE")"
: > "\$LOG_FILE"
cd "\$PROJECT_DIR"

{
  echo "==== remote moe run ===="
  echo "started_at=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host=\$(uname -n 2>/dev/null || true)"
  echo "project=\$PWD"
  echo "python=\$PYTHON_BIN"
  "\$PYTHON_BIN" --version 2>&1 || true
  echo "cuda_visible_devices=\${CUDA_DEVICES:-<all>}"
  echo "command=\$RUN_CMD"
  echo
} | tee -a "\$LOG_FILE"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="\${HF_HOME:-\$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="\${TORCH_HOME:-\$PROJECT_DIR/.cache/torch}"
if [[ -n "\$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="\$CUDA_DEVICES"
fi

set +e
bash -lc "\$RUN_CMD" 2>&1 | tee -a "\$LOG_FILE"
status=\${PIPESTATUS[0]}
set -e

{
  echo
  echo "finished_at=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_status=\$status"
  echo "log_file=\$LOG_FILE"
} | tee -a "\$LOG_FILE"

exit "\$status"
REMOTE
)

echo "Running on $REMOTE_HOST; refreshed log: $REMOTE_LOG"
ssh "$REMOTE_HOST" "bash -s" <<<"$REMOTE_SCRIPT"

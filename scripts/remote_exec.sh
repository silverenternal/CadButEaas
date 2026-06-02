#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-47.110.35.232}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/hugo/codes/CadButEaas}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$REMOTE_PROJECT/.venv/bin/python}"
REMOTE_LOG="${REMOTE_LOG:-$REMOTE_PROJECT/logs/remote_exec.log}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/remote_exec.sh -- <remote shell command>
  scripts/remote_exec.sh py <script.py> [args...]
  scripts/remote_exec.sh moe [image_only_moe_v17_pipeline.py args...]
  scripts/remote_exec.sh --dry-run -- <remote shell command>

Examples:
  scripts/remote_exec.sh -- nvidia-smi
  scripts/remote_exec.sh -- git status --short
  scripts/remote_exec.sh py scripts/vlm/run_cadstruct_moe_smoke_v18.py
  scripts/remote_exec.sh moe run-all --limit 256
  CUDA_VISIBLE_DEVICES=0 scripts/remote_exec.sh -- python -m py_compile scripts/vlm/foo.py

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "remote_exec.sh: missing remote shell command after --" >&2
        exit 2
      fi
      REMOTE_RUN_CMD="$(shell_join "$@")"
      break
      ;;
    py)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "remote_exec.sh: missing Python script after py" >&2
        exit 2
      fi
      REMOTE_RUN_CMD="$(shell_join "$REMOTE_PYTHON" "$@")"
      break
      ;;
    moe)
      shift
      if [[ "$#" -eq 0 ]]; then
        set -- run-all
      fi
      REMOTE_RUN_CMD="$(shell_join "$REMOTE_PYTHON" "scripts/vlm/image_only_moe_v17_pipeline.py" "$@")"
      break
      ;;
    *)
      echo "remote_exec.sh: first argument must be --, py, moe, --dry-run, or --help" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${REMOTE_RUN_CMD:-}" ]]; then
  usage >&2
  exit 2
fi

REMOTE_SCRIPT=$(cat <<REMOTE
set -euo pipefail

PROJECT_DIR=$(printf '%q' "$REMOTE_PROJECT")
PYTHON_BIN=$(printf '%q' "$REMOTE_PYTHON")
LOG_FILE=$(printf '%q' "$REMOTE_LOG")
RUN_CMD=$(printf '%q' "$REMOTE_RUN_CMD")
CUDA_DEVICES=$(printf '%q' "$CUDA_VISIBLE_DEVICES")

mkdir -p "\$(dirname "\$LOG_FILE")"
cd "\$PROJECT_DIR"

{
  echo "==== remote exec run ===="
  echo "started_at=\$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host=\$(uname -n 2>/dev/null || true)"
  echo "project=\$PWD"
  echo "python=\$PYTHON_BIN"
  "\$PYTHON_BIN" --version 2>&1 || true
  echo "cuda_visible_devices=\${CUDA_DEVICES:-<all>}"
  echo "command=\$RUN_CMD"
  echo
} | tee "\$LOG_FILE"

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

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "host: $REMOTE_HOST"
  echo "project: $REMOTE_PROJECT"
  echo "log: $REMOTE_LOG"
  echo "command: $REMOTE_RUN_CMD"
  exit 0
fi

echo "Running on $REMOTE_HOST; log: $REMOTE_LOG"
ssh "$REMOTE_HOST" "bash -s" <<<"$REMOTE_SCRIPT"

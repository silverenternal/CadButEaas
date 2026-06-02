#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="${REMOTE_USER:-hugo}"
REMOTE_HOST="${REMOTE_HOST:-47.110.35.232}"
REMOTE_PORT="${REMOTE_PORT:-33022}"
REMOTE_PROJECT="${REMOTE_PROJECT:-/home/hugo/codes/CadButEaas}"
REMOTE_PYTHON="${REMOTE_PYTHON:-$REMOTE_PROJECT/.venv/bin/python}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-$REMOTE_PROJECT/logs/remote-runs}"
REMOTE_ENV_FILE="${REMOTE_ENV_FILE:-$REMOTE_PROJECT/.remote_env}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-./remote-outputs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
REMOTE_RETRIES="${REMOTE_RETRIES:-3}"
REMOTE_RETRY_DELAY="${REMOTE_RETRY_DELAY:-2}"
REMOTE_CONNECT_TIMEOUT="${REMOTE_CONNECT_TIMEOUT:-10}"
REMOTE_ALIVE_INTERVAL="${REMOTE_ALIVE_INTERVAL:-15}"
REMOTE_ALIVE_COUNT="${REMOTE_ALIVE_COUNT:-2}"
REMOTE_CONTROL_DIR="${REMOTE_CONTROL_DIR:-${XDG_RUNTIME_DIR:-/tmp}/cadbut-remote-ctl}"
REMOTE_ENABLE_MULTIPLEX="${REMOTE_ENABLE_MULTIPLEX:-1}"

SSH_OPTS=(
  -p "$REMOTE_PORT"
  -o BatchMode=yes
  -o ServerAliveInterval="$REMOTE_ALIVE_INTERVAL"
  -o ServerAliveCountMax="$REMOTE_ALIVE_COUNT"
  -o TCPKeepAlive=yes
  -o ConnectTimeout="$REMOTE_CONNECT_TIMEOUT"
  -o ConnectionAttempts=1
)

if [[ "$REMOTE_ENABLE_MULTIPLEX" == "1" ]]; then
  mkdir -p "$REMOTE_CONTROL_DIR"
  chmod 700 "$REMOTE_CONTROL_DIR" 2>/dev/null || true
  SSH_OPTS+=(
    -o ControlMaster=auto
    -o ControlPersist=10m
    -o ControlPath="$REMOTE_CONTROL_DIR/%r@%h:%p"
  )
fi

usage() {
  cat <<'USAGE'
Usage:
  scripts/remote_ctl.sh ping
  scripts/remote_ctl.sh run -- <remote shell command>
  scripts/remote_ctl.sh py <script.py> [args...]
  scripts/remote_ctl.sh cargo [cargo args...]
  scripts/remote_ctl.sh start <name> -- <remote shell command>
  scripts/remote_ctl.sh start-py <name> <script.py> [args...]
  scripts/remote_ctl.sh tail <name|latest> [-f]
  scripts/remote_ctl.sh status [name|latest]
  scripts/remote_ctl.sh stop <name|latest>
  scripts/remote_ctl.sh list
  scripts/remote_ctl.sh doctor
  scripts/remote_ctl.sh fetch <remote-path> [local-dir]

Examples:
  scripts/remote_ctl.sh ping
  scripts/remote_ctl.sh run -- nvidia-smi
  scripts/remote_ctl.sh py scripts/vlm/run_cadstruct_moe_smoke_v18.py
  scripts/remote_ctl.sh cargo test -p common-types
  scripts/remote_ctl.sh start moe-smoke -- python scripts/vlm/image_only_moe_v17_pipeline.py run-all --limit 64
  scripts/remote_ctl.sh start-py smoke scripts/vlm/run_cadstruct_moe_smoke_v18.py
  scripts/remote_ctl.sh tail latest -f
  scripts/remote_ctl.sh status latest
  scripts/remote_ctl.sh fetch reports/vlm/some_report.json

Environment overrides:
  REMOTE_USER=hugo
  REMOTE_HOST=47.110.35.232
  REMOTE_PORT=33022
  REMOTE_PROJECT=/home/hugo/codes/CadButEaas
  REMOTE_PYTHON=/home/hugo/codes/CadButEaas/.venv/bin/python
  REMOTE_RUN_DIR=/home/hugo/codes/CadButEaas/logs/remote-runs
  REMOTE_ENV_FILE=/home/hugo/codes/CadButEaas/.remote_env
  LOCAL_OUTPUT_DIR=./remote-outputs
  CUDA_VISIBLE_DEVICES=0
  REMOTE_RETRIES=3
  REMOTE_RETRY_DELAY=2
  REMOTE_ENABLE_MULTIPLEX=1
USAGE
}

err() { echo "remote_ctl.sh: $*" >&2; }
remote_addr() { printf '%s@%s' "$REMOTE_USER" "$REMOTE_HOST"; }

shell_join() {
  local out="" arg
  for arg in "$@"; do
    printf -v arg "%q" "$arg"
    out+="${out:+ }$arg"
  done
  printf '%s' "$out"
}

command_from_args() {
  if [[ "$#" -eq 1 ]]; then
    printf '%s' "$1"
  else
    shell_join "$@"
  fi
}

retry() {
  local attempt=1 status=0
  while true; do
    "$@" && return 0
    status=$?
    if [[ "$attempt" -ge "$REMOTE_RETRIES" ]]; then
      return "$status"
    fi
    echo "remote_ctl.sh: command failed with status $status; retry $attempt/$REMOTE_RETRIES in ${REMOTE_RETRY_DELAY}s" >&2
    sleep "$REMOTE_RETRY_DELAY"
    attempt=$((attempt + 1))
  done
}

ssh_remote() {
  local remote_cmd
  remote_cmd="$(shell_join "$@")"
  retry ssh "${SSH_OPTS[@]}" "$(remote_addr)" "$remote_cmd"
}

scp_from_remote() {
  retry scp -P "$REMOTE_PORT" \
    -o BatchMode=yes \
    -o ServerAliveInterval="$REMOTE_ALIVE_INTERVAL" \
    -o ServerAliveCountMax="$REMOTE_ALIVE_COUNT" \
    -o TCPKeepAlive=yes \
    -o ConnectTimeout="$REMOTE_CONNECT_TIMEOUT" \
    "$@"
}

sanitize_name() { printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '-'; }

run_remote_foreground() {
  local run_cmd="$1"
  echo "Running on $(remote_addr):$REMOTE_PORT; project: $REMOTE_PROJECT"
  ssh_remote bash -s -- "$REMOTE_PROJECT" "$REMOTE_PYTHON" "$REMOTE_RUN_DIR" "$REMOTE_ENV_FILE" "$CUDA_VISIBLE_DEVICES" "$run_cmd" <<'REMOTE'
set -euo pipefail
PROJECT_DIR="$1"
PYTHON_BIN="$2"
RUN_DIR="$3"
ENV_FILE="$4"
CUDA_DEVICES="$5"
RUN_CMD="$6"
mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.runner.sh' -mtime +7 -delete 2>/dev/null || true
find "$RUN_DIR" -maxdepth 1 -type f -name '*.log' -mtime +30 -delete 2>/dev/null || true
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
if [[ -n "$CUDA_DEVICES" ]]; then export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"; fi
LOG_FILE="$RUN_DIR/manual-$(date '+%Y%m%d-%H%M%S').log"
{
  echo "==== CadButEaas remote run ===="
  echo "name=manual"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host=$(uname -n 2>/dev/null || true)"
  echo "project=$PWD"
  echo "python=$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1 || true
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<all>}"
  echo "command=$RUN_CMD"
  echo
} | tee "$LOG_FILE"
set +e
bash -lc "$RUN_CMD" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
{
  echo
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_status=$status"
  echo "log_file=$LOG_FILE"
} | tee -a "$LOG_FILE"
exit "$status"
REMOTE
}

start_remote_background() {
  local name="$1" run_cmd="$2" safe_name
  safe_name="$(sanitize_name "$name")"
  ssh_remote bash -s -- "$REMOTE_PROJECT" "$REMOTE_PYTHON" "$REMOTE_RUN_DIR" "$REMOTE_ENV_FILE" "$CUDA_VISIBLE_DEVICES" "$safe_name" "$run_cmd" <<'REMOTE'
set -euo pipefail
PROJECT_DIR="$1"
PYTHON_BIN="$2"
RUN_DIR="$3"
ENV_FILE="$4"
CUDA_DEVICES="$5"
RUN_NAME="$6"
RUN_CMD="$7"
mkdir -p "$RUN_DIR"
cd "$PROJECT_DIR"
find "$RUN_DIR" -maxdepth 1 -type f -name '*.runner.sh' -mtime +7 -delete 2>/dev/null || true
find "$RUN_DIR" -maxdepth 1 -type f -name '*.log' -mtime +30 -delete 2>/dev/null || true
STAMP="$(date '+%Y%m%d-%H%M%S')"
BASE="$RUN_DIR/${STAMP}-${RUN_NAME}"
LOG_FILE="$BASE.log"
PID_FILE="$BASE.pid"
META_FILE="$BASE.meta"
RUNNER_FILE="$BASE.runner.sh"
LATEST_FILE="$RUN_DIR/latest"
cat > "$RUNNER_FILE" <<'RUNNER'
#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$1"
PYTHON_BIN="$2"
ENV_FILE="$3"
CUDA_DEVICES="$4"
RUN_NAME="$5"
RUN_CMD="$6"
LOG_FILE="$7"
cd "$PROJECT_DIR"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
fi
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/.cache/torch}"
if [[ -n "$CUDA_DEVICES" ]]; then export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"; fi
{
  echo "==== CadButEaas remote run ===="
  echo "name=$RUN_NAME"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "host=$(uname -n 2>/dev/null || true)"
  echo "project=$PWD"
  echo "python=$PYTHON_BIN"
  "$PYTHON_BIN" --version 2>&1 || true
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<all>}"
  echo "pid=$$"
  echo "command=$RUN_CMD"
  echo
} | tee -a "$LOG_FILE"
set +e
bash -lc "$RUN_CMD" 2>&1 | tee -a "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
{
  echo
  echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "exit_status=$status"
  echo "log_file=$LOG_FILE"
} | tee -a "$LOG_FILE"
exit "$status"
RUNNER
chmod +x "$RUNNER_FILE"
{
  echo "name=$RUN_NAME"
  echo "command=$RUN_CMD"
  echo "log_file=$LOG_FILE"
  echo "pid_file=$PID_FILE"
  echo "meta_file=$META_FILE"
  echo "started_at=$(date '+%Y-%m-%d %H:%M:%S %z')"
} > "$META_FILE"
: > "$LOG_FILE"
nohup "$RUNNER_FILE" "$PROJECT_DIR" "$PYTHON_BIN" "$ENV_FILE" "$CUDA_DEVICES" "$RUN_NAME" "$RUN_CMD" "$LOG_FILE" >/dev/null 2>&1 &
pid=$!
echo "$pid" > "$PID_FILE"
echo "$BASE" > "$LATEST_FILE"
echo "pid=$pid" >> "$META_FILE"
echo "started name=$RUN_NAME pid=$pid log=$LOG_FILE"
REMOTE
}

remote_base_for() {
  local target="${1:-latest}"
  ssh_remote bash -s -- "$REMOTE_RUN_DIR" "$target" <<'REMOTE'
set -euo pipefail
RUN_DIR="$1"
target="$2"
if [[ "$target" == "latest" ]]; then
  [[ -f "$RUN_DIR/latest" ]] || { echo "no latest run" >&2; exit 1; }
  cat "$RUN_DIR/latest"
elif [[ -f "$target" ]]; then
  printf '%s\n' "${target%.log}"
elif [[ -f "$RUN_DIR/$target.log" ]]; then
  printf '%s\n' "$RUN_DIR/$target"
else
  match=$(ls -1t "$RUN_DIR"/*-"$target".log 2>/dev/null | head -n 1 || true)
  [[ -n "$match" ]] || { echo "run not found: $target" >&2; exit 1; }
  printf '%s\n' "${match%.log}"
fi
REMOTE
}

cmd_tail() {
  local target="${1:-latest}" follow="${2:-}" base log_file
  base="$(remote_base_for "$target")"
  log_file="$base.log"
  if [[ "$follow" == "-f" || "$follow" == "--follow" ]]; then
    ssh_remote tail -n 80 -f "$log_file"
  else
    ssh_remote tail -n 120 "$log_file"
  fi
}

cmd_status() {
  local target="${1:-latest}" base
  base="$(remote_base_for "$target")"
  ssh_remote bash -s -- "$base" <<'REMOTE'
set -euo pipefail
BASE="$1"
LOG="$BASE.log"
PID="$BASE.pid"
META="$BASE.meta"
[[ -f "$META" ]] && cat "$META" || true
if [[ -f "$PID" ]]; then
  pid=$(cat "$PID")
  if kill -0 "$pid" 2>/dev/null; then
    echo "state=running"
  else
    echo "state=exited"
  fi
fi
if [[ -f "$LOG" ]]; then
  grep -E '^(finished_at|exit_status|log_file)=' "$LOG" | tail -n 3 || true
fi
REMOTE
}

cmd_stop() {
  local target="${1:-latest}" base
  base="$(remote_base_for "$target")"
  ssh_remote bash -s -- "$base" <<'REMOTE'
set -euo pipefail
BASE="$1"
PID="$BASE.pid"
[[ -f "$PID" ]] || { echo "pid file missing: $PID" >&2; exit 1; }
pid=$(cat "$PID")
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" || true
    echo "force-stopped pid=$pid"
  else
    echo "stopped pid=$pid"
  fi
else
  echo "not running pid=$pid"
fi
REMOTE
}

cmd_list() {
  ssh_remote bash -s -- "$REMOTE_RUN_DIR" <<'REMOTE'
set -euo pipefail
RUN_DIR="$1"
mkdir -p "$RUN_DIR"
find "$RUN_DIR" -maxdepth 1 -name '*.meta' -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn \
  | head -n 30 \
  | cut -d' ' -f2- \
  | while read -r meta; do echo "--- $meta"; cat "$meta"; done
REMOTE
}

cmd_fetch() {
  local remote_path="${1:-}" local_dir="${2:-$LOCAL_OUTPUT_DIR}"
  [[ -n "$remote_path" ]] || { err "fetch requires a remote path"; exit 2; }
  mkdir -p "$local_dir"
  if [[ "$remote_path" != /* ]]; then remote_path="$REMOTE_PROJECT/$remote_path"; fi
  scp_from_remote -r "$(remote_addr):$remote_path" "$local_dir/"
}

cmd_doctor() {
  echo "local_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "remote=$(remote_addr):$REMOTE_PORT"
  echo "project=$REMOTE_PROJECT"
  echo "cwd=$PWD"
  if command -v mountpoint >/dev/null 2>&1 && mountpoint -q "$PWD"; then
    echo "sshfs_mount=mounted"
  else
    echo "sshfs_mount=not-mounted-or-not-current-cwd"
  fi
  ssh_remote bash -s -- "$REMOTE_PROJECT" "$REMOTE_PYTHON" "$REMOTE_RUN_DIR" <<'REMOTE'
set -euo pipefail
PROJECT_DIR="$1"
PYTHON_BIN="$2"
RUN_DIR="$3"
echo "remote_host=$(uname -n 2>/dev/null || true)"
echo "remote_time=$(date '+%Y-%m-%d %H:%M:%S %z')"
echo "remote_project_exists=$(test -d "$PROJECT_DIR" && echo yes || echo no)"
echo "remote_project_writable=$(test -w "$PROJECT_DIR" && echo yes || echo no)"
echo "remote_python=$($PYTHON_BIN --version 2>&1 || true)"
mkdir -p "$RUN_DIR"
echo "remote_run_dir=$RUN_DIR"
echo "remote_run_dir_writable=$(test -w "$RUN_DIR" && echo yes || echo no)"
echo "disk=$(df -h "$PROJECT_DIR" | tail -n 1)"
REMOTE
}

[[ $# -gt 0 ]] || { usage; exit 2; }

case "$1" in
  -h|--help) usage ;;
  ping)
    ssh_remote bash -s -- "$REMOTE_PROJECT" "$REMOTE_PYTHON" <<'REMOTE'
set -euo pipefail
cd "$1"
echo "project=$(pwd)"
echo "host=$(uname -n 2>/dev/null || true)"
echo "time=$(date '+%Y-%m-%d %H:%M:%S %z')"
"$2" --version
REMOTE
    ;;
  run)
    shift
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -gt 0 ]] || { err "run requires a command"; exit 2; }
    run_remote_foreground "$(command_from_args "$@")"
    ;;
  py)
    shift
    [[ $# -gt 0 ]] || { err "py requires a script"; exit 2; }
    run_remote_foreground "$(shell_join "$REMOTE_PYTHON" "$@")"
    ;;
  cargo)
    shift
    run_remote_foreground "$(shell_join cargo "$@")"
    ;;
  start)
    shift
    name="${1:-}"
    [[ -n "$name" ]] || { err "start requires a name"; exit 2; }
    shift
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -gt 0 ]] || { err "start requires a command"; exit 2; }
    start_remote_background "$name" "$(command_from_args "$@")"
    ;;
  start-py)
    shift
    name="${1:-}"
    [[ -n "$name" ]] || { err "start-py requires a name"; exit 2; }
    shift
    [[ $# -gt 0 ]] || { err "start-py requires a script"; exit 2; }
    start_remote_background "$name" "$(shell_join "$REMOTE_PYTHON" "$@")"
    ;;
  tail)
    shift
    cmd_tail "${1:-latest}" "${2:-}"
    ;;
  status)
    shift
    cmd_status "${1:-latest}"
    ;;
  stop)
    shift
    cmd_stop "${1:-latest}"
    ;;
  list) cmd_list ;;
  doctor) cmd_doctor ;;
  fetch)
    shift
    cmd_fetch "${1:-}" "${2:-$LOCAL_OUTPUT_DIR}"
    ;;
  *)
    err "unknown command: $1"
    usage >&2
    exit 2
    ;;
esac

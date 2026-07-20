#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hugo/codes/CadButEaas}"
GPU_INDEX="${GPU_INDEX:-0}"
RUN_NAME="${RUN_NAME:-floorplancad_high_rq_contract_dev_smoke}"
RUN_DIR="$ROOT/reports/vlm/$RUN_NAME"
REPORT="$ROOT/results/${RUN_NAME}_train.json"
LOG="$RUN_DIR/train.log"
PID_FILE="$RUN_DIR/train.pid.json"
LOCK_FILE="$RUN_DIR/train.lock"
PYTHON="$ROOT/.venv-vlm/bin/python"
CHECKPOINT="$ROOT/reports/vlm/floorplancad_v4_same_set_overfit_32_v4_protocol_replay/best.pt"
CACHE="$ROOT/datasets/floorplancad_v4_same_set_overfit_32_v1/same_set_windowed_primitive_cache.jsonl"

cd "$ROOT"
mkdir -p "$RUN_DIR/diagnostic_topk" "$RUN_DIR/checkpoint_archive"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing python environment: $PYTHON" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "missing high-RQ checkpoint: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$CACHE" ]]; then
  echo "missing same-set cache: $CACHE" >&2
  exit 2
fi
if nvidia-smi pmon -i "$GPU_INDEX" -c 1 | awk 'NR > 2 && $2 != "-" {found=1} END {exit found ? 0 : 1}'; then
  echo "GPU $GPU_INDEX already has a compute process; refusing to start" >&2
  exit 3
fi

"$PYTHON" -m pytest \
  tests/test_floorplancad_high_rq_moe_regression.py \
  tests/test_floorplancad_pcgrad_production.py::test_training_presets_share_the_same_content_query_and_typed_stuff_architecture \
  tests/test_floorplancad_panoptic_training_losses.py::test_production_preset_forces_joint_fail_closed_gates \
  -q

command=(
  "$PYTHON" -u
  experiments/floorplancad_train_line_token_panoptic_moe.py
  --train "$CACHE"
  --val "$CACHE"
  --input-feature-schema v4
  --require-target-schema-v4
  --training-preset dev
  --device cuda:0
  --epochs 2
  --hidden-dim 256
  --layers 4
  --heads 8
  --query-decoder-layers 1
  --num-queries 256
  --max-tokens-per-record 2048
  --batch-records 2
  --amp bf16
  --geometry-attention-tile-size 128
  --train-prefetch-records 16
  --train-prefetch-workers 4
  --disable-bottleneck-weights
  --init-checkpoint "$CHECKPOINT"
  --model-output "$RUN_DIR/best.pt"
  --last-model-output "$RUN_DIR/last.pt"
  --report "$REPORT"
  --diagnostic-checkpoint-dir "$RUN_DIR/diagnostic_topk"
  --diagnostic-checkpoint-top-k 2
  --checkpoint-archive-dir "$RUN_DIR/checkpoint_archive"
  --checkpoint-archive-keep 2
  --progress-status-records 8
  --progress-status-seconds 60
  --progress-checkpoint-records 8
  --progress-checkpoint-seconds 120
  --lr 0.00002
  --lr-warmup-steps 50
  --lr-decay-steps 0
  --seed 20260720
)

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "duplicate launch blocked by lock: $LOCK_FILE" >&2
  exit 73
fi

"$PYTHON" - "$PID_FILE" "$REPORT" "$LOG" "$LOCK_FILE" "$GPU_INDEX" "$RUN_NAME" "${command[@]}" <<'PY'
import datetime
import json
import pathlib
import sys

pid_file, report, log, lock_file, gpu, run_name, *command = sys.argv[1:]
payload = {
    "schema_version": "floorplancad_high_rq_background_launch_v1",
    "status": "starting",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "launcher_pid": None,
    "gpu": int(gpu),
    "run_name": run_name,
    "report": report,
    "log": log,
    "lock": lock_file,
    "command": command,
    "contract": {
        "init_checkpoint": "reports/vlm/floorplancad_v4_same_set_overfit_32_v4_protocol_replay/best.pt",
        "forbidden_init_checkpoint": "reports/vlm/floorplancad_v6_test_train/last.pt",
        "expected_router_enabled": True,
        "expected_content_seeded_queries": False,
        "expected_partial_component_policy": "exclude",
    },
}
pathlib.Path(pid_file).write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"event": "launch_prepared", **payload}), flush=True)
PY

setsid env CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 "${command[@]}" >"$LOG" 2>&1 < /dev/null &

train_pid=$!
"$PYTHON" - "$PID_FILE" "$train_pid" <<'PY'
import json
import pathlib
import sys

pid_file, train_pid = sys.argv[1:]
path = pathlib.Path(pid_file)
payload = json.loads(path.read_text())
payload["status"] = "running"
payload["pid"] = int(train_pid)
path.write_text(json.dumps(payload, indent=2) + "\n")
print(json.dumps({"event": "launch_started", "pid": int(train_pid), "pid_file": pid_file}), flush=True)
PY

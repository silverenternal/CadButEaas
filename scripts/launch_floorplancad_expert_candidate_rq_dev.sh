#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/hugo/codes/CadButEaas}"
GPU_INDEX="${GPU_INDEX:-0}"
RUN_NAME="${RUN_NAME:-floorplancad_expert_candidate_rq_dev}"
RUN_DIR="$ROOT/reports/vlm/$RUN_NAME"
REPORT="$ROOT/results/${RUN_NAME}_train.json"
LOG="$RUN_DIR/train.log"
PID_FILE="$RUN_DIR/train.pid.json"
LOCK_FILE="$RUN_DIR/train.lock"
PYTHON="$ROOT/.venv-vlm/bin/python"

TRAIN_CACHE="$ROOT/reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/train_windowed_primitive_cache.jsonl"
VAL_CACHE="$ROOT/reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/val_windowed_primitive_cache.jsonl"
TRAIN_CANDIDATES="$ROOT/reports/vlm/floorplancad_expert_candidate_proposals/train_vecformer_expert_candidates.jsonl"
VAL_CANDIDATES="$ROOT/reports/vlm/floorplancad_expert_candidate_proposals/val_vecformer_expert_candidates.jsonl"

cd "$ROOT"
mkdir -p "$RUN_DIR/diagnostic_topk" "$RUN_DIR/checkpoint_archive"

for required in "$PYTHON" "$TRAIN_CACHE" "$VAL_CACHE" "$TRAIN_CANDIDATES" "$VAL_CANDIDATES"; do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done

if nvidia-smi pmon -i "$GPU_INDEX" -c 1 | awk 'NR > 2 && $2 != "-" {found=1} END {exit found ? 0 : 1}'; then
  echo "GPU $GPU_INDEX already has a compute process; refusing to start" >&2
  exit 3
fi

"$PYTHON" -m pytest \
  tests/test_floorplancad_candidate_aware_moe.py::test_expert_candidate_adapter_drops_audit_fields_and_maps_window \
  tests/test_floorplancad_candidate_aware_moe.py::test_dev_preset_preserves_explicit_expert_candidate_source \
  tests/test_floorplancad_high_rq_moe_regression.py::test_dev_preset_rejects_bad_v6_defaults_and_promotes_safe_route_dense_mainline \
  -q

command=(
  "$PYTHON" -u
  experiments/floorplancad_train_line_token_panoptic_moe.py
  --train "$TRAIN_CACHE"
  --val "$VAL_CACHE"
  --input-feature-schema v3
  --require-target-schema-v3
  --training-preset dev
  --device cuda:0
  --epochs "${EPOCHS:-2}"
  --hidden-dim "${HIDDEN_DIM:-256}"
  --layers "${LAYERS:-4}"
  --heads "${HEADS:-8}"
  --query-decoder-layers "${QUERY_DECODER_LAYERS:-1}"
  --num-queries "${NUM_QUERIES:-256}"
  --max-tokens-per-record "${MAX_TOKENS_PER_RECORD:-2048}"
  --batch-records "${BATCH_RECORDS:-2}"
  --amp "${AMP:-bf16}"
  --geometry-attention-tile-size "${GEOMETRY_ATTENTION_TILE_SIZE:-128}"
  --train-prefetch-records "${TRAIN_PREFETCH_RECORDS:-16}"
  --train-prefetch-workers "${TRAIN_PREFETCH_WORKERS:-4}"
  --disable-bottleneck-weights
  --candidate-proposals "$TRAIN_CANDIDATES"
  --val-candidate-proposals "$VAL_CANDIDATES"
  --candidate-feature-dim 57
  --max-candidate-queries "${MAX_CANDIDATE_QUERIES:-64}"
  --candidate-mask-prior-logit "${CANDIDATE_MASK_PRIOR_LOGIT:-0.75}"
  --candidate-mask-prior-loss-weight "${CANDIDATE_MASK_PRIOR_LOSS_WEIGHT:-0.02}"
  --candidate-ablation-tag vecformer_expert_soft_prior_rq_reuse_v1
  --model-output "$RUN_DIR/best.pt"
  --last-model-output "$RUN_DIR/last.pt"
  --report "$REPORT"
  --diagnostic-checkpoint-dir "$RUN_DIR/diagnostic_topk"
  --diagnostic-checkpoint-top-k 2
  --checkpoint-archive-dir "$RUN_DIR/checkpoint_archive"
  --checkpoint-archive-keep 2
  --progress-status-records 32
  --progress-status-seconds 60
  --progress-checkpoint-records 128
  --progress-checkpoint-seconds 180
  --seed "${SEED:-20260720}"
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
    "schema_version": "floorplancad_expert_candidate_rq_launch_v1",
    "status": "starting",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gpu": int(gpu),
    "run_name": run_name,
    "report": report,
    "log": log,
    "lock": lock_file,
    "command": command,
    "contract": {
        "candidate_source": "vecformer_expert_predictions_gt_free_adapter",
        "candidate_use": "soft_prior_query_initialization_and_mask_prior",
        "matching_policy": "not_hungarian_constraint",
        "candidate_feature_dim": 57,
        "max_candidate_queries": 64,
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

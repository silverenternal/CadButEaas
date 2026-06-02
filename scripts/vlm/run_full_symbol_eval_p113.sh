#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="$ROOT_DIR/reports/vlm/full_symbol_eval_p113_logs"
mkdir -p "$LOG_DIR"

case "$MODE" in
  sanity)
    LIMIT_TILES="${LIMIT_TILES:-20}"
    BATCH="${PREDICT_BATCH:-4}"
    OUT_PREFIX="full_public_raster_symbol_eval_sanity_p113_${RUN_ID}"
    ;;
  subset)
    LIMIT_TILES="${LIMIT_TILES:-2000}"
    BATCH="${PREDICT_BATCH:-8}"
    OUT_PREFIX="full_public_raster_symbol_eval_subset_p113_${RUN_ID}"
    ;;
  full)
    LIMIT_TILES="${LIMIT_TILES:-0}"
    BATCH="${PREDICT_BATCH:-16}"
    OUT_PREFIX="full_public_raster_symbol_eval_locked_p113_${RUN_ID}"
    ;;
  *)
    echo "Usage: $0 [sanity|subset|full]" >&2
    exit 2
    ;;
esac

EVAL_OUTPUT="$ROOT_DIR/reports/vlm/${OUT_PREFIX}.json"
PRED_OUTPUT="$ROOT_DIR/reports/vlm/${OUT_PREFIX}_predictions.jsonl"
LOG_FILE="$LOG_DIR/${OUT_PREFIX}.log"
PID_FILE="$LOG_DIR/${OUT_PREFIX}.pid"
STATUS_FILE="$LOG_DIR/${OUT_PREFIX}.status.json"

CMD=(
  "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/vlm/eval_symbol_yolo_tile_detector_v22.py"
  --data "$ROOT_DIR/datasets/symbol_tile_detector_tiny_sahi_v21"
  --yolo-dir "$ROOT_DIR/datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22_locked_full"
  --weights "$ROOT_DIR/runs/segment/runs/segment/runs/vlm/symbol_yolov8s_seg_rect_v28/weights/best.pt"
  --split locked
  --limit-tiles "$LIMIT_TILES"
  --imgsz 640
  --decode-conf 0.001
  --decode-iou 0.7
  --max-det-per-tile 300
  --predict-batch "$BATCH"
  --score-threshold-grid 0.02,0.05
  --nms-threshold-grid 0.45,0.55
  --max-per-page 500
  --selection-mode balanced_f1
  --eval-output "$EVAL_OUTPUT"
  --predictions-output "$PRED_OUTPUT"
  --device "${DEVICE:-0}"
)

cat > "$STATUS_FILE" <<JSON
{
  "run_id": "$RUN_ID",
  "mode": "$MODE",
  "state": "starting",
  "limit_tiles": $LIMIT_TILES,
  "predict_batch": $BATCH,
  "eval_output": "${EVAL_OUTPUT#$ROOT_DIR/}",
  "predictions_output": "${PRED_OUTPUT#$ROOT_DIR/}",
  "log_file": "${LOG_FILE#$ROOT_DIR/}",
  "pid_file": "${PID_FILE#$ROOT_DIR/}",
  "command": "${CMD[*]}"
}
JSON

(
  echo "[$(date -Is)] START mode=$MODE run_id=$RUN_ID limit_tiles=$LIMIT_TILES batch=$BATCH"
  echo "[$(date -Is)] CMD: ${CMD[*]}"
  set +e
  "${CMD[@]}"
  code=$?
  echo "[$(date -Is)] EXIT code=$code"
  python - <<PY
import json
from pathlib import Path
status = Path('$STATUS_FILE')
data = json.loads(status.read_text())
data['state'] = 'completed' if $code == 0 else 'failed'
data['exit_code'] = $code
data['completed_at'] = '$(date -Is)'
status.write_text(json.dumps(data, indent=2) + '\n')
PY
  exit $code
) > "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

python - <<PY
import json
from pathlib import Path
status = Path('$STATUS_FILE')
data = json.loads(status.read_text())
data['state'] = 'running'
data['pid'] = $PID
data['started_at'] = '$(date -Is)'
status.write_text(json.dumps(data, indent=2) + '\n')
PY

echo "Started $MODE run"
echo "PID: $PID"
echo "Status: ${STATUS_FILE#$ROOT_DIR/}"
echo "Log: ${LOG_FILE#$ROOT_DIR/}"
echo "Eval output: ${EVAL_OUTPUT#$ROOT_DIR/}"
echo "Predictions: ${PRED_OUTPUT#$ROOT_DIR/}"

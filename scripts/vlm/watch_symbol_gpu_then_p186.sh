#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 4 ]]; then
  echo "usage: $0 EXP_ID MODEL_PATH DEVICE TAG [PREDICT_BATCH]" >&2
  exit 2
fi
EXP_ID="$1"
MODEL_PATH="$2"
DEVICE="$3"
TAG="$4"
PREDICT_BATCH="${5:-16}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs/vlm reports/vlm configs/vlm
printf "[%s] waiting for %s\n" "$(date -Is)" "$MODEL_PATH"
while [[ ! -s "$MODEL_PATH" ]]; do
  sleep 60
done
printf "[%s] found model, starting P101 eval for %s\n" "$(date -Is)" "$EXP_ID"
EVAL_JSON="reports/vlm/${TAG}_p101_eval.json"
PRED_JSONL="reports/vlm/${TAG}_p101_predictions.jsonl"
FUSE_JSON="configs/vlm/${TAG}_p186_fusion.json"
FUSE_MD="reports/vlm/${TAG}_p186_fusion.md"
FUSE_OVERLAY="reports/vlm/${TAG}_p186_overlay.jsonl"
.venv/bin/python scripts/vlm/eval_symbol_yolo_p101_tiles_p186.py \
  --weights "$MODEL_PATH" \
  --yolo-dir datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22_locked_full \
  --device "$DEVICE" \
  --predict-batch "$PREDICT_BATCH" \
  --eval-output "$EVAL_JSON" \
  --predictions-output "$PRED_JSONL" \
  --selection-mode balanced_f1
printf "[%s] detector eval complete, starting P186 fusion for %s\n" "$(date -Is)" "$EXP_ID"
.venv/bin/python scripts/vlm/fuse_symbol_detector_with_p182_p186.py \
  --detector-predictions "$PRED_JSONL" \
  --output-json "$FUSE_JSON" \
  --output-md "$FUSE_MD" \
  --output-overlay "$FUSE_OVERLAY"
printf "[%s] done %s outputs: %s %s %s\n" "$(date -Is)" "$EXP_ID" "$EVAL_JSON" "$FUSE_JSON" "$FUSE_OVERLAY"

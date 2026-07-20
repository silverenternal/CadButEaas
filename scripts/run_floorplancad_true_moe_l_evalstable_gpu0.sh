#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs"
OUT_DIR="reports/vlm/floorplancad_line_token_panoptic_moe_l_evalstable"
TEACHER_PROPOSALS="${TEACHER_PROPOSALS:-reports/vlm/floorplancad_vecformer_teacher_dataset/train_val_teacher_proposals.jsonl}"
TRAIN_CACHE="${TRAIN_CACHE:-reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/train_windowed_primitive_cache.jsonl}"
VAL_CACHE="${VAL_CACHE:-reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/val_windowed_primitive_cache.jsonl}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ "$CUDA_VISIBLE_DEVICES" != "0" ]]; then
  echo "Refusing to launch: this eval-stable CadStruct-MoE run is GPU0-only while GPU1 is reserved."
  exit 2
fi

for required in \
  "$TEACHER_PROPOSALS" \
  "$TRAIN_CACHE" \
  "$VAL_CACHE"
do
  if [[ ! -f "$required" ]]; then
    echo "missing required input: $required"
    exit 2
  fi
done

LOG="$LOG_DIR/floorplancad_true_moe_l_evalstable_gpu0_${TS}.out"
PIDFILE="$LOG_DIR/floorplancad_true_moe_l_evalstable_gpu0.pid"

CMD=(
  .venv-vlm/bin/python
  experiments/floorplancad_train_line_token_panoptic_moe.py
  --train "$TRAIN_CACHE"
  --val "$VAL_CACHE"
  --input-feature-schema v3
  --require-target-schema-v3
  --device cuda:0
  --epochs "${EPOCHS:-80}"
  --hidden-dim "${HIDDEN_DIM:-512}"
  --layers "${LAYERS:-10}"
  --heads "${HEADS:-8}"
  --num-queries "${NUM_QUERIES:-256}"
  --query-decoder-layers "${QUERY_DECODER_LAYERS:-6}"
  --dropout "${DROPOUT:-0.0}"
  --checkpoint-metric pq_aware_component_proxy
  --min-val-object-recall-for-checkpoint "${MIN_VAL_OBJECT_RECALL:-0.80}"
  --min-val-mask-recall-for-checkpoint "${MIN_VAL_MASK_RECALL:-0.62}"
  --min-val-mask-precision-for-checkpoint "${MIN_VAL_MASK_PRECISION:-0.50}"
  --max-val-positive-rate-ratio-for-checkpoint "${MAX_VAL_POSITIVE_RATE_RATIO:-2.50}"
  --objectness-warmup-epochs "${OBJECTNESS_WARMUP_EPOCHS:-4}"
  --objectness-warmup-loss-multiplier "${OBJECTNESS_WARMUP_LOSS_MULTIPLIER:-4.0}"
  --objectness-warmup-positive-multiplier "${OBJECTNESS_WARMUP_POSITIVE_MULTIPLIER:-4.0}"
  --mask-tversky-loss-weight "${MASK_TVERSKY_LOSS_WEIGHT:-0.25}"
  --mask-positive-prob-floor-loss-weight "${MASK_POSITIVE_PROB_FLOOR_LOSS_WEIGHT:-0.10}"
  --zero-admission-patience-epochs "${ZERO_ADMISSION_PATIENCE_EPOCHS:-3}"
  --zero-admission-min-epoch "${ZERO_ADMISSION_MIN_EPOCH:-5}"
  --teacher-proposals "$TEACHER_PROPOSALS"
  --teacher-loss-weight "${TEACHER_LOSS_WEIGHT:-0.15}"
  --teacher-mask-loss-weight "${TEACHER_MASK_LOSS_WEIGHT:-1.0}"
  --teacher-query-loss-weight "${TEACHER_QUERY_LOSS_WEIGHT:-0.5}"
  --teacher-min-gt-iou "${TEACHER_MIN_GT_IOU:-0.50}"
  --extra-recall-labels "${EXTRA_RECALL_LABELS:-11,12,25,26,32}"
  --extra-grouping-labels "${EXTRA_GROUPING_LABELS:-30,31,32,33,34}"
  --model-output "$OUT_DIR/panoptic_component_moe_l_evalstable_best.pt"
  --last-model-output "$OUT_DIR/panoptic_component_moe_l_evalstable_last.pt"
  --report "results/floorplancad_true_moe_l_evalstable_gpu0_train.json"
  "$@"
)

printf '%q ' "${CMD[@]}" > "$OUT_DIR/evalstable_train_command_${TS}.txt"
printf '\n' >> "$OUT_DIR/evalstable_train_command_${TS}.txt"

setsid "${CMD[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDFILE"
echo "started pid=$PID log=$LOG pidfile=$PIDFILE"

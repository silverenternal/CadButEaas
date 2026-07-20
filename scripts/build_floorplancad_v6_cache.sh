#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv-vlm/bin/python}"

"$PYTHON_BIN" experiments/floorplancad_build_line_json_primitive_cache.py \
  --schema-version v6 \
  --output-dir reports/vlm/floorplancad_line_json_primitive_cache_v6 \
  --report results/floorplancad_line_json_primitive_cache_v6.json \
  --splits train val test \
  --reuse-existing-splits

"$PYTHON_BIN" experiments/floorplancad_build_windowed_line_token_cache.py \
  --cache-dir reports/vlm/floorplancad_line_json_primitive_cache_v6 \
  --output-dir reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v6 \
  --report results/floorplancad_windowed_line_token_v6.json \
  --window-size 2048 \
  --stride 1536 \
  --max-components-per-window 256 \
  --splits train val test \
  --reuse-existing-splits \
  --include-component-centered-train-aux

"$PYTHON_BIN" experiments/floorplancad_make_query_safe_window_cache.py \
  --source-dir reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v6 \
  --output-dir reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v6_p2_q256 \
  --report results/floorplancad_windowed_line_token_v6_p2_q256.json \
  --query-capacity 256 \
  --splits train val

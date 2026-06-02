#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TEMPLATE="${1:-}"
OUT="${2:-reports/vlm/submission_template_inserted.tex}"

if [[ -n "$TEMPLATE" ]]; then
  python scripts/vlm/insert_p266_into_template_p269.py \
    --template "$TEMPLATE" \
    --out "$OUT" \
    --apply
  SOURCE="$OUT"
else
  SOURCE="reports/vlm/p266_generic_submission_manuscript.tex"
fi

python scripts/vlm/check_p268_template_or_compile_readiness.py \
  --source "$SOURCE" \
  --compile

python scripts/vlm/build_p267_submission_handoff_bundle.py
python scripts/vlm/build_p270_portable_submission_bundle.py

echo "Submission resume check complete."
echo "Source checked: $SOURCE"
echo "Readiness: reports/vlm/p268_template_or_compile_readiness.md"
echo "Bundle: reports/vlm/p270_submission_bundle.tar.gz"

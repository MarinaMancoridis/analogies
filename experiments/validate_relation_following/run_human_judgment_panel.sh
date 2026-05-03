#!/usr/bin/env bash
# Run all three judge shards in parallel. Usage: ./run_human_judgment_panel.sh smoke|full
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python}"
IN="${IN:-$DIR/human_completions_flat.json}"

mode="${1:-}"
case "$mode" in
  smoke)
    OUT="${OUT:-$DIR/human_judgments_panel_smoke.ndjson}"
    EXTRA=(--smoke)
    ;;
  full)
    OUT="${OUT:-$DIR/human_judgments_panel.ndjson}"
    EXTRA=()
    ;;
  *)
    echo "Usage: $0 smoke|full" >&2
    echo "  smoke  -- 10 completions per shard, default OUT: human_judgments_panel_smoke.ndjson" >&2
    echo "  full   -- all eligible completions per shard, default OUT: human_judgments_panel.ndjson" >&2
    echo "Env: IN=path/to/human_completions_flat.json OUT=shared.ndjson PYTHON=python3" >&2
    echo "Progress: pip install tqdm for per-shard bars; this script passes --tqdm-position 0,1,2." >&2
    exit 1
    ;;
esac

echo "IN=$IN OUT=$OUT mode=$mode" >&2
pos=0
for SHARD in \
  openai/gpt-5.4-mini \
  google/gemini-3.1-flash-lite-preview \
  deepseek/deepseek-v3.2
do
  "$PY" "$DIR/judge_human_completions.py" "${EXTRA[@]}" \
    --in-json "$IN" \
    --out-ndjson "$OUT" \
    --shard-model "$SHARD" \
    --tqdm-position "$pos" &
  pos=$((pos + 1))
done
wait
echo "All shards finished." >&2

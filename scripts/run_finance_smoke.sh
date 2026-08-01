#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
MODE=${MODE:-direct_llm}
TASK_TYPE=${TASK_TYPE:-formula}
PORT=${PORT:-8010}
MODEL=${MODEL:-${CACHESPEC_MODEL:-qwen3-32b-fp8}}
DATA_FILE=${DATA_FILE:-"$ROOT/data/cache_friendly_v2_cacheable/formula_grouped_optimized_cacheable.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/outputs/${TASK_TYPE}_smoke_${MODE}"}
WORKSPACE_DIR=${WORKSPACE_DIR:-"$OUTPUT_DIR/service"}
NUM_EXAMPLES=${NUM_EXAMPLES:-10}

mkdir -p "$OUTPUT_DIR"

"$PYTHON" -m cachespec_api_service.app \
  --mode "$MODE" \
  --task-type "$TASK_TYPE" \
  --workspace-dir "$WORKSPACE_DIR" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --default-model "$MODEL" \
  > "$OUTPUT_DIR/api_stdout.log" 2>&1 &
SERVICE_PID=$!

cleanup() {
  kill "$SERVICE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null; then
    break
  fi
  sleep 1
done

"$PYTHON" -m cachespec_api_service.run_finance_task \
  --task-type "$TASK_TYPE" \
  --data-file "$DATA_FILE" \
  --output-dir "$OUTPUT_DIR/eval" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "$MODEL" \
  --num-examples "$NUM_EXAMPLES" \
  --use-cache

curl -fsS "http://127.0.0.1:${PORT}/admin/stats" | "$PYTHON" -m json.tool

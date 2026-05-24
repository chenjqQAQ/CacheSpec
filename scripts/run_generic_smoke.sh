#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
MODE=${MODE:-exact_cache}
TASK_TYPE=${TASK_TYPE:-generic}
PORT=${PORT:-8000}
MODEL=${MODEL:-${MINICACHE_MODEL:-qwen3-32b-fp8}}
DATA_FILE=${DATA_FILE:-"$ROOT/data/generic/gt_param-w-synonym_data_large.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT/outputs/generic_smoke_${MODE}"}
WORKSPACE_DIR=${WORKSPACE_DIR:-"$OUTPUT_DIR/service"}
NUM_EXAMPLES=${NUM_EXAMPLES:-10}

mkdir -p "$OUTPUT_DIR"

"$PYTHON" -m gencache_api_service.app \
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

(
  cd "$ROOT/LASER"
  GENCACHE_SERVICE_BASE_URL="http://127.0.0.1:${PORT}/v1" \
  "$PYTHON" caching_wo_agent.py \
    --data-file "$DATA_FILE" \
    --dir_name "$OUTPUT_DIR/laser" \
    --start 0 \
    --num_examples "$NUM_EXAMPLES"
)

curl -fsS "http://127.0.0.1:${PORT}/admin/stats" | "$PYTHON" -m json.tool

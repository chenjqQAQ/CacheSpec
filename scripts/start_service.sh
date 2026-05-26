#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON=${PYTHON:-python}
MODE=${MODE:-direct_llm}
TASK_TYPE=${TASK_TYPE:-generic}
PORT=${PORT:-8000}
HOST=${HOST:-127.0.0.1}
MODEL=${MODEL:-${MINICACHE_MODEL:-qwen3-32b-fp8}}
WORKSPACE_DIR=${WORKSPACE_DIR:-"$ROOT/outputs/service_${MODE}_${TASK_TYPE}"}

mkdir -p "$WORKSPACE_DIR"

exec "$PYTHON" -m minicache_api_service.app \
  --mode "$MODE" \
  --task-type "$TASK_TYPE" \
  --workspace-dir "$WORKSPACE_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --default-model "$MODEL"

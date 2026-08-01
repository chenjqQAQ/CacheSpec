# CacheSpec

CacheSpec is a research prototype for evaluating LLM response caching. It provides an OpenAI-compatible API service and task runners for exact matching, semantic matching, reproduced GenCache baselines, and CacheSpec program-cache variants.

The repository is organized for reproducible experiments:

- `cachespec_api_service/`: OpenAI-compatible CacheSpec service.
- `cachespec/`: CacheSpec library components used by the original and modified modes.
- `LASER/`: generic prompt and WebShop runners.
- `data/`: datasets used by the public experiments.
- `data_gen/`: scripts for regenerating the generic and structural datasets.
- `scripts/`: small, configurable smoke-test scripts.
- `outputs/`: default location for local experiment outputs.

## Modes

The API service supports the following modes:

- `direct_llm`: always forwards the request to the backend LLM.
- `exact_cache`: returns a cached answer only when the full prompt is exactly identical.
- `gptcache`: returns a cached answer when the full-prompt embedding similarity exceeds a threshold.
- `original`: reproduced GenCache baseline with the original rule-based extraction path.
- `modified`: CacheSpec program-cache path with small-model semantic variable extraction where supported.
- `formula_modified_parallel`: parallel formula-task variant with concurrent extractor calls and optional cold-start monitoring.

Supported task types are:

- `generic`
- `webshop`
- `formula`
- `codetatqa`
- `codefinqa`
- `bizbench`

## Data

The public copy includes these experiment files:

```text
data/generic/
  gt_param-w-synonym_data_large.jsonl
  gt_param-w-synonym_data_large_structural_10k.jsonl

data/cache_friendly_v2_cacheable/
  formula_grouped_optimized_cacheable.jsonl
  CodeTAT-QA_grouped_optimized_cacheable.jsonl
  CodeFinQA_grouped_optimized_cacheable.jsonl
  SEC-NUM_grouped_optimized_cacheable.jsonl
  build_summary.json
  token_length_distribution_qwen3_32b.json
```

WebShop code is included under `LASER/web_agent_site/`, but the full WebShop product index can be large and may have separate licensing requirements. To run full WebShop experiments, prepare the official WebShop data and point `LASER/data`, `LASER/search_engine`, and `LASER/web_agent_site/search_engine` to the prepared files.

## Installation

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e cachespec
```

If you use WebShop, also follow the official WebShop setup instructions and install the required Java/Lucene and spaCy dependencies.

## Configuration

Copy the example environment file and edit it for your own model endpoint:

```bash
cp .env.example .env
source .env
```

The service talks to any OpenAI-compatible chat-completions endpoint. Important variables:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:9000/v1
export OPENAI_API_KEY=EMPTY
export CACHESPEC_MODEL=qwen3-32b-fp8
export SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

For GPTCache-style matching:

```bash
export GENCACHE_GPTCACHE_SIMILARITY_THRESHOLD=0.95
```

For formula, CodeTAT-QA, CodeFinQA, and BizBench, code-generation fallback is enabled by default. Disable it with:

```bash
export GENCACHE_CODEGEN_FALLBACK=false
```

## Start The API Service

Start a direct-LLM service:

```bash
MODE=direct_llm TASK_TYPE=generic PORT=8000 MODEL="$CACHESPEC_MODEL" \
  bash scripts/start_service.sh
```

Start an exact-cache service:

```bash
MODE=exact_cache TASK_TYPE=generic PORT=8000 MODEL="$CACHESPEC_MODEL" \
  bash scripts/start_service.sh
```

Start a GPTCache-style service:

```bash
MODE=gptcache TASK_TYPE=generic PORT=8000 MODEL="$CACHESPEC_MODEL" \
  bash scripts/start_service.sh
```

The service exposes:

- `POST /v1/chat/completions`
- `GET /health`
- `GET /admin/stats`
- `POST /admin/reset`

## Run Smoke Tests

Generic prompt smoke test:

```bash
MODE=exact_cache NUM_EXAMPLES=10 PORT=8000 bash scripts/run_generic_smoke.sh
MODE=gptcache NUM_EXAMPLES=10 PORT=8001 bash scripts/run_generic_smoke.sh
```

Formula smoke test:

```bash
MODE=direct_llm TASK_TYPE=formula NUM_EXAMPLES=10 PORT=8010 \
  bash scripts/run_finance_smoke.sh
```

CodeTAT-QA smoke test:

```bash
MODE=gptcache TASK_TYPE=codetatqa \
DATA_FILE=data/cache_friendly_v2_cacheable/CodeTAT-QA_grouped_optimized_cacheable.jsonl \
NUM_EXAMPLES=10 PORT=8011 \
  bash scripts/run_finance_smoke.sh
```

Outputs are written under `outputs/` by default. Each service workspace records:

- `service_stats.json`
- `logs/request_trace.jsonl`
- cache files under `cache/`
- aggregate metrics under `results/`

## Regenerate Generic Data

The included generic datasets can be regenerated with:

```bash
python data_gen/restore_cachespec_generic_data.py \
  --large-output data/generic/gt_param-w-synonym_data_large.jsonl \
  --structural-output data/generic/gt_param-w-synonym_data_large_structural_10k.jsonl \
  --workers 32 \
  --judge-workers 32
```

The script reads model endpoints from `OPENAI_BASE_URL` or `CACHESPEC_STRUCTURAL_BASE_URLS`.

## Notes For Reproducibility

- No private API keys or private endpoints are stored in this repository.
- All model endpoints should be provided through environment variables.
- Experiment outputs are intentionally ignored by git.
- Large WebShop indexes should be prepared separately and linked into the `LASER/` directory before running WebShop experiments.

## Compute and Hardware

The experiments are designed to run against OpenAI-compatible model servers.
The paper experiments used NVIDIA A100 80GB GPUs, with one Qwen3-32B target
model server per GPU and a Qwen3-1.7B small model for semantic variable
extraction and speculative drafting. Exact GPU-hours depend on the selected
task, concurrency setting, backend throughput, and whether only smoke tests or
the full benchmark suite are reproduced. The released runners write timing and
request-level logs that can be used to compute the exact GPU-hour budget for a
new reproduction run.

## License

This repository is released under the MIT License. See `LICENSE` for details.

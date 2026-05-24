from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from .formula_modified_runtime import FormulaModifiedRuntime
from .runtime import MiniCacheRuntime


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: Optional[str] = None
    messages: list = Field(default_factory=list)
    temperature: Optional[float] = 0
    top_p: Optional[float] = 1
    max_tokens: Optional[int] = None
    stop: Any = None
    n: Optional[int] = 1
    functions: Any = None
    function_call: Any = None
    tools: Any = None
    tool_choice: Any = None
    response_format: Any = None
    extra_body: Optional[Dict[str, Any]] = None
    use_cache: bool = True
    pretrain: bool = False
    ground_truth: Any = None
    test_mode: bool = False


def create_app(runtime: Any) -> FastAPI:
    app = FastAPI(title="MiniCache baseline service")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "ok": True,
            "mode": runtime.mode,
            "task_type": runtime.task_type,
            "cache_entries": runtime.cache_entries(),
        }

    @app.get("/admin/stats")
    def stats() -> Dict[str, Any]:
        return runtime.admin_stats()

    @app.post("/admin/reset")
    def reset() -> Dict[str, Any]:
        runtime.reset()
        return {"ok": True}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest) -> Dict[str, Any]:
        payload = req.model_dump()
        extra_body = payload.pop("extra_body", None) or {}
        if isinstance(extra_body, dict):
            payload.update(extra_body)
        result = runtime.predict(payload)
        message: Dict[str, Any] = {"role": "assistant", "content": result["answer"]}
        if result.get("function_call") is not None:
            message["function_call"] = result["function_call"]
        if result.get("tool_calls") is not None:
            message["tool_calls"] = result["tool_calls"]
        return {
            "id": f"minicache-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or runtime.default_model,
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": result["usage"],
            "cache_hit": result["cache_hit"],
            "cached_key": result["cached_key"],
            "pretrain": result["pretrain"],
            "latency_sec": result["latency_sec"],
            "stage_timings": result["stage_timings"],
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["direct_llm", "exact_cache", "gptcache", "original", "modified", "formula_modified_parallel"],
        required=True,
    )
    parser.add_argument(
        "--task-type",
        choices=["generic", "webshop", "formula", "bizbench", "codetatqa", "codefinqa"],
        required=True,
    )
    parser.add_argument("--workspace-dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-model", default="qwen3-32b-fp8")
    args = parser.parse_args()

    runtime_cls = FormulaModifiedRuntime if args.mode == "formula_modified_parallel" else MiniCacheRuntime
    runtime = runtime_cls(mode=args.mode, task_type=args.task_type, workspace_dir=Path(args.workspace_dir), default_model=args.default_model)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

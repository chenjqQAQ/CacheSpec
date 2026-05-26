from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .baseline_caches import ExactCacheBaseline, GptCacheBaseline, RuleProgramCacheBaseline
from .codegen import build_codegen_messages, execute_code, extract_code


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def usage_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    raw = usage.model_dump() if hasattr(usage, "model_dump") else usage if isinstance(usage, dict) else {}
    return {
        "prompt_tokens": int(raw.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(raw.get("completion_tokens", 0) or 0),
        "total_tokens": int(raw.get("total_tokens", 0) or 0),
    }


@dataclass
class ServiceStats:
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_calls: int = 0
    pretrain_calls: int = 0
    codegen_calls: int = 0
    codegen_successes: int = 0
    codegen_failures: int = 0
    request_latency_sum: float = 0.0
    cache_hit_latency_sum: float = 0.0
    cache_miss_latency_sum: float = 0.0
    fallback_latency_sum: float = 0.0
    cache_lookup_latency_sum: float = 0.0
    cache_store_latency_sum: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_latencies: List[float] = field(default_factory=list)
    cache_hit_latencies: List[float] = field(default_factory=list)
    cache_miss_latencies: List[float] = field(default_factory=list)
    cache_lookup_latencies: List[float] = field(default_factory=list)
    fallback_latencies: List[float] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @staticmethod
    def percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return ordered[idx]

    def snapshot(self, cache_entries: int, mode: str, task_type: str, workspace_dir: Path) -> Dict[str, Any]:
        with self.lock:
            total = self.total_requests
            hits = self.cache_hits
            misses = self.cache_misses
            hit_rate = hits / total if total else 0.0
            return {
                "mode": mode,
                "variant": mode,
                "task_type": task_type,
                "workspace_dir": str(workspace_dir),
                "total_calls": total,
                "total_requests": total,
                "cache_hits": hits,
                "num_cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_rate": hit_rate,
                "fallback_calls": self.fallback_calls,
                "pretrain_calls": self.pretrain_calls,
                "codegen_calls": self.codegen_calls,
                "codegen_successes": self.codegen_successes,
                "codegen_failures": self.codegen_failures,
                "total_latency_sec": self.request_latency_sum,
                "avg_latency_sec": self.request_latency_sum / total if total else 0.0,
                "avg_request_latency_sec": self.request_latency_sum / total if total else 0.0,
                "avg_cache_hit_latency_sec": self.cache_hit_latency_sum / hits if hits else 0.0,
                "avg_cache_miss_latency_sec": self.cache_miss_latency_sum / misses if misses else 0.0,
                "avg_fallback_latency_sec": self.fallback_latency_sum / self.fallback_calls if self.fallback_calls else 0.0,
                "avg_cache_lookup_latency_sec": self.cache_lookup_latency_sum / total if total else 0.0,
                "avg_cache_store_latency_sec": self.cache_store_latency_sum / misses if misses else 0.0,
                "p50_request_latency_sec": self.percentile(self.request_latencies, 50),
                "p95_request_latency_sec": self.percentile(self.request_latencies, 95),
                "p99_request_latency_sec": self.percentile(self.request_latencies, 99),
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cache_entries": cache_entries,
                "cache_generation_attempts": 0,
                "cache_generation_successes": 0,
                "cache_generation_failures": 0,
                "cache_generation_success_rate": 0.0,
                "avg_cache_generation_latency_sec": 0.0,
            }

    def metrics_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "request_latency": list(self.request_latencies),
                "cache_hit_latency": list(self.cache_hit_latencies),
                "cache_miss_latency": list(self.cache_miss_latencies),
                "cache_find_time": list(self.cache_lookup_latencies),
                "fallback_latency": list(self.fallback_latencies),
            }


class MiniCacheRuntime:
    def __init__(self, mode: str, task_type: str, workspace_dir: Path, default_model: str):
        self.mode = mode
        self.task_type = task_type
        self.workspace_dir = workspace_dir
        self.default_model = default_model
        self.cache_dir = workspace_dir / mode / "cache"
        self.logs_dir = workspace_dir / mode / "logs"
        self.results_dir = workspace_dir / mode / "results"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.stats = ServiceStats()
        self.trace_path = self.logs_dir / "request_trace.jsonl"
        self.stats_path = workspace_dir / mode / "service_stats.json"
        self.service_log_path = self.logs_dir / "service.log"
        self.metric_path = self.results_dir / "metric.json"
        self.results_path = self.results_dir / "results.json"
        self.trace_lock = threading.Lock()
        self.log_lock = threading.Lock()

        self.cache = None
        if mode == "exact_cache":
            self.cache = ExactCacheBaseline(self.cache_dir / "exact_cache.json")
        elif mode == "gptcache":
            threshold = float(os.getenv("GENCACHE_GPTCACHE_SIMILARITY_THRESHOLD", "0.95"))
            self.cache = GptCacheBaseline(self.cache_dir / "gptcache.json", threshold=threshold)
        elif mode in {"original", "modified"}:
            num_records = int(os.getenv("GENCACHE_NUM_RECORDS_BEFORE_CACHING", "3"))
            self.cache = RuleProgramCacheBaseline(
                self.cache_dir / "global_cache.json",
                workspace_dir / mode / "database" / "database_clusters.json",
                task_type=task_type,
                mode=mode,
                num_records_before_caching=num_records,
            )
        elif mode != "direct_llm":
            raise ValueError(f"unknown mode: {mode}")

        base_url = (
            os.getenv("GENCACHE_FALLBACK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_ENDPOINT")
            or "http://127.0.0.1:9000/v1"
        )
        api_key = os.getenv("GENCACHE_FALLBACK_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
        timeout = float(os.getenv("GENCACHE_FALLBACK_TIMEOUT", "180"))
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.enable_thinking = env_bool("GENCACHE_FALLBACK_ENABLE_THINKING", False)
        self.codegen_fallback = env_bool("GENCACHE_CODEGEN_FALLBACK", task_type in {"formula", "codetatqa", "codefinqa", "bizbench"})
        self.fallback_base_url = base_url
        self.fallback_concurrency = int(os.getenv("GENCACHE_FALLBACK_CONCURRENCY", "1000000"))
        self.fallback_semaphore = threading.Semaphore(self.fallback_concurrency)
        self._log(
            "INFO",
            (
                f"starting MiniCache mode={mode} task_type={task_type} "
                f"workspace={workspace_dir} model={default_model} "
                f"fallback_base_url={base_url} thinking={self.enable_thinking} "
                f"codegen_fallback={self.codegen_fallback} fallback_concurrency={self.fallback_concurrency}"
            ),
        )
        self._write_stats()

    def _log(self, level: str, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with self.log_lock:
            with self.service_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ts} - {level} - {message}\n")

    def cache_entries(self) -> int:
        return self.cache.count() if self.cache is not None else 0

    def admin_stats(self) -> Dict[str, Any]:
        stats = self.stats.snapshot(self.cache_entries(), self.mode, self.task_type, self.workspace_dir)
        if self.cache is not None and hasattr(self.cache, "generation_stats"):
            stats.update(self.cache.generation_stats())
        return stats

    def _write_stats(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path.write_text(
            json.dumps(self.admin_stats(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.metric_path.write_text(
            json.dumps(self.stats.metrics_snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.results_path.write_text(
            json.dumps(
                {
                    "total_calls": self.stats.total_requests,
                    "num_cache_hits": self.stats.cache_hits,
                    "num_cache_misses": self.stats.cache_misses,
                    "cache_entries": self.cache_entries(),
                    "mode": self.mode,
                    "task_type": self.task_type,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _trace(self, payload: Dict[str, Any]) -> None:
        with self.trace_lock:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        if self.cache is not None:
            self.cache.reset()
        self.stats = ServiceStats()
        if self.trace_path.exists():
            self.trace_path.unlink()
        self._log("INFO", "admin reset")
        self._write_stats()

    def _call_llm(self, request: Dict[str, Any], messages: Optional[list] = None) -> Tuple[str, Optional[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
        wait_t0 = time.time()
        self.fallback_semaphore.acquire()
        fallback_wait = time.time() - wait_t0
        t0 = time.time()
        model = request.get("model") or self.default_model
        try:
            extra_body = request.get("extra_body") or {}
            if not self.enable_thinking:
                extra_body = dict(extra_body)
                extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages or request.get("messages") or [],
                "temperature": request.get("temperature", 0),
                "top_p": request.get("top_p", 1),
                "max_tokens": int(request.get("max_tokens") or os.getenv("GENCACHE_STRONG_MAX_TOKENS", "1024")),
                "extra_body": extra_body,
            }
            if request.get("stop") is not None:
                kwargs["stop"] = request.get("stop")
            if request.get("functions") is not None:
                kwargs["functions"] = request.get("functions")
                kwargs["function_call"] = request.get("function_call") or "auto"
            if request.get("tools") is not None:
                kwargs["tools"] = request.get("tools")
                if request.get("tool_choice") is not None:
                    kwargs["tool_choice"] = request.get("tool_choice")

            resp = self.client.chat.completions.create(**kwargs)
            message = resp.choices[0].message
            content = message.content or ""
            function_call = None
            tool_calls = None
            if getattr(message, "function_call", None) is not None:
                function_call = message.function_call.model_dump() if hasattr(message.function_call, "model_dump") else dict(message.function_call)
            if getattr(message, "tool_calls", None) is not None:
                tool_calls = [
                    item.model_dump() if hasattr(item, "model_dump") else dict(item)
                    for item in (message.tool_calls or [])
                ]
            return content, function_call, usage_dict(resp.usage), {
                "finish_reason": resp.choices[0].finish_reason,
                "fallback_model": model,
                "fallback_base_url": self.fallback_base_url,
                "fallback_latency_sec": time.time() - t0,
                "fallback_wait_sec": fallback_wait,
                "tool_calls": tool_calls,
            }
        finally:
            self.fallback_semaphore.release()

    def _call_codegen(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        t0 = time.time()
        raw, _, usage, meta = self._call_llm(request, messages=build_codegen_messages(request.get("messages") or []))
        code = extract_code(raw)
        if not code:
            return raw, usage, {"codegen_error": "no_code_block", "raw_output": raw[:500], **meta}
        result, error, exec_latency = execute_code(code)
        stage = {
            "codegen_generation_sec": time.time() - t0 - exec_latency,
            "codegen_exec_latency_sec": exec_latency,
            "codegen_error": error,
            "code": code[:1000],
            **meta,
        }
        if error:
            return raw, usage, stage
        return f"[[{result}]]", usage, stage

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        use_cache = bool(request.get("use_cache", True))
        pretrain = bool(request.get("pretrain", False))
        test_mode = bool(request.get("test_mode", False))
        ground_truth = request.get("ground_truth")
        stage: Dict[str, Any] = {}
        cache_hit = False
        cached_key = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        function_call = None

        with self.stats.lock:
            self.stats.total_requests += 1

        try:
            if pretrain and ground_truth is not None and self.cache is not None:
                cached_key = self.cache.store(request, ground_truth, {"pretrain": True, "task_type": self.task_type})
                answer = str(ground_truth)
                with self.stats.lock:
                    self.stats.pretrain_calls += 1
            else:
                entry = None
                if use_cache and self.cache is not None:
                    entry, stage = self.cache.lookup(request)
                    lookup_latency = float(stage.get("cache_lookup_sec", 0.0) or 0.0)
                    with self.stats.lock:
                        self.stats.cache_lookup_latency_sum += lookup_latency
                        self.stats.cache_lookup_latencies.append(lookup_latency)
                if entry is not None:
                    cache_hit = True
                    cached_key = entry.get("key")
                    cached_answer = entry.get("answer", "")
                    if isinstance(cached_answer, dict):
                        answer = str(cached_answer.get("content") or "")
                        function_call = cached_answer.get("function_call")
                        if cached_answer.get("tool_calls") is not None:
                            stage["tool_calls"] = cached_answer.get("tool_calls")
                    else:
                        answer = str(cached_answer)
                    with self.stats.lock:
                        self.stats.cache_hits += 1
                else:
                    with self.stats.lock:
                        self.stats.cache_misses += 1
                        self.stats.fallback_calls += 1
                    if self.codegen_fallback and self.task_type in {"formula", "codetatqa", "codefinqa", "bizbench"}:
                        with self.stats.lock:
                            self.stats.codegen_calls += 1
                        answer, usage, codegen_stage = self._call_codegen(request)
                        stage.update(codegen_stage)
                        with self.stats.lock:
                            if codegen_stage.get("codegen_error"):
                                self.stats.codegen_failures += 1
                            else:
                                self.stats.codegen_successes += 1
                    else:
                        answer, function_call, usage, llm_stage = self._call_llm(request)
                        stage.update(llm_stage)
                    fallback_latency = float(stage.get("fallback_latency_sec", 0.0) or 0.0)
                    if fallback_latency:
                        with self.stats.lock:
                            self.stats.fallback_latency_sum += fallback_latency
                            self.stats.fallback_latencies.append(fallback_latency)
                    if use_cache and self.cache is not None and not test_mode:
                        store_t0 = time.time()
                        stored_answer: Any = answer
                        if function_call is not None:
                            stored_answer = {"content": answer, "function_call": function_call}
                        if stage.get("tool_calls") is not None:
                            stored_answer = {"content": answer, "function_call": function_call, "tool_calls": stage.get("tool_calls")}
                        cached_key = self.cache.store(request, stored_answer, {"pretrain": False, "task_type": self.task_type})
                        stage["cache_store_sec"] = time.time() - store_t0
                        with self.stats.lock:
                            self.stats.cache_store_latency_sum += stage["cache_store_sec"]
        except Exception as exc:
            answer = ""
            stage["error"] = repr(exc)
            self._log("ERROR", f"request failed: {exc!r}")

        latency = time.time() - started
        with self.stats.lock:
            self.stats.request_latency_sum += latency
            self.stats.request_latencies.append(latency)
            self.stats.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
            self.stats.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
            self.stats.total_tokens += int(usage.get("total_tokens", 0) or 0)
            if cache_hit:
                self.stats.cache_hit_latency_sum += latency
                self.stats.cache_hit_latencies.append(latency)
            else:
                self.stats.cache_miss_latency_sum += latency
                self.stats.cache_miss_latencies.append(latency)
        stage.setdefault("total_request_sec", latency)
        self._write_stats()
        trace_payload = {
            "ts": time.time(),
            "mode": self.mode,
            "variant": self.mode,
            "task_type": self.task_type,
            "workspace_dir": str(self.workspace_dir),
            "model": request.get("model") or self.default_model,
            "pretrain": pretrain,
            "ground_truth": ground_truth,
            "use_cache": use_cache,
            "test_mode": test_mode,
            "cache_hit": cache_hit,
            "cached_key": cached_key,
            "latency_sec": latency,
            "duration_sec": latency,
            "usage": usage,
            "answer": answer,
            "function_call": function_call,
            "tool_calls": stage.get("tool_calls"),
            "stage_timings": stage,
            "messages": request.get("messages") or [],
            "functions": request.get("functions"),
            "function_call_request": request.get("function_call"),
            "tools": request.get("tools"),
            "tool_choice": request.get("tool_choice"),
        }
        self._trace(
            trace_payload
        )
        self._log(
            "INFO",
            (
                f"request mode={self.mode} task={self.task_type} "
                f"hit={cache_hit} latency={latency:.4f}s key={cached_key} "
                f"tokens={usage.get('total_tokens', 0)}"
            ),
        )
        return {
            "answer": answer,
            "function_call": function_call,
            "tool_calls": stage.get("tool_calls"),
            "usage": usage,
            "cache_hit": cache_hit,
            "cached_key": cached_key,
            "pretrain": pretrain,
            "latency_sec": latency,
            "stage_timings": stage,
        }

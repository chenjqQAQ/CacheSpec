from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .codegen import SAFE_BUILTINS, build_codegen_messages, execute_code, extract_code
from .runtime import ServiceStats, env_bool, usage_dict


def normalize_answer(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"\[\[(.*?)\]\]", text, flags=re.DOTALL)
    if match:
        text = match.group(1).strip()
    text = text.replace(",", "").replace("$", "")
    text = re.sub(r"\s+", " ", text)
    return text


def answers_match(expected: Any, actual: Any, tolerance: float = 1e-4) -> bool:
    e = normalize_answer(expected)
    a = normalize_answer(actual)
    if e == a:
        return True
    try:
        ef = float(e.rstrip("%"))
        af = float(a.rstrip("%"))
    except Exception:
        return False
    return abs(ef - af) <= tolerance * max(1.0, abs(ef), abs(af))


def request_text(request: Dict[str, Any]) -> str:
    parts: List[str] = []
    for message in request.get("messages") or []:
        content = message.get("content", "")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
        else:
            parts.append(str(content))
    return "\n\n".join(parts)


def formula_name_from_request(request: Dict[str, Any]) -> str:
    extra = request.get("extra_body") or {}
    for key in ("formula_name", "_formula_name", "group_key"):
        if request.get(key):
            return str(request[key]).strip()
        if isinstance(extra, dict) and extra.get(key):
            return str(extra[key]).strip()
    text = request_text(request)
    patterns = [
        r"Use formula\s+(.+?)\s+to answer",
        r"Formula name\s*:\s*(.+)",
        r"_formula_name\s*[:=]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().strip(".")
    match = re.search(r"Formula\s*:\s*([^\n,]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "unknown_formula"


def stable_cache_id(group_key: str) -> str:
    return hashlib.sha256(group_key.encode("utf-8")).hexdigest()[:12]


def compact_json(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    think_end = raw.rfind("</think>")
    if think_end >= 0:
        raw = raw[think_end + len("</think>") :].strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end >= start:
        raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(escape_control_chars_in_json_strings(raw))


def escape_control_chars_in_json_strings(raw: str) -> str:
    out: List[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
            continue
        out.append(ch)
        if ch == '"':
            in_string = True
    return "".join(out)


def run_solve_program(program_code: str, slots: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], float]:
    started = time.time()
    namespace = {"__builtins__": dict(SAFE_BUILTINS)}
    local_ns: Dict[str, Any] = {}
    try:
        parsed = ast.parse(program_code)
        for node in ast.walk(parsed):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.AsyncWith, ast.Lambda, ast.ClassDef)):
                return None, f"unsafe_node: {type(node).__name__}", time.time() - started
        exec(compile(parsed, "<formula_cache_program>", "exec"), namespace, local_ns)
        solve = local_ns.get("solve") or namespace.get("solve")
        if not callable(solve):
            return None, "missing_solve_function", time.time() - started
        try:
            result = solve(slots)
        except Exception:
            result = solve(coerce_slots(slots))
    except Exception as exc:
        return None, f"exec_error: {exc}", time.time() - started
    if result is None:
        return None, "empty_result", time.time() - started
    text = str(result).strip()
    if not re.search(r"\[\[.*?\]\]", text, flags=re.DOTALL):
        text = f"[[{text}]]"
    return text, None, time.time() - started


def coerce_slots(slots: Dict[str, Any]) -> Dict[str, Any]:
    return {key: coerce_number(value) for key, value in slots.items()}


def coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return value
    multiplier = 1.0
    if "%" in text:
        multiplier = 0.01
        text = text.replace("%", "")
    scale_map = [
        ("billion", 1e9),
        ("bn", 1e9),
        ("million", 1e6),
        ("mm", 1e6),
        ("thousand", 1e3),
        ("k", 1e3),
    ]
    for marker, scale in scale_map:
        if marker in text:
            multiplier *= scale
            text = text.replace(marker, "")
    text = text.replace("$", "").replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return value
    try:
        return float(match.group(0)) * multiplier
    except Exception:
        return value


@dataclass
class FormulaSample:
    request: Dict[str, Any]
    answer: str
    group_key: str
    ts: float = field(default_factory=time.time)


@dataclass
class FormulaCacheEntry:
    group_key: str
    slot_spec: Dict[str, str]
    program_code: str
    created_at: float
    generation_samples: int
    validation: Dict[str, Any]
    cache_id: str
    fail_count: int = 0
    hit_count: int = 0

    def to_json(self) -> Dict[str, Any]:
        return {
            "group_key": self.group_key,
            "spec": {
                "representation_type": "program",
                "action_type": "formula_extraction",
                "cluster_summary": self.group_key,
                "slot_spec": self.slot_spec,
                "program_code": self.program_code,
                "_program_validation": self.validation,
            },
            "created_at": self.created_at,
            "generation_samples": self.generation_samples,
            "validation": self.validation,
            "cache_id": self.cache_id,
            "fail_count": self.fail_count,
            "hit_count": self.hit_count,
        }


@dataclass
class FormulaModifiedStats:
    cache_generation_attempts: int = 0
    cache_generation_successes: int = 0
    cache_generation_failures: int = 0
    cache_generation_latency_sum: float = 0.0
    cache_generation_latencies: List[float] = field(default_factory=list)
    cache_generation_enqueued: int = 0
    cache_generation_skips: int = 0
    extractor_calls: int = 0
    extractor_successes: int = 0
    extractor_failures: int = 0
    extractor_latency_sum: float = 0.0
    program_successes: int = 0
    program_failures: int = 0
    fallback_serial_wait_sum: float = 0.0
    fallback_serial_waits: List[float] = field(default_factory=list)
    background_running: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class FormulaModifiedRuntime:
    def __init__(self, mode: str, task_type: str, workspace_dir: Path, default_model: str):
        if mode != "formula_modified_parallel" or task_type != "formula":
            raise ValueError("FormulaModifiedRuntime only supports mode=formula_modified_parallel and task_type=formula")
        self.mode = mode
        self.task_type = task_type
        self.workspace_dir = workspace_dir
        self.default_model = default_model
        self.root_dir = workspace_dir / mode
        self.cache_dir = self.root_dir / "cache"
        self.logs_dir = self.root_dir / "logs"
        self.results_dir = self.root_dir / "results"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.stats = ServiceStats()
        self.modified_stats = FormulaModifiedStats()
        self.trace_path = self.logs_dir / "request_trace.jsonl"
        self.stats_path = self.root_dir / "service_stats.json"
        self.service_log_path = self.logs_dir / "service.log"
        self.metric_path = self.results_dir / "metric.json"
        self.results_path = self.results_dir / "results.json"
        self.global_cache_path = self.cache_dir / "global_cache.json"
        self.database_path = self.cache_dir / "formula_groups.json"
        self.trace_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.state_lock = threading.RLock()

        strong_base_url = (
            os.getenv("GENCACHE_STRONG_BASE_URL")
            or os.getenv("GENCACHE_FALLBACK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "http://127.0.0.1:9001/v1"
        )
        extractor_base_url = os.getenv("GENCACHE_EXTRACTOR_BASE_URL") or "http://127.0.0.1:9002/v1"
        extractor_base_urls = [
            item.strip()
            for item in (os.getenv("GENCACHE_EXTRACTOR_BASE_URLS") or extractor_base_url).split(",")
            if item.strip()
        ]
        api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
        timeout = float(os.getenv("GENCACHE_FALLBACK_TIMEOUT", "240"))
        self.strong_client = OpenAI(api_key=os.getenv("GENCACHE_STRONG_API_KEY") or api_key, base_url=strong_base_url, timeout=timeout)
        self.extractor_clients = [
            OpenAI(api_key=os.getenv("GENCACHE_EXTRACTOR_API_KEY") or api_key, base_url=url, timeout=timeout)
            for url in extractor_base_urls
        ]
        self.extractor_rr_lock = threading.Lock()
        self.extractor_rr_next = 0
        self.strong_base_url = strong_base_url
        self.extractor_base_url = extractor_base_urls[0]
        self.extractor_base_urls = extractor_base_urls
        self.strong_model = os.getenv("GENCACHE_STRONG_MODEL", default_model)
        self.extractor_model = os.getenv("GENCACHE_EXTRACTOR_MODEL", "qwen3-1.7B")
        self.enable_thinking = env_bool("GENCACHE_FALLBACK_ENABLE_THINKING", False)
        self.group_trigger_size = int(os.getenv("GENCACHE_FORMULA_GROUP_TRIGGER_SIZE", "3"))
        self.validation_size = int(os.getenv("GENCACHE_FORMULA_VALIDATION_SIZE", "3"))
        self.validation_pass = int(os.getenv("GENCACHE_FORMULA_VALIDATION_PASS", "2"))
        self.regeneration_fail_threshold = int(os.getenv("GENCACHE_FORMULA_REGEN_FAIL_THRESHOLD", "3"))
        self.async_cache_gen = env_bool("GENCACHE_FORMULA_ASYNC_CACHE_GEN", True)
        self.readonly_cache = env_bool("GENCACHE_FORMULA_READONLY_CACHE", False)
        self.codegen_fallback = env_bool("GENCACHE_CODEGEN_FALLBACK", True)
        self.extractor_semaphore = threading.Semaphore(int(os.getenv("GENCACHE_FORMULA_EXTRACTOR_CONCURRENCY", "32")))
        self.strong_concurrency = int(os.getenv("GENCACHE_FORMULA_STRONG_CONCURRENCY", "1"))
        self.strong_lock = threading.Semaphore(self.strong_concurrency)
        self.background_pool = ThreadPoolExecutor(max_workers=int(os.getenv("GENCACHE_FORMULA_CACHE_GEN_WORKERS", "1")))
        self.groups: Dict[str, List[FormulaSample]] = {}
        self.caches: Dict[str, FormulaCacheEntry] = {}
        self.inflight_generations: set[str] = set()

        self._log(
            "INFO",
            (
                f"starting FormulaModifiedRuntime workspace={workspace_dir} strong={self.strong_model}@{strong_base_url} "
                f"extractor={self.extractor_model}@{extractor_base_urls} async_cache_gen={self.async_cache_gen} "
                f"readonly_cache={self.readonly_cache} codegen_fallback={self.codegen_fallback}"
            ),
        )
        self._load_warm_cache_from_env()
        self._write_stats()

    def _log(self, level: str, message: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with self.log_lock:
            with self.service_log_path.open("a", encoding="utf-8") as f:
                f.write(f"{ts} - {level} - {message}\n")

    def cache_entries(self) -> int:
        with self.state_lock:
            return len(self.caches)

    def admin_stats(self) -> Dict[str, Any]:
        return self._snapshot_stats()

    def _snapshot_stats(self) -> Dict[str, Any]:
        base = self.stats.snapshot(self.cache_entries(), self.mode, self.task_type, self.workspace_dir)
        with self.modified_stats.lock:
            attempts = self.modified_stats.cache_generation_attempts
            successes = self.modified_stats.cache_generation_successes
            extractor_calls = self.modified_stats.extractor_calls
            base.update(
                {
                    "cache_generation_attempts": attempts,
                    "cache_generation_successes": successes,
                    "cache_generation_failures": self.modified_stats.cache_generation_failures,
                    "cache_generation_success_rate": successes / attempts if attempts else 0.0,
                    "avg_cache_generation_latency_sec": (
                        self.modified_stats.cache_generation_latency_sum / attempts if attempts else 0.0
                    ),
                    "cache_generation_enqueued": self.modified_stats.cache_generation_enqueued,
                    "cache_generation_skips": self.modified_stats.cache_generation_skips,
                    "extractor_calls": extractor_calls,
                    "extractor_successes": self.modified_stats.extractor_successes,
                    "extractor_failures": self.modified_stats.extractor_failures,
                    "avg_extractor_latency_sec": (
                        self.modified_stats.extractor_latency_sum / extractor_calls if extractor_calls else 0.0
                    ),
                    "program_successes": self.modified_stats.program_successes,
                    "program_failures": self.modified_stats.program_failures,
                    "fallback_serial_wait_sum_sec": self.modified_stats.fallback_serial_wait_sum,
                    "avg_fallback_serial_wait_sec": (
                        self.modified_stats.fallback_serial_wait_sum / len(self.modified_stats.fallback_serial_waits)
                        if self.modified_stats.fallback_serial_waits
                        else 0.0
                    ),
                    "background_running": self.modified_stats.background_running,
                    "cluster_count": len(self.groups),
                    "strong_base_url": self.strong_base_url,
                    "extractor_base_url": self.extractor_base_url,
                    "extractor_base_urls": self.extractor_base_urls,
                    "strong_model": self.strong_model,
                    "extractor_model": self.extractor_model,
                    "readonly_cache": self.readonly_cache,
                }
            )
        return base

    def _load_warm_cache_from_env(self) -> None:
        path = os.getenv("GENCACHE_FORMULA_WARM_CACHE_PATH")
        if not path:
            return
        loaded = self.load_warm_cache(Path(path))
        self._log("INFO", f"loaded warm cache entries={loaded} path={path}")

    def load_warm_cache(self, path: Path) -> int:
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded = 0
        with self.state_lock:
            for cache_id, raw in payload.items():
                spec = raw.get("spec") or {}
                group_key = str(raw.get("group_key") or spec.get("cluster_summary") or "").strip()
                slot_spec = spec.get("slot_spec") or {}
                program_code = spec.get("program_code") or ""
                if not group_key or not isinstance(slot_spec, dict) or not program_code:
                    continue
                entry = FormulaCacheEntry(
                    group_key=group_key,
                    slot_spec={str(k): str(v) for k, v in slot_spec.items()},
                    program_code=str(program_code),
                    created_at=float(raw.get("created_at") or time.time()),
                    generation_samples=int(raw.get("generation_samples") or 0),
                    validation=raw.get("validation") or spec.get("_program_validation") or {},
                    cache_id=str(raw.get("cache_id") or cache_id or stable_cache_id(group_key)),
                    fail_count=int(raw.get("fail_count") or 0),
                    hit_count=int(raw.get("hit_count") or 0),
                )
                self.caches[group_key] = entry
                loaded += 1
        return loaded

    def _write_stats(self) -> None:
        with self.write_lock:
            self.stats_path.write_text(json.dumps(self._snapshot_stats(), ensure_ascii=False, indent=2), encoding="utf-8")
            self.metric_path.write_text(json.dumps(self.stats.metrics_snapshot(), ensure_ascii=False, indent=2), encoding="utf-8")
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
            with self.state_lock:
                cache_payload = {entry.cache_id: entry.to_json() for entry in self.caches.values()}
                groups_payload = {
                    key: [
                        {
                            "answer": sample.answer,
                            "ts": sample.ts,
                            "prompt": request_text(sample.request),
                        }
                        for sample in samples[-20:]
                    ]
                    for key, samples in self.groups.items()
                }
            suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
            tmp_cache = self.global_cache_path.with_name(self.global_cache_path.name + suffix)
            tmp_cache.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_cache, self.global_cache_path)
            tmp_db = self.database_path.with_name(self.database_path.name + suffix)
            tmp_db.write_text(json.dumps(groups_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_db, self.database_path)

    def _trace(self, payload: Dict[str, Any]) -> None:
        with self.trace_lock:
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def reset(self) -> None:
        with self.state_lock:
            self.groups.clear()
            self.caches.clear()
            self.inflight_generations.clear()
        self.stats = ServiceStats()
        self.modified_stats = FormulaModifiedStats()
        if self.trace_path.exists():
            self.trace_path.unlink()
        self._log("INFO", "admin reset")
        self._write_stats()

    def _extra_body(self, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        extra: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": self.enable_thinking}}
        if max_tokens is not None:
            extra["max_tokens"] = max_tokens
        return extra

    def _call_strong(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        wait_started = time.time()
        with self.strong_lock:
            wait = time.time() - wait_started
            started = time.time()
            resp = self.strong_client.chat.completions.create(
                model=self.strong_model,
                messages=messages,
                temperature=0,
                top_p=1,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
        content = resp.choices[0].message.content or ""
        meta = {
            "fallback_model": self.strong_model,
            "fallback_base_url": self.strong_base_url,
            "fallback_latency_sec": time.time() - started,
            "fallback_serial_wait_sec": wait,
        }
        with self.modified_stats.lock:
            self.modified_stats.fallback_serial_wait_sum += wait
            self.modified_stats.fallback_serial_waits.append(wait)
        return content, usage_dict(resp.usage), meta

    def _fallback(self, request: Dict[str, Any]) -> Tuple[str, Dict[str, int], Dict[str, Any]]:
        if not self.codegen_fallback:
            messages = request.get("messages") or []
            return self._call_strong(messages, max_tokens=int(request.get("max_tokens") or os.getenv("GENCACHE_STRONG_MAX_TOKENS", "1024")))
        started = time.time()
        raw, usage, meta = self._call_strong(build_codegen_messages(request.get("messages") or []), max_tokens=2048)
        code = extract_code(raw)
        if not code:
            meta.update({"codegen_error": "no_code_block", "raw_output": raw[:500], "codegen_total_sec": time.time() - started})
            return raw, usage, meta
        result, error, exec_latency = execute_code(code)
        meta.update(
            {
                "codegen_error": error,
                "codegen_exec_latency_sec": exec_latency,
                "codegen_total_sec": time.time() - started,
                "code": code[:1000],
            }
        )
        if error:
            return raw, usage, meta
        return f"[[{result}]]", usage, meta

    def _call_extractor(self, entry: FormulaCacheEntry, request: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        started = time.time()
        extractor_client, extractor_base_url = self._next_extractor_client()
        prompt = [
            {
                "role": "system",
                "content": (
                    "You extract variables from formula questions. Return ONLY a JSON object. "
                    "Use the exact keys from the slot spec. Values should be short strings copied or normalized from the prompt."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Formula group: {entry.group_key}\n"
                    f"Slot spec JSON:\n{json.dumps(entry.slot_spec, ensure_ascii=False)}\n\n"
                    f"Prompt:\n{request_text(request)}\n\nReturn JSON only."
                ),
            },
        ]
        with self.extractor_semaphore:
            resp = extractor_client.chat.completions.create(
                model=self.extractor_model,
                messages=prompt,
                temperature=0,
                top_p=1,
                max_tokens=int(os.getenv("GENCACHE_EXTRACTOR_MAX_TOKENS", "512")),
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
        latency = time.time() - started
        raw = resp.choices[0].message.content or ""
        stage = {
            "extractor_latency_sec": latency,
            "extractor_model": self.extractor_model,
            "extractor_base_url": extractor_base_url,
            "extractor_raw": raw[:1000],
        }
        with self.modified_stats.lock:
            self.modified_stats.extractor_calls += 1
            self.modified_stats.extractor_latency_sum += latency
        try:
            slots = compact_json(raw)
        except Exception as exc:
            with self.modified_stats.lock:
                self.modified_stats.extractor_failures += 1
            stage["extractor_error"] = repr(exc)
            return None, stage
        missing = [key for key in entry.slot_spec if key not in slots or slots.get(key) in (None, "")]
        if missing:
            with self.modified_stats.lock:
                self.modified_stats.extractor_failures += 1
            stage["extractor_error"] = f"missing_slots: {missing}"
            stage["slots"] = slots
            return None, stage
        with self.modified_stats.lock:
            self.modified_stats.extractor_successes += 1
        stage["slots"] = slots
        return slots, stage

    def _next_extractor_client(self) -> Tuple[OpenAI, str]:
        with self.extractor_rr_lock:
            idx = self.extractor_rr_next % len(self.extractor_clients)
            self.extractor_rr_next += 1
        return self.extractor_clients[idx], self.extractor_base_urls[idx]

    def _generate_cache_prompt(self, group_key: str, samples: List[FormulaSample]) -> List[Dict[str, str]]:
        examples = []
        for idx, sample in enumerate(samples, 1):
            examples.append(
                f"Example {idx}\nPrompt:\n{request_text(sample.request)}\nAnswer:\n{sample.answer}"
            )
        return [
            {
                "role": "system",
                "content": (
                    "You build a reusable program cache for a group of formula QA prompts. "
                    "Return ONLY JSON with keys slot_spec and program_code.\n"
                    "slot_spec must map variable names to extraction instructions. "
                    "program_code must define solve(slots) and return the final answer in [[VALUE]] format. "
                    "The code must not import modules or read files."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Formula group: {group_key}\n\n"
                    + "\n\n".join(examples)
                    + "\n\nCreate a general extractor slot spec and Python solve(slots) program for this formula group."
                ),
            },
        ]

    def _examples_text(self, samples: List[FormulaSample]) -> str:
        return "\n\n".join(
            f"Example {idx}\nPrompt:\n{request_text(sample.request)}\nAnswer:\n{sample.answer}"
            for idx, sample in enumerate(samples, 1)
        )

    def _generate_slot_spec_prompt(self, group_key: str, samples: List[FormulaSample]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You design variable extraction specs for formula QA. Return ONLY JSON with one key: slot_spec. "
                    "slot_spec must map variable names to short extraction instructions."
                ),
            },
            {
                "role": "user",
                "content": f"Formula group: {group_key}\n\n{self._examples_text(samples)}\n\nReturn slot_spec JSON only.",
            },
        ]

    def _generate_program_prompt(self, group_key: str, slot_spec: Dict[str, str], samples: List[FormulaSample]) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You write reusable Python formula programs. Return ONLY a ```python code block. "
                    "The code must define solve(slots). It must not import modules or read files. "
                    "It should convert string slot values like '$1,000', '32%', '1.5 million' to numbers when needed. "
                    "Return the final answer in [[VALUE]] format."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Formula group: {group_key}\n"
                    f"Slot spec:\n{json.dumps(slot_spec, ensure_ascii=False, indent=2)}\n\n"
                    f"{self._examples_text(samples)}\n\nWrite solve(slots)."
                ),
            },
        ]

    def _validate_entry(self, entry: FormulaCacheEntry, samples: List[FormulaSample]) -> Dict[str, Any]:
        checked = valid = 0
        failures: List[Dict[str, Any]] = []
        for sample in samples:
            checked += 1
            slots, extract_stage = self._call_extractor(entry, sample.request)
            if slots is None:
                failures.append({"answer": sample.answer, "error": extract_stage.get("extractor_error")})
                continue
            answer, error, exec_latency = run_solve_program(entry.program_code, slots)
            if error or not answers_match(sample.answer, answer):
                failures.append(
                    {
                        "answer": sample.answer,
                        "predicted": answer,
                        "error": error,
                        "slots": slots,
                        "exec_latency_sec": exec_latency,
                    }
                )
                continue
            valid += 1
        return {"checked": checked, "valid": valid, "failed_samples": failures[:5]}

    def _build_cache_entry(self, group_key: str, samples: List[FormulaSample]) -> FormulaCacheEntry:
        raw, _, _ = self._call_strong(self._generate_cache_prompt(group_key, samples), max_tokens=2048)
        try:
            spec = compact_json(raw)
            slot_spec = spec.get("slot_spec")
            program_code = spec.get("program_code") or extract_code(str(spec.get("code") or raw))
        except Exception as exc:
            self._log("INFO", f"combined cache spec parse failed group={group_key}: {exc!r}; retrying two-stage generation")
            slot_raw, _, _ = self._call_strong(self._generate_slot_spec_prompt(group_key, samples), max_tokens=1024)
            slot_spec = compact_json(slot_raw).get("slot_spec")
            if not isinstance(slot_spec, dict) or not slot_spec:
                raise ValueError("slot_spec generation failed after combined parse failure")
            code_raw, _, _ = self._call_strong(self._generate_program_prompt(group_key, slot_spec, samples), max_tokens=2048)
            program_code = extract_code(code_raw) or code_raw.strip()
        if not isinstance(slot_spec, dict) or not slot_spec or not program_code:
            raise ValueError("cache generation did not return slot_spec and program_code")
        return FormulaCacheEntry(
            group_key=group_key,
            slot_spec={str(k): str(v) for k, v in slot_spec.items()},
            program_code=str(program_code),
            created_at=time.time(),
            generation_samples=len(samples),
            validation={},
            cache_id=stable_cache_id(group_key),
        )

    def _run_cache_generation(self, group_key: str) -> None:
        started = time.time()
        with self.modified_stats.lock:
            self.modified_stats.cache_generation_attempts += 1
            self.modified_stats.background_running += 1
        try:
            with self.state_lock:
                samples = list(self.groups.get(group_key, []))
            if len(samples) < self.group_trigger_size:
                raise ValueError("not enough samples")
            gen_samples = random.sample(samples, min(self.group_trigger_size, len(samples)))
            val_samples = random.sample(samples, min(self.validation_size, len(samples)))
            entry = self._build_cache_entry(group_key, gen_samples)
            validation = self._validate_entry(entry, val_samples)
            entry.validation = validation
            if validation.get("valid", 0) < min(self.validation_pass, validation.get("checked", 0)):
                raise ValueError(f"validation failed: {validation}")
            with self.state_lock:
                self.caches[group_key] = entry
                self.inflight_generations.discard(group_key)
            latency = time.time() - started
            with self.modified_stats.lock:
                self.modified_stats.cache_generation_successes += 1
                self.modified_stats.cache_generation_latency_sum += latency
                self.modified_stats.cache_generation_latencies.append(latency)
            self._log("INFO", f"cache generation success group={group_key} latency={latency:.3f}s valid={validation.get('valid')}/{validation.get('checked')}")
        except Exception as exc:
            latency = time.time() - started
            with self.state_lock:
                self.inflight_generations.discard(group_key)
            with self.modified_stats.lock:
                self.modified_stats.cache_generation_failures += 1
                self.modified_stats.cache_generation_latency_sum += latency
                self.modified_stats.cache_generation_latencies.append(latency)
            self._log("ERROR", f"cache generation failed group={group_key}: {exc!r}")
        finally:
            with self.modified_stats.lock:
                self.modified_stats.background_running -= 1
            self._write_stats()

    def _maybe_enqueue_generation(self, group_key: str) -> bool:
        with self.state_lock:
            if group_key in self.inflight_generations:
                with self.modified_stats.lock:
                    self.modified_stats.cache_generation_skips += 1
                return False
            sample_count = len(self.groups.get(group_key, []))
            if sample_count < self.group_trigger_size:
                return False
            self.inflight_generations.add(group_key)
        with self.modified_stats.lock:
            self.modified_stats.cache_generation_enqueued += 1
        self._log("INFO", f"cache generation enqueue group={group_key}")
        if self.async_cache_gen:
            self.background_pool.submit(self._run_cache_generation, group_key)
        else:
            self._run_cache_generation(group_key)
        return True

    def _record_sample(self, group_key: str, request: Dict[str, Any], answer: str) -> None:
        with self.state_lock:
            self.groups.setdefault(group_key, []).append(FormulaSample(request=dict(request), answer=answer, group_key=group_key))
            cache = self.caches.get(group_key)
            should_regen = bool(cache and cache.fail_count >= self.regeneration_fail_threshold)
            if should_regen:
                cache.fail_count = 0
        if self.readonly_cache:
            return
        if should_regen or group_key not in self.caches:
            self._maybe_enqueue_generation(group_key)

    def predict(self, request: Dict[str, Any]) -> Dict[str, Any]:
        started = time.time()
        group_key = formula_name_from_request(request)
        use_cache = bool(request.get("use_cache", True))
        pretrain = bool(request.get("pretrain", False))
        test_mode = bool(request.get("test_mode", False))
        ground_truth = request.get("ground_truth")
        cache_hit = False
        cached_key = None
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        stage: Dict[str, Any] = {"group_key": group_key}

        with self.stats.lock:
            self.stats.total_requests += 1

        answer = ""
        try:
            entry = None
            with self.state_lock:
                entry = self.caches.get(group_key)
            if use_cache and entry is not None:
                slots, extract_stage = self._call_extractor(entry, request)
                stage.update(extract_stage)
                if slots is not None:
                    predicted, error, exec_latency = run_solve_program(entry.program_code, slots)
                    stage["program_exec_latency_sec"] = exec_latency
                    if error is None and predicted:
                        answer = predicted
                        cache_hit = True
                        cached_key = entry.cache_id
                        with self.state_lock:
                            entry.hit_count += 1
                        with self.stats.lock:
                            self.stats.cache_hits += 1
                        with self.modified_stats.lock:
                            self.modified_stats.program_successes += 1
                    else:
                        stage["program_error"] = error
                        with self.state_lock:
                            entry.fail_count += 1
                        with self.modified_stats.lock:
                            self.modified_stats.program_failures += 1
                if not cache_hit:
                    self._log("INFO", f"cache ineffective group={group_key} reason={stage.get('extractor_error') or stage.get('program_error')}")
            if not cache_hit:
                with self.stats.lock:
                    self.stats.cache_misses += 1
                    self.stats.fallback_calls += 1
                    if self.codegen_fallback:
                        self.stats.codegen_calls += 1
                answer, usage, fallback_stage = self._fallback(request)
                stage.update(fallback_stage)
                if self.codegen_fallback:
                    with self.stats.lock:
                        if stage.get("codegen_error"):
                            self.stats.codegen_failures += 1
                        else:
                            self.stats.codegen_successes += 1
                fallback_latency = float(stage.get("fallback_latency_sec", 0.0) or 0.0)
                with self.stats.lock:
                    self.stats.fallback_latency_sum += fallback_latency
                    self.stats.fallback_latencies.append(fallback_latency)
                if use_cache and not test_mode and not self.readonly_cache:
                    self._record_sample(group_key, request, answer)
        except Exception as exc:
            stage["error"] = repr(exc)
            self._log("ERROR", f"request failed group={group_key}: {exc!r}")

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
        self._trace(
            {
                "ts": time.time(),
                "mode": self.mode,
                "variant": "modified",
                "task_type": self.task_type,
                "workspace_dir": str(self.workspace_dir),
                "model": request.get("model") or self.default_model,
                "group_key": group_key,
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
                "stage_timings": stage,
                "messages": request.get("messages") or [],
            }
        )
        self._log(
            "INFO",
            f"request mode={self.mode} task=formula group={group_key} hit={cache_hit} latency={latency:.4f}s key={cached_key}",
        )
        return {
            "answer": answer,
            "function_call": None,
            "usage": usage,
            "cache_hit": cache_hit,
            "cached_key": cached_key,
            "pretrain": pretrain,
            "latency_sec": latency,
            "stage_timings": stage,
        }

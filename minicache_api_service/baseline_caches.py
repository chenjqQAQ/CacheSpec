from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def build_prompt_object(request: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("messages", "functions", "function_call", "tools", "tool_choice", "response_format")
    return {key: request.get(key) for key in keys if request.get(key) is not None}


def full_prompt_text(request: Dict[str, Any]) -> str:
    return stable_json(build_prompt_object(request))


def prompt_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class ExactCacheBaseline:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.lock = threading.Lock()
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                self.entries = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = {}

    def lookup(self, request: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.time()
        prompt = full_prompt_text(request)
        key = prompt_sha256(prompt)
        with self.lock:
            entry = self.entries.get(key)
        return entry, {"exact_cache_key": key, "cache_lookup_sec": time.time() - t0}

    def store(self, request: Dict[str, Any], answer: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        prompt = full_prompt_text(request)
        key = prompt_sha256(prompt)
        entry = {
            "key": key,
            "prompt": prompt,
            "answer": answer,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        with self.lock:
            self.entries[key] = entry
            _atomic_write_json(self.cache_path, self.entries)
        return key

    def reset(self) -> None:
        with self.lock:
            self.entries = {}
            _atomic_write_json(self.cache_path, self.entries)

    def count(self) -> int:
        with self.lock:
            return len(self.entries)


class HashEmbeddingBackend:
    name = "hash_fallback"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class SentenceTransformerBackend:
    name = "sentence_transformers"

    def __init__(self, model_path: str):
        from sentence_transformers import SentenceTransformer

        self.model_path = model_path
        self.model = SentenceTransformer(model_path)

    def encode(self, text: str) -> np.ndarray:
        vec = np.asarray(self.model.encode(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


def load_embedding_backend() -> Any:
    model_path = os.getenv("GENCACHE_GPTCACHE_EMBEDDING_MODEL") or os.getenv("SENTENCE_TRANSFORMER_MODEL")
    if model_path:
        try:
            return SentenceTransformerBackend(model_path)
        except Exception as exc:
            print(f"[MiniCache] sentence-transformer backend failed, using hash fallback: {exc}", flush=True)
    return HashEmbeddingBackend()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class GptCacheBaseline:
    def __init__(self, cache_path: Path, threshold: float = 0.95):
        self.cache_path = cache_path
        self.threshold = threshold
        self.lock = threading.Lock()
        self.backend = load_embedding_backend()
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                self.entries = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.entries = []

    def lookup(self, request: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.time()
        prompt = full_prompt_text(request)
        vec = self.backend.encode(prompt)
        best_entry = None
        best_similarity = -math.inf
        with self.lock:
            entries = list(self.entries)
        for entry in entries:
            emb = np.asarray(entry.get("embedding", []), dtype=np.float32)
            if emb.size == 0:
                continue
            score = cosine(vec, emb)
            if score > best_similarity:
                best_similarity = score
                best_entry = entry
        hit = best_entry if best_entry is not None and best_similarity >= self.threshold else None
        return hit, {
            "best_similarity": 0.0 if best_similarity == -math.inf else best_similarity,
            "similarity_threshold": self.threshold,
            "embedding_backend": self.backend.name,
            "cache_lookup_sec": time.time() - t0,
        }

    def store(self, request: Dict[str, Any], answer: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        prompt = full_prompt_text(request)
        key = prompt_sha256(prompt)
        emb = self.backend.encode(prompt)
        entry = {
            "key": key,
            "prompt": prompt,
            "embedding": emb.astype(float).tolist(),
            "answer": answer,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        with self.lock:
            for i, old in enumerate(self.entries):
                if old.get("key") == key:
                    self.entries[i] = entry
                    break
            else:
                self.entries.append(entry)
            _atomic_write_json(self.cache_path, self.entries)
        return key

    def reset(self) -> None:
        with self.lock:
            self.entries = []
            _atomic_write_json(self.cache_path, self.entries)

    def count(self) -> int:
        with self.lock:
            return len(self.entries)


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or item))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(content or "")
    return str(message)


def task_group_key(request: Dict[str, Any], task_type: str) -> str:
    for key in ("group_key", "_group_key", "formula_name", "_formula_name", "task_name"):
        value = request.get(key)
        if value:
            return f"{task_type}:{value}"
    messages = request.get("messages") or []
    text = "\n".join(_message_text(m) for m in messages)
    if task_type == "formula":
        patterns = [
            r"formula_name\s*[:=]\s*([^\n,;]+)",
            r"_formula_name\s*[:=]\s*([^\n,;]+)",
            r"Use formula\s+(.+?)\s+to answer",
            r"Formula\s*:\s*([^\n,;]+)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return f"{task_type}:{match.group(1).strip()}"
    if task_type == "webshop":
        functions = request.get("functions") or request.get("tools") or []
        if functions:
            try:
                names = []
                for item in functions:
                    fn = item.get("function", item) if isinstance(item, dict) else {}
                    names.append(str(fn.get("name") or "unknown"))
                return f"{task_type}:functions:{','.join(names)}"
            except Exception:
                pass
    digest = prompt_sha256(full_prompt_text(request))[:16]
    return f"{task_type}:prompt:{digest}"


class RuleProgramCacheBaseline:
    """Compatibility cache for original/modified modes.

    This keeps the old GenCache file shape alive for experiments that need
    original/modified service directories. It intentionally uses conservative
    grouping and stores reusable answers only after a group has enough samples.
    """

    def __init__(
        self,
        cache_path: Path,
        database_path: Path,
        task_type: str,
        mode: str,
        num_records_before_caching: int = 3,
    ):
        self.cache_path = cache_path
        self.database_path = database_path
        self.task_type = task_type
        self.mode = mode
        self.num_records_before_caching = num_records_before_caching
        self.lock = threading.Lock()
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.database: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_generation_attempts = 0
        self.cache_generation_successes = 0
        self.cache_generation_failures = 0
        self.cache_generation_latency_sum = 0.0
        self._load()

    def _load(self) -> None:
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        if self.database_path.exists():
            try:
                self.database = json.loads(self.database_path.read_text(encoding="utf-8"))
            except Exception:
                self.database = {}

    def _write(self) -> None:
        _atomic_write_json(self.cache_path, self.cache)
        _atomic_write_json(self.database_path, self.database)

    def _entry_for_group(self, group_key: str) -> Optional[Dict[str, Any]]:
        cache_id = prompt_sha256(group_key)[:16]
        entry = self.cache.get(cache_id)
        if entry is not None:
            entry = dict(entry)
            entry["key"] = cache_id
        return entry

    def lookup(self, request: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        t0 = time.time()
        group_key = task_group_key(request, self.task_type)
        with self.lock:
            entry = self._entry_for_group(group_key)
        return entry, {
            "group_key": group_key,
            "cache_lookup_sec": time.time() - t0,
            "compat_cache_mode": self.mode,
        }

    def store(self, request: Dict[str, Any], answer: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        group_key = task_group_key(request, self.task_type)
        cache_id = prompt_sha256(group_key)[:16]
        prompt = full_prompt_text(request)
        sample = {
            "prompt": prompt,
            "answer": answer,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        with self.lock:
            samples = self.database.setdefault(group_key, [])
            samples.append(sample)
            if cache_id not in self.cache and len(samples) >= self.num_records_before_caching:
                t0 = time.time()
                self.cache_generation_attempts += 1
                try:
                    self.cache[cache_id] = {
                        "key": cache_id,
                        "group_key": group_key,
                        "answer": answer,
                        "count": len(samples),
                        "regex": "compatibility_group_key",
                        "plan": "compatibility_replay_answer",
                        "metadata": {
                            "mode": self.mode,
                            "task_type": self.task_type,
                            "generated_by": "MiniCache RuleProgramCacheBaseline",
                        },
                        "created_at": time.time(),
                    }
                    self.cache_generation_successes += 1
                except Exception:
                    self.cache_generation_failures += 1
                    raise
                finally:
                    self.cache_generation_latency_sum += time.time() - t0
            self._write()
        return cache_id

    def reset(self) -> None:
        with self.lock:
            self.cache = {}
            self.database = {}
            self.cache_generation_attempts = 0
            self.cache_generation_successes = 0
            self.cache_generation_failures = 0
            self.cache_generation_latency_sum = 0.0
            self._write()

    def count(self) -> int:
        with self.lock:
            return len(self.cache)

    def generation_stats(self) -> Dict[str, Any]:
        with self.lock:
            attempts = self.cache_generation_attempts
            successes = self.cache_generation_successes
            return {
                "cache_generation_attempts": attempts,
                "cache_generation_successes": successes,
                "cache_generation_failures": self.cache_generation_failures,
                "cache_generation_success_rate": successes / attempts if attempts else 0.0,
                "avg_cache_generation_latency_sec": self.cache_generation_latency_sum / attempts if attempts else 0.0,
                "cluster_count": len(self.database),
            }

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


SYSTEM_PROMPT = """You are an expert at checking whether two WebShop Search API calls are semantically equivalent for the same shopping instruction.

The instruction is a user request to buy an item under a price limit. You will compare:
- Ground Truth Phrase: the reference Search API call
- Algorithm Phrase: the model output Search API call

Judge whether the Algorithm Phrase preserves the important item attributes from the instruction. Different word order or synonyms are acceptable. Do not require exact wording. Ignore minor formatting differences. Do not over-penalize keyword order. The max_price should be compatible with the instruction.

Answer only JSON:
{"same_answer": true/false, "reason": "short reason"}
"""


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_bool(raw: str) -> Optional[bool]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("same_answer"), bool):
            return bool(data["same_answer"])
    except Exception:
        pass
    lowered = raw.lower()
    if re.search(r'"same_answer"\s*:\s*true', lowered) or re.search(r"\byes\b", lowered):
        return True
    if re.search(r'"same_answer"\s*:\s*false', lowered) or re.search(r"\bno\b", lowered):
        return False
    return None


def make_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def judge_one(
    idx: int,
    row: Dict[str, Any],
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> Dict[str, Any]:
    prompt = (
        f"Instruction:\n{row.get('instruction', '')}\n\n"
        f"Ground Truth Phrase:\n{row.get('actual_response', '')}\n\n"
        f"Algorithm Phrase:\n{row.get('llm_response', '')}\n"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            client = make_client(base_url, api_key, timeout)
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                top_p=1,
                max_tokens=128,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = resp.choices[0].message.content or ""
            same = parse_bool(raw)
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else {}
            return {
                "index": idx,
                "instruction": row.get("instruction", ""),
                "ground_truth": row.get("actual_response", ""),
                "llm_response": row.get("llm_response", ""),
                "cache_hit": row.get("cache_hit"),
                "same_answer": same,
                "judge_raw": raw,
                "judge_model": model,
                "judge_base_url": base_url,
                "usage": usage,
            }
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(min(2 ** attempt, 8))
    return {
        "index": idx,
        "instruction": row.get("instruction", ""),
        "ground_truth": row.get("actual_response", ""),
        "llm_response": row.get("llm_response", ""),
        "cache_hit": row.get("cache_hit"),
        "same_answer": None,
        "judge_raw": "",
        "judge_model": model,
        "judge_base_url": base_url,
        "error": last_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--base-urls", required=True, help="Comma-separated OpenAI-compatible /v1 base URLs.")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3-32B-FP8-spec")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    base_urls = [x.strip().rstrip("/") for x in args.base_urls.split(",") if x.strip()]
    rows = read_jsonl(input_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    done = set()
    if args.resume and output_path.exists():
        for item in read_jsonl(output_path):
            done.add(int(item.get("index", -1)))
    elif output_path.exists():
        output_path.unlink()

    lock = threading.Lock()
    tasks = [(i, row) for i, row in enumerate(rows) if i not in done]
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for pos, (idx, row) in enumerate(tasks):
            base_url = base_urls[pos % len(base_urls)]
            futures.append(pool.submit(judge_one, idx, row, base_url, args.api_key, args.model, args.timeout, args.max_retries))
        for future in as_completed(futures):
            append_jsonl(output_path, future.result(), lock)

    judged = read_jsonl(output_path)
    valid = [x for x in judged if x.get("same_answer") is not None]
    correct = [x for x in valid if x.get("same_answer") is True]
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "total_rows": len(rows),
        "judged_rows": len(judged),
        "valid_judgements": len(valid),
        "same_answer_count": len(correct),
        "accuracy": len(correct) / len(valid) if valid else 0.0,
        "elapsed_sec": time.time() - started,
        "judge_model": args.model,
        "base_urls": base_urls,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

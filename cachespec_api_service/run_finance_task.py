from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI


def iter_jsonl(path: Path, start: int, limit: int) -> Iterable[Dict[str, Any]]:
    end = start + limit
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if idx >= end:
                break
            if line.strip():
                yield json.loads(line)


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", choices=["formula", "codetatqa", "codefinqa", "bizbench"], required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="qwen3-32b-fp8")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "cache_hit_responses.jsonl"
    summary_path = out_dir / "finance_eval_summary.json"
    client = OpenAI(api_key=args.api_key, base_url=args.base_url.rstrip("/"), timeout=180)

    processed = correct = failures = cache_hits = 0
    total_latency = 0.0
    started = time.time()

    with rows_path.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(Path(args.data_file), args.start, args.num_examples):
            messages: List[Dict[str, Any]] = row.get("messages") or row.get("prompt_action") or [
                {"role": "system", "content": "Return only the final answer in the format [[VALUE]]."},
                {"role": "user", "content": row.get("instruction") or row.get("question") or ""},
            ]
            expected = row.get("ground_truth") or row.get("response_action") or row.get("answer")
            t0 = time.time()
            response_text = ""
            err = None
            cache_hit = False
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    temperature=0,
                    top_p=1,
                    max_tokens=args.max_tokens,
                    extra_body={
                        "use_cache": args.use_cache,
                        "pretrain": args.pretrain,
                        "ground_truth": expected,
                        "test_mode": args.test_mode,
                    },
                )
                response_text = resp.choices[0].message.content or ""
                cache_hit = bool(getattr(resp, "cache_hit", False))
            except Exception as exc:
                failures += 1
                err = repr(exc)
            latency = time.time() - t0
            total_latency += latency
            is_correct = err is None and answers_match(expected, response_text)
            processed += 1
            correct += int(is_correct)
            cache_hits += int(cache_hit)
            out.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "task_type": args.task_type,
                        "expected": expected,
                        "response": response_text,
                        "correct": is_correct,
                        "cache_hit": cache_hit,
                        "latency_sec": latency,
                        "error": err,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    summary = {
        "task_type": args.task_type,
        "data_file": args.data_file,
        "processed": processed,
        "failures": failures,
        "correct": correct,
        "accuracy": correct / processed if processed else 0.0,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / processed if processed else 0.0,
        "avg_latency_sec": total_latency / processed if processed else 0.0,
        "total_time_sec": time.time() - started,
        "rows_file": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

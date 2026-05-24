from __future__ import annotations

import argparse
import json
import queue
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError


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


def post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"URL error: {exc}") from exc


def fetch_service_stats(url: str, timeout: float = 5.0) -> Dict[str, Any]:
    if not url:
        return {}
    try:
        req = urlrequest.Request(url, method="GET")
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def build_snapshot(counters: Dict[str, Any], started: float, current_round: int, service_stats: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    elapsed = now - started
    processed = counters["processed"]
    return {
        "ts": now,
        "elapsed_sec": elapsed,
        "round": current_round,
        "total_requests": processed,
        "correct": counters["correct"],
        "accuracy": counters["correct"] / processed if processed else 0.0,
        "failures": counters["failures"],
        "cache_hits": counters["cache_hits"],
        "cache_hit_rate": counters["cache_hits"] / processed if processed else 0.0,
        "avg_time_per_request": elapsed / processed if processed else 0.0,
        "throughput_eps": processed / elapsed if elapsed > 0 else 0.0,
        "avg_latency_sec": counters["latency_sum"] / processed if processed else 0.0,
        "avg_cache_hit_latency_sec": service_stats.get("avg_cache_hit_latency_sec", 0.0),
        "avg_cache_miss_latency_sec": service_stats.get("avg_cache_miss_latency_sec", 0.0),
        "cache_entries": service_stats.get("cache_entries", 0),
        "cache_generation_successes": service_stats.get("cache_generation_successes", 0),
        "cache_generation_attempts": service_stats.get("cache_generation_attempts", 0),
        "extractor_calls": service_stats.get("extractor_calls", 0),
        "avg_extractor_latency_sec": service_stats.get("avg_extractor_latency_sec", 0.0),
        "background_running": service_stats.get("background_running", 0),
    }


def append_snapshot(path: Path, snapshot: Dict[str, Any], lock: threading.Lock) -> None:
    line = json.dumps(snapshot, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


def time_snapshot_worker(
    path: Path,
    interval: float,
    counters: Dict[str, Any],
    started: float,
    round_ref: List[int],
    stats_url: str,
    counter_lock: threading.Lock,
    snapshot_lock: threading.Lock,
    stop_event: threading.Event,
) -> None:
    while not stop_event.wait(interval):
        with counter_lock:
            c_copy = dict(counters)
        if c_copy["processed"] == 0:
            continue
        svc = fetch_service_stats(stats_url)
        snap = build_snapshot(c_copy, started, round_ref[0], svc)
        append_snapshot(path, snap, snapshot_lock)


def build_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    return row.get("messages") or row.get("prompt_action") or [
        {"role": "system", "content": "Return only the final answer in the format [[VALUE]]."},
        {"role": "user", "content": row.get("instruction") or row.get("question") or ""},
    ]


def worker(
    work_q: "queue.Queue[Dict[str, Any]]",
    rows_path: Path,
    lock: threading.Lock,
    args: argparse.Namespace,
    counters: Dict[str, Any],
    snapshot_path: Path | None,
    snapshot_lock: threading.Lock | None,
    round_ref: List[int],
    started: float,
) -> None:
    url = args.base_url.rstrip("/") + "/chat/completions"
    while True:
        try:
            row = work_q.get_nowait()
        except queue.Empty:
            return
        expected = row.get("ground_truth") or row.get("response_action") or row.get("answer")
        payload = {
            "model": args.model,
            "messages": build_messages(row),
            "temperature": 0,
            "top_p": 1,
            "max_tokens": args.max_tokens,
            "use_cache": args.use_cache,
            "test_mode": args.test_mode,
            "ground_truth": expected,
            "extra_body": {
                "formula_name": row.get("_formula_name"),
                "enable_thinking": False,
            },
        }
        req_started = time.time()
        error = None
        response_text = ""
        cache_hit = False
        stage: Dict[str, Any] = {}
        try:
            resp = post_json(url, payload, timeout=args.timeout)
            response_text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            cache_hit = bool(resp.get("cache_hit"))
            stage = resp.get("stage_timings") or {}
        except Exception as exc:
            error = repr(exc)
        latency = time.time() - req_started
        correct = error is None and answers_match(expected, response_text)
        record = {
            "id": row.get("id"),
            "task_type": "formula",
            "formula_name": row.get("_formula_name") or stage.get("group_key"),
            "expected": expected,
            "response": response_text,
            "correct": correct,
            "cache_hit": cache_hit,
            "latency_sec": latency,
            "error": error,
            "stage_timings": stage,
        }
        do_snapshot = False
        with lock:
            counters["processed"] += 1
            counters["correct"] += int(correct)
            counters["failures"] += int(error is not None)
            counters["cache_hits"] += int(cache_hit)
            counters["latency_sum"] += latency
            with rows_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if counters["processed"] % max(1, args.progress_every) == 0:
                print(
                    f"[round {round_ref[0]}] processed={counters['processed']}/{counters['total']} "
                    f"acc={counters['correct']/counters['processed']:.4f} "
                    f"hit={counters['cache_hits']/counters['processed']:.4f} "
                    f"avg={counters['latency_sum']/counters['processed']:.4f}s",
                    flush=True,
                )
            if args.snapshot_every > 0 and counters["processed"] % args.snapshot_every == 0:
                do_snapshot = True
                c_copy = dict(counters)

        if do_snapshot and snapshot_path is not None and snapshot_lock is not None:
            svc = fetch_service_stats(args.service_stats_url)
            snap = build_snapshot(c_copy, started, round_ref[0], svc)
            append_snapshot(snapshot_path, snap, snapshot_lock)

        work_q.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3-32b-fp8")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--snapshot-interval", type=float, default=0)
    parser.add_argument("--service-stats-url", default="")
    parser.add_argument("--shuffle-seed", type=int, default=-1)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "cache_hit_responses.jsonl"
    summary_path = out_dir / "finance_eval_summary.json"
    request_snapshot_path = out_dir / "metrics_timeseries_by_request.jsonl"
    time_snapshot_path = out_dir / "metrics_timeseries_by_time.jsonl"

    if rows_path.exists():
        rows_path.unlink()
    if request_snapshot_path.exists():
        request_snapshot_path.unlink()
    if time_snapshot_path.exists():
        time_snapshot_path.unlink()

    all_rows = list(iter_jsonl(Path(args.data_file), args.start, args.num_examples))
    if args.shuffle_seed >= 0:
        random.seed(args.shuffle_seed)
        random.shuffle(all_rows)
        print(f"Shuffled {len(all_rows)} rows with seed={args.shuffle_seed}", flush=True)
    total_across_rounds = len(all_rows) * args.rounds

    counters = {
        "processed": 0,
        "correct": 0,
        "failures": 0,
        "cache_hits": 0,
        "latency_sum": 0.0,
        "total": total_across_rounds,
    }
    lock = threading.Lock()
    snapshot_lock = threading.Lock()
    round_ref = [1]
    started = time.time()

    stop_event = threading.Event()
    if args.snapshot_interval > 0:
        timer_thread = threading.Thread(
            target=time_snapshot_worker,
            args=(
                time_snapshot_path,
                args.snapshot_interval,
                counters,
                started,
                round_ref,
                args.service_stats_url,
                lock,
                snapshot_lock,
                stop_event,
            ),
            daemon=True,
        )
        timer_thread.start()

    for round_num in range(1, args.rounds + 1):
        round_ref[0] = round_num
        print(f"\n=== Round {round_num}/{args.rounds} ({len(all_rows)} rows) ===", flush=True)

        work_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        for row in all_rows:
            work_q.put(row)

        threads = [
            threading.Thread(
                target=worker,
                args=(
                    work_q,
                    rows_path,
                    lock,
                    args,
                    counters,
                    request_snapshot_path if args.snapshot_every > 0 else None,
                    snapshot_lock if args.snapshot_every > 0 else None,
                    round_ref,
                    started,
                ),
                daemon=True,
            )
            for _ in range(max(1, args.concurrency))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    stop_event.set()
    total_time = time.time() - started

    processed = counters["processed"]
    summary = {
        "task_type": "formula",
        "data_file": args.data_file,
        "processed": processed,
        "rounds": args.rounds,
        "rows_per_round": len(all_rows),
        "shuffle_seed": args.shuffle_seed,
        "concurrency": args.concurrency,
        "failures": counters["failures"],
        "correct": counters["correct"],
        "accuracy": counters["correct"] / processed if processed else 0.0,
        "cache_hits": counters["cache_hits"],
        "cache_hit_rate": counters["cache_hits"] / processed if processed else 0.0,
        "avg_latency_sec": counters["latency_sum"] / processed if processed else 0.0,
        "avg_time_per_request": total_time / processed if processed else 0.0,
        "throughput_eps": processed / total_time if total_time > 0 else 0.0,
        "total_time_sec": total_time,
        "rows_file": str(rows_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

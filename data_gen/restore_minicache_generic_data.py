from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI


random.seed(0)

PROMPT_RATIONALE = [
    {
        "role": "system",
        "content": (
            "You are an intelligent shopping assistant that can help users find the right item. "
            "You are given an observation of the current web navigation session, in the following format:\n\n"
            "Current observation:\nWebShop\nInstruction:\n{the user instruction}\n"
            "[button] Search [button_] (generate a search query based on the user instruction and select this button to find relevant items)\n\n"
            "Every button in the observation represents a possible action you can take. Based on the current observation, "
            "your task is to generate a rationale about the next action you should take."
        ),
    },
    {"role": "user", "content": [{"type": "text", "text": ""}]},
]

PROMPT_ACTION = [
    {
        "role": "system",
        "content": (
            "You are a intelligent shopping assistant that can help users find the right item. "
            "You are given an observation of the current environment and a rationale for the next action to be taken.\n\n"
            "Your task is to output exactly one search action in this format: "
            "Search({\"keywords\":\"...\",\"max_price\":\"...\"}). Do not output any explanation."
        ),
    },
    {"role": "user", "content": [{"type": "text", "text": ""}]},
]

FUNC_DESCRIPTION = [
    {
        "name": "Search",
        "description": "Use this function to search for the target item in the inventory based on keywords",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "The keywords that describe the item to be searched for"},
                "max_price": {
                    "type": "string",
                    "description": "The upper bound of the item price, if unspecified set to 1000000.",
                },
            },
            "required": ["keywords"],
        },
    }
]

STRUCTURAL_SYSTEM = (
    "You are an intelligent sentence generator. Given a shopping instruction, generate 10 structurally different "
    "sentences with the same item, attributes, and price constraint. Return exactly 10 lines, one sentence per line."
)

JUDGE_SYSTEM = (
    "You judge whether two WebShop Search actions are semantically equivalent for the same shopping intent. "
    "They are equivalent if the keywords preserve the same product intent and key attributes, and the max_price "
    "constraint is the same or equally restrictive for the request. Return only JSON: "
    "{\"same_answer\": true/false, \"reason\": \"short reason\"}."
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
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


def load_instructions(items_human_ins: Optional[Path], limit: Optional[int]) -> List[str]:
    if items_human_ins and items_human_ins.exists():
        raw = json.loads(items_human_ins.read_text(encoding="utf-8"))
        instructions = []
        for _, description in raw.items():
            instr = description[0]["instruction"][:-1]
            price = random.randint(20, 1000)
            instructions.append(instr + f", and price lower than {price} dollars")
        return instructions[:limit] if limit else instructions
    prompts_path = Path(__file__).with_name("human_prompts.txt")
    instructions = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return instructions[:limit] if limit else instructions


def call_with_retries(client: OpenAI, model: str, **kwargs: Any) -> Any:
    last_exc = None
    for attempt in range(5):
        try:
            return client.chat.completions.create(model=model, temperature=0, top_p=1, **kwargs)
        except Exception as exc:
            last_exc = exc
            time.sleep(min(30, 2**attempt))
    raise last_exc


def parse_json_bool(text: str) -> bool:
    text = str(text or "").strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return bool(json.loads(match.group(0)).get("same_answer"))
        except Exception:
            pass
    return '"same_answer": true' in text.lower() or "'same_answer': true" in text.lower()


class EndpointPool:
    def __init__(self, base_urls: str, api_key: str, timeout: float):
        urls = [x.strip().rstrip("/") for x in base_urls.split(",") if x.strip()]
        if not urls:
            raise ValueError("at least one structural base url is required")
        self.urls = urls
        self.api_key = api_key
        self.timeout = timeout
        self.lock = threading.Lock()
        self.next_idx = 0

    def client(self) -> OpenAI:
        with self.lock:
            url = self.urls[self.next_idx % len(self.urls)]
            self.next_idx += 1
        return OpenAI(api_key=self.api_key, base_url=url, timeout=self.timeout)


def build_large_row(client: OpenAI, model: str, instruction: str) -> Dict[str, Any]:
    prompt_rationale = json.loads(json.dumps(PROMPT_RATIONALE, ensure_ascii=False))
    prompt_rationale[1]["content"][0]["text"] = (
        f"Current observation:\nWebShop\nInstruction:\n{instruction}\n[button] Search [button_]"
    )
    rationale_resp = call_with_retries(client, model, messages=prompt_rationale, max_tokens=2000, stop="\n\n")
    rationale = rationale_resp.choices[0].message.content or ""

    prompt_action = json.loads(json.dumps(PROMPT_ACTION, ensure_ascii=False))
    prompt_action[1]["content"][0]["text"] = (
        f"Current observation:\nWebShop\nInstruction:\n{instruction}\n[button] Search [button_]\n\n"
        f"Next action rationale:{rationale}."
    )
    action_resp = call_with_retries(client, model, messages=prompt_action, max_tokens=300, stop="\n\n")
    processed_action = (action_resp.choices[0].message.content or "").strip()
    return {
        "instruction": instruction,
        "prompt_rationale": prompt_rationale,
        "response_rationale": rationale,
        "prompt_action": prompt_action,
        "response_action": processed_action,
    }


def generate_large(args: argparse.Namespace) -> None:
    done = {row["instruction"] for row in read_jsonl(Path(args.large_output))}
    instructions = [x for x in load_instructions(Path(args.items_human_ins) if args.items_human_ins else None, args.limit) if x not in done]
    lock = threading.Lock()

    def worker(instr: str) -> Dict[str, Any]:
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"], timeout=args.timeout)
        return build_large_row(client, args.model, instr)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(worker, instr): instr for instr in instructions}
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(Path(args.large_output), row, lock)


def generate_structural(args: argparse.Namespace) -> None:
    large_rows = read_jsonl(Path(args.large_output))
    if args.structural_source_limit is not None:
        large_rows = large_rows[: args.structural_source_limit]
    out_path = Path(args.structural_output)
    target = args.structural_limit
    if out_path.exists() and not args.resume_structural:
        out_path.unlink()
    lock = threading.Lock()
    accepted = 0
    accepted_lock = threading.Lock()

    endpoint_pool = EndpointPool(args.structural_base_urls or args.structural_base_url, args.structural_api_key, args.timeout)
    qwen_model = args.structural_model

    def build_records(row: Dict[str, Any], lines: List[str]) -> List[Dict[str, Any]]:
        records = []
        for prompt in lines[:10]:
            new_row = dict(row)
            new_row["instruction"] = prompt
            new_row["prompt_rationale"] = json.loads(json.dumps(row["prompt_rationale"], ensure_ascii=False))
            new_row["prompt_rationale"][1]["content"][0]["text"] = (
                f"Current observation:\nWebShop\nInstruction:\n{prompt}\n[button] Search [button_]"
            )
            new_row["prompt_action"] = json.loads(json.dumps(row["prompt_action"], ensure_ascii=False))
            new_row["prompt_action"][1]["content"][0]["text"] = (
                f"Current observation:\nWebShop\nInstruction:\n{prompt}\n[button] Search [button_]\n\n"
                f"Next action rationale:{row['response_rationale']}."
            )
            new_row["variant_source_instruction"] = row["instruction"]
            new_row["candidate_kind"] = "variant"
            records.append(new_row)
        return records

    def generate_candidates(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        client = endpoint_pool.client()
        resp = call_with_retries(
            client,
            qwen_model,
            messages=[{"role": "system", "content": STRUCTURAL_SYSTEM}, {"role": "user", "content": row["instruction"]}],
            max_tokens=1200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        lines = [x.strip(" -0123456789.\t") for x in (resp.choices[0].message.content or "").splitlines() if x.strip()]
        return build_records(row, lines)

    def generate_action_for_candidate(record: Dict[str, Any]) -> str:
        client = endpoint_pool.client()
        resp = call_with_retries(
            client,
            qwen_model,
            messages=record["prompt_action"],
            max_tokens=300,
            stop="\n\n",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return (resp.choices[0].message.content or "").strip()

    def judge_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if record.get("candidate_kind") == "original":
            record["variant_generated_action"] = record.get("response_action", "")
            record["equivalence_judge"] = {"model": qwen_model, "raw": "{\"same_answer\": true, \"reason\": \"original sample\"}"}
            return record
        generated_action = generate_action_for_candidate(record)
        client = endpoint_pool.client()
        content = (
            f"Original instruction:\n{record.get('variant_source_instruction', '')}\n\n"
            f"Variant instruction:\n{record['instruction']}\n\n"
            f"Original action:\n{record.get('response_action', '')}\n\n"
            f"Variant generated action:\n{generated_action}\n\n"
            "Are the original action and variant generated action semantically equivalent for the variant instruction?"
        )
        resp = call_with_retries(
            client,
            qwen_model,
            messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}],
            max_tokens=200,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        if parse_json_bool(raw):
            record["variant_generated_action"] = generated_action
            record["equivalence_judge"] = {"model": qwen_model, "raw": raw}
            return record
        return None

    candidate_path = out_path.with_suffix(".candidates.jsonl")
    if candidate_path.exists() and not args.reuse_candidates:
        candidate_path.unlink()

    def build_all_candidates() -> List[Dict[str, Any]]:
        if args.reuse_candidates and candidate_path.exists():
            return read_jsonl(candidate_path)
        candidates: List[Dict[str, Any]] = []
        for row in large_rows:
            original = dict(row)
            original["variant_source_instruction"] = row["instruction"]
            original["candidate_kind"] = "original"
            append_jsonl(candidate_path, original, lock)
            candidates.append(original)
        with ThreadPoolExecutor(max_workers=args.workers) as gen_pool:
            futures = {gen_pool.submit(generate_candidates, row): row for row in large_rows}
            for future in as_completed(futures):
                try:
                    records = future.result()
                except Exception:
                    continue
                for record in records:
                    append_jsonl(candidate_path, record, lock)
                    candidates.append(record)
        return candidates

    candidates = build_all_candidates()
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(candidates)
    if out_path.exists() and args.resume_structural:
        accepted = len(read_jsonl(out_path))
        candidates = candidates[accepted:]

    def judge_and_append(record: Dict[str, Any]) -> bool:
        nonlocal accepted
        with accepted_lock:
            if accepted >= target:
                return False
        judged = judge_record(record)
        if judged is None:
            return False
        with accepted_lock:
            if accepted >= target:
                return False
            append_jsonl(out_path, judged, lock)
            accepted += 1
            return True

    for start in range(0, len(candidates), args.candidate_batch_size):
        with accepted_lock:
            if accepted >= target:
                break
        batch = candidates[start : start + args.candidate_batch_size]
        with ThreadPoolExecutor(max_workers=args.judge_workers) as judge_pool:
            futures = [judge_pool.submit(judge_and_append, record) for record in batch]
            for future in as_completed(futures):
                _ = future.result()
                with accepted_lock:
                    if accepted >= target:
                        break
            with accepted_lock:
                if accepted >= target:
                    break

    if accepted < target:
        raise RuntimeError(
            f"Only accepted {accepted} structurally varied examples; target is {target}. "
            f"Generate more candidates or lower filtering strictness."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-human-ins", default=None)
    parser.add_argument("--large-output", default="gt_param-w-synonym_data_large.jsonl")
    parser.add_argument("--structural-output", default="gt_param-w-synonym_data_large_structural_10k.jsonl")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--structural-base-url", default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1"))
    parser.add_argument(
        "--structural-base-urls",
        default=os.getenv("MINICACHE_STRUCTURAL_BASE_URLS", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:9000/v1")),
    )
    parser.add_argument("--structural-api-key", default="EMPTY")
    parser.add_argument("--structural-model", default="qwen3-32b-fp8")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--judge-workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--structural-limit", type=int, default=10000)
    parser.add_argument("--structural-source-limit", type=int, default=None)
    parser.add_argument("--skip-large", action="store_true")
    parser.add_argument("--skip-structural", action="store_true")
    parser.add_argument("--resume-structural", action="store_true")
    parser.add_argument("--reuse-candidates", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--source-batch-size", type=int, default=256)
    parser.add_argument("--candidate-batch-size", type=int, default=2048)
    args = parser.parse_args()
    if not args.skip_large:
        generate_large(args)
    if not args.skip_structural:
        generate_structural(args)


if __name__ == "__main__":
    main()

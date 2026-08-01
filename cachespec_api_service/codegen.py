from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple


CODEGEN_SYSTEM = (
    "You are a financial QA assistant. Read the question and context carefully. "
    "Write a Python program that computes the answer from the given context.\n\n"
    "Rules:\n"
    "- Hardcode all values from the context as variables in the program.\n"
    "- Compute the answer step by step using basic arithmetic (+, -, *, /).\n"
    "- Assign the final answer to a variable named 'answer'.\n"
    "- Do NOT use import statements.\n"
    "- Do NOT use print(), input(), or file operations.\n"
    "- Return ONLY the Python code in a ```python code block. No other text."
)


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "None": None,
    "True": True,
    "False": False,
}


def extract_code(text: str) -> Optional[str]:
    text = str(text or "").strip()
    think_end = text.rfind("</think>")
    if think_end >= 0:
        text = text[think_end + len("</think>") :].strip()
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def execute_code(code: str) -> Tuple[Optional[str], Optional[str], float]:
    t0 = time.time()
    namespace = {"__builtins__": dict(SAFE_BUILTINS), "math": math}
    local_ns: Dict[str, Any] = {}
    try:
        exec(code, namespace, local_ns)
    except Exception as exc:
        return None, f"exec_error: {exc}", time.time() - t0
    for name in ("answer", "result", "ans", "value"):
        if name in local_ns:
            return str(local_ns[name]), None, time.time() - t0
    if local_ns:
        last_val = list(local_ns.values())[-1]
        if last_val is not None:
            return str(last_val), None, time.time() - t0
    return None, "no_return_value", time.time() - t0


def build_codegen_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    user_text = "\n\n".join(str(m.get("content", "")) for m in messages if m.get("role") != "system")
    if not user_text:
        user_text = "\n\n".join(str(m.get("content", "")) for m in messages)
    return [
        {"role": "system", "content": CODEGEN_SYSTEM},
        {
            "role": "user",
            "content": (
                "Read the following context and answer the question by writing a Python program.\n\n"
                f"{user_text}\n\nWrite the Python code to compute the answer:"
            ),
        },
    ]


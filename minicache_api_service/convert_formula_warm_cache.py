from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

try:
    from .formula_modified_runtime import stable_cache_id
except ImportError:
    from formula_modified_runtime import stable_cache_id


def load_cluster_map(path: Path) -> Dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clusters = payload.get("clusters", payload if isinstance(payload, list) else [])
    mapping: Dict[str, str] = {}
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id"))
        formula_name = str(cluster.get("formula_name") or "").strip()
        if cluster_id and formula_name:
            mapping[cluster_id] = formula_name
    return mapping


def convert_entry(cluster_id: str, formula_name: str, old_entry: Dict[str, Any]) -> Dict[str, Any]:
    spec = old_entry.get("spec") or {}
    cache_id = stable_cache_id(formula_name)
    validation = spec.get("_program_validation") or spec.get("_slot_validation") or {}
    return {
        "group_key": formula_name,
        "spec": {
            "representation_type": spec.get("representation_type", "program"),
            "action_type": spec.get("action_type", "formula_extraction"),
            "cluster_summary": spec.get("cluster_summary") or formula_name,
            "slot_spec": spec.get("slot_spec") or {},
            "program_code": spec.get("program_code") or "",
            "_program_validation": validation,
            "_source_cluster_id": cluster_id,
        },
        "created_at": time.time(),
        "generation_samples": int(validation.get("checked", 0) or 0),
        "validation": validation,
        "cache_id": cache_id,
        "fail_count": 0,
        "hit_count": 0,
        "source_cluster_id": cluster_id,
        "source": "legacy_modified_formula_cache",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-global-cache", required=True)
    parser.add_argument("--old-clusters", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    old_cache_path = Path(args.old_global_cache)
    old_clusters_path = Path(args.old_clusters)
    output_path = Path(args.output)

    old_cache = json.loads(old_cache_path.read_text(encoding="utf-8"))
    cluster_map = load_cluster_map(old_clusters_path)
    converted: Dict[str, Dict[str, Any]] = {}
    skipped = []
    for cluster_id, old_entry in old_cache.items():
        formula_name = cluster_map.get(str(cluster_id))
        if not formula_name:
            skipped.append({"cluster_id": cluster_id, "reason": "missing_formula_name"})
            continue
        spec = old_entry.get("spec") or {}
        if not spec.get("slot_spec") or not spec.get("program_code"):
            skipped.append({"cluster_id": cluster_id, "formula_name": formula_name, "reason": "missing_spec_or_program"})
            continue
        entry = convert_entry(str(cluster_id), formula_name, old_entry)
        converted[entry["cache_id"]] = entry

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "old_global_cache": str(old_cache_path),
        "old_clusters": str(old_clusters_path),
        "output": str(output_path),
        "converted": len(converted),
        "skipped": skipped,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

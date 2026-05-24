#!/usr/bin/env python3
"""Plot cold-start timeseries metrics for multiple concurrency levels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_timeseries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


METRICS = [
    ("accuracy", "Accuracy", "Accuracy"),
    ("cache_hit_rate", "Cache Hit Rate", "Cache Hit Rate"),
    ("avg_time_per_request", "Avg Time per Request (s)", "Avg Time per Request (s)"),
    ("throughput_eps", "Throughput (episodes/s)", "Throughput (eps)"),
    ("cache_entries", "Cache Entries (out of 25)", "Cache Entries"),
]

COLORS = plt.cm.tab10.colors


def plot_metric(
    data_by_c: Dict[int, List[Dict]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    for idx, (c, rows) in enumerate(sorted(data_by_c.items())):
        xs = [r.get(x_key, 0) for r in rows]
        ys = [r.get(y_key, 0) for r in rows]
        if not xs:
            continue
        ax.plot(xs, ys, label=f"c={c}", color=COLORS[idx % len(COLORS)], linewidth=1.5, alpha=0.85)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--concurrencies", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)
    out = Path(args.output_dir) if args.output_dir else base / "plots"
    out.mkdir(parents=True, exist_ok=True)

    for source, filename, x_key, xlabel, suffix in [
        ("request", "metrics_timeseries_by_request.jsonl", "total_requests", "Total Requests", "requests"),
        ("time", "metrics_timeseries_by_time.jsonl", "elapsed_sec", "Elapsed Time (s)", "time"),
    ]:
        data_by_c: Dict[int, List[Dict]] = {}
        for c in args.concurrencies:
            path = base / f"c{c}" / "laser" / filename
            rows = load_timeseries(path)
            if rows:
                data_by_c[c] = rows
                print(f"loaded c={c} {source}: {len(rows)} points")
            else:
                print(f"SKIP c={c} {source}: no data at {path}")

        if not data_by_c:
            print(f"no data for {source}, skipping plots")
            continue

        for y_key, title, ylabel in METRICS:
            plot_metric(
                data_by_c,
                x_key=x_key,
                y_key=y_key,
                title=f"{title} vs {xlabel}",
                xlabel=xlabel,
                ylabel=ylabel,
                output_path=out / f"{y_key}_vs_{suffix}.png",
            )

    print(f"\nAll plots saved to {out}")


if __name__ == "__main__":
    main()

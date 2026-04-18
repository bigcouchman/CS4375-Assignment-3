from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import isinf
from pathlib import Path
from statistics import mean
from typing import Any
from typing import Iterable, List, Sequence

from .jaccard_kmeans import JaccardKMeans
from .types import TweetRecord


@dataclass
class ExperimentRow:
    k: int
    sse: float
    iterations: int
    cluster_sizes: List[int]
    tweets_count: int

    @property
    def sse_per_tweet(self) -> float:
        if self.tweets_count <= 0:
            return 0.0
        return self.sse / self.tweets_count

    @property
    def min_cluster_size(self) -> int:
        if not self.cluster_sizes:
            return 0
        return min(self.cluster_sizes)

    @property
    def max_cluster_size(self) -> int:
        if not self.cluster_sizes:
            return 0
        return max(self.cluster_sizes)

    @property
    def empty_cluster_count(self) -> int:
        return sum(1 for size in self.cluster_sizes if size == 0)

    @property
    def imbalance_ratio(self) -> float:
        minimum = self.min_cluster_size
        if minimum == 0:
            return float("inf")
        return self.max_cluster_size / minimum


def run_experiment(
    tweets: Sequence[TweetRecord],
    k_values: Iterable[int],
    max_iter: int = 40,
    base_seed: int = 42,
) -> List[ExperimentRow]:
    rows: List[ExperimentRow] = []

    for offset, k in enumerate(k_values):
        model = JaccardKMeans(k=k, max_iter=max_iter, random_state=base_seed + offset)
        result = model.fit(tweets)
        rows.append(
            ExperimentRow(
                k=k,
                sse=result.sse,
                iterations=result.iterations,
                cluster_sizes=result.cluster_sizes,
                tweets_count=len(tweets),
            )
        )

    return rows


def format_results_table(rows: Sequence[ExperimentRow]) -> str:
    header = (
        f"{'K':>4} | {'SSE':>12} | {'SSE/Tweet':>10} | {'Iter':>6} | "
        f"{'Min/Max':>9} | {'Empty':>5} | Cluster sizes"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for row in rows:
        sizes = ", ".join(f"{idx + 1}:{size}" for idx, size in enumerate(row.cluster_sizes))
        min_max = f"{row.min_cluster_size}/{row.max_cluster_size}"
        lines.append(
            f"{row.k:>4} | {row.sse:>12.6f} | {row.sse_per_tweet:>10.6f} | "
            f"{row.iterations:>6} | {min_max:>9} | {row.empty_cluster_count:>5} | {sizes}"
        )

    return "\n".join(lines)


def build_run_summary(rows: Sequence[ExperimentRow]) -> dict[str, Any]:
    if not rows:
        return {
            "k_count": 0,
            "best_sse_k": None,
            "best_sse": None,
            "best_sse_per_tweet_k": None,
            "best_sse_per_tweet": None,
            "most_balanced_k": None,
            "most_balanced_ratio": None,
            "avg_iterations": 0.0,
            "max_iterations": 0,
        }

    best_sse_row = min(rows, key=lambda row: (row.sse, row.k))
    best_normalized_row = min(rows, key=lambda row: (row.sse_per_tweet, row.k))
    balanced_row = min(rows, key=lambda row: (row.imbalance_ratio, row.empty_cluster_count, row.k))

    balanced_ratio = None if isinf(balanced_row.imbalance_ratio) else balanced_row.imbalance_ratio

    return {
        "k_count": len(rows),
        "best_sse_k": best_sse_row.k,
        "best_sse": best_sse_row.sse,
        "best_sse_per_tweet_k": best_normalized_row.k,
        "best_sse_per_tweet": best_normalized_row.sse_per_tweet,
        "most_balanced_k": balanced_row.k,
        "most_balanced_ratio": balanced_ratio,
        "avg_iterations": mean(row.iterations for row in rows),
        "max_iterations": max(row.iterations for row in rows),
    }


def format_run_summary(rows: Sequence[ExperimentRow], duration_seconds: float) -> str:
    summary = build_run_summary(rows)

    lines = ["Important run metrics"]
    lines.append(f"- Runtime (seconds): {duration_seconds:.3f}")
    lines.append(f"- K values tested: {summary['k_count']}")

    best_sse_k = summary["best_sse_k"]
    best_sse = summary["best_sse"]
    if best_sse_k is not None and best_sse is not None:
        lines.append(f"- Lowest SSE: K={best_sse_k}, SSE={best_sse:.6f}")

    best_norm_k = summary["best_sse_per_tweet_k"]
    best_norm = summary["best_sse_per_tweet"]
    if best_norm_k is not None and best_norm is not None:
        lines.append(f"- Lowest SSE/tweet: K={best_norm_k}, SSE/tweet={best_norm:.6f}")

    balanced_k = summary["most_balanced_k"]
    balanced_ratio = summary["most_balanced_ratio"]
    if balanced_k is not None:
        if balanced_ratio is None:
            lines.append(f"- Most balanced clustering: K={balanced_k}, ratio=inf (empty cluster)")
        else:
            lines.append(
                f"- Most balanced clustering: K={balanced_k}, max/min cluster ratio={balanced_ratio:.3f}"
            )

    lines.append(f"- Avg iterations: {summary['avg_iterations']:.2f}")
    lines.append(f"- Max iterations: {summary['max_iterations']}")
    return "\n".join(lines)


def _format_float(value: float) -> str:
    if isinf(value):
        return "inf"
    return f"{value:.6f}"


def row_to_metrics_dict(row: ExperimentRow) -> dict[str, Any]:
    imbalance_ratio: float | None
    if isinf(row.imbalance_ratio):
        imbalance_ratio = None
    else:
        imbalance_ratio = row.imbalance_ratio

    return {
        "k": row.k,
        "sse": row.sse,
        "sse_per_tweet": row.sse_per_tweet,
        "iterations": row.iterations,
        "min_cluster_size": row.min_cluster_size,
        "max_cluster_size": row.max_cluster_size,
        "empty_clusters": row.empty_cluster_count,
        "imbalance_ratio": imbalance_ratio,
        "cluster_sizes": row.cluster_sizes,
        "cluster_sizes_compact": ";".join(str(size) for size in row.cluster_sizes),
    }


def save_results_csv(rows: Sequence[ExperimentRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "k",
                "sse",
                "sse_per_tweet",
                "iterations",
                "min_cluster_size",
                "max_cluster_size",
                "empty_clusters",
                "imbalance_ratio",
                "cluster_sizes",
            ]
        )
        for row in rows:
            metrics = row_to_metrics_dict(row)
            writer.writerow(
                [
                    metrics["k"],
                    _format_float(float(metrics["sse"])),
                    _format_float(float(metrics["sse_per_tweet"])),
                    metrics["iterations"],
                    metrics["min_cluster_size"],
                    metrics["max_cluster_size"],
                    metrics["empty_clusters"],
                    "" if metrics["imbalance_ratio"] is None else _format_float(float(metrics["imbalance_ratio"])),
                    metrics["cluster_sizes_compact"],
                ]
            )


def append_history_csv(
    rows: Sequence[ExperimentRow],
    history_path: Path,
    run_metadata: dict[str, Any],
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not history_path.exists()

    with history_path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        if should_write_header:
            writer.writerow(
                [
                    "run_id",
                    "run_timestamp_utc",
                    "input_file",
                    "raw_lines",
                    "tweets_used",
                    "dropped_tweets",
                    "k",
                    "max_iter",
                    "seed",
                    "duration_seconds",
                    "sse",
                    "sse_per_tweet",
                    "iterations",
                    "min_cluster_size",
                    "max_cluster_size",
                    "empty_clusters",
                    "imbalance_ratio",
                    "cluster_sizes",
                ]
            )

        for row in rows:
            metrics = row_to_metrics_dict(row)
            writer.writerow(
                [
                    run_metadata["run_id"],
                    run_metadata["run_timestamp_utc"],
                    run_metadata["input_file"],
                    run_metadata["raw_lines"],
                    run_metadata["tweets_used"],
                    run_metadata["dropped_tweets"],
                    metrics["k"],
                    run_metadata["max_iter"],
                    run_metadata["seed"],
                    _format_float(float(run_metadata["duration_seconds"])),
                    _format_float(float(metrics["sse"])),
                    _format_float(float(metrics["sse_per_tweet"])),
                    metrics["iterations"],
                    metrics["min_cluster_size"],
                    metrics["max_cluster_size"],
                    metrics["empty_clusters"],
                    "" if metrics["imbalance_ratio"] is None else _format_float(float(metrics["imbalance_ratio"])),
                    metrics["cluster_sizes_compact"],
                ]
            )


def save_run_metrics_json(
    rows: Sequence[ExperimentRow],
    output_path: Path,
    run_metadata: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_metadata": run_metadata,
        "summary": build_run_summary(rows),
        "rows": [row_to_metrics_dict(row) for row in rows],
    }

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from math import isinf
from pathlib import Path
from statistics import mean, pstdev
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
    init_strategy: str = "random"
    init_trials: int = 1
    best_init_seed: int | None = None
    best_init_strategy: str | None = None
    mean_trial_sse: float | None = None
    std_trial_sse: float | None = None

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


HISTORY_COLUMNS = [
    "run_id",
    "run_timestamp_utc",
    "input_file",
    "raw_lines",
    "tweets_used",
    "dropped_tweets",
    "k",
    "max_iter",
    "seed",
    "init_strategy",
    "n_init",
    "duration_seconds",
    "sse",
    "sse_per_tweet",
    "iterations",
    "best_init_seed",
    "best_init_strategy",
    "mean_trial_sse",
    "std_trial_sse",
    "min_cluster_size",
    "max_cluster_size",
    "empty_clusters",
    "imbalance_ratio",
    "cluster_sizes",
]


def run_experiment(
    tweets: Sequence[TweetRecord],
    k_values: Iterable[int],
    max_iter: int = 40,
    base_seed: int = 42,
    n_init: int = 1,
    init_strategy: str = "hybrid",
) -> List[ExperimentRow]:
    if n_init <= 0:
        raise ValueError("n_init must be a positive integer")
    if init_strategy not in {"random", "kmedoids++", "hybrid"}:
        raise ValueError("init_strategy must be 'random', 'kmedoids++', or 'hybrid'")

    k_values_list = list(k_values)
    if not k_values_list:
        return []

    rows: List[ExperimentRow] = []
    shared_distance_cache: dict[tuple[int, int], float] = {}

    for offset, k in enumerate(k_values_list):
        trial_sses: List[float] = []
        best_result = None
        best_seed = None
        best_strategy = None
        trial_strategies = [init_strategy]
        if init_strategy == "hybrid":
            # Run both strategies per restart and keep the best SSE.
            trial_strategies = ["random", "kmedoids++"]

        for trial in range(n_init):
            # Keep trial 0 aligned with historical seed progression for fair comparison.
            trial_seed = base_seed + offset + (trial * len(k_values_list))
            for trial_strategy in trial_strategies:
                model = JaccardKMeans(
                    k=k,
                    max_iter=max_iter,
                    random_state=trial_seed,
                    init_strategy=trial_strategy,
                    distance_cache=shared_distance_cache,
                )
                result = model.fit(tweets)
                trial_sses.append(result.sse)

                if best_result is None or result.sse < best_result.sse:
                    best_result = result
                    best_seed = trial_seed
                    best_strategy = trial_strategy

        if best_result is None:
            raise RuntimeError("Failed to produce clustering result")

        rows.append(
            ExperimentRow(
                k=k,
                sse=best_result.sse,
                iterations=best_result.iterations,
                cluster_sizes=best_result.cluster_sizes,
                tweets_count=len(tweets),
                init_strategy=init_strategy,
                init_trials=len(trial_sses),
                best_init_seed=best_seed,
                best_init_strategy=best_strategy,
                mean_trial_sse=mean(trial_sses),
                std_trial_sse=pstdev(trial_sses) if len(trial_sses) > 1 else 0.0,
            )
        )

    return rows


def format_results_table(rows: Sequence[ExperimentRow]) -> str:
    header = (
        f"{'K':>4} | {'SSE':>12} | {'SSE/Tweet':>10} | {'Iter':>6} | {'Trials':>6} | "
        f"{'BestSeed':>8} | {'BestInit':>9} | {'Min/Max':>9} | {'Empty':>5} | Cluster sizes"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for row in rows:
        sizes = ", ".join(f"{idx + 1}:{size}" for idx, size in enumerate(row.cluster_sizes))
        min_max = f"{row.min_cluster_size}/{row.max_cluster_size}"
        best_seed_display = "-" if row.best_init_seed is None else str(row.best_init_seed)
        best_init_display = row.best_init_strategy or "-"
        lines.append(
            f"{row.k:>4} | {row.sse:>12.6f} | {row.sse_per_tweet:>10.6f} | "
            f"{row.iterations:>6} | {row.init_trials:>6} | {best_seed_display:>8} | "
            f"{best_init_display:>9} | {min_max:>9} | {row.empty_cluster_count:>5} | {sizes}"
        )

    return "\n".join(lines)


def format_assignment_table(rows: Sequence[ExperimentRow]) -> str:
    """Format table.

    Columns: Value of K | SSE | Size of each cluster
    """
    header = f"{'Value of K':>10} | {'SSE':>12} | Size of each cluster"
    divider = "-" * len(header)
    lines = [header, divider]

    for row in rows:
        size_desc = "; ".join(
            f"{idx + 1}: {size} tweets" for idx, size in enumerate(row.cluster_sizes)
        )
        lines.append(f"{row.k:>10} | {row.sse:>12.6f} | {size_desc}")

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
            "init_strategy": None,
            "trial_count_per_k": 0,
            "avg_restart_gain": 0.0,
            "max_restart_gain": 0.0,
        }

    best_sse_row = min(rows, key=lambda row: (row.sse, row.k))
    best_normalized_row = min(rows, key=lambda row: (row.sse_per_tweet, row.k))
    balanced_row = min(rows, key=lambda row: (row.imbalance_ratio, row.empty_cluster_count, row.k))

    balanced_ratio = None if isinf(balanced_row.imbalance_ratio) else balanced_row.imbalance_ratio
    gains = []
    for row in rows:
        if row.mean_trial_sse is None:
            continue
        gains.append(max(0.0, row.mean_trial_sse - row.sse))

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
        "init_strategy": rows[0].init_strategy,
        "trial_count_per_k": max(row.init_trials for row in rows),
        "avg_restart_gain": mean(gains) if gains else 0.0,
        "max_restart_gain": max(gains) if gains else 0.0,
    }


def format_run_summary(rows: Sequence[ExperimentRow], duration_seconds: float) -> str:
    summary = build_run_summary(rows)

    lines = ["Important run metrics"]
    lines.append(f"- Runtime (seconds): {duration_seconds:.3f}")
    lines.append(f"- K values tested: {summary['k_count']}")
    lines.append(f"- Init strategy: {summary['init_strategy']}")
    lines.append(f"- Trials per K (executed): {summary['trial_count_per_k']}")

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
    if summary["trial_count_per_k"] and summary["trial_count_per_k"] > 1:
        lines.append(f"- Avg restart gain (mean_trial_sse - best_sse): {summary['avg_restart_gain']:.6f}")
        lines.append(f"- Max restart gain (mean_trial_sse - best_sse): {summary['max_restart_gain']:.6f}")
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
        "init_strategy": row.init_strategy,
        "init_trials": row.init_trials,
        "best_init_seed": row.best_init_seed,
        "best_init_strategy": row.best_init_strategy,
        "mean_trial_sse": row.mean_trial_sse,
        "std_trial_sse": row.std_trial_sse,
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
                "init_strategy",
                "init_trials",
                "best_init_seed",
                "best_init_strategy",
                "mean_trial_sse",
                "std_trial_sse",
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
                    metrics["init_strategy"],
                    metrics["init_trials"],
                    "" if metrics["best_init_seed"] is None else metrics["best_init_seed"],
                    "" if metrics["best_init_strategy"] is None else metrics["best_init_strategy"],
                    "" if metrics["mean_trial_sse"] is None else _format_float(float(metrics["mean_trial_sse"])),
                    "" if metrics["std_trial_sse"] is None else _format_float(float(metrics["std_trial_sse"])),
                    metrics["min_cluster_size"],
                    metrics["max_cluster_size"],
                    metrics["empty_clusters"],
                    "" if metrics["imbalance_ratio"] is None else _format_float(float(metrics["imbalance_ratio"])),
                    metrics["cluster_sizes_compact"],
                ]
            )


def save_assignment_results_csv(rows: Sequence[ExperimentRow], output_path: Path) -> None:
    """
    Output columns:
    - Value of K
    - SSE
    - Size of each cluster
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Value of K", "SSE", "Size of each cluster"])
        for row in rows:
            sizes = "; ".join(
                f"{idx + 1}: {size} tweets" for idx, size in enumerate(row.cluster_sizes)
            )
            writer.writerow([row.k, _format_float(row.sse), sizes])


def append_history_csv(
    rows: Sequence[ExperimentRow],
    history_path: Path,
    run_metadata: dict[str, Any],
) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows: List[dict[str, Any]] = []
    if history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_rows.append({col: row.get(col, "") for col in HISTORY_COLUMNS})

    new_rows: List[dict[str, Any]] = []
    for row in rows:
        metrics = row_to_metrics_dict(row)
        new_rows.append(
            {
                "run_id": run_metadata["run_id"],
                "run_timestamp_utc": run_metadata["run_timestamp_utc"],
                "input_file": run_metadata["input_file"],
                "raw_lines": run_metadata["raw_lines"],
                "tweets_used": run_metadata["tweets_used"],
                "dropped_tweets": run_metadata["dropped_tweets"],
                "k": metrics["k"],
                "max_iter": run_metadata["max_iter"],
                "seed": run_metadata["seed"],
                "init_strategy": run_metadata.get("init_strategy", ""),
                "n_init": run_metadata.get("n_init", ""),
                "duration_seconds": _format_float(float(run_metadata["duration_seconds"])),
                "sse": _format_float(float(metrics["sse"])),
                "sse_per_tweet": _format_float(float(metrics["sse_per_tweet"])),
                "iterations": metrics["iterations"],
                "best_init_seed": "" if metrics["best_init_seed"] is None else metrics["best_init_seed"],
                "best_init_strategy": ""
                if metrics["best_init_strategy"] is None
                else metrics["best_init_strategy"],
                "mean_trial_sse": ""
                if metrics["mean_trial_sse"] is None
                else _format_float(float(metrics["mean_trial_sse"])),
                "std_trial_sse": ""
                if metrics["std_trial_sse"] is None
                else _format_float(float(metrics["std_trial_sse"])),
                "min_cluster_size": metrics["min_cluster_size"],
                "max_cluster_size": metrics["max_cluster_size"],
                "empty_clusters": metrics["empty_clusters"],
                "imbalance_ratio": ""
                if metrics["imbalance_ratio"] is None
                else _format_float(float(metrics["imbalance_ratio"])),
                "cluster_sizes": metrics["cluster_sizes_compact"],
            }
        )

    with history_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for record in existing_rows:
            writer.writerow({col: record.get(col, "") for col in HISTORY_COLUMNS})
        for record in new_rows:
            writer.writerow({col: record.get(col, "") for col in HISTORY_COLUMNS})


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

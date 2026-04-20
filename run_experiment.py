from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tweet_clustering.experiment import (
    append_history_csv,
    format_assignment_table,
    format_results_table,
    format_run_summary,
    run_experiment,
    save_assignment_results_csv,
    save_results_csv,
    save_run_metrics_json,
)
from tweet_clustering.preprocessing import load_and_preprocess_tweets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tweet clustering with Jaccard distance and K-means (medoid update)."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to one UCI health tweet file (e.g., usnewshealth.txt).",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 15, 20, 25],
        help="Space-separated list of K values. Use at least 5 values.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=40,
        help="Maximum K-means iterations per K value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducible centroid initialization.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=1,
        help="Number of restart trials per K. Best SSE across trials is kept.",
    )
    parser.add_argument(
        "--init-strategy",
        type=str,
        choices=["random", "kmedoids++", "hybrid"],
        default="random",
        help="Centroid initialization strategy.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results") / "kmeans_results.csv",
        help="Path to save experiment summary CSV.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tweets that become empty after preprocessing (default: drop them).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    unique_k_values = sorted(set(args.k_values))
    if any(k <= 0 for k in unique_k_values):
        raise ValueError("All K values must be positive integers.")
    if args.n_init <= 0:
        raise ValueError("--n-init must be a positive integer.")
    if len(unique_k_values) < 5:
        raise ValueError(
            "At least 5 different K values are required. "
            f"Current unique K count: {len(unique_k_values)}"
        )

    load_result = load_and_preprocess_tweets(
        args.input_file, drop_empty=not args.keep_empty
    )
    tweets = load_result.tweets

    if not tweets:
        raise ValueError(
            "No tweets available after preprocessing."
        )

    max_k = max(unique_k_values)
    if max_k > len(tweets):
        raise ValueError(
            f"Largest K ({max_k}) cannot exceed number of tweets ({len(tweets)})."
        )

    print("Tweet Clustering using Jaccard Distance + K-means (medoid centroid update)")
    print(f"Input file         : {args.input_file}")
    print(f"Raw lines          : {load_result.raw_line_count}")
    print(f"Tweets used        : {len(tweets)}")
    print(f"Dropped as empty   : {load_result.dropped_tweet_count}")
    print(f"K values           : {unique_k_values}")
    print(f"Max iterations     : {args.max_iter}")
    print(f"Base random seed   : {args.seed}")
    print(f"Init strategy      : {args.init_strategy}")
    print(f"Restarts per K     : {args.n_init}")
    print("-" * 84)

    run_timestamp = datetime.now(timezone.utc)
    run_id = run_timestamp.strftime("%Y%m%dT%H%M%S_%fZ")

    start_time = time.perf_counter()
    rows = run_experiment(
        tweets=tweets,
        k_values=unique_k_values,
        max_iter=args.max_iter,
        base_seed=args.seed,
        n_init=args.n_init,
        init_strategy=args.init_strategy,
    )
    duration_seconds = time.perf_counter() - start_time

    print(format_results_table(rows))
    print()
    print("Assignment-required table")
    print(format_assignment_table(rows))
    print()
    print(format_run_summary(rows, duration_seconds))

    results_dir = args.output_csv.parent
    history_csv_path = results_dir / "run_history.csv"
    latest_metrics_json_path = results_dir / "latest_run_metrics.json"
    assignment_csv_path = results_dir / "assignment_results.csv"
    per_run_csv_path = results_dir / "runs" / f"run_{run_id}.csv"
    per_run_assignment_csv_path = results_dir / "runs" / f"run_{run_id}_assignment.csv"

    run_metadata = {
        "run_id": run_id,
        "run_timestamp_utc": run_timestamp.isoformat(),
        "input_file": str(args.input_file),
        "raw_lines": load_result.raw_line_count,
        "tweets_used": len(tweets),
        "dropped_tweets": load_result.dropped_tweet_count,
        "k_values": unique_k_values,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "n_init": args.n_init,
        "init_strategy": args.init_strategy,
        "duration_seconds": duration_seconds,
    }

    save_results_csv(rows, args.output_csv)
    save_assignment_results_csv(rows, assignment_csv_path)
    save_results_csv(rows, per_run_csv_path)
    save_assignment_results_csv(rows, per_run_assignment_csv_path)
    append_history_csv(rows, history_csv_path, run_metadata)
    save_run_metrics_json(rows, latest_metrics_json_path, run_metadata)

    print(f"\nSaved latest table to : {args.output_csv}")
    print(f"Saved assignment CSV : {assignment_csv_path}")
    print(f"Saved run snapshot to : {per_run_csv_path}")
    print(f"Saved run assignment : {per_run_assignment_csv_path}")
    print(f"Saved run history to  : {history_csv_path}")
    print(f"Saved JSON metrics to : {latest_metrics_json_path}")


if __name__ == "__main__":
    main()

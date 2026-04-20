from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from tweet_clustering.experiment import (
    ExperimentRow,
    append_history_csv,
    build_run_summary,
    format_assignment_table,
    run_experiment,
    save_assignment_results_csv,
    save_results_csv,
    save_run_metrics_json,
)
from tweet_clustering.types import TweetRecord


class TestExperimentOutputs(unittest.TestCase):
    @staticmethod
    def _record(idx: int, text: str) -> TweetRecord:
        tokens = tuple(text.split())
        return TweetRecord(
            tweet_id=str(idx),
            timestamp="",
            raw_text=text,
            normalized_text=text,
            tokens=tokens,
            token_set=frozenset(tokens),
        )

    def test_summary_and_metrics_properties(self) -> None:
        rows = [
            ExperimentRow(k=2, sse=1.50, iterations=4, cluster_sizes=[3, 1], tweets_count=4),
            ExperimentRow(k=3, sse=1.25, iterations=2, cluster_sizes=[2, 1, 1], tweets_count=4),
        ]

        summary = build_run_summary(rows)

        self.assertEqual(summary["k_count"], 2)
        self.assertEqual(summary["best_sse_k"], 3)
        self.assertEqual(summary["best_sse_per_tweet_k"], 3)
        self.assertEqual(summary["max_iterations"], 4)

    def test_writes_csv_history_and_json(self) -> None:
        rows = [
            ExperimentRow(k=2, sse=1.50, iterations=4, cluster_sizes=[3, 1], tweets_count=4),
            ExperimentRow(k=3, sse=1.25, iterations=2, cluster_sizes=[2, 1, 1], tweets_count=4),
        ]

        run_metadata = {
            "run_id": "test-run-1",
            "run_timestamp_utc": "2026-04-17T12:00:00+00:00",
            "input_file": "data/Health-Tweets/usnewshealth.txt",
            "raw_lines": 10,
            "tweets_used": 4,
            "dropped_tweets": 1,
            "k_values": [2, 3],
            "max_iter": 40,
            "seed": 42,
            "duration_seconds": 0.12,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            results_csv = tmp_path / "kmeans_results.csv"
            history_csv = tmp_path / "run_history.csv"
            metrics_json = tmp_path / "latest_run_metrics.json"

            save_results_csv(rows, results_csv)
            append_history_csv(rows, history_csv, run_metadata)
            save_run_metrics_json(rows, metrics_json, run_metadata)

            self.assertTrue(results_csv.exists())
            self.assertTrue(history_csv.exists())
            self.assertTrue(metrics_json.exists())

            with results_csv.open("r", encoding="utf-8") as f:
                header = f.readline().strip()
            self.assertIn("sse_per_tweet", header)
            self.assertIn("imbalance_ratio", header)

            with history_csv.open("r", encoding="utf-8", newline="") as f:
                rows_read = list(csv.DictReader(f))
            self.assertEqual(len(rows_read), 2)
            self.assertEqual(rows_read[0]["run_id"], "test-run-1")
            self.assertIn("cluster_sizes", rows_read[0])
            self.assertIn("init_strategy", rows_read[0])
            self.assertIn("n_init", rows_read[0])

            with metrics_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["run_metadata"]["run_id"], "test-run-1")
            self.assertEqual(len(payload["rows"]), 2)
            self.assertIn("summary", payload)

    def test_assignment_table_and_csv_format(self) -> None:
        rows = [
            ExperimentRow(k=10, sse=200.0, iterations=4, cluster_sizes=[10, 25, 20], tweets_count=55),
        ]

        table_text = format_assignment_table(rows)
        self.assertIn("Value of K", table_text)
        self.assertIn("SSE", table_text)
        self.assertIn("Size of each cluster", table_text)
        self.assertIn("10", table_text)
        self.assertIn("1: 10 tweets", table_text)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            assignment_csv = tmp_path / "assignment_results.csv"
            save_assignment_results_csv(rows, assignment_csv)

            self.assertTrue(assignment_csv.exists())
            with assignment_csv.open("r", encoding="utf-8", newline="") as f:
                csv_rows = list(csv.reader(f))

            self.assertEqual(csv_rows[0], ["Value of K", "SSE", "Size of each cluster"])
            self.assertEqual(csv_rows[1][0], "10")
            self.assertEqual(csv_rows[1][1], "200.000000")
            self.assertIn("1: 10 tweets", csv_rows[1][2])

    def test_run_experiment_with_restarts_populates_metadata(self) -> None:
        tweets = [
            self._record(1, "heart health tips"),
            self._record(2, "heart health advice"),
            self._record(3, "flu vaccine update"),
            self._record(4, "flu vaccine report"),
        ]

        rows = run_experiment(
            tweets=tweets,
            k_values=[2],
            max_iter=20,
            base_seed=10,
            n_init=2,
            init_strategy="kmedoids++",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.init_trials, 2)
        self.assertEqual(row.init_strategy, "kmedoids++")
        self.assertIsNotNone(row.best_init_seed)
        self.assertIsNotNone(row.mean_trial_sse)
        self.assertIsNotNone(row.std_trial_sse)

    def test_run_experiment_hybrid_executes_both_strategies(self) -> None:
        tweets = [
            self._record(1, "heart health tips"),
            self._record(2, "heart health advice"),
            self._record(3, "flu vaccine update"),
            self._record(4, "flu vaccine report"),
        ]

        rows = run_experiment(
            tweets=tweets,
            k_values=[2],
            max_iter=20,
            base_seed=10,
            n_init=2,
            init_strategy="hybrid",
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row.init_trials, 4)
        self.assertEqual(row.init_strategy, "hybrid")
        self.assertIn(row.best_init_strategy, {"random", "kmedoids++"})


if __name__ == "__main__":
    unittest.main()

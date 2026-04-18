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
    save_results_csv,
    save_run_metrics_json,
)


class TestExperimentOutputs(unittest.TestCase):
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

            with metrics_json.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["run_metadata"]["run_id"], "test-run-1")
            self.assertEqual(len(payload["rows"]), 2)
            self.assertIn("summary", payload)


if __name__ == "__main__":
    unittest.main()

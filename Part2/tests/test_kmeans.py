from __future__ import annotations

import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from Part2.src.tweet_clustering.jaccard_kmeans import JaccardKMeans, jaccard_distance
from Part2.src.tweet_clustering.types import TweetRecord


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


class TestKMeans(unittest.TestCase):
    def test_jaccard_distance_properties(self) -> None:
        self.assertEqual(jaccard_distance(frozenset({"a"}), frozenset({"a"})), 0.0)
        self.assertEqual(jaccard_distance(frozenset({"a"}), frozenset({"b"})), 1.0)
        self.assertAlmostEqual(
            jaccard_distance(frozenset({"a", "b"}), frozenset({"b", "c"})),
            1.0 - (1.0 / 3.0),
        )

    def test_clustering_separates_two_groups(self) -> None:
        tweets = [
            _record(1, "apple banana"),
            _record(2, "apple banana orange"),
            _record(3, "cat dog"),
            _record(4, "cat dog mouse"),
        ]

        model = JaccardKMeans(k=2, max_iter=25, random_state=7)
        result = model.fit(tweets)

        clusters = [set(indices) for indices in result.clusters.values()]
        self.assertIn({0, 1}, clusters)
        self.assertIn({2, 3}, clusters)
        self.assertGreaterEqual(result.sse, 0.0)
        self.assertEqual(sum(result.cluster_sizes), 4)

    def test_kmeans_handles_more_than_two_clusters(self) -> None:
        tweets = [
            _record(1, "heart health tips"),
            _record(2, "heart health advice"),
            _record(3, "covid vaccine update"),
            _record(4, "covid variant report"),
            _record(5, "exercise fitness routine"),
            _record(6, "fitness training plan"),
        ]

        model = JaccardKMeans(k=3, max_iter=25, random_state=3)
        result = model.fit(tweets)

        self.assertEqual(len(result.cluster_sizes), 3)
        self.assertEqual(sum(result.cluster_sizes), len(tweets))

    def test_kmedoids_plus_plus_initialization(self) -> None:
        tweets = [
            _record(1, "heart health"),
            _record(2, "heart care"),
            _record(3, "flu vaccine"),
            _record(4, "flu study"),
            _record(5, "fitness workout"),
        ]

        model = JaccardKMeans(
            k=3,
            max_iter=20,
            random_state=11,
            init_strategy="kmedoids++",
        )
        result = model.fit(tweets)

        self.assertEqual(result.k, 3)
        self.assertEqual(len(set(result.centroid_indices)), 3)

    def test_invalid_init_strategy_raises(self) -> None:
        with self.assertRaises(ValueError):
            JaccardKMeans(k=2, init_strategy="bad-strategy")


if __name__ == "__main__":
    unittest.main()

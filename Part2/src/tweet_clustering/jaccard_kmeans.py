from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, Sequence, Tuple

from .types import TweetRecord


def jaccard_distance(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


@dataclass
class ClusterResult:
    k: int
    centroid_indices: List[int]
    assignments: List[int]
    clusters: Dict[int, List[int]]
    sse: float
    iterations: int

    @property
    def cluster_sizes(self) -> List[int]:
        return [len(self.clusters[i]) for i in range(self.k)]


class JaccardKMeans:
    """K-means style clustering for text using Jaccard distance.

    Since tweets do not live in Euclidean space, centroid update uses a medoid:
    the tweet in each cluster with minimum total distance to others in that cluster.
    """

    def __init__(
        self,
        k: int,
        max_iter: int = 40,
        random_state: int | None = None,
        initial_centroid_indices: Sequence[int] | None = None,
        init_strategy: str = "random",
        distance_cache: MutableMapping[Tuple[int, int], float] | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if init_strategy not in {"random", "kmedoids++"}:
            raise ValueError("init_strategy must be 'random' or 'kmedoids++'")

        self.k = k
        self.max_iter = max_iter
        self.random_state = random_state
        self.initial_centroid_indices = list(initial_centroid_indices or [])
        self.init_strategy = init_strategy
        self._rng = random.Random(random_state)

        self._tweets: Sequence[TweetRecord] = []
        self._n = 0
        if distance_cache is None:
            self._distance_cache: MutableMapping[Tuple[int, int], float] = {}
            self._owns_distance_cache = True
        else:
            self._distance_cache = distance_cache
            self._owns_distance_cache = False

    def fit(self, tweets: Sequence[TweetRecord]) -> ClusterResult:
        self._tweets = tweets
        self._n = len(tweets)

        if self.k > self._n:
            raise ValueError(f"k={self.k} cannot be greater than number of tweets={self._n}")

        if self._owns_distance_cache:
            self._distance_cache.clear()
        centroids = self._initialize_centroids()

        clusters: Dict[int, List[int]] = {i: [] for i in range(self.k)}
        assignments: List[int] = [-1] * self._n
        converged = False
        iterations = 0

        for step in range(1, self.max_iter + 1):
            assignments = self._assign_points(centroids)
            clusters = self._build_clusters(assignments)
            new_centroids = self._update_centroids(clusters)
            iterations = step

            if new_centroids == centroids:
                converged = True
                break

            centroids = new_centroids

        if not converged:
            assignments = self._assign_points(centroids)
            clusters = self._build_clusters(assignments)

        sse = self._compute_sse(assignments, centroids)

        return ClusterResult(
            k=self.k,
            centroid_indices=centroids,
            assignments=assignments,
            clusters=clusters,
            sse=sse,
            iterations=iterations,
        )

    def _initialize_centroids(self) -> List[int]:
        if self.initial_centroid_indices:
            if len(self.initial_centroid_indices) != self.k:
                raise ValueError("Length of initial_centroid_indices must be exactly k")

            unique = set(self.initial_centroid_indices)
            if len(unique) != self.k:
                raise ValueError("initial_centroid_indices must be unique")

            if any(idx < 0 or idx >= self._n for idx in unique):
                raise ValueError("All initial centroid indices must be in range [0, n)")

            return list(self.initial_centroid_indices)

        if self.init_strategy == "random":
            return self._rng.sample(range(self._n), self.k)

        return self._initialize_centroids_kmedoids_pp()

    def _initialize_centroids_kmedoids_pp(self) -> List[int]:
        """Distance centroid initialization for k-medoids.

        """
        if self.k == 1:
            return [self._rng.randrange(self._n)]

        centroids = [self._rng.randrange(self._n)]
        selected = {centroids[0]}

        while len(centroids) < self.k:
            candidates: List[int] = []
            weights: List[float] = []

            for point_idx in range(self._n):
                if point_idx in selected:
                    continue

                nearest = min(self._distance(point_idx, c_idx) for c_idx in centroids)
                weight = nearest * nearest
                candidates.append(point_idx)
                weights.append(weight)

            if not candidates:
                break

            total_weight = sum(weights)
            if total_weight <= 0:
                next_idx = self._rng.choice(candidates)
            else:
                threshold = self._rng.random() * total_weight
                cumulative = 0.0
                next_idx = candidates[-1]
                for candidate, weight in zip(candidates, weights):
                    cumulative += weight
                    if cumulative >= threshold:
                        next_idx = candidate
                        break

            centroids.append(next_idx)
            selected.add(next_idx)

        if len(centroids) < self.k:
            remaining = [idx for idx in range(self._n) if idx not in selected]
            fill_count = self.k - len(centroids)
            centroids.extend(self._rng.sample(remaining, fill_count))

        return centroids

    def _assign_points(self, centroids: Sequence[int]) -> List[int]:
        assignments = [-1] * self._n
        for point_idx in range(self._n):
            best_cluster = 0
            best_distance = float("inf")

            for cluster_idx, centroid_idx in enumerate(centroids):
                distance = self._distance(point_idx, centroid_idx)
                if distance < best_distance or (
                    distance == best_distance and cluster_idx < best_cluster
                ):
                    best_distance = distance
                    best_cluster = cluster_idx

            assignments[point_idx] = best_cluster

        return assignments

    def _build_clusters(self, assignments: Sequence[int]) -> Dict[int, List[int]]:
        clusters = {i: [] for i in range(self.k)}
        for point_idx, cluster_idx in enumerate(assignments):
            clusters[cluster_idx].append(point_idx)
        return clusters

    def _update_centroids(self, clusters: Dict[int, List[int]]) -> List[int]:
        new_centroids: List[int] = []
        selected = set()

        for cluster_idx in range(self.k):
            members = clusters[cluster_idx]
            if not members:
                replacement = self._select_replacement_centroid(selected)
                new_centroids.append(replacement)
                selected.add(replacement)
                continue

            medoid = self._find_medoid(members)
            new_centroids.append(medoid)
            selected.add(medoid)

        return new_centroids

    def _find_medoid(self, members: Sequence[int]) -> int:
        best_member = members[0]
        best_total_distance = float("inf")

        for candidate in members:
            total_distance = 0.0
            for other in members:
                if candidate == other:
                    continue
                total_distance += self._distance(candidate, other)

            if total_distance < best_total_distance or (
                total_distance == best_total_distance and candidate < best_member
            ):
                best_total_distance = total_distance
                best_member = candidate

        return best_member

    def _select_replacement_centroid(self, already_selected: set[int]) -> int:
        candidates = [idx for idx in range(self._n) if idx not in already_selected]
        if not candidates:
            return self._rng.randrange(self._n)

        if not already_selected:
            return candidates[0]

        best_idx = candidates[0]
        best_score = -1.0

        for candidate in candidates:
            nearest_selected = min(
                self._distance(candidate, selected_idx)
                for selected_idx in already_selected
            )
            if nearest_selected > best_score or (
                nearest_selected == best_score and candidate < best_idx
            ):
                best_score = nearest_selected
                best_idx = candidate

        return best_idx

    def _compute_sse(self, assignments: Sequence[int], centroids: Sequence[int]) -> float:
        sse = 0.0
        for point_idx, cluster_idx in enumerate(assignments):
            centroid_idx = centroids[cluster_idx]
            distance = self._distance(point_idx, centroid_idx)
            sse += distance * distance
        return sse

    def _distance(self, idx_a: int, idx_b: int) -> float:
        if idx_a == idx_b:
            return 0.0

        key = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
        cached = self._distance_cache.get(key)
        if cached is not None:
            return cached

        dist = jaccard_distance(self._tweets[idx_a].token_set, self._tweets[idx_b].token_set)
        self._distance_cache[key] = dist
        return dist

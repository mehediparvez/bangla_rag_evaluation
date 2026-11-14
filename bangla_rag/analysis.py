"""Statistics for comparing the evaluation conditions on the same test questions.

* ``mcnemar_midp``: paired test on exact match using the mid-p version (Fagerland et al., 2013).
* ``ClusterBootstrap``: percentile bootstrap that resamples whole passages, because about five test
  questions share a passage and are not independent. The same resampled passages are used for every
  condition, so differences between conditions are paired.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def mcnemar_midp(only_a: int, only_b: int) -> float:
    """Two-sided mid-p McNemar p-value from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    m = min(only_a, only_b)
    return float(min(1.0, 2 * binom.cdf(m, n, 0.5) - binom.pmf(m, n, 0.5)))


class ClusterBootstrap:
    """Resamples clusters (passages) with replacement and evaluates ratio statistics on each resample."""

    def __init__(self, cluster_of_item, n_boot: int = 10_000, seed: int = 13, alpha: float = 0.05):
        _, self.cluster = np.unique(np.asarray(cluster_of_item), return_inverse=True)
        self.n_clusters = int(self.cluster.max()) + 1
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, self.n_clusters, size=(n_boot, self.n_clusters))
        self.q = (100 * alpha / 2, 100 * (1 - alpha / 2))

    def _per_cluster(self, values, mask):
        values = np.asarray(values, dtype=float)
        mask = np.asarray(mask, dtype=float)
        num = np.bincount(self.cluster, weights=values * mask, minlength=self.n_clusters)
        den = np.bincount(self.cluster, weights=mask, minlength=self.n_clusters)
        return num, den

    def _draws(self, values, mask):
        num, den = self._per_cluster(values, mask)
        return num[self.idx].sum(1) / np.maximum(den[self.idx].sum(1), 1e-12)

    def mean_ci(self, values, mask):
        """(point estimate, low, high) of the mean of ``values`` over the items selected by ``mask``."""
        num, den = self._per_cluster(values, mask)
        point = num.sum() / den.sum() if den.sum() else float("nan")
        lo, hi = np.percentile(self._draws(values, mask), self.q)
        return float(point), float(lo), float(hi)

    def diff_ci(self, values_a, values_b, mask):
        """(point, low, high) of mean(a) - mean(b) on the same items, resampled together."""
        na, den = self._per_cluster(values_a, mask)
        nb, _ = self._per_cluster(values_b, mask)
        point = (na.sum() - nb.sum()) / den.sum()
        draws = self._draws(values_a, mask) - self._draws(values_b, mask)
        lo, hi = np.percentile(draws, self.q)
        return float(point), float(lo), float(hi)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(proportion, low, high): Wilson score interval for k successes out of n, for small hand-labelled samples."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(p), float(centre - half), float(centre + half)

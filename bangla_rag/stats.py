"""Small statistics helpers (numpy/pandas only, so they run anywhere)."""
from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean with a percentile bootstrap confidence interval: (mean, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(arr.mean()), float(lo), float(hi)


def mcnemar_exact(correct_a, correct_b):
    """Exact two-sided McNemar test on paired binary outcomes.

    Returns (a_only, b_only, p) where ``a_only`` counts items A got right and B wrong.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return a_only, b_only, float(min(1.0, p))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[name] = running
    return adjusted


def spearman(x, y) -> float:
    xs, ys = pd.Series(list(x), dtype=float), pd.Series(list(y), dtype=float)
    rx, ry = xs.rank().to_numpy(), ys.rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Spearman rho with a bootstrap CI over resampled pairs: (rho, lo, hi)."""
    x, y = np.asarray(list(x), float), np.asarray(list(y), float)
    rho = spearman(x, y)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        r = spearman(x[i], y[i])
        if not np.isnan(r):
            vals.append(r)
    if not vals:
        return rho, float("nan"), float("nan")
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rho, float(lo), float(hi)


def cohen_kappa(a, b, weights: str | None = None) -> float:
    """Cohen's kappa; ``weights='quadratic'`` for ordinal scales such as 1-3."""
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    n = obs.sum()
    if n == 0 or k < 2:
        return float("nan")
    exp = np.outer(obs.sum(1), obs.sum(0)) / n
    if weights == "quadratic":
        w = np.subtract.outer(np.arange(k), np.arange(k)) ** 2 / (k - 1) ** 2
    else:
        w = 1 - np.eye(k)
    denom = (w * exp).sum()
    return float("nan") if denom == 0 else float(1 - (w * obs).sum() / denom)

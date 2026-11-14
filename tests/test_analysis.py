import numpy as np
import pytest
from scipy.stats import binomtest

from bangla_rag.analysis import ClusterBootstrap, mcnemar_midp
from bangla_rag.t5_data import question_key


def test_mcnemar_midp_is_exact_p_minus_half_the_point_probability():
    from scipy.stats import binom

    only_a, only_b = 30, 12
    exact = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue
    assert mcnemar_midp(only_a, only_b) == pytest.approx(exact - binom.pmf(12, 42, 0.5))
    assert mcnemar_midp(only_a, only_b) < exact          # mid-p is less conservative than exact
    assert mcnemar_midp(5, 5) == pytest.approx(1 - binom.pmf(5, 10, 0.5))   # tie: p just below 1
    assert mcnemar_midp(0, 0) == 1.0
    assert mcnemar_midp(30, 12) == mcnemar_midp(12, 30)                      # symmetric


def test_cluster_bootstrap_point_estimate_and_ci_cover_the_mean():
    rng = np.random.default_rng(0)
    clusters = np.repeat(np.arange(100), 5)               # 100 passages x 5 questions
    values = rng.integers(0, 2, size=500).astype(float)
    boot = ClusterBootstrap(clusters, n_boot=2000, seed=1)
    point, lo, hi = boot.mean_ci(values, np.ones(500))
    assert point == pytest.approx(values.mean())
    assert lo < point < hi
    # a mask selects a subset
    mask = np.arange(500) % 2 == 0
    p2, _, _ = boot.mean_ci(values, mask)
    assert p2 == pytest.approx(values[mask].mean())


def test_cluster_bootstrap_difference_is_paired():
    clusters = np.repeat(np.arange(50), 4)
    a = np.ones(200)
    b = np.ones(200)
    boot = ClusterBootstrap(clusters, n_boot=500, seed=2)
    point, lo, hi = boot.diff_ci(a, b, np.ones(200))
    assert (point, lo, hi) == (0.0, 0.0, 0.0)             # identical systems: no difference at all


def test_cluster_bootstrap_is_wider_when_questions_in_a_passage_agree():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(100), 5)
    per_passage = rng.integers(0, 2, size=100).astype(float)
    correlated = np.repeat(per_passage, 5)                # all 5 questions of a passage share the outcome
    independent = rng.permutation(correlated)             # same mean, no clustering
    boot = ClusterBootstrap(clusters, n_boot=2000, seed=4)
    _, lo_c, hi_c = boot.mean_ci(correlated, np.ones(500))
    _, lo_i, hi_i = boot.mean_ci(independent, np.ones(500))
    assert (hi_c - lo_c) > (hi_i - lo_i)


def test_question_key_separates_reused_ids():
    assert question_key("bn_wiki_0059_01", " বেকেরেল কী? ") != question_key("bn_wiki_0059_01", "শাহ আমানত সেতু কোথায়?")

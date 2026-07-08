import numpy as np

from src.eval.metrics import (
    expected_calibration_error,
    mean_ndcg_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_perfect_calibration_has_zero_ece():
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = np.full(10, 0.5)
    assert expected_calibration_error(y, p) < 1e-9


def test_ece_detects_miscalibration():
    y = np.zeros(100)
    p = np.full(100, 0.9)  # confidently wrong every time
    assert expected_calibration_error(y, p) > 0.5


def test_ndcg_perfect_order_is_one():
    rel = np.array([1, 1, 0, 0, 0])
    assert abs(ndcg_at_k(rel, k=5) - 1.0) < 1e-9


def test_ndcg_worst_order_is_low():
    rel = np.array([0, 0, 0, 1, 1])
    assert ndcg_at_k(rel, k=5) < ndcg_at_k(np.array([1, 1, 0, 0, 0]), k=5)


def test_mean_ndcg_ignores_groups_with_no_relevant_items():
    groups = [np.array([0, 0, 0]), np.array([1, 0, 0])]
    result = mean_ndcg_at_k(groups, k=3)
    assert result == ndcg_at_k(np.array([1, 0, 0]), k=3)


def test_recall_at_k_counts_hits_within_cutoff():
    retrieved = np.array([5, 2, 9, 1, 7])
    relevant = {2, 7, 100}
    assert recall_at_k(retrieved, relevant, k=5) == 2 / 3


def test_recall_at_k_respects_cutoff():
    retrieved = np.array([5, 2, 9, 1, 7])
    relevant = {7}
    assert recall_at_k(retrieved, relevant, k=2) == 0.0

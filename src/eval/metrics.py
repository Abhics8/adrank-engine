"""Metrics for ranking quality, calibration, and retrieval recall."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def logloss(y_true: np.ndarray, y_score: np.ndarray) -> float:
    eps = 1e-7
    return float(log_loss(y_true, np.clip(y_score, eps, 1 - eps)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Mean, over equal-width probability bins, of |predicted rate - actual
    rate|, weighted by bin size. Zero means perfectly calibrated."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    mean_pred, mean_actual, counts = [], [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            mean_pred.append(None)
            mean_actual.append(None)
            counts.append(0)
            continue
        mean_pred.append(float(y_prob[mask].mean()))
        mean_actual.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return {"mean_predicted": mean_pred, "mean_actual": mean_actual, "counts": counts}


def ndcg_at_k(relevance_ranked: np.ndarray, k: int = 10) -> float:
    """relevance_ranked: 1-D array of 0/1 relevance labels already sorted by
    the model's predicted rank (best first), truncated conceptually at k."""
    rel = relevance_ranked[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    dcg = float((rel * discounts).sum())
    ideal = np.sort(rel)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg > 0 else 0.0


def mean_ndcg_at_k(groups: list[np.ndarray], k: int = 10) -> float:
    scores = [ndcg_at_k(g, k) for g in groups if g.sum() > 0]
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(retrieved_ids: np.ndarray, relevant_ids: set, k: int) -> float:
    """Of the truly relevant (clicked) ads for a user, what fraction survive
    into the top-k retrieved candidate set? Low recall@k means the reranker
    never even sees the ad the user would have clicked -- the classic
    two-stage-system failure mode."""
    if not relevant_ids:
        return None
    hit = len(set(retrieved_ids[:k]) & relevant_ids)
    return hit / len(relevant_ids)

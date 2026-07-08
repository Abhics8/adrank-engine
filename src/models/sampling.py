"""
Negative downsampling for CTR training.

Real ad-click datasets are ~95%+ negatives; training on all of them is
wasteful, so production CTR systems commonly keep every positive but only a
fraction of negatives to cut training data volume and speed. This is
completely standard practice -- and it has a well-known side effect: the
raw model output is no longer a valid probability. If you keep only
`keep_ratio` of negatives, the model has effectively been trained on a
world where clicks are `keep_ratio` times more common than they really are,
so its raw score systematically overestimates true CTR. Ranking quality
(AUC) is preserved almost exactly, because downsampling negatives uniformly
at random doesn't change their relative order -- but using the raw score
directly for bidding (`bid x raw_score`) would consistently overpay. This
is the calibration stage's actual job: without it, `models/gbdt.py` trained
this way would be a realistic example of the miscalibration a real ad
system has to correct for.
"""
from __future__ import annotations

import numpy as np


def downsample_negatives(
    X: np.ndarray, y: np.ndarray, keep_ratio: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    keep_neg = rng.choice(neg_idx, size=int(len(neg_idx) * keep_ratio), replace=False)
    keep_idx = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]

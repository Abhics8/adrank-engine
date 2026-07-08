"""
Probability calibration.

A ranker only needs scores that *order* candidates correctly -- AUC doesn't
care if the top score is 0.9 or 0.09. But an ad auction charges roughly
bid x pCTR, so the auction needs scores that are correct *probabilities*.
A model can have great AUC and still be badly calibrated (e.g. it says 0.3
when the true rate is 0.08); isotonic regression fixes that by learning a
monotonic map from raw score to calibrated probability, without touching
the ranking.
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class Calibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_scores: np.ndarray, y: np.ndarray) -> "Calibrator":
        self.iso.fit(raw_scores, y)
        return self

    def transform(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.iso.transform(raw_scores)

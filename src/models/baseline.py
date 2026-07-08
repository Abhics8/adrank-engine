"""Logistic regression over hashed features -- the honest baseline every
CTR model should be measured against before reaching for a GBDT."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class LRBaseline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=200, C=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LRBaseline":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

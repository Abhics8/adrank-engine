"""GBDT CTR ranker. Unlike the two-tower model, this one sees user and ad
features *together* in the same tree split, so it can express interactions
(e.g. "this user's affinity x this ad's category") that a dot-product of
separately-computed embeddings cannot."""
from __future__ import annotations

import numpy as np
import lightgbm as lgb


class GBDTRanker:
    def __init__(self, **params):
        self.params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            **params,
        }
        self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "GBDTRanker":
        eval_set = [(X_val, y_val)] if X_val is not None else None
        callbacks = [lgb.early_stopping(30, verbose=False)] if eval_set else None
        self.model.fit(X, y, eval_set=eval_set, callbacks=callbacks)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

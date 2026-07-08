"""
Toy second-price auction, used to demonstrate *why* calibration matters
beyond AUC.

Setup: for each impression, two hypothetical bidders compete for the same
slot with the same underlying bid value. One bidder uses raw (uncalibrated)
GBDT scores as its pCTR estimate; the other uses the isotonic-calibrated
scores. Both have identical ranking ability (same AUC) -- only their
probability estimates differ. We simulate expected revenue/cost under
each and show the miscalibrated bidder systematically over- or under-pays
relative to the true click-through rate, even though it ranks candidates
identically.
"""
from __future__ import annotations

import numpy as np


def simulate_auction(y_true: np.ndarray, pctr_estimate: np.ndarray, bid: np.ndarray, floor: float) -> dict:
    """
    Second-price-style toy: this bidder wins whenever pctr_estimate*bid
    clears a fixed reserve `floor`, and pays pctr_estimate*bid if it wins.
    `floor` is a single fixed number applied identically to both the raw
    and calibrated scenarios (see compare_calibration_impact) -- if instead
    each scenario used its own distribution's median as the floor, a
    uniformly-inflated raw score would just shift its own threshold up
    with it and the miscalibration's effect would cancel out, hiding the
    exact problem this simulation exists to show.
    """
    expected_value = pctr_estimate * bid
    wins = expected_value > floor

    spend = expected_value[wins].sum()
    true_value_generated = (y_true[wins] * bid[wins]).sum()
    roi = (true_value_generated - spend) / spend if spend > 0 else 0.0

    return {
        "impressions_won": int(wins.sum()),
        "total_spend": float(spend),
        "true_value_generated": float(true_value_generated),
        "roi": float(roi),
    }


def compare_calibration_impact(y_true: np.ndarray, raw_score: np.ndarray, calibrated_score: np.ndarray, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    bid = rng.uniform(0.5, 5.0, size=len(y_true))  # dollars per click, if it converts
    # A reserve floor set from the true population CTR and typical bid --
    # i.e. a platform policy calibrated to reality, not to either model's
    # own (possibly wrong) output scale. Held fixed across both scenarios.
    floor = float(np.median(bid) * y_true.mean())
    return {
        "uncalibrated": simulate_auction(y_true, raw_score, bid, floor),
        "calibrated": simulate_auction(y_true, calibrated_score, bid, floor),
    }

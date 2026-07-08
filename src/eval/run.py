"""
End-to-end pipeline: generate/load data -> features -> temporal split ->
train LR baseline + GBDT -> calibrate -> train two-tower + build ANN index ->
evaluate everything -> write results/metrics.json.

This is the one command that produces every number the README reports:

    python -m src.eval.run --config configs/gbdt.yaml

Every metric in results/metrics.json is reproducible by re-running this
script with the same --seed; nothing in the README is hand-typed.
"""
from __future__ import annotations

from src import _threading_fix  # noqa: F401  (must import first, see module docstring)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from data.generate import generate
from src.eval import metrics as M
from src.eval.auction_sim import compare_calibration_impact
from src.features.pipeline import FEATURE_COLS, LABEL_COL, build_features, temporal_split
from src.models.baseline import LRBaseline
from src.models.calibrate import Calibrator
from src.models.gbdt import GBDTRanker
from src.models.sampling import downsample_negatives
from src.retrieval.index import AdIndex
from src.retrieval.two_tower import embed_ads, embed_users, train_two_tower

NEGATIVE_KEEP_RATIO = 0.2  # must match src/train.py -- see src/models/sampling.py


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def to_xy(df: pl.DataFrame):
    X = df.select(FEATURE_COLS).to_numpy()
    y = df[LABEL_COL].to_numpy()
    return X, y


def sampled_ranking_eval(
    test: pl.DataFrame,
    history: pl.DataFrame,
    ad_catalog: pl.DataFrame,
    gbdt: GBDTRanker,
    n_neg: int = 100,
    max_users: int = 2000,
    k: int = 10,
    seed: int = 0,
) -> dict:
    """
    Standard implicit-feedback ranking eval: for each user with at least one
    click in the test window, build a candidate set of {the ad(s) they
    clicked} + `n_neg` sampled ads they did *not* click, score the whole set,
    and check whether the true click surfaces near the top.

    This matters because re-ranking only the 2-4 ads a user happened to be
    shown that day (an earlier version of this function did that) gives
    almost no room for a ranker to distinguish itself from popularity --
    with so few candidates, whichever ad the log happened to show is
    usually also the popular one. Injecting genuine distractor candidates
    is what makes NDCG here actually test personalization.
    """
    rng = np.random.default_rng(seed)

    # Per-(user, category) CTR "as of" the test day: aggregated over all
    # history strictly before the test day (train+val), matching exactly
    # what a leakage-safe cumulative feature would show a real user on
    # that day, whether or not this user actually saw this category before.
    cat_hist = history.group_by(["user_id", "ad_category"]).agg(
        pl.len().alias("impr"), pl.col(LABEL_COL).sum().alias("clicks")
    )
    cat_ctr_lookup = {
        (row["user_id"], row["ad_category"]): (row["clicks"] + 1) / (row["impr"] + 20)
        for row in cat_hist.to_dicts()
    }

    pop = history.group_by("ad_id").agg(pl.col(LABEL_COL).sum().alias("pop"))
    pop_counts = dict(zip(pop["ad_id"].to_list(), pop["pop"].to_list()))

    ad_ids_all = ad_catalog["ad_id"].to_numpy()
    ad_meta = {
        row["ad_id"]: (row["ad_category"], row["ad_id_hash"], row["site_id_hash"], row["ad_category_hash"])
        for row in ad_catalog.to_dicts()
    }

    clicked_by_user = (
        test.filter(pl.col(LABEL_COL) == 1)
        .group_by("user_id")
        .agg(pl.col("ad_id").unique().alias("clicked_ads"))
    )
    user_ctx = test.group_by("user_id", maintain_order=True).agg(
        pl.col("user_hist_ctr").first(),
        pl.col("user_hist_impressions").first(),
        pl.col("hour_sin").first(),
        pl.col("hour_cos").first(),
        pl.col("device_id_hash").first(),
    )
    ctx_by_user = {row["user_id"]: row for row in user_ctx.to_dicts()}

    gbdt_ndcgs, pop_ndcgs = [], []
    user_rows = clicked_by_user.to_dicts()[:max_users]
    for row in user_rows:
        uid = row["user_id"]
        positives = list(dict.fromkeys(row["clicked_ads"]))  # de-dupe, keep order
        ctx = ctx_by_user.get(uid)
        if ctx is None:
            continue

        pos_set = set(positives)
        neg_pool = rng.choice(ad_ids_all, size=min(n_neg * 2, len(ad_ids_all)), replace=False)
        negatives = [int(a) for a in neg_pool if int(a) not in pos_set][:n_neg]
        candidates = positives + negatives
        relevance = np.array([1] * len(positives) + [0] * len(negatives))

        feat_rows, pops = [], []
        for ad_id in candidates:
            cat, ad_id_hash, site_id_hash, ad_category_hash = ad_meta[ad_id]
            cat_ctr = cat_ctr_lookup.get((uid, cat), 0.05)
            feat_rows.append(
                [
                    ctx["user_hist_ctr"],
                    ctx["user_hist_impressions"],
                    cat_ctr,
                    ctx["hour_sin"],
                    ctx["hour_cos"],
                    ad_id_hash,
                    site_id_hash,
                    ctx["device_id_hash"],
                    ad_category_hash,
                ]
            )
            pops.append(pop_counts.get(ad_id, 0))

        X = np.array(feat_rows, dtype=np.float32)
        gbdt_scores = gbdt.predict_proba(X)
        gbdt_ndcgs.append(M.ndcg_at_k(relevance[np.argsort(-gbdt_scores)], k))
        pop_ndcgs.append(M.ndcg_at_k(relevance[np.argsort(-np.array(pops))], k))

    return {
        "gbdt": float(np.mean(gbdt_ndcgs)) if gbdt_ndcgs else None,
        "popularity_baseline": float(np.mean(pop_ndcgs)) if pop_ndcgs else None,
        "n_users_evaluated": len(gbdt_ndcgs),
        "n_negatives_per_user": n_neg,
    }


def run(config: dict) -> dict:
    t0 = time.time()
    seed = config.get("seed", 42)

    raw = generate(config["data"]["days"], config["data"]["rows_per_day"], seed)
    df = build_features(raw)
    train, val, test = temporal_split(df, train_days=config["data"]["days"] - 2, val_days=1)

    X_train, y_train = to_xy(train)
    X_val, y_val = to_xy(val)
    X_test, y_test = to_xy(test)

    results = {"seed": seed, "n_train": train.height, "n_val": val.height, "n_test": test.height}

    # --- LR baseline ---
    lr = LRBaseline().fit(X_train, y_train)
    lr_scores = lr.predict_proba(X_test)
    results["lr_baseline"] = {"auc": M.auc(y_test, lr_scores), "logloss": M.logloss(y_test, lr_scores)}

    # --- GBDT ranker ---
    # Trained on negative-downsampled data, a standard practice for
    # large-scale CTR training (see src/models/sampling.py). This makes the
    # raw score a realistic example of a miscalibrated probability -- AUC is
    # essentially unaffected, but the raw score systematically overstates
    # true CTR, which is exactly what the calibration stage below corrects.
    X_train_ds, y_train_ds = downsample_negatives(X_train, y_train, NEGATIVE_KEEP_RATIO, seed=seed)
    gbdt = GBDTRanker().fit(X_train_ds, y_train_ds, X_val, y_val)
    gbdt_val_scores = gbdt.predict_proba(X_val)
    gbdt_test_scores = gbdt.predict_proba(X_test)
    results["gbdt"] = {
        "auc": M.auc(y_test, gbdt_test_scores),
        "logloss": M.logloss(y_test, gbdt_test_scores),
        "ece_before_calibration": M.expected_calibration_error(y_test, gbdt_test_scores),
    }

    # --- Calibration (fit on val, applied to test) ---
    calib = Calibrator().fit(gbdt_val_scores, y_val)
    calibrated_test_scores = calib.transform(gbdt_test_scores)
    results["calibration"] = {
        "ece_after_calibration": M.expected_calibration_error(y_test, calibrated_test_scores),
        "reliability_curve_raw": M.reliability_curve(y_test, gbdt_test_scores),
        "reliability_curve_calibrated": M.reliability_curve(y_test, calibrated_test_scores),
    }

    # Full ad catalog (id, category, hashed features), shared by the ranking
    # eval below and the retrieval index.
    ad_catalog = (
        train.select(["ad_id", "ad_category", "ad_id_hash", "site_id_hash", "ad_category_hash"])
        .unique(subset=["ad_id"])
        .sort("ad_id")
    )

    # --- Ranking quality: NDCG@10 vs popularity baseline, on candidate sets
    # of {true click} + sampled distractors (see sampled_ranking_eval) ---
    history = pl.concat([train, val])
    results["ndcg_at_10"] = sampled_ranking_eval(test, history, ad_catalog, gbdt, seed=seed)

    # --- Two-tower retrieval + ANN recall@k ---
    tt_cfg = config.get("two_tower", {})
    if tt_cfg.get("enabled", True):
        user_feat_cols = ["user_hist_ctr", "user_hist_impressions", "hour_sin", "hour_cos"]
        ad_feat_cols = ["ad_id_hash", "site_id_hash", "ad_category_hash"]
        Xu_train = train.select(user_feat_cols).to_numpy().astype(np.float32)
        Xa_train = train.select(ad_feat_cols).to_numpy().astype(np.float32)

        model = train_two_tower(Xu_train, Xa_train, epochs=tt_cfg.get("epochs", 5))

        ad_lookup = ad_catalog.select(["ad_id"] + ad_feat_cols)
        ad_ids = ad_lookup["ad_id"].to_numpy()
        ad_embeds = embed_ads(model, ad_lookup.select(ad_feat_cols).to_numpy().astype(np.float32))
        index = AdIndex(ad_embeds, ad_ids)

        test_users = test.select(["user_id"] + user_feat_cols).unique(subset=["user_id"]).sort("user_id")
        user_embeds = embed_users(model, test_users.select(user_feat_cols).to_numpy().astype(np.float32))
        k = tt_cfg.get("retrieve_k", 200)
        retrieved = index.search(user_embeds, k)

        clicked_by_user = (
            test.filter(pl.col(LABEL_COL) == 1)
            .group_by("user_id")
            .agg(pl.col("ad_id").unique().alias("clicked_ads"))
        )
        clicked_map = dict(zip(clicked_by_user["user_id"].to_list(), clicked_by_user["clicked_ads"].to_list()))

        recalls = []
        for i, uid in enumerate(test_users["user_id"].to_list()):
            relevant = set(clicked_map.get(uid, []))
            if not relevant:
                continue
            r = M.recall_at_k(retrieved[i], relevant, k)
            if r is not None:
                recalls.append(r)

        results["retrieval"] = {
            "backend": index.backend,
            "k": k,
            "recall_at_k": float(np.mean(recalls)) if recalls else None,
            "n_users_evaluated": len(recalls),
        }

    # --- Auction simulation: calibration's dollar impact ---
    results["auction_sim"] = compare_calibration_impact(y_test, gbdt_test_scores, calibrated_test_scores, seed=seed)

    results["runtime_seconds"] = round(time.time() - t0, 1)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gbdt.yaml")
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()

    config = load_config(args.config)
    results = run(config)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

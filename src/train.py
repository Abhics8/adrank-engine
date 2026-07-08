"""
Trains every artifact the serving layer needs and writes them to a
versioned run directory: models/{run_id}/{gbdt,calibrator,index,two_tower}.pkl
plus metrics.json. `src/serve/app.py` only ever reads from this directory --
it never calls fit() on anything.
"""
from __future__ import annotations

from src import _threading_fix  # noqa: F401  (must import first, see module docstring)

import argparse
import time

import numpy as np
import polars as pl
import yaml

from data.generate import generate
from src.eval.run import to_xy
from src.features.pipeline import FEATURE_COLS, LABEL_COL, build_features, temporal_split
from src.models.calibrate import Calibrator
from src.models.gbdt import GBDTRanker
from src.models.sampling import downsample_negatives
from src.retrieval.index import AdIndex
from src.retrieval.two_tower import embed_ads, train_two_tower
from src.serve.artifacts import save_artifacts

USER_FEAT_COLS = ["user_hist_ctr", "user_hist_impressions", "hour_sin", "hour_cos"]
AD_FEAT_COLS = ["ad_id_hash", "site_id_hash", "ad_category_hash"]
NEGATIVE_KEEP_RATIO = 0.2  # keep all clicks, 20% of non-clicks -- see src/models/sampling.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gbdt.yaml")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = f"models/{run_id}"

    raw = generate(config["data"]["days"], config["data"]["rows_per_day"], config.get("seed", 42))
    df = build_features(raw)
    train, val, _ = temporal_split(df, train_days=config["data"]["days"] - 2, val_days=1)

    X_train, y_train = to_xy(train)
    X_val, y_val = to_xy(val)
    X_train_ds, y_train_ds = downsample_negatives(
        X_train, y_train, NEGATIVE_KEEP_RATIO, seed=config.get("seed", 42)
    )

    gbdt = GBDTRanker().fit(X_train_ds, y_train_ds, X_val, y_val)
    val_scores = gbdt.predict_proba(X_val)
    calibrator = Calibrator().fit(val_scores, y_val)

    tt_cfg = config.get("two_tower", {})
    Xu_train = train.select(USER_FEAT_COLS).to_numpy().astype(np.float32)
    Xa_train = train.select(AD_FEAT_COLS).to_numpy().astype(np.float32)
    two_tower = train_two_tower(Xu_train, Xa_train, epochs=tt_cfg.get("epochs", 5))

    ad_lookup = train.select(["ad_id"] + AD_FEAT_COLS).unique(subset=["ad_id"]).sort("ad_id")
    ad_ids = ad_lookup["ad_id"].to_numpy()
    ad_embeds = embed_ads(two_tower, ad_lookup.select(AD_FEAT_COLS).to_numpy().astype(np.float32))
    index = AdIndex(ad_embeds, ad_ids)

    # Per-ad hashed features (id/site/category), keyed by ad_id, so the
    # serving layer can assemble a full GBDT feature row for any candidate
    # ad without re-deriving hashes on the request path. device_id_hash is
    # a per-request context feature (which device *this* user is on), so
    # it's supplied by the caller at request time, not stored per ad.
    # Raw ad_category (not just its hash) is kept too, so the serving layer
    # can look up this user's category-specific history for the feature.
    ad_category_lookup = train.select(["ad_id", "ad_category"]).unique(subset=["ad_id"])
    ad_category_by_id = dict(
        zip(ad_category_lookup["ad_id"].to_list(), ad_category_lookup["ad_category"].to_list())
    )
    ad_feature_lookup = {
        int(row["ad_id"]): {
            **{c: row[c] for c in AD_FEAT_COLS},
            "ad_category": ad_category_by_id[int(row["ad_id"])],
        }
        for row in ad_lookup.to_dicts()
    }

    save_artifacts(
        run_dir,
        gbdt=gbdt,
        calibrator=calibrator,
        index=index,
        two_tower=two_tower,
        ad_feature_lookup=ad_feature_lookup,
        metrics={"run_id": run_id, "config": config},
    )
    print(f"saved artifacts -> {run_dir}")


if __name__ == "__main__":
    main()

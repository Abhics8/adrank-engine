"""
Feature engineering + temporal split.

The one rule every function here must respect: a feature for a row on day D
may only use information from days < D. Violating this is "leakage" and is
the single most common way an offline CTR/ranking result fails to replicate
online. `tests/test_no_leakage.py` asserts this holds for every feature.
"""
from __future__ import annotations

import numpy as np
import polars as pl

HASH_BUCKETS = 2**18  # feature-hashing space for high-cardinality IDs

CATEGORICAL_COLS = ["ad_id", "site_id", "device_id", "ad_category"]


def hash_col(df: pl.DataFrame, col: str, n_buckets: int = HASH_BUCKETS) -> pl.Series:
    """Feature-hashing trick: map an unbounded-cardinality ID space into a
    fixed-size bucket space, so the model doesn't need a growing vocabulary
    for new ad_ids/site_ids seen only at serve time."""
    return (pl.col(col).hash(seed=0) % n_buckets).alias(f"{col}_hash")


def add_user_history(df: pl.DataFrame) -> pl.DataFrame:
    """Per-user trailing CTR and impression count, computed strictly from
    days *before* the current row's day (a same-day or future click could
    not have been observed yet at serving time)."""
    daily = (
        df.group_by(["user_id", "day"])
        .agg(
            pl.len().alias("day_impressions"),
            pl.col("click").sum().alias("day_clicks"),
        )
        .sort(["user_id", "day"])
    )

    daily = daily.with_columns(
        pl.col("day_impressions").cum_sum().shift(1).over("user_id").alias("cum_impr"),
        pl.col("day_clicks").cum_sum().shift(1).over("user_id").alias("cum_clicks"),
    )

    daily = daily.with_columns(
        pl.col("cum_impr").fill_null(0),
        pl.col("cum_clicks").fill_null(0),
    ).with_columns(
        (
            (pl.col("cum_clicks") + 1) / (pl.col("cum_impr") + 20)
        ).alias("user_hist_ctr"),  # Laplace-smoothed: cold-start users get ~5%, not 0 or 1
        pl.col("cum_impr").alias("user_hist_impressions"),
    )

    return df.join(
        daily.select(["user_id", "day", "user_hist_ctr", "user_hist_impressions"]),
        on=["user_id", "day"],
        how="left",
    )


def add_user_category_history(df: pl.DataFrame) -> pl.DataFrame:
    """Per-(user, ad_category) trailing CTR, computed the same
    strictly-backward-looking way as `add_user_history` but conditioned on
    category. This is the feature that lets a model exploit "this user
    likes this category of ad" -- a real personalization signal a global
    popularity baseline cannot see, since popularity is aggregated across
    all users."""
    daily = (
        df.group_by(["user_id", "ad_category", "day"])
        .agg(
            pl.len().alias("day_impressions"),
            pl.col("click").sum().alias("day_clicks"),
        )
        .sort(["user_id", "ad_category", "day"])
    )

    daily = daily.with_columns(
        pl.col("day_impressions").cum_sum().shift(1).over(["user_id", "ad_category"]).alias("cum_impr"),
        pl.col("day_clicks").cum_sum().shift(1).over(["user_id", "ad_category"]).alias("cum_clicks"),
    )

    daily = daily.with_columns(
        pl.col("cum_impr").fill_null(0),
        pl.col("cum_clicks").fill_null(0),
    ).with_columns(
        ((pl.col("cum_clicks") + 1) / (pl.col("cum_impr") + 20)).alias("user_cat_hist_ctr"),
    )

    return df.join(
        daily.select(["user_id", "ad_category", "day", "user_cat_hist_ctr"]),
        on=["user_id", "ad_category", "day"],
        how="left",
    )


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = add_user_history(df)
    df = add_user_category_history(df)
    df = df.with_columns(
        [hash_col(df, c) for c in CATEGORICAL_COLS]
        + [
            (2 * np.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
        ]
    )
    return df


FEATURE_COLS = [
    "user_hist_ctr",
    "user_hist_impressions",
    "user_cat_hist_ctr",
    "hour_sin",
    "hour_cos",
    "ad_id_hash",
    "site_id_hash",
    "device_id_hash",
    "ad_category_hash",
]
LABEL_COL = "click"


def temporal_split(df: pl.DataFrame, train_days: int, val_days: int = 1):
    """Train on the earliest days, validate on the next day, test on the
    last day. Never shuffle across the day boundary -- a random split would
    let the model see "future" user history at train time."""
    max_day = df["day"].max()
    val_day = max_day - val_days
    train = df.filter(pl.col("day") < val_day)
    val = df.filter(pl.col("day") == val_day)
    test = df.filter(pl.col("day") == max_day)
    return train, val, test

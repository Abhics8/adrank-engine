"""
The one test that matters most in this repo.

If a feature for a row on day D can see clicks from day D or later, offline
metrics will look great and the model will underperform online -- the
single most common way CTR/ranking projects fail to replicate in
production. This test asserts the user-history feature is strictly
backward-looking, by construction, for every row.
"""
import polars as pl

from data.generate import generate
from src.features.pipeline import add_user_category_history, add_user_history


def test_user_history_uses_only_past_days():
    raw = generate(n_days=5, rows_per_day=500, seed=1)
    df = add_user_history(raw)

    # Recompute each user's cumulative clicks/impressions *strictly before*
    # each row's day using only future-blind data, then confirm it matches
    # exactly what the pipeline produced.
    daily = (
        raw.group_by(["user_id", "day"])
        .agg(pl.len().alias("impr"), pl.col("click").sum().alias("clicks"))
        .sort(["user_id", "day"])
    )

    for (user_id,), group in daily.group_by(["user_id"]):
        group = group.sort("day")
        days = group["day"].to_list()
        impr = group["impr"].to_list()
        clicks = group["clicks"].to_list()

        cum_impr, cum_clicks = 0, 0
        for i, day in enumerate(days):
            expected_ctr = (cum_clicks + 1) / (cum_impr + 20)
            actual = df.filter((pl.col("user_id") == user_id) & (pl.col("day") == day))[
                "user_hist_ctr"
            ]
            if actual.len() > 0:
                assert abs(actual[0] - expected_ctr) < 1e-9, (
                    f"leakage: user {user_id} day {day} used data from day {day} or later"
                )
            cum_impr += impr[i]
            cum_clicks += clicks[i]


def test_user_category_history_uses_only_past_days():
    """Same guard as above, but for the (user, ad_category) feature -- the
    one most likely to leak, since it's easy to accidentally group by day
    without also partitioning by category and let one category's future
    clicks leak into another's history."""
    raw = generate(n_days=5, rows_per_day=500, seed=3)
    df = add_user_category_history(raw)

    daily = (
        raw.group_by(["user_id", "ad_category", "day"])
        .agg(pl.len().alias("impr"), pl.col("click").sum().alias("clicks"))
        .sort(["user_id", "ad_category", "day"])
    )

    for (user_id, ad_category), group in daily.group_by(["user_id", "ad_category"]):
        group = group.sort("day")
        days = group["day"].to_list()
        impr = group["impr"].to_list()
        clicks = group["clicks"].to_list()

        cum_impr, cum_clicks = 0, 0
        for i, day in enumerate(days):
            expected_ctr = (cum_clicks + 1) / (cum_impr + 20)
            actual = df.filter(
                (pl.col("user_id") == user_id)
                & (pl.col("ad_category") == ad_category)
                & (pl.col("day") == day)
            )["user_cat_hist_ctr"]
            if actual.len() > 0:
                assert abs(actual[0] - expected_ctr) < 1e-9, (
                    f"leakage: user {user_id} category {ad_category} day {day} "
                    "used data from day {day} or later"
                )
            cum_impr += impr[i]
            cum_clicks += clicks[i]


def test_temporal_split_has_no_overlap():
    from src.features.pipeline import build_features, temporal_split

    raw = generate(n_days=6, rows_per_day=300, seed=2)
    df = build_features(raw)
    train, val, test = temporal_split(df, train_days=4, val_days=1)

    assert train["day"].max() < val["day"].min()
    assert val["day"].max() < test["day"].min()
    assert train.height + val.height + test.height <= df.height

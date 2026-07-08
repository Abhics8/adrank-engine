"""
Online serving path: retrieve -> rerank -> calibrate -> auction.

POST /rank {user_id, device_id, hour, bids: {ad_id: bid}} returns the top-k
ads by calibrated-pCTR x bid, with a stage-by-stage latency breakdown. The
expensive GBDT only ever scores the few hundred candidates the retriever
returns, never the full ad corpus -- that funnel is the reason two-stage
systems exist.
"""
from __future__ import annotations

from src import _threading_fix  # noqa: F401  (must import first, see module docstring)

import math
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import polars as pl
from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from src.features.pipeline import FEATURE_COLS, HASH_BUCKETS
from src.serve.artifacts import Artifacts, load_artifacts

RUN_DIR = os.environ.get("YIELDGUARD_RUN_DIR", "models/latest")

stage_latency = Histogram(
    "yieldguard_stage_latency_seconds", "Latency per serving stage", ["stage"]
)

_artifacts: Artifacts | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _artifacts
    _artifacts = load_artifacts(RUN_DIR)
    yield


app = FastAPI(title="AdRank serving API", lifespan=lifespan)


def _hash(value: int) -> int:
    return int(pl.Series([value]).hash(seed=0)[0] % HASH_BUCKETS)


class RankRequest(BaseModel):
    user_id: int
    device_id: int
    hour: int
    user_hist_ctr: float = 0.05
    user_hist_impressions: int = 0
    # This user's trailing CTR specifically within each ad's category, e.g.
    # {"3": 0.09} means a 9% historical CTR for this user on category 3.
    # In a production system this would come from a feature store keyed by
    # (user, category); here the caller supplies it directly, same as
    # user_hist_ctr above. Missing categories fall back to user_hist_ctr.
    user_category_hist_ctr: dict[str, float] = {}
    candidate_bids: dict[str, float]  # ad_id (str) -> bid
    top_k: int = 10


class RankedAd(BaseModel):
    ad_id: int
    raw_score: float
    calibrated_pctr: float
    bid: float
    expected_value: float


class RankResponse(BaseModel):
    ranked: list[RankedAd]
    latency_ms: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "run_dir": RUN_DIR}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest):
    assert _artifacts is not None
    latency = {}

    # --- retrieve ---
    t0 = time.perf_counter()
    hour_sin = math.sin(2 * math.pi * req.hour / 24)
    hour_cos = math.cos(2 * math.pi * req.hour / 24)
    user_vec = np.array(
        [[req.user_hist_ctr, req.user_hist_impressions, hour_sin, hour_cos]], dtype=np.float32
    )
    from src.retrieval.two_tower import embed_users

    user_embed = embed_users(_artifacts.two_tower, user_vec)
    k = min(500, len(_artifacts.ad_feature_lookup))
    retrieved_ids = _artifacts.index.search(user_embed, k)[0]
    latency["retrieve_ms"] = (time.perf_counter() - t0) * 1000
    stage_latency.labels(stage="retrieve").observe(time.perf_counter() - t0)

    # Only rank candidates the caller actually has a bid for.
    device_hash = _hash(req.device_id)
    candidates = [
        int(aid) for aid in retrieved_ids if str(int(aid)) in req.candidate_bids
    ]
    if not candidates:
        candidates = [int(a) for a in req.candidate_bids if int(a) in _artifacts.ad_feature_lookup]

    # --- rerank ---
    t0 = time.perf_counter()
    rows = []
    for ad_id in candidates:
        ad_feat = _artifacts.ad_feature_lookup[ad_id]
        cat_ctr = req.user_category_hist_ctr.get(
            str(ad_feat["ad_category"]), req.user_hist_ctr
        )
        row = {
            "user_hist_ctr": req.user_hist_ctr,
            "user_hist_impressions": req.user_hist_impressions,
            "user_cat_hist_ctr": cat_ctr,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "ad_id_hash": ad_feat["ad_id_hash"],
            "site_id_hash": ad_feat["site_id_hash"],
            "device_id_hash": device_hash,
            "ad_category_hash": ad_feat["ad_category_hash"],
        }
        rows.append([row[c] for c in FEATURE_COLS])
    X = np.array(rows, dtype=np.float32) if rows else np.zeros((0, len(FEATURE_COLS)))
    raw_scores = _artifacts.gbdt.predict_proba(X) if len(X) else np.array([])
    latency["rerank_ms"] = (time.perf_counter() - t0) * 1000
    stage_latency.labels(stage="rerank").observe(time.perf_counter() - t0)

    # --- calibrate ---
    t0 = time.perf_counter()
    calibrated = _artifacts.calibrator.transform(raw_scores) if len(raw_scores) else np.array([])
    latency["calibrate_ms"] = (time.perf_counter() - t0) * 1000
    stage_latency.labels(stage="calibrate").observe(time.perf_counter() - t0)

    # --- auction: rank by calibrated pCTR x bid ---
    t0 = time.perf_counter()
    ranked = []
    for i, ad_id in enumerate(candidates):
        bid = req.candidate_bids.get(str(ad_id), 0.0)
        ranked.append(
            RankedAd(
                ad_id=ad_id,
                raw_score=float(raw_scores[i]),
                calibrated_pctr=float(calibrated[i]),
                bid=bid,
                expected_value=float(calibrated[i]) * bid,
            )
        )
    ranked.sort(key=lambda r: r.expected_value, reverse=True)
    latency["auction_ms"] = (time.perf_counter() - t0) * 1000
    stage_latency.labels(stage="auction").observe(time.perf_counter() - t0)

    return RankResponse(ranked=ranked[: req.top_k], latency_ms=latency)

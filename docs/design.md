# Design notes

Decisions and trade-offs, recorded as they were made. This doubles as interview
prep: every choice below is something a reviewer of a real ranking system
would ask about.

## Why synthetic data instead of Avazu/Criteo

Avazu (40M rows) and Criteo need a Kaggle account, a multi-GB download, and
don't reproduce in CI. `data/generate.py` produces a smaller dataset with the
same structural properties that make CTR modeling non-trivial: extreme class
imbalance, high-cardinality IDs that must be hashed, a real time axis, and
user-level heterogeneity so history features carry signal. Every number in
this README comes from this generator with a fixed seed, so the whole
pipeline reproduces on a laptop with no external downloads. Swapping in real
Avazu/Criteo data means writing a loader with the same output schema as
`generate()` — the rest of the pipeline doesn't change.

## Why a temporal split, not a random split

A random train/test split lets the model see a user's future clicks when
computing that user's *past* history feature for an earlier row — the model
learns to "predict" using information that wouldn't exist yet in production.
Random splits reliably overstate offline AUC relative to what the same model
achieves online. `temporal_split()` trains on the earliest days only, and
`test_no_leakage.py` asserts every user-history feature is computed strictly
from days before the row's own day.

## Why two-stage (retrieve, then rerank) instead of one model

Scoring every ad in the corpus with a GBDT for every request doesn't scale —
if there are 1M ads, that's 1M tree traversals per request. A two-tower model
lets you precompute all ad embeddings offline once; at request time you do one
user-tower forward pass and one approximate-nearest-neighbor lookup (FAISS),
cutting the candidate set to a few hundred before the expensive reranker runs.
The cost of this: the two-tower model can't express interactions between user
and ad features (it only ever computes a dot product of two independently-
computed vectors), which is why the GBDT reranker — which sees user and ad
features *together* in the same tree split — still adds value on top of
retrieval. `recall@k` in `results/metrics.json` measures how much true signal
survives the retrieval cut; if it's low, the reranker never even sees the ad
the user would have clicked, regardless of how good the reranker is.

## Why calibration is a separate stage from ranking

AUC only measures whether the model orders candidates correctly — it's blind
to whether a score of 0.3 means the true click rate is actually 30% or 8%.
But an ad auction charges roughly `bid x pCTR`, so a bidder using
miscalibrated scores will systematically over- or under-pay relative to true
value, even with perfect ranking (same AUC). `auction_sim.py` demonstrates
this directly: two identical rankers, one using raw GBDT scores and one using
isotonic-calibrated scores, are run through a toy auction, and the ROI
difference is the practical cost of skipping calibration. Isotonic regression
(rather than Platt scaling) was chosen because it makes no assumption about
the shape of the miscalibration curve — it only assumes monotonicity, which
holds for any reasonable ranker.

## Why isotonic regression is fit on the validation set, not train

Fitting the calibrator on the same rows the GBDT trained on would let it
calibrate against scores the model has effectively memorized, understating
real-world miscalibration. It's fit on the held-out validation split instead,
so the correction reflects genuine out-of-sample overconfidence/underconfidence.

## Why the GBDT is trained on negative-downsampled data

An earlier version of this project trained the GBDT on the full, naturally
imbalanced data and found it was already well-calibrated out of the box
(ECE ~0.001) — meaning the calibration stage had nothing real to fix, and the
whole "calibration matters" story was assertion, not evidence. Rather than
force a fabricated miscalibration, `src/models/sampling.py` introduces one
that's genuinely realistic: negative downsampling (keep all clicks, 20% of
non-clicks), a completely standard practice for large-scale CTR training
since it cuts training volume without hurting ranking quality (uniformly
downsampling negatives doesn't change their relative order). Its well-known
side effect is that the raw score no longer reflects the true click rate —
in this repo, training this way pushes `ece_before_calibration` to ~0.15 and
isotonic regression brings it back to ~0.002 (see `results/metrics.json`).
The auction simulation quantifies what that miscalibration costs directly:
ROI recovers from roughly -75% (uncalibrated) to near break-even
(calibrated) under a fixed reserve-price policy.

## Why the auction simulation uses a fixed floor, not a per-run median

The first version of `auction_sim.py` set each bidder's win threshold to the
median of *its own* score distribution. That's a bug in the experiment
design, not just the code: if a score is uniformly inflated, its own median
inflates by the same factor, so the miscalibration cancels itself out and
the simulation can't show the problem it exists to demonstrate. The fixed
version anchors the floor to the true population CTR and the median bid —
a stand-in for "the platform's reserve price is set from real historical
performance, not from whatever a given model happens to output" — applied
identically to both the raw and calibrated bidder.

## Why users are far fewer than one might expect (2,000, not 20,000+)

The `user_cat_hist_ctr` feature (this user's trailing CTR within a specific
ad category) is the whole point of the personalization story — it's the
signal a global popularity baseline structurally cannot see. But with
20,000 users, 20 categories, and ~600K rows, each (user, category) pair sees
on average 1.5 impressions — far too sparse for a Laplace-smoothed
historical CTR to mean anything, and NDCG@10 confirmed it: the popularity
baseline beat the GBDT reranker. Dropping to 2,000 users (holding data
volume fixed) raises that to ~15 impressions per (user, category) pair,
enough for the feature to actually carry signal — after which GBDT clearly
beats popularity (NDCG@10 0.20 vs 0.17). This is an honest scale trade-off,
not a free lunch: a real system with millions of users would need either
much longer history windows or a learned embedding (e.g., matrix
factorization on the user x category matrix) instead of a raw historical
aggregate, precisely because real production data is this sparse per user.

## Why NDCG@10 samples negative candidates instead of re-ranking the day's ads

`sampled_ranking_eval` in `src/eval/run.py` builds each user's candidate set
as {ad(s) they clicked} + 100 sampled ads they didn't interact with, then
checks whether the true click surfaces near the top of the scored list —
the standard implicit-feedback recsys evaluation. An earlier version just
re-ranked the 2-4 ads a user happened to be shown that day; with so few
candidates there's almost no room for a personalized ranker to distinguish
itself from popularity, since whichever ad the log happened to serve is
usually also the popular one. Injecting real distractors is what makes the
comparison actually test personalization rather than measurement noise.

## What would break this at real scale

- **Cold start**: a brand-new user has no history, so `user_hist_ctr` falls
  back to the Laplace-smoothed prior (~5%) — reasonable, but a real system
  would want a proper cold-start model or content-based fallback.
- **Position bias**: the training data has no notion of "this ad converted
  because it was shown first," which real click logs are full of. A
  production version needs propensity scoring or a position feature.
- **Single-machine scale**: FAISS `IndexFlatIP` is exact brute-force; at real
  ad-corpus scale (tens of millions of ads) this would need `IndexIVFPQ` or a
  sharded ScaNN index, trading recall for latency.
- **Feature hashing collisions**: `HASH_BUCKETS = 2^18` was chosen so
  collision rate stays low for the corpus sizes here; a real ad marketplace
  with orders of magnitude more IDs would need a larger space or a learned
  embedding table instead of raw hashing.

## What isn't built

- No production feature store — training and serving share code paths in
  this repo but wouldn't in a system with independent train/serve
  infrastructure and the online/offline skew that implies.
- No online A/B testing harness. Every result here is an offline metric;
  the design notes above call out where offline and online are expected to
  diverge.

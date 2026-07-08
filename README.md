# YieldGuard

**What this is, in one sentence:** a system that decides which ad to show
someone, guesses how likely they are to click it, and — the important part —
makes sure that guess is actually *right*, not just a good guess dressed up
as a number.

## The problem, explained without jargon

Imagine a weather forecaster who says "70% chance of rain" every single day,
rain or shine. If it only actually rains on 40% of those days, their
forecasts are useless — even if they're technically better than a coin
flip. The number they say has to *mean* what it says.

Ad systems have the exact same problem. When an ad platform decides which ad
to show you, it estimates "there's an 8% chance this person clicks this ad."
Advertisers then pay based on that number. If the system is overconfident —
saying 8% when the real number is closer to 2% — the platform ends up
selling advertisers a bunch of clicks that were never going to happen,
advertisers overpay, and everyone's trust in the numbers erodes. That quiet
overpayment is where the name **YieldGuard** comes from: yield (how much
money a platform actually earns per ad shown) leaks out when the numbers
being sold aren't calibrated to reality, and this project exists to catch
that before it ships.

Most portfolio projects on this topic stop at "did I rank things well?" This
one goes a step further and checks "are my *numbers* trustworthy enough to
put a dollar figure on?" — and then puts an actual dollar figure on what
happens if you skip that check.

## How it works, in plain terms

There are two problems to solve, in order:

1. **Find the right ads fast.** If a platform has millions of possible ads,
   it can't carefully evaluate every single one for every single visitor —
   that would be too slow. So the system first does a quick, rough pass to
   narrow millions of ads down to a few hundred plausible candidates (this
   step is called **retrieval**).
2. **Carefully rank and price the survivors.** Once the candidate list is
   small, a more careful model scores each one properly, and a final step
   converts that score into an honest, trustworthy percentage before any
   money changes hands (this step is **reranking + calibration**).

Concretely: raw activity logs → a fast matching step (like a librarian
narrowing "which shelf" before you browse) → a more careful ranking model →
a calibration step that corrects the model's confidence to match reality →
a mock auction that shows, in dollars, why that last step matters.

## The headline result

The single most important finding in this project: **fixing the confidence
of the model's predictions — without changing what it ranks first or last —
turned a badly losing simulated auction into a roughly break-even one.**

| | before fixing confidence | after fixing confidence |
|---|---|---|
| how far off the predicted click-rate was from reality | way off (see below) | almost exactly right |
| simulated return on ad spend | **-75%** (losing 75 cents per dollar) | **-2%** (nearly break-even) |

Nothing about which ads got shown changed between these two columns — only
whether the system's stated confidence could be trusted. That's the whole
point of calibration: it's invisible in a simple "did it rank things
correctly" check, but it's the difference between a healthy business and
one quietly bleeding money.

## Results, for the technically curious

Everything below is regenerated from scratch by one command and never
hand-typed:

```
python -m src.eval.run --config configs/gbdt.yaml
```

`results/metrics.json` is what these numbers are read from. `docs/design.md`
explains the reasoning behind every decision, including two mistakes made
along the way and how they were caught (see below) — written to double as
interview prep.

All numbers are on synthetic data standing in for real ad-click logs (see
"Why synthetic data" below), seed 42, 480K training / 60K validation / 60K
test rows.

### Ranking quality

| model | AUC (higher = better ordering) | log loss |
|---|---|---|
| logistic regression (simple baseline) | 0.716 | 0.177 |
| GBDT (the main model) | 0.715 | 0.274 |

These are basically tied — on this data, the underlying pattern is fairly
simple, so a basic model captures most of the ranking signal. The GBDT's
real advantage shows up on *personalized* ranking, not this overall score:

| | NDCG@10 (higher = better; measures if the ad someone actually wanted ends up near the top) |
|---|---|
| GBDT (personalized) | **0.202** |
| just showing the most generally popular ads | 0.166 |

(1,374 users tested, each with their actual clicked ad mixed in among 100
random decoys, to check whether the model can find the needle in the
haystack.) The GBDT wins because it uses something a "just show what's
popular" approach structurally can't: *this specific person's* history with
*this specific category* of ad. Getting this comparison to be fair took two
real fixes — see "What went wrong" below.

### Calibration: the centerpiece result

The main model is deliberately trained the way large real-world systems
often are — on a shrunk-down, resampled version of the data to save
computing costs. This is standard practice, but it has a well-known side
effect: it makes the model's raw confidence score inflated and untrustworthy.

| | Expected Calibration Error (0 = perfectly trustworthy numbers) |
|---|---|
| raw model score | 0.147 |
| after the calibration fix | 0.0018 |

That gap isn't just an abstract statistic — it has a direct dollar cost,
shown by running a mock auction (two identical bidders, same items, same
bids, one trusting the raw score and one trusting the calibrated score):

| | ads won | money spent | actual value received | return on spend |
|---|---|---|---|---|
| trusting the raw (uncalibrated) score | 56,068 | $31,783 | $7,893 | **-75.2%** |
| trusting the calibrated score | 19,609 | $5,504 | $5,376 | **-2.3%** |

The uncalibrated bidder wins way more auctions than it should, because its
inflated confidence makes cheap, low-quality traffic look like a great deal.
It ends up paying for a lot of clicks that were never going to happen.

### Retrieval (the "find candidates fast" step)

The fast-matching step correctly recovers about **1 in 10** of the ads a
person would have actually clicked, when narrowing 2,000 ads down to the
top 200. That's an honest, unflattering number — the matching model here is
intentionally small and lightly trained, and this is the clearest place a
more serious version of this project would improve next (a bigger model,
more training time, or smarter negative examples would all likely help;
none of that was done here, specifically so this number stays real instead
of tuned to look good).

## What went wrong along the way (and how it was caught)

Two of the results above only exist because an earlier version of this
project gave a *worse*, less flattering result, and that was investigated
instead of quietly reported as-is:

- **The personalized model first lost to "just show what's popular."** The
  feature meant to capture personal preference existed, but there wasn't
  enough repeat data per person for it to mean anything (each person had
  seen a given ad category only 1-2 times on average) — so the model had
  nothing real to learn from, and popularity fairly won. Fixed by shrinking
  the simulated audience size so the same amount of data gives each person
  enough repeat history to actually learn from (see `docs/design.md`).
- **The model was already almost perfectly confident-and-correct**, which
  meant there was nothing for the calibration fix to actually fix, and the
  entire "calibration matters" story would have been an unproven claim.
  Fixed by training the model the more realistic way described above, which
  reproduces the exact kind of overconfidence a real system has to correct
  for.

Both are written up in full, with the actual before/after numbers, in
`docs/design.md`.

## Why synthetic data instead of real ad logs

Real public ad-click datasets (Avazu, Criteo) need a Kaggle account and a
multi-gigabyte download, neither of which works inside an automated test
pipeline. Instead, `data/generate.py` creates a smaller, fake-but-realistic
dataset with the same tricky properties real ad data has: very few clicks
relative to views (~5%), huge numbers of distinct ads/sites/devices, a real
sense of time (so the model can't accidentally peek at the future), and a
genuine "this person tends to like this category" pattern. Every number in
this README is generated from this data with a fixed random seed, so anyone
can reproduce the exact same results on their own laptop with no downloads
or accounts needed.

## Repo layout

```
configs/          experiment settings (yaml)
data/generate.py  the fake-but-realistic click-log generator
src/features/     turns raw logs into model-ready features, safely (no peeking at the future)
src/models/       the baseline model, the main model, the calibration fix, the resampling trick
src/retrieval/    the fast "find candidates" step
src/eval/         all the scoring/metrics code, the mock auction, and the one-command results runner
src/serve/        the actual API that would run in production
src/train.py      trains and saves everything the API needs
tests/            automated checks, including the most important one: no peeking at the future
docs/design.md    every decision explained, including the two mistakes above
```

## Running it yourself

```bash
pip install -r requirements.txt
pytest                                          # runs all the automated checks
python -m src.eval.run --config configs/gbdt.yaml   # reproduces every number in this README
python -m src.train --config configs/gbdt.yaml --run-id latest  # trains the real API's model
YIELDGUARD_RUN_DIR=models/latest uvicorn src.serve.app:app --reload  # starts the live API
```

A GitHub Actions workflow automatically runs the checks and a smaller
version of the results pipeline on every change, so nothing gets pushed
without being verified.

## What's intentionally not built

This is a portfolio-scale project, not a production system, so a few things
a real company would need are deliberately left out: a proper system for
storing/serving features at scale, live A/B testing against real users,
correcting for "this ad only got clicked because it was shown first"
effects, and a real strategy for brand-new users the system has no history
on. Every one of these is called out honestly in `docs/design.md`, along
with what it would actually take to add them.

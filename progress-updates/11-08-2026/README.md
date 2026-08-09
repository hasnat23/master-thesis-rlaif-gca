# Progress Update — 11 August 2026

**Student:** Muhammad Hasnat
**Meeting:** Tuesday, 11 August 2026

---

## The professor's question

> "I believe the steps of training reward models using holistic/GCA scores
> and evaluating those reward models haven't been done yet."

**This has been done.** It's the main experiment of the thesis. Two reward
models were trained — one on "holistic" preferences, one on "GCA" preferences
— and compared by how accurately each one predicts which of two summaries is
better. That comparison, at three dataset sizes, is below.

---

## Reward model results: Holistic vs. GCA

A reward model is "accurate" when it correctly picks the better of two
summaries. 50% accuracy means it's guessing; higher is better.

| Training set size | Holistic accuracy | GCA accuracy | Who wins |
|---:|---:|---:|---|
| 1,000 examples | 52.95% | 56.03% | **GCA, clearly** — confirmed across 20 repeated runs |
| 5,000 examples | 58.55% | 58.57% | **Tie** — confirmed across 6 repeated runs |
| 10,000 examples | 58.57% | 58.54% | **Tie** — confirmed across 6 repeated runs |

**How to read this:**
- With a **small** training set (1,000 examples), the reward model trained on
  GCA-style preferences is clearly and reliably more accurate than the one
  trained on holistic preferences — about 3 points better, every single time
  it was tried (20 out of 20 runs).
- With **larger** training sets (5,000 or 10,000 examples), there is no real
  difference between the two anymore — both land around 58.5%, repeatedly.
- Accuracy for both goes up as the training set grows (from ~53–56% to
  ~58.5%), which makes sense — more data generally helps. What changes is
  that GCA's edge over holistic disappears once there's enough data.

**The takeaway to say out loud:** GCA-based preferences make the reward model
meaningfully better than holistic preferences, but only when training data is
limited. Once there's more data, it doesn't matter which one you use — they
perform the same.

The rest of this update explains how confident we can be in these numbers,
since that changed significantly in the last few days.

---

## Why we can trust these numbers

Each row in the table isn't from one lucky run — it's an average over several
repeats (20 repeats at 1,000 examples, 6 repeats each at 5,000 and 10,000),
using a version of the training code where we fixed a bug that had been
letting results vary randomly between runs. That's what lets us say "GCA
wins" or "it's a tie" with confidence instead of "it looked that way once."

This is new since the last progress note — the 1,000-example result went
from "looks promising, can't fully prove it" to statistically solid, and the
5,000/10,000 results went from "one run, not sure if meaningful" to a
confirmed, repeated tie.

---

## Still open (not urgent)

One smaller side-experiment — checking whether the reward model performs
badly because it isn't shown enough of the article — is half-finished. The
other half is stuck because a server the cluster relies on to download an AI
model is currently down. Not something we can fix on our end; the thesis
already reports what we have so far and clearly marks the rest as future
work.

---

## Questions for Tuesday

1. Does this fully answer the "has RM training been done" concern, or is
   there something else expected?
2. Given both results are now solid, is there anything else you'd want added
   before submission, or is this ready?

---

## For reference

All code, results, and the full thesis are on GitHub, up to date:
https://github.com/hasnat23/master-thesis-rlaif-gca

Full technical detail (exact numbers, confidence intervals, job logs) is in
`thesis/chapters/06_results.tex` §6.6–6.8 and `thesis/chapters/07_discussion.tex`
§7.5, not repeated here to keep this update short.

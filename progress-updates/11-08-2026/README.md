# Progress Update — 11 August 2026

**Student:** Muhammad Hasnat
**Meeting:** Tuesday, 11 August 2026

---

## The professor's question

> "I believe the steps of training reward models using holistic/GCA scores
> and evaluating those reward models haven't been done yet."

**This has been done.** It's the main experiment of the thesis. Training two
reward models (one on "holistic" preferences, one on "GCA" preferences) and
comparing their accuracy is what Chapter 6 of the thesis reports. This was
already finished before this update — the confusion is just that the last
progress note (16 July) was written before the thesis document itself made
this clear.

The rest of this update is about what's new since then: the result got
**much stronger**, and one open question got **fully answered**.

---

## What's new: two results

### 1. The main result is now statistically significant

Before: we trained 6 reward models and GCA won 5 times, with a ~3 point
average improvement. But 6 tries is too few to prove anything statistically
— even a perfect result would not have counted as "significant" by the usual
standard.

We found and fixed a bug in the training code (some randomness wasn't
properly controlled) and reran the experiment **20 times**.

**Result: GCA won all 20 times.** Average improvement: **+3.95 percentage
points**. This time it clears the standard bar for statistical significance
easily.

**In plain terms: at this dataset size, GCA reliably makes the reward model
better, and we can now say that with confidence instead of just "it looked
that way."**

### 2. At bigger datasets, the advantage genuinely disappears

The thesis also tested bigger datasets (5,000 and 10,000 examples instead of
1,000), but originally only once each — not enough to know if the advantage
was really gone or if we just got unlucky.

We reran each of those **6 times** too.

**Result: at both bigger sizes, GCA and holistic are statistically tied.**
Not "a bit worse," genuinely no difference — the numbers land almost exactly
on top of each other, 6 times in a row.

**In plain terms: GCA clearly helps when training data is limited (1,000
examples), and clearly stops mattering once there's more data (5,000+). This
is now a confirmed finding, not a guess.**

---

## What this means for the thesis

- The headline claim went from "looks promising but can't prove it" to
  **"proven, with a clear explanation of when it does and doesn't apply."**
- This is a stronger, more complete thesis than the one we had a few days
  ago.

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

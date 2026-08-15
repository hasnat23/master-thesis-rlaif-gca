# Progress Update, 18 August 2026

Student: Muhammad Hasnat
Meeting: Tuesday, 18 August 2026

---

## Context

At the 11 August meeting, Lingxiao raised a question about how precise the
reward models actually are, and followed up by email with two possible
directions for the remaining weeks:

- Direction 1: fine-tune small LLMs directly on the AlignScore signal.
- Direction 2: keep the existing trained reward models, and evaluate them
  against an independent ground truth built from AlignScore.

Given the submission deadline, Direction 1 was ruled out. It is close to the
DPO/policy fine-tuning approach that was already tried and dropped in June,
for the same reason it would be risky now: it is hard to tell whether a
change in output comes from the preference data or from the fine-tuning
itself.

This update reports on Direction 2.

---

## What this experiment does

Three steps, on top of the reward models already trained for the 11 August
result.

Step 1. Build an independent ground truth. 500 new CNN/DailyMail articles
were selected, guaranteed to be ones neither reward model was trained or
tested on before. Two candidate summaries were generated for each, exactly
as before. Each summary was then scored with AlignScore's sentence-level
scoring, aggregated the same way GCA aggregates it. For each article, the
higher-scoring summary becomes the one that "should" be preferred, and the
lower-scoring one the one that "should" be rejected.

Step 2. Retrain reward models with their weights saved. The 11 August
results only ever kept the accuracy numbers, not the trained models
themselves, so there was nothing to test against a new set of articles.
Both the holistic and the GCA reward model were retrained the same way as
before, across 20 random seeds, this time saving the actual trained model
for each run.

Step 3. Test both reward models against the same ground truth. For every
one of the 500 ground-truth pairs, each reward model is asked to score both
the "should be preferred" and the "should be rejected" summary. A model
succeeds on a pair if it scores the correct one higher. The success rate is
recorded for every one of the 20 holistic models and every one of the 20
GCA models.

This directly answers Lingxiao's question: does the GCA reward model, more
often than the holistic one, give the higher score to the summary that
should get it.

---

## Results

All 500 ground-truth pairs were scored with zero ties. 20 holistic and 20
GCA reward models were retrained and tested against the same 500 pairs.

| | Holistic RM | GCA RM |
|---|---:|---:|
| Mean success rate | 55.2% | 55.3% |
| Runs favouring | — | 9 / 20 |

Mean gap (GCA minus holistic): +0.14 percentage points
95% confidence interval: [-0.76, +1.06] points
Wilcoxon signed-rank p (two-sided): 0.76
Cohen's d: 0.07

This does not match what Lingxiao's email predicted. The two reward models
are indistinguishable on this test: GCA is ahead in 9 of 20 seeds, holistic
in the other 11, essentially a coin flip, and the gap is nowhere near
statistically significant.

The same equivalence check used for the 5,000 and 10,000-example result on
11 August was run here too: it rules out any real difference bigger than
about 1 percentage point in either direction. So this is not a case of not
enough data to see a difference — with 500 ground-truth pairs and 20 seeds
per condition, the test could have detected a true gap as small as 1.3
percentage points, and no such gap is there.

How this fits with the 11 August result. That result measured each reward
model's accuracy on examples drawn from the same preference-labelling
process it was trained on, holistic tested against holistic-style labels,
GCA against GCA-style labels. It found GCA clearly better at 1,000 training
examples. This week's test asks a different, stricter question: when both
reward models are scored against one shared, independent yardstick that
neither was built around, does GCA still come out ahead. It does not.
Put plainly: GCA preferences are easier for a reward model to learn
consistently from a small training set, but that does not carry over into
GCA producing a reward model that is a better judge of factuality by an
outside standard.

---

## What this means for the thesis

Both findings are real and both are being reported, not just the one that
was expected. Taken together they say something more specific than either
result alone: sentence-level scoring changes how learnable the preference
data is, without changing how accurate the resulting reward model is
against independent ground truth. Learnability and precision turn out to
be different properties, and this thesis's main result was always about the
former.

---

## Questions for Tuesday

1. The result is a genuine null, not a confirmation of the expected
   direction — is this still a useful answer to the precision question, or
   is there a different test you'd want run given the time remaining?
2. Given the deadline, is it enough to report both findings honestly
   (learnable but not more precise), or is further investigation needed?
3. Is anything else needed before submission, given the 3 September
   deadline?

---

## For reference

Full code is on GitHub: https://github.com/hasnat23/master-thesis-rlaif-gca

Pipeline: `scripts/03_prepare_ground_truth_subset.py`,
`slurm/generate_candidates_groundtruth.sh`, `slurm/build_ground_truth.sh`,
`slurm/retrain_for_ground_truth.sh`, `analysis/evaluate_ground_truth.py`,
`analysis/ground_truth_stats.py`.

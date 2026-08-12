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

Pending — experiments are running on MOGON as of this write-up. This
section will be filled in with the per-seed success rates, the mean gap
between GCA and holistic, and the same statistical checks used for the 11
August result (significance test, confidence interval, effect size) before
Tuesday's meeting.

---

## Questions for Tuesday

1. Does this ground-truth evaluation answer the precision question from the
   11 August meeting?
2. Is anything else needed before submission, given the 3 September
   deadline?

---

## For reference

Full code is on GitHub: https://github.com/hasnat23/master-thesis-rlaif-gca

Pipeline: `scripts/03_prepare_ground_truth_subset.py`,
`slurm/generate_candidates_groundtruth.sh`, `slurm/build_ground_truth.sh`,
`slurm/retrain_for_ground_truth.sh`, `analysis/evaluate_ground_truth.py`,
`analysis/ground_truth_stats.py`.

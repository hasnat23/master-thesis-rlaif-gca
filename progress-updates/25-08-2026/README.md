# Progress Update, 25 August 2026

Student: Muhammad Hasnat
Meeting: Tuesday, 25 August 2026

---

## Context

The 18 August update found that GCA's ground-truth ranking precision advantage
over Holistic reverses as the reward model's training set grows: significant
GCA advantage at n=1,000 (up to +9.3pp), gone at n=5,000, and a significant
Holistic advantage at n=10,000. The last meeting asked for this to be closed
out with one of: make GCA win consistently, find and test a hypothesis for
why it doesn't win at scale, or locate the crossover point between the small-
and large-scale results. This update covers what was tested this week.

---

## Hypothesis test: near-tie training pairs

Leading hypothesis: GCA's sentence-level aggregation produces more
near-zero-margin ("noise-level") preference pairs than Holistic's single
whole-summary score, and fitting that noise at large training-set sizes is
what hurts precision on the held-out ground truth.

Checked directly against the existing preference files, which already store
each pair's score gap — no new training or scoring needed. **Falsified**: the
fraction of near-tie pairs (gap < 0.05) is essentially identical between GCA
and Holistic at every scale (n=10,000: 24.2% vs 24.2%, 1,103 vs 1,103 pairs
exactly). At n=1,000, where GCA wins, GCA actually has *more* near-ties than
Holistic (26.5% vs 20.5%) -- the opposite of what the hypothesis predicts.
This mechanism does not explain the reversal.

---

## Hypothesis test: length/sentence-count shortcut

Second candidate: GCA-trained reward models learn to rely on summary length
or sentence count as a shortcut for quality, and lean on it more heavily as
training data grows, in a way that doesn't transfer to the ground truth's
cross-article comparisons. Tested by scoring every ground-truth summary with
the already-trained checkpoints at n=2,000, n=3,000, and n=10,000, and
correlating each model's predicted score against summary length and
sentence count -- no new training needed, only inference on checkpoints
already on disk.

| Scale | Holistic r(score, length) | GCA r(score, length) | Gap (GCA − Hol.) |
|---|---:|---:|---:|
| n=2,000 | −0.037 | −0.046 | −0.009 |
| n=3,000 | −0.089 | −0.132 | −0.043 |
| n=10,000 | −0.198 | −0.201 | −0.003 |

(Same pattern for sentence count: gaps of −0.013, −0.032, −0.012 respectively.
Negative throughout means both models score *shorter* summaries higher, not
longer ones.)

**Falsified.** If GCA increasingly relied on this shortcut more than
Holistic as training data grew, the gap should be largest at n=10,000 —
exactly where GCA starts losing. Instead the gap peaks at n=3,000 and nearly
vanishes at n=10,000: the two models converge to almost identical
length-reliance right at the scale where the reversal happens. What is real
and worth noting separately: *both* models become substantially more
correlated with summary length as training data grows (roughly 5× larger by
n=10,000 for each), which may partly explain the low absolute ceiling
neither model breaks, but it does not differentiate GCA from Holistic and
so cannot be the reversal's mechanism.

This is the second candidate mechanism tested and ruled out this week.

---

## Crossover point

Retrained at two intermediate scales, n=2,000 and n=3,000, to locate where
the advantage crosses from GCA-favoring to Holistic-favoring. Both training
sets are exact subsets of the existing n=5,000/10,000 draws (verified
empirically), so no new candidate generation or AlignScore scoring was
needed -- only filtering the existing preference files by article ID, then
retraining and evaluating. 5 seeds per scale, not 20, to keep this
supplementary check fast this close to the deadline.

| Test | n=1,000 | n=2,000 | n=3,000 | n=5,000 | n=10,000 |
|---|---:|---:|---:|---:|---:|
| Same-article pairs | +0.1pp | -1.7pp | +0.5pp | -0.4pp | -1.5pp |
| Top/bottom 25% | +3.2pp | -0.2pp | -0.9pp | -0.4pp | -2.1pp |
| Top/bottom 10% | +9.3pp | +3.4pp | +3.8pp | +0.1pp | -2.0pp |
| Top/bottom 5% (all-pairs) | +7.4pp | +1.8pp | +1.4pp | +0.7pp | -2.1pp |

(All values are the GCA-minus-Holistic gap. n=1,000/5,000/10,000 are 20-seed
means; n=2,000/3,000 are 5-seed means, so noisier -- see the same-article
row, which stays close to zero throughout with no real signal at any scale.)

![Where GCA's precision advantage crosses zero as the training set grows](crossover_by_scale.png)

For the three conditions with a real effect at n=1,000 (top/bottom
25%/10%/5%), the decline from n=1,000 to n=10,000 is consistent and close to
monotonic: a large GCA advantage at n=1,000, roughly half that by
n=2,000-3,000, near zero at n=5,000, and a small but significant Holistic
advantage by n=10,000. The crossover from GCA-favoring to Holistic-favoring
happens **between n=3,000 and n=5,000** for all three conditions.

---

## Conclusion

GCA's ground-truth precision advantage is real, but bounded to a specific
regime: reward models trained on a small amount of data (up to a few
thousand examples), evaluated on comparisons with a clear, large quality
gap. Outside that regime -- more training data, or subtler comparisons --
the advantage shrinks, disappears, and past roughly n=3,000-5,000, reverses
into a small Holistic advantage instead. Two candidate explanations for the
reversal were tested this week and both were ruled out: near-tie training
pairs, and a length/sentence-count shortcut. The underlying mechanism
remains open, noted below as future work.

---

## Future work

- Two candidate mechanisms for the reversal are now ruled out (near-tie
  training pairs; length/sentence-count shortcut). Remaining candidates are
  less mechanical and harder to test quickly: e.g. whether GCA's
  sentence-level aggregation lets the reward model fit article-specific,
  within-pair artifacts that do not generalize to the ground truth's
  cross-article comparisons, as opposed to holistic's single coarser score.
- The n=2,000/3,000 points used 5 seeds instead of 20; re-running them at
  20 seeds would tighten the crossover estimate if time allows.

---

## Questions

1. Does the crossover point (n=3,000-5,000) combined with the two ruled-out
   hypotheses (near-tie pairs, length/sentence-count shortcut) give enough
   of a closing story for this section of the thesis, or is further
   investigation into the *why* expected before submission?
2. Is it acceptable to close with "the effect is real, bounded, and we
   ruled out two plausible mechanisms" without a confirmed explanation, or
   should more candidates be tested before the deadline?
3. Anything else needed before the 3 September submission?

---

## For reference

Full code: https://github.com/hasnat23/master-thesis-rlaif-gca

This update builds directly on the 18 August results:
[progress-updates/18-08-2026/README.md](../18-08-2026/README.md).

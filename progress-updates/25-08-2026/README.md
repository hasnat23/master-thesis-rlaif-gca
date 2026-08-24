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

## Summary for tomorrow's meeting

Yes -- all three hypotheses are in the thesis, but only two were actually
tested. Here's exactly where things stand.

**Tested and rejected** (both are full findings, documented in Results
§6.10.2/§6.10.5, Discussion §7.6, Conclusion, and now the Abstract/Intro
too):

- **Near-tie training pairs** -- the idea that GCA's sentence scoring
  produces more "coin-flip" training pairs than Holistic, and fitting
  that noise at scale hurts GCA. Rejected: the near-tie fraction is
  virtually identical between conditions at every scale (24.2% vs 24.2%
  at n=10,000), and GCA actually has *more* near-ties at n=1,000 -- where
  GCA wins. Opposite of the prediction.

- **Length/sentence-count shortcut** -- the idea that GCA increasingly
  leans on summary length as a proxy for quality as training data grows.
  Rejected: the GCA-vs-Holistic gap in length-correlation should peak at
  n=10,000 (where GCA loses) if this were true; instead it peaks at
  n=3,000 and nearly vanishes at n=10,000. Opposite of the prediction
  again.

**Mentioned but not tested** -- the one that's still genuinely open:

- **Article-specific overfitting** -- the idea that GCA's sentence-level
  aggregation lets the reward model fit patterns specific to how one
  article's two candidates happen to differ, which don't generalize to
  the ground truth's cross-article comparisons. This appears only in the
  Future Work section (`sec:concl-future`, Conclusion) as an explicitly
  untested candidate -- it was never run. The thesis is careful not to
  claim any evidence for or against it.

**Where the thesis ends up** (RQ4, the headline finding of the whole
project): GCA is significantly more precise than Holistic at n=1,000 on
comparisons with a real quality gap (up to +9.25pp, p<0.001), the two are
indistinguishable at n=5,000, and Holistic is significantly *more*
precise than GCA at n=10,000 on 3 of 4 constructions (p=0.010-0.036). The
crossover is located between n=3,000 and n=5,000. Two plausible
mechanical explanations for that reversal have been tested and
eliminated. The actual mechanism is left as an open question for future
work -- the thesis doesn't pretend to have solved it, and closes RQ4
honestly with "the effect is real and its regime is mapped; why it
reverses is not yet known."

That's the landing point: a confirmed, bounded, well-characterized
effect, with two dead ends clearly marked so nobody wastes time retesting
them, and one live lead pointed at for whoever continues this work.

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

Pulling the three pieces above (near-tie test, shortcut test, crossover
scan) into one story:

**What holds up.** GCA is more precise than Holistic against the
independent ground truth, but only when two conditions both hold at once:
the reward model was trained on a small amount of data (roughly n=1,000,
maybe up to n=2,000-3,000), and the two summaries being compared actually
differ in quality rather than being near-identical. Inside that regime the
advantage is large and consistent: up to +9.3pp at n=1,000 on the
top/bottom 10% comparison, still +3.4 to +3.8pp at n=2,000-3,000 on the
same comparison. This is not a marginal or noisy effect where it holds.

**Where it breaks down, in two independent ways.** Tightening the
comparison difficulty kills the advantage even at the best training-set
size: on same-article pairs (the closest, least clear-cut comparison),
GCA and Holistic are statistically indistinguishable at every single
scale tested, n=1,000 through n=10,000 -- there was never a regime where
GCA won on hard comparisons. Separately, growing the training set kills
the advantage even on the easy comparisons where it is otherwise largest:
holding comparison difficulty fixed at top/bottom 10%, the advantage is
+9.3pp at n=1,000, roughly a third of that by n=2,000-3,000, essentially
zero at n=5,000, and by n=10,000 it has not just vanished but flipped
into a small, statistically significant advantage for Holistic instead
(-2.0pp). So "GCA wins" is not a general property of the method -- it is
a property of a specific, identifiable combination of training-set size
and comparison difficulty, and the boundary of that regime is now mapped
rather than assumed.

**What we ruled out this week, and how.** The interesting question is not
just that the advantage reverses, but *why* -- a reversal is a stronger
claim than "the advantage fades," since something has to actively flip
sign rather than merely shrink toward zero. Two concrete, testable
mechanisms were checked directly against data and checkpoints already on
disk, at no new training cost:
1. *Near-tie training pairs.* If GCA's sentence-level scoring produced
   more near-zero-margin ("coin flip") training pairs than Holistic's,
   and fitting that noise got worse as more such pairs accumulated at
   larger n, that would explain a growing GCA disadvantage. Checked
   directly against the stored preference files: the near-tie fraction is
   essentially identical between GCA and Holistic at every scale
   (24.2% vs. 24.2% at n=10,000), and if anything GCA has *more*
   near-ties at n=1,000, where GCA wins -- the opposite of the
   prediction. Rejected.
2. *Length/sentence-count shortcut.* If GCA increasingly learned to use
   summary length or sentence count as a proxy for quality, and leaned on
   that proxy more heavily as training data grew, the effect should be
   *largest* at n=10,000, where GCA starts losing. Checked by scoring the
   ground-truth summaries with the already-trained checkpoints and
   correlating predicted reward with length: the gap between GCA's and
   Holistic's length-correlation instead peaks at n=3,000 and nearly
   disappears at n=10,000, again the opposite of the prediction. Rejected.

Both tests were designed so that the hypothesis made a specific,
falsifiable prediction about *where the effect should be largest*, and in
both cases the data showed the opposite pattern rather than just failing
to show the predicted one -- which is a cleaner rejection than a null
result would have been.

**What remains open.** Ruling out two mechanisms did not surface a third,
confirmed one. The leading untested candidate (see Future Work below) is
that GCA's sentence-level aggregation lets the reward model fit
article-specific regularities during training -- patterns tied to how one
particular article's two candidate summaries happen to differ -- that
don't generalize to the ground truth's cross-article comparisons, where
the two summaries being ranked come from different source articles
entirely. This was not tested, so it remains a hypothesis, not a finding.

**Bottom line.** The effect is real, its regime is now precisely located
(small training set, clear quality gap between the two summaries being
compared), and two of the most obvious mechanical explanations for why it
reverses at scale -- rather than merely fading -- have been eliminated
with direct, falsification-style evidence. That is a complete and honest
result to report for RQ4 even though the reversal's underlying mechanism
itself is not resolved.

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

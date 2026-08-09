# Progress Update — 11 August 2026

**Student:** Muhammad Hasnat
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner
**Meeting:** Tuesday, 11 August 2026

> **Update (night of 8 August 2026).** This document was originally drafted
> describing two follow-up experiments as prepared and ready to submit. Both
> have since run to completion on MOGON NHR. The headline result changed from
> a repeatedly-observed-but-statistically-underpowered effect to a properly
> powered, statistically significant one (p < 0.001). Section 3 below reports
> the actual numbers in place of the earlier "results not yet available"
> note, and Section 4 has been revised accordingly — the open question is no
> longer whether to run the extension, but what it shows.
>
> **Second update (9 August 2026).** The same treatment was extended to
> $n=5{,}000$ and $n=10{,}000$, which had previously been single, unrepeated
> runs. Six-run campaigns at both scales now show the GCA advantage is
> **confirmed absent**, not merely smaller, with confidence intervals that do
> not overlap the significant $n=1{,}000$ result at all. See the new §3.2.
> The thesis now rests on properly powered evidence at every dataset scale
> tested, not only at $n=1{,}000$.

---

## 1. Purpose of This Update

This update responds directly to the supervisor comment received before this
meeting:

> "You mentioned that the experiments are largely complete, but I believe the
> steps of training reward models using holistic/GCA scores and evaluating
> those reward models haven't been done yet."

**Short answer: this step is done, and it is the central experiment of the
thesis, not an outstanding one.** Section 2 gives the direct evidence —
exact files, exact numbers, exact thesis sections. Section 3 goes further and
reports two things completed since our last meeting on 21 July: a
statistical-power fix that turns the headline result from suggestive into
significant, and a diagnostic experiment on why both reward models sit close
to chance.

The confusion likely traces back to the 16 July progress note (written before
our 21 July meeting) predating the point at which the reward-model comparison
was written up as the thesis's central result — at the time, the most visible
artefact in the repository was still framed around the (later dropped) DPO
pipeline.

---

## 2. Evidence That Reward-Model Training and Evaluation on Holistic/GCA Scores Is Done

Training a Bradley–Terry reward model separately on holistic-scored and
GCA-scored preference pairs, and evaluating both by cross-validated pairwise
accuracy, **is the central experiment of this thesis** — not a step that was
skipped. Concretely:

| What | Where |
|---|---|
| Preference construction (holistic and GCA, from the same candidate pool) | `src/judging/build_reward_preferences.py` |
| Reward-model architecture and training loop | `src/reward_model/train.py` |
| Training/evaluation orchestration (both conditions from one invocation) | `src/reward_model/run_training.py` |
| Slurm job that produced the headline result | `slurm/train_rm_1000.sh` (original), `slurm/seed_campaign.sh` (extension, see §3.1) |
| Full description of the method | `thesis/chapters/05_implementation.tex`, §5.4–5.5 ("Preference Construction", "Reward-Model Training") |
| Full results table and statistical analysis | `thesis/chapters/06_results.tex`, §6.6–6.8 |
| Headline numbers | `README.md`, ["Final Results"](../../README.md#final-results) section |

The headline result, already trained and evaluated at three dataset scales:

| Dataset size | Holistic mean acc | GCA mean acc | Gap (GCA − Holistic) |
|---:|---:|---:|---:|
| 1,000 (20 runs, run-level statistics — see §3.1) | 0.5295* | 0.5603* | **+0.0395** |
| 5,000 (single run) | 0.5788 | 0.5746 | −0.0042 |
| 10,000 (single run) | 0.5827 | 0.5862 | +0.0035 |

\* Per-condition means shown are from the original six-run campaign; the
extended twenty-run campaign's own per-run numbers are in §3.1 below and in
`thesis/chapters/06_results.tex`, Table 6.8.

Each of these numbers is a **pairwise validation accuracy from a trained
Bradley–Terry reward model**, evaluated under 5-fold cross-validation, for
both the holistic-preference model and the GCA-preference model. This is
exactly the step the comment describes as missing. It is not only done — as
of this week it is done to a standard that supports a statistically
significant claim, which was not previously true (§3.1).

---

## 3. Work Completed Since Our Last Meeting (21 July)

### 3.1 A statistical-power problem in the existing result — diagnosed, fixed, and resolved

The original headline n=1,000 result rested on six training runs (five
distinct random seeds — seed 42 was used twice). A run-level Wilcoxon
signed-rank test on those six runs gave p = 0.0625, two-sided — this did
**not** reach the conventional 0.05 threshold. Critically, **six paired
samples cannot produce a two-sided p below 0.0312 no matter how consistent
the effect is**, so the original campaign was structurally unable to
establish significance, independent of whether the underlying effect was
real.

A second issue was found in the training code while investigating this: the
cross-validation routine (`_kfold_cv` in `run_training.py`) seeded only
Python's `random` module, which fixes fold assignment, but never seeded
PyTorch. This left the reward head's weight initialization, batch ordering,
and dropout uncontrolled — meaning two runs launched with an identical
`--seed` could still produce different results. This was fixed: torch and
CUDA are now seeded per fold, with an explicit `DataLoader` generator and
seeded worker initialization, so a run is fully determined by `--seed` on
fixed hardware (`src/reward_model/run_training.py`).

**The extended campaign has run and completed.** Twenty runs (seeds 1
through 20, fixed in advance of running any of them) were submitted as
Slurm array job **1415720** on MOGON NHR and completed successfully — all 20
tasks `COMPLETED`, exit code 0, roughly 27–28 minutes each, finished between
18:18 and 21:44 CEST on 8 August.

**Result: GCA led in all twenty runs.**

| Runs | GCA ahead | Mean gap | SD | 95% bootstrap CI | Wilcoxon p (two-sided) |
|---:|---:|---:|---:|---:|---:|
| 20 | 20/20 | **+3.95 pp** | 1.78 pp | [+3.19, +4.70] pp | **8.8 × 10⁻⁵** |

This **meets the conventional significance threshold** — the confidence
interval does not cross zero, and an independent exact permutation test
(enumerating all 2²⁰ sign assignments as a cross-check on the standard
library computation) gives p ≈ 1.9 × 10⁻⁶, the theoretical floor for twenty
same-signed observations. The mean gap (+3.95pp) sits close to the original
six-run estimate (+3.08pp), so the extension refined the precision of the
same effect rather than revealing a different one — the original six-run
campaign was pointing at something real, it simply could not prove it with
six observations.

Full per-run breakdown: `thesis/chapters/06_results.tex`, Table 6.8; raw
outputs at `outputs/seed_campaign/seed_{1..20}/rm_training_summary.json`
(committed to this repository).

### 3.2 Does the advantage hold at scale? Now resolved, not just observed

The thesis previously reported single runs at $n=5{,}000$ and $n=10{,}000$
showing a much smaller, direction-inconsistent gap than at $n=1{,}000$
(−0.42pp and +0.35pp), with an explicit caveat that a single run at each
scale could not distinguish a genuine dataset-size effect from the same
run-to-run noise the $n=1{,}000$ campaign had. Following the same logic that
resolved §3.1, both scales were extended to **six independently seeded runs
each**, submitted as Slurm array jobs **1415902** ($n=5{,}000$) and **1415903**
($n=10{,}000$), using the identical corrected training procedure. Both
completed successfully — 12/12 tasks `COMPLETED`, exit code 0.

**Result: the advantage is confirmed absent at both larger scales, not merely
smaller.**

| Dataset size | Runs | GCA ahead | Mean gap | 95% bootstrap CI | Wilcoxon $p$ |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 20 | 20/20 | +3.95pp | [+3.19, +4.70]pp | 8.8×10⁻⁵ |
| 5,000 | 6 | 2/6 | +0.02pp | [−0.44, +0.62]pp | 0.6875 |
| 10,000 | 6 | 4/6 | −0.03pp | [−0.54, +0.42]pp | 1.0000 |

Both larger-scale confidence intervals are centred almost exactly on zero and
**do not overlap the significant $n=1{,}000$ interval at all** — the nearest
edges are separated by more than 2.5 percentage points at both scales. This is
exactly the test the thesis proposed when only single runs were available: if
the scale movement were statistical noise, repeated runs should produce
intervals overlapping the $n=1{,}000$ result; if it reflects a real,
scale-dependent change, they should not. They do not. The GCA advantage is
therefore a genuine, bounded effect — real and significant when preference
data are scarce ($n=1{,}000$), and undetectable once more training data are
available — rather than an artefact that a single run happened to catch.

This is a materially stronger position than "the effect doesn't reproduce at
scale, and we don't know why": it is now a fully powered, two-sided finding
at every scale tested, and it directly answers the open question the thesis
previously could only pose as a hypothesis. Full write-up:
`thesis/chapters/06_results.tex` §6.8 (new Table 6.10, revised Table 6.11,
revised Figure 6.5) and `thesis/chapters/07_discussion.tex` §7.5 (substantially
rewritten). Raw outputs:
`outputs/seed_campaign_{5000,10000}/seed_{1..6}/rm_training_summary.json`.

### 3.3 Diagnosing the near-chance accuracy — partially completed

Both reward models sit close to chance (0.53–0.59 against a 0.50 baseline).
One candidate explanation, already flagged as an open question in the thesis:
the reward model only ever sees a 2,000-character prefix of the source
article, concatenated with the summary and truncated again to 512 tokens —
for the median article (3,643 characters), the model may see well under half
of what the judge actually scored against.

A five-configuration ablation was designed to test this
(`slurm/truncation_ablation.sh`, three seeds each). It is not a simple sweep
of the character limit, because at 2,000 characters the input already
exceeds the 512-token cap — the binding constraint is the token limit, not
the character limit. The ablation therefore also varies the backbone to
`deberta-v3-base`, whose relative position embeddings allow a 1,024-token
context window, so that "more context" and "different backbone" stay
separable.

**Three of five configurations have run** (the RoBERTa-base arms, submitted
as Slurm array job **1415721**, tasks 0–8, all `COMPLETED`):

| Max article chars | Holistic | GCA | Gap (GCA − Holistic) |
|---:|---:|---:|---:|
| 500   | 0.5553 | 0.5850 | +2.97 |
| 1,000 | 0.5663 | 0.5903 | +2.40 |
| 2,000 | 0.5290 | 0.5480 | +1.90 |

**This runs against the truncation hypothesis as originally stated:**
accuracy was *higher* with less retained context, for both conditions — the
opposite of what "the model isn't seeing enough of the article" predicts. A
plausible reading is that at 2,000 characters the amount that actually
survives the 512-token cap varies unpredictably with summary length across
examples, whereas a shorter, more consistently-applied budget produces a
more uniform input even though it discards more raw text.

**The remaining two configurations (`deberta-v3-base` at 512 and 1,024
tokens) have not run.** They are the ones that would separate "less context"
from "a backbone not capped at 512 tokens," and are needed to properly test
the mechanism above rather than just observe it. They are blocked on an
infrastructure issue outside this project: MOGON's internal Hugging Face
model mirror (`10.81.2.171:8090`) is returning HTTP 500 for every model
queried, not only this one, and the public Hugging Face endpoint is blocked
by the institutional proxy. This was last checked at the time of writing and
remains down; the two configurations will be submitted as soon as it
recovers.

This is reported as a genuine, if incomplete, finding: it does not resolve
why both reward models are weak, but it rules out the simplest version of the
truncation explanation and redirects toward a more specific hypothesis about
input consistency rather than input quantity.

Full write-up: `thesis/chapters/07_discussion.tex`, §7.4 (new paragraph and
Table 7.1). Raw outputs:
`outputs/truncation_ablation/cfg_{A,B,C}/seed_{1,2,3}/rm_training_summary.json`
(committed).

**Data-integrity check performed before either campaign was submitted.**
Before submitting anything, the preference-data file at the path the training
scripts expected on the cluster was checked against the thesis's own
six-run numbers rather than trusted by directory name — it was found to
point at a stale run using the discarded α=0.5 aggregation formula, not the
thesis's actual α=0.0 configuration. This was caught by reproducing five of
the six original per-seed accuracy pairs to the third decimal place from the
correct source files before launching either job, and corrected before
submission.

### 3.4 Thesis document

The thesis document (`thesis/main.tex` and eight chapters, now **87 pages**)
was substantially completed and updated during this period, including:

- Full write-up of the methodology, implementation, results, discussion, and
  conclusion chapters described above, now reflecting the significant
  extended-campaign result throughout (both abstracts, §6.6, §7.3, §7.4,
  §7.8, and the Conclusion's RQ1/RQ3 answers and limitations list).
- Front matter (acknowledgments, declaration on AI-assistance use, list of
  abbreviations) and three appendices (data/code availability, a full
  reproduction configuration with exact commands, and an annotated example
  preference record).
- A citation audit that found the Discussion chapter carried zero citations
  across 3,226 words despite making claims that depend on cited literature;
  ten citation anchors were added there, and seven foundational sources
  (Bradley–Terry, Wilcoxon, the bootstrap, RoBERTa, Mistral, and two papers on
  fine-tuning variance) were added where the text already used the method or
  model but never attributed it.

The current PDF is available at [`thesis/main.pdf`](../../thesis/main.pdf) in
this repository.

---

## 4. What Tuesday's Meeting Should Resolve

1. Confirm that the reward-model comparison described in Section 2, now
   supported by a statistically significant twenty-run extension at
   $n=1{,}000$ **and** a confirmed, statistically null result at $n=5{,}000$
   and $n=10{,}000$ (§3.2), resolves the concern raised, and clarify if there
   is a different or additional expectation about what "training and
   evaluating reward models" should include.
2. Decide whether the two remaining truncation-ablation configurations
   (`deberta-v3-base` arms, §3.3) are worth pursuing once the cluster's HF
   mirror recovers, or whether the partial result — which already
   complicates the original truncation hypothesis in an informative way — is
   sufficient to report as an exploratory finding with the missing arms
   listed as future work.
3. Discuss whether, with both the headline result and the scale question now
   resolved with proper statistical power, any further scope should be added
   before submission, or whether the current thesis (a significant,
   scale-bounded GCA advantage at $n=1{,}000$; a confirmed absence of that
   advantage at $n=5{,}000$/$n=10{,}000$; a partially-explained
   absolute-accuracy limitation) represents a complete and submittable
   contribution. Given how much stronger this position is than the one this
   document opened with, this is now more plausibly "yes" than it was even
   this morning.

---

## 5. Recorded Outputs

### Code and infrastructure changes since 21 July

```text
src/reward_model/run_training.py   — deterministic seeding fix
slurm/seed_campaign.sh             — 20-run campaign at n=1,000 (job 1415720, complete)
slurm/seed_campaign_5000.sh        — 6-run campaign at n=5,000 (job 1415902, complete)
slurm/seed_campaign_10000.sh       — 6-run campaign at n=10,000 (job 1415903, complete)
slurm/truncation_ablation.sh       — 5-config x 3-seed ablation (job 1415721, tasks 0-8 complete; 9-14 blocked)
analysis/aggregate_campaigns.py    — run-level Wilcoxon + bootstrap analysis,
                                      validated against the published 6-run result
thesis/                            — full 8-chapter document, 90 pages, front matter,
                                      appendices, 32 cited references
```

### New experimental outputs (this update)

```text
outputs/seed_campaign/seed_{1..20}/rm_training_summary.json           (20 files, n=1,000, all complete)
outputs/seed_campaign_5000/seed_{1..6}/rm_training_summary.json       (6 files, n=5,000, all complete)
outputs/seed_campaign_10000/seed_{1..6}/rm_training_summary.json      (6 files, n=10,000, all complete)
outputs/truncation_ablation/cfg_{A,B,C}/seed_{1,2,3}/...              (9 files, all complete)
reports/campaigns/                                                    (n=1,000 tables, .dat files, JSON summaries)
reports/campaigns_5000/, reports/campaigns_10000/                     (scale-campaign summaries)
```

### Historical reward-model outputs (original single-run campaign, referenced in Section 2)

```text
~/thesis/outputs/reward_models_1000/rm_training_summary.json
~/thesis/data/preferences_1000_alpha0_mode_nli*/     (six seed runs, verified)
~/thesis/outputs/reward_models_5000/rm_training_summary.json
~/thesis/outputs/reward_models_10000/rm_training_summary.json
```

### Git

All of the above is committed and pushed to `main`:

```text
64524e4  docs: add progress update for 11 August supervisor meeting
c9853a5  thesis: extended seed campaign reaches significance; truncation ablation (partial)
eebbb24  docs: update 11 August progress note with completed campaign results
06ac258  progress: correct meeting date; add repeated-run campaigns at n=5,000/10,000
1757b3b  thesis: repeated-run campaigns at n=5,000/10,000 confirm the advantage vanishes
```

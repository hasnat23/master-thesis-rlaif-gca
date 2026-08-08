# Progress Update — 11 August 2026

**Student:** Muhammad Hasnat
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner
**Meeting:** Tuesday, 11 August 2026

---

## 1. Purpose of This Update

This update responds directly to the supervisor comment received before this
meeting:

> "You mentioned that the experiments are largely complete, but I believe the
> steps of training reward models using holistic/GCA scores and evaluating
> those reward models haven't been done yet."

This is not correct, and this document exists to show precisely where that
work is, so the misunderstanding does not recur. The confusion most likely
carries over from an earlier progress update (dated 16 July) that predates the
reward-model comparison being written up in the thesis proper — at the time,
the most visible artefact in the repo was still framed around the (later
dropped) DPO pipeline. Section 2 below points at the exact evidence.

Section 3 then covers what has changed since 16 July: a statistical-power
extension of the existing reward-model result, and a diagnostic experiment
aimed at explaining why both reward models sit close to chance. Neither of
these existed at the time of the last progress note.

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
| Slurm job that produced the headline result | `slurm/train_rm_1000.sh` |
| Full description of the method | `thesis/chapters/05_implementation.tex`, §5.4–5.5 ("Preference Construction", "Reward-Model Training") |
| Full results table and statistical analysis | `thesis/chapters/06_results.tex`, §6.6–6.7 |
| Headline numbers | `README.md`, ["Final Results"](../../README.md#final-results) section |

The headline result, already trained and evaluated:

| Dataset size | Holistic mean acc | GCA mean acc | Gap (GCA − Holistic) |
|---:|---:|---:|---:|
| 1,000 (6 runs, 30 folds pooled) | 0.5295 | 0.5603 | +0.0308 |
| 5,000 (single run) | 0.5788 | 0.5746 | −0.0042 |
| 10,000 (single run) | 0.5827 | 0.5862 | +0.0035 |

Each of these numbers is a **pairwise validation accuracy from a trained
Bradley–Terry reward model**, evaluated under 5-fold cross-validation, for
both the holistic-preference model and the GCA-preference model, at three
dataset scales. This is exactly the step the comment describes as missing.
The full per-seed breakdown (six independent training runs at n=1,000, with
individual seeds and results) is in `thesis/chapters/06_results.tex`, Table
6.8, and is reproduced in `progress-updates/16-7-2026/README.md` (see the
correction note at the top of that file for the statistical caveats that were
added after 16 July).

If the concern is instead about *rigour* — whether one run per configuration
is enough to trust — that is a fair question, and it is exactly what the new
work in Section 3 addresses.

---

## 3. Work Completed Since 16 July

### 3.1 A statistical-power problem in the existing result, and its fix

The headline n=1,000 result rests on six training runs (five distinct random
seeds — seed 42 was used twice). A run-level Wilcoxon signed-rank test on
those six runs gives p = 0.0625, two-sided — this does **not** reach the
conventional 0.05 threshold. Critically, **six paired samples cannot produce a
two-sided p below 0.0312 no matter how consistent the effect is**, so the
existing campaign was structurally unable to establish significance,
independent of whether the underlying effect is real.

A second issue was found in the training code while investigating this: the
cross-validation routine (`_kfold_cv` in `run_training.py`) seeded only
Python's `random` module, which fixes fold assignment, but never seeded
PyTorch. This left the reward head's weight initialization, batch ordering,
and dropout uncontrolled — meaning two runs launched with an identical
`--seed` could still produce different results. This has been fixed: torch
and CUDA are now seeded per fold, with an explicit `DataLoader` generator and
seeded worker initialization, so a run is now fully determined by `--seed` on
fixed hardware. (`src/reward_model/run_training.py`, committed 27 July.)

**Extended seed campaign.** A 20-run array (`slurm/seed_campaign.sh`), seeds 1
through 20 fixed in advance, is prepared to give the run-level Wilcoxon test
the resolution needed to either establish the effect properly or rule it out.
This uses the corrected, fully-seeded training code, so — unlike the original
six runs — each of the twenty is a genuinely independent, reproducible sample.

### 3.2 Diagnosing the near-chance accuracy

Both reward models sit close to chance (0.53–0.59 against a 0.50 baseline).
One candidate explanation, already flagged as an open question in the thesis
(`thesis/chapters/05_implementation.tex`, "Source-document truncation"): the
reward model only ever sees a 2,000-character prefix of the source article,
concatenated with the summary and truncated again to 512 tokens — for the
median article in the dataset (3,643 characters), the model may be seeing
well under half of what the judge actually scored against.

A five-configuration ablation (`slurm/truncation_ablation.sh`, three seeds
each) is prepared to test this. It is not a simple sweep of the character
limit, because at 2,000 characters the input already exceeds the 512-token
cap — the binding constraint is the token limit, not the character limit.
The ablation therefore also varies the backbone to `deberta-v3-base`, whose
relative position embeddings allow a 1,024-token context window, so that the
effect of "more context" and the effect of "different backbone" stay
separable across the five arms.

**Status:** both scripts are written, use the corrected preference data
(confirmed against the thesis's own reported numbers — see the note below),
and are ready to submit on MOGON NHR. Submission was in progress as of this
writing and results were not yet available at the time this update was
prepared; the intention is to have partial results in hand before or shortly
after Tuesday's meeting, and the full campaign will be summarized in the next
progress update.

**Data-integrity check performed before launching anything.** Before
submitting either campaign, the exact preference file being pointed at was
verified against the thesis's own reported results rather than trusted by
directory name alone: five of the six per-seed accuracy pairs in Table 6.8
were reproduced to the third decimal place from
`data/preferences_1000_alpha0_mode_nli*` on the cluster, confirming this is
the correct (α=0.0, margin=0, `nli`-mode) dataset and not an older,
superseded run that used the discarded α=0.5 aggregation formula.

### 3.3 Thesis document

The thesis document itself (`thesis/main.tex` and eight chapters, 81 pages)
was substantially completed and pushed to the repository during this period,
including:

- Full write-up of the methodology, implementation, results, discussion, and
  conclusion chapters described in the table above.
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

1. Confirm that the reward-model comparison described in Section 2 satisfies
   the concern raised, and clarify if there is a different or additional
   expectation about what "training and evaluating reward models" should
   include.
2. Agree whether the extended seed campaign (Section 3.1) is worth completing
   before submission, given the deadline, or whether the honestly-reported
   p = 0.0625 result is acceptable as a repeatedly-observed-but-not-significant
   finding.
3. Flag the truncation ablation (Section 3.2) as optional strengthening work
   rather than a requirement — it explains a limitation the thesis already
   states plainly, it does not change the headline claim either way.

---

## 5. Recorded Outputs

### Code changes since 16 July

```text
src/reward_model/run_training.py   — deterministic seeding fix
slurm/seed_campaign.sh             — 20-run extended campaign (prepared)
slurm/truncation_ablation.sh       — 5-config x 3-seed ablation (prepared)
analysis/aggregate_campaigns.py    — run-level Wilcoxon + bootstrap analysis,
                                      validated against the published 6-run result
thesis/                            — full 8-chapter document, front matter,
                                      appendices, 32 cited references
```

### Historical reward-model outputs (already completed, referenced in Section 2)

```text
~/thesis/outputs/reward_models_1000/rm_training_summary.json
~/thesis/data/preferences_1000_alpha0_mode_nli*/     (six seed runs, verified)
~/thesis/outputs/reward_models_5000/rm_training_summary.json
~/thesis/outputs/reward_models_10000/rm_training_summary.json
```

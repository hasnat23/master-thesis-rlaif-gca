# Master Thesis: Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)

**University of Koblenz** | Master's Thesis | 2025–2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner  
**Mentor:** Lingxiao Kong  
**GitHub:** [@hasnat23](https://github.com/hasnat23)

---

## Overview

> Status (as of 24 August 2026): experimentation is complete and the thesis is written. Four research questions are answered. RQ1–RQ3 compare holistic and GCA preference construction by reward-model learnability (see [Learnability Results](#learnability-results-rq1rq3)). RQ4, added in August, re-evaluates the same trained models against an independent held-out ground truth and produced the thesis's most striking finding: GCA's advantage does not merely fade at scale, it reverses (see [Ground-Truth Results](#ground-truth-results-rq4)). Early-phase DPO fine-tuning experiments (Mar–May 2026) are preserved in `src/dpo/` and earlier `progress-updates/` entries but are not part of the final reported pipeline — DPO was dropped on 2 June 2026 in favour of a pure reward-model comparison, per supervisor feedback.

This thesis investigates a controlled question: for long news summaries, are holistic A/B preferences sufficient to produce a learnable factual-reliability signal for reward-model training, or does supervision become more effective when the judge evaluates aligned sentence pairs whose local decisions are then aggregated into a summary-level preference via Granular Credit Assignment (GCA)?

The base candidate pool, judge, and reward-model training recipe are held constant across conditions. The only manipulated factor is the granularity of AI-generated preference labels used to construct the pairwise training data.

- **Dataset:** CNN/DailyMail (main runs at 1,000 / 5,000 / 10,000 samples, plus 2,000 / 3,000 for the RQ4 crossover check; note these subsets are *nested*, see [Learnability Results](#learnability-results-rq1rq3)). A separate, disjoint 500-article held-out set (subset seed 999) is used for the RQ4 ground-truth evaluation.
- **Candidate generation model:** Mistral-7B-Instruct-v0.3 (instruction-tuned 7B, Apache 2.0) — two-temperature sampling (T=0.7 / T=1.0)
- **Judge:** `yzha/AlignScore`, `nli` mode — a fixed factual-consistency metric used as the primary automatic judge (no OpenAI API, no generative LLM calls). Locked as the final judge configuration.
- **Reward model:** Bradley-Terry pairwise reward model, `FacebookAI/roberta-base` backbone, trained separately on holistic vs GCA preference sets, evaluated via 5-fold cross-validation pairwise accuracy
- **Primary metric:** RM pairwise validation accuracy (Holistic vs GCA), with bootstrap 95% CIs and Wilcoxon significance tests
- **Superseded/early-phase only:** ROUGE-1/2/L, BERTScore F1, SummaC, QAFactEval, FineSurE were used during the earlier DPO-based prototype (see `progress-updates/19-03-2026/` through `21-04-2026/`) but are not part of the final reward-model comparison

---

## Research Questions

These are the four research questions as stated in the thesis (Chapter 4, §4.1) and answered in Chapter 8, §8.2.

**RQ1:** Does sentence-level factuality feedback aggregated via GCA produce a more learnable pairwise preference signal than holistic summary-level feedback, as measured by reward-model pairwise validation accuracy?
→ *Yes at n=1,000 (+3.95 pp, 20/20 runs, p<0.001); confirmed absent at n=5,000 and n=10,000.*

**RQ2:** How do evaluator configuration and sentence-score aggregation strategy affect the relative learnability of holistic and GCA-derived preferences?
→ *Both matter substantially, but the evidence is associational: each judge mode was run once.*

**RQ3:** How stable are the observed differences across repeated reward-model training runs and dataset sizes?
→ *Highly stable at fixed scale (all 20 runs agree in direction at n=1,000); not stable across scale.*

**RQ4:** Does the learnability advantage measured in RQ1 also hold when the trained reward models are evaluated against an independent, held-out ground truth rather than against their own training-style labels, and does that hold as the training-set size grows?
→ *Yes at n=1,000, and more sharply than for learnability — but it reverses at n=10,000. See [Ground-Truth Results](#ground-truth-results-rq4).*

### Hypotheses

- **H1:** Sentence-level aggregated preferences are more learnable than holistic preferences, because they localise the factuality signal within a long output.
  → **Supported at n=1,000, refuted at larger scale.** Holds decisively where data are scarce (20/20 runs, Cohen's *d* = +2.22), but equivalence tests at n=5,000/10,000 exclude effects larger than ±0.61 and ±0.36 pp. The hypothesis as originally stated was unconditional and is not supported in that form; what survives is a scale-bounded version.
- **H2:** The evaluator's configuration affects whether any such difference is observable.
  → **Supported, associationally.** Whether GCA leads at all depends on the AlignScore mode, and the two modes where it does not lead are precisely the two where the evaluator performs its own internal decomposition. Each mode was run once, so the association is established more securely than the mechanism behind it.
- **H3:** Effects observed at one dataset scale need not persist at another.
  → **Supported, and the hypothesis the evidence bears on most directly.** Included as a methodological precaution, it turned out to determine the shape of the final result. Had the study stopped at n=1,000, it would have reported an unqualified positive finding the fuller evidence does not support.

*Note:* an earlier version of this README listed a fourth hypothesis about error categories (entity/relation mistakes). That line of work was not carried out and is recorded in the thesis under Future Work, not as a hypothesis under test.

---

## GCA: Granular Credit Assignment

GCA is an aggregation layer that converts sentence-level factuality scores into a single summary-level preference pair usable for reward-model training.

**Final (locked) configuration:**

$$\text{score} = \bar{s} \quad (\alpha = 0.0 \text{, simple mean})$$

- $\bar{s}$ = mean sentence score
- No margin filter (`margin = 0`) — every scored pair is used, since AlignScore is continuous and exact ties are effectively impossible
- This is the result of an optimisation campaign (see `OPTIMIZATION_CAMPAIGN.md`) that started from a penalty formula ($\alpha=0.5$, `tie_margin=0.05`) and found the simple mean gives the most learnable, least noisy signal

**Reliability of the judge:**
- Fixed, deterministic sequence-classifier (AlignScore, `nli` mode) — no prompt sensitivity or hallucinated rationales
- No margin gating in the final pipeline; all pairs are used for reward-model training

## Why Sentence-Level Segmentation?

Full-summary scoring is the holistic baseline and provides one factuality score per summary. It is useful, but it can hide where factual errors occur when a long summary mixes correct and incorrect claims.

Sentence-level segmentation makes each local claim independently scoreable against the source article. This enables the pipeline to localise weak factual segments, reduce credit-assignment ambiguity, and create granular supervision that can differ from holistic scoring when one summary has a few critical unsupported sentences.

## What GCA Does

GCA takes sentence-level factuality scores and aggregates them into a single summary-level comparison score for A vs B, which is then used to construct a chosen/rejected pair for Bradley-Terry reward-model training.

---

## Methodology Summary

| Factor | Value |
|--------|-------|
| Task | Single-document abstractive summarisation (news) |
| Dataset | CNN/DailyMail — runs at 1,000 / 5,000 / 10,000 samples (all subset seed 200; nested, not disjoint) |
| Candidate generation model | Mistral-7B-Instruct-v0.3, two-temperature sampling (T=0.7 / T=1.0) |
| Judge | `yzha/AlignScore`, mode `nli` (locked final configuration) |
| Main variable | Feedback granularity: holistic A/B vs sentence-level + GCA |
| GCA alpha | 0.0 (simple mean; see optimisation campaign) |
| Margin | 0 (no filtering; all pairs used) |
| Reward model | Bradley-Terry, `FacebookAI/roberta-base` backbone, mean-pool + linear scalar head |
| RM training | epochs=5, lr=2e-5, batch=8, max_length=512, 5-fold CV |
| Primary metric | RM pairwise validation accuracy (Holistic vs GCA) |
| Stats | Bootstrap resampling (10,000), 95% CI, Wilcoxon signed-rank test |
| Superseded (early prototype only) | DPO+LoRA fine-tuning, ROUGE-1/2/L, BERTScore F1, SummaC, QAFactEval, FineSurE |

---

## Learnability Results (RQ1–RQ3)

Reward models were trained on holistic vs GCA preference pairs and compared by 5-fold cross-validation pairwise accuracy, at three dataset scales:

| Dataset size | Runs | Holistic mean acc | GCA mean acc | Mean gap | 95% CI | Wilcoxon p |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 20 | 0.5297 | 0.5691 | +3.95 pp | [+3.19, +4.70] pp | 8.8×10⁻⁵ |
| 5,000 | 20 | 0.5842 | 0.5820 | −0.22 pp | [−0.66, +0.21] pp | 0.3410 |
| 10,000 | 20 | 0.5859 | 0.5848 | −0.11 pp | [−0.39, +0.18] pp | 0.5882 |

Interpretation. An initial six-run campaign at n=1,000 gave GCA ahead in 5 of 6 runs, mean advantage +3.08 pp, but a two-sided Wilcoxon test over the six run-level differences returned p = 0.0625 — the smallest two-sided value attainable with only six paired observations, so the campaign was structurally unable to reach significance regardless of the true effect size. After fixing a determinism gap in the training code (PyTorch's generator was not seeded in the cross-validation path, so two runs at the same `--seed` could still differ), the campaign was extended to 20 independently seeded runs. GCA led in all 20, mean advantage +3.95 pp, bootstrap 95% CI [+3.19, +4.70] pp (entirely positive), run-level Wilcoxon p < 0.001 — now statistically significant.

The n=5,000 and n=10,000 rows were originally single runs (−0.42 pp and +0.35 pp respectively), which could not distinguish a genuine scale effect from ordinary run-to-run noise. Both were extended to the same 20-run resolution as n=1,000, under the same corrected procedure. The result is decisive: mean gaps of −0.22 pp and −0.11 pp, both statistically indistinguishable from zero, with confidence intervals that do not overlap the n=1,000 result at all (nearest edges over 2.5 pp apart). This resolves the question definitively: GCA's advantage is confirmed real and significant at n=1,000, and confirmed absent at n=5,000 and n=10,000, not merely smaller — and all three conclusions now rest on equally powered evidence. Full per-run numbers and the aggregation script are in `thesis/chapters/06_results.tex` §6.6–6.9 and `analysis/aggregate_campaigns.py`.

Two further caveats bound the result:

- Both models are weak in absolute terms at n=1,000. 0.5297 and 0.5691 sit only 3.0 and 6.9 pp above the 0.50 chance level, even though the difference between them is significant. Both conditions rise to roughly 0.58–0.59 at the larger scales.
- The three subsets are nested, not disjoint. All use subset seed 200 and differ only in sample count: n=5,000 is entirely contained in n=10,000, and 999 of the 1,000 n=1,000 samples appear in n=10,000. The larger-scale runs therefore measure behaviour as data is added to the same pool; they are not out-of-sample replication, independent of how many times each was repeated.

Higher reward-model accuracy means the preference relation is more consistently recoverable by the chosen architecture. It does not establish better agreement with human factuality judgments, more accurate labels, or better generated summaries.

---

## Ground-Truth Results (RQ4)

The learnability results above validate each reward model against labels produced by the same evaluator that built its training preferences. RQ4 was added to close that gap: **500 held-out articles, disjoint from every training pool used in this thesis**, scored by the same locked AlignScore-GCA configuration but never used to train any reward model. No new reward models were trained — the checkpoints from RQ1–RQ3 were simply re-evaluated against a new criterion.

Success rate = the fraction of comparisons in which a reward model scores the higher-quality summary above the lower-quality one. Four constructions were used, differing only in how large a quality gap separates the two summaries being compared:

| Construction | Summaries/side | Score gap | n=1,000 gap | n=5,000 gap | n=10,000 gap | p (n=10,000) |
|---|---:|---:|---:|---:|---:|---:|
| Same-article pairs | 500 pairs | ≈0.04 | +0.14 pp | −0.43 pp | −1.45 pp | 0.010 |
| Top/bottom 25% | 250 | ≥0.36 | +3.24 pp | −0.42 pp | −2.12 pp | 0.036 |
| Top/bottom 10% | 100 | ≥0.66 | **+9.25 pp** | +0.10 pp | −1.95 pp | 0.058 |
| Top/bottom 5% (all-pairs) | 50 | ≥0.79 | **+7.38 pp** | +0.73 pp | −2.12 pp | 0.025 |

**Two findings, in order.**

*At n=1,000, GCA is substantially more precise — but only on comparisons that are clear-cut.* The advantage scales with the quality gap: nothing on same-article pairs (where the two candidates are close in quality by construction, p=0.76, equivalence rules out any true difference above ±0.96 pp), rising to +9.25 pp on top/bottom 10% (p<0.001, Cohen's *d* = +1.06). This is a sharper result than the learnability finding, and it was obtained on data the models never saw.

*At scale, the advantage reverses.* By n=10,000 holistic is significantly **more** precise than GCA on three of four constructions (p=0.010–0.036). This is the first result in the thesis, on any metric, where holistic significantly beats GCA rather than the two being indistinguishable.

**The reversal is holistic improving faster, not GCA degrading.** The gaps above hide this; the absolute rates make it visible. On top/bottom 10%, GCA rises 86.00% → 90.00% between n=1,000 and n=10,000. Holistic rises 76.75% → 91.95% over the same interval — a gain of 15.2 pp against GCA's 4.0. Both benefit from more data; holistic simply benefits far more, and overtakes. Any explanation of the reversal has to account for GCA's flatter returns to scale, not for a decline that does not occur.

**Where the crossover sits.** Supplementary five-seed runs at n=2,000 and n=3,000 place the transition in a region rather than at a point. Significance pins it down only at the two ends — GCA significantly ahead at n=1,000, holistic significantly ahead at n=10,000 — while nothing at n=2,000, n=3,000 or n=5,000 is distinguishable from zero in either direction. The honest statement is that the decline passes through a null region around n=3,000–5,000.

**Two candidate mechanisms tested and rejected.** Both were chosen because they make a falsifiable prediction about *where* their effect should be largest, and both required no new training:

1. **Near-tie training pairs** — that GCA produces more noise-level preference pairs than holistic, and fitting that noise hurts more as data grow. *Rejected:* the near-tie fraction is essentially identical between conditions at every scale (24.2% vs 24.2% at n=10,000), and at n=1,000 — where GCA wins — GCA has *more* near-ties, not fewer. Opposite of the prediction.
2. **Length/sentence-count shortcut** — that GCA increasingly leans on summary length as a proxy for quality. *Rejected:* the GCA-minus-holistic gap in length-correlation should peak at n=10,000 where GCA loses; instead it peaks at n=3,000 (−0.043) and nearly vanishes at n=10,000 (−0.003). Opposite of the prediction again.

In both cases the data contradicted the hypothesis's own prediction rather than merely failing to confirm it, which is a cleaner rejection than a null result. **No mechanism has been confirmed.** The leading untested candidate — that GCA's sentence-level aggregation lets the reward model fit article-specific, within-pair regularities that do not generalise to the ground truth's cross-article comparisons — is recorded under Future Work and was not attempted.

One side-finding worth noting: *both* conditions become roughly five times more correlated with summary length as training data grow. Because this affects them symmetrically it cannot explain the reversal, but it is a plausible contributor to the low absolute ceiling neither condition breaks through.

Full detail: `thesis/chapters/06_results.tex` §6.10 (six subsections), Discussion §7.6, Conclusion §8.2. Pipeline: `scripts/03_prepare_ground_truth_subset.py`, `analysis/build_biased_ground_truth.py`, `analysis/evaluate_ground_truth.py`, `analysis/ground_truth_stats.py`, `analysis/check_length_shortcut.py`.

---

Full experimental history is in `progress-updates/` and `OPTIMIZATION_CAMPAIGN.md`.

> Canonical source. The written thesis in [`thesis/`](thesis/) supersedes all earlier summaries in this repository. Where a `progress-updates/` entry or `reports/` file disagrees with it, the thesis is correct: those files are dated records of what was believed at the time, and several predate the statistical and dataset-nesting corrections above. In particular, `reports/reward_model_judging_results.md` is an April 2026 artefact (200 samples, margin=0.05, α=0.5, DPO-era) and should not be cited as a current result.

---

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── subset.yaml           #   Data subset selection params
│   ├── generation.yaml       #   Candidate generation params
│   └── judging.yaml          #   AI judging params
├── scripts/                  # Pipeline entry points (run in order)
│   ├── 01_prepare_subset.py  #   Select N CNN/DM articles (seeded, deterministic)
│   ├── 02_generate_candidates.py  # Generate summary pairs (two temperatures)
│   ├── 03_prepare_ground_truth_subset.py  # RQ4: 500-article held-out set, disjoint from all training pools
│   ├── 03_run_judge_test.py  #   Test judging prompts (mock or live)
│   ├── 04_evaluate_baseline.py    # Compute ROUGE/BERTScore baseline
│   └── 05_plot_results.py    #   Generate comparison plots
├── src/                      # Core library
│   ├── data/                 #   Schema (SubsetSample→CandidatePair→Judgment→PreferencePair), subset selection
│   ├── generation/           #   Model loading, two-temperature candidate generation
│   ├── judging/              #   Holistic, sentence-level, GCA aggregation, preference construction
│   ├── reward_model/         #   Bradley-Terry RM training (train.py, run_training.py) — final comparison target
│   ├── eval/                 #   ROUGE, BERTScore, SummaC, QAFactEval (early-prototype metrics only)
│   ├── dpo/                  #   DPO fine-tuning (early prototype, dropped 2 June 2026 — kept for history)
│   └── utils/                #   Config loader, run-metadata logging
├── slurm/                    # MOGON NHR job scripts
│   ├── smoke_test.sh         #   5-sample validation (A100, ~1 min)
│   ├── generate_candidates.sh / generate_candidates_scale.sh
│   ├── build_reward_preferences.sh / build_preferences_scale.sh
│   ├── train_reward_models.sh / train_rm_1000.sh / train_rm_scale.sh  #  Bradley-Terry RM training (holistic + GCA)
│   ├── submit_gca_hpsearch.sh #  RM hyperparameter search
│   ├── mode_nli_seed_confirm.sh #  Seed-validation reruns for the locked `nli` judge mode
│   └── RQ4 ground-truth chain:
│       generate_candidates_groundtruth.sh → build_ground_truth.sh →
│       retrain_for_ground_truth{,_5000,_10000,_curve}.sh →
│       evaluate_ground_truth_{scale,curve}.sh
├── analysis/                 # Optimisation, statistics, and RQ4 evaluation scripts
│   ├── aggregate_campaigns.py         # Run-level means, bootstrap CIs, Wilcoxon tests
│   ├── effect_sizes_and_equivalence.py # Cohen's d, TOST equivalence, MDE
│   ├── build_biased_ground_truth.py   # RQ4: top/bottom X% construction
│   ├── evaluate_ground_truth.py / ground_truth_stats.py  # RQ4: scoring + statistics
│   ├── check_length_shortcut.py       # RQ4: length/sentence-count shortcut test
│   ├── surface_feature_baselines.py   # Confound check: can length alone predict the label?
│   └── test_gca_formulas.py / test_advanced_aggregation.py  # GCA formula sweep
├── reports/campaigns/        # Committed per-campaign JSON results (all reported numbers regenerate from these)
├── thesis/                   # LaTeX source — the canonical write-up (main.pdf, chapters/)
├── progress-updates/         # Biweekly meeting reports (chronological — 25-08-2026/ is the most recent)
├── proposal/                 # Thesis proposal
├── data/                     # Generated data artifacts (gitignored)
├── outputs/                  # Metrics, plots, run metadata (gitignored)
└── requirements.txt
```

---

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Prepare an N-sample subset (seed configurable in configs/subset.yaml)
python scripts/01_prepare_subset.py --config configs/subset.yaml

# 2. (Requires GPU + model) Generate candidates (two temperatures)
python scripts/02_generate_candidates.py --config configs/generation.yaml

# 3. Build reward-model preferences (holistic + GCA), locked final config
python src/judging/build_reward_preferences.py \
    --candidates data/candidates/candidates_1000.jsonl \
    --output-dir data/preferences \
    --mode both \
    --alpha 0.0 \
    --margin 0 \
    --alignscore-evaluation-mode nli

# 4. Train Bradley-Terry reward models (holistic + GCA), 5-fold CV
python src/reward_model/run_training.py \
    --holistic data/preferences/holistic_1000.jsonl \
    --gca data/preferences/gca_1000.jsonl \
    --output-dir outputs/reward_models_1000 \
    --kfold 5 --epochs 5 --lr 2e-5 --batch-size 8
```

---

## MOGON NHR Execution (HPC)

```bash
# Sync code (preserving directory structure)
rsync -avz --relative src/ scripts/ configs/ slurm/ \
  -e "ssh -o 'ControlPath=/tmp/mogon-nhr-cm.sock'" mogon:~/thesis/

# SSH in (requires 2FA via freeOTP → hpcgate.zdv.uni-mainz.de ProxyJump)
ssh mogon

# Activate conda environment
module load lang/Anaconda3/2024.06-1
source $(conda info --base)/etc/profile.d/conda.sh && conda activate thesis_env

# Run smoke test, then candidate generation
sbatch slurm/smoke_test.sh
sbatch slurm/generate_candidates.sh

# Build reward-model preferences (holistic + GCA) on A100
sbatch slurm/build_reward_preferences.sh

# Train Bradley-Terry reward models (holistic + GCA), 5-fold CV
sbatch slurm/train_rm_1000.sh      # or train_rm_scale.sh for 5k/10k reruns

# Monitor
squeue -u muhhas01
```

---

## Key Design Decisions

- **Controlled experiment:** Only feedback granularity (holistic vs GCA) varies — candidate pool, judge, and RM training recipe are identical across conditions
- **Config-driven:** All parameters in YAML files / CLI flags, overridable via `--override key=value` or explicit flags
- **Reproducible:** Seeded randomness, SHA256-based sample IDs, run metadata JSON per execution, multi-seed validation for the headline result
- **Fixed factuality judge:** Preferences are derived from `yzha/AlignScore` (`nli` mode, locked). Deterministic, testable, avoids any OpenAI / generative LLM dependency.
- **No margin gating in the final pipeline:** unlike the early prototype (`tie_margin=0.05`), the final config uses `margin=0` — all pairs are used, based on supervisor feedback (2 June 2026) that margin filtering discarded ~20% of usable pairs without clear benefit.
- **DPO dropped from the final comparison:** the original proposal's DPO fine-tuning step (`src/dpo/`) was used in the early prototype (Mar–May 2026) but removed on 2 June 2026 so the thesis focuses purely on reward-model learnability — the core IRL framing suggested by the mentor.
- **Compatibility with accepted proposal:** the core thesis comparison is unchanged — holistic AI feedback vs granular AI feedback via GCA. What changed is (a) the feedback source (generative LLM → fixed factuality classifier) and (b) the downstream target (DPO-tuned policy → Bradley-Terry reward model), both agreed with supervisors as within-scope pivots.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Candidate generation model | Mistral-7B-Instruct-v0.3 |
| Dataset | CNN/DailyMail |
| Frameworks | Hugging Face Transformers 5.x, PEFT, BitsAndBytes, PyTorch 2.4 |
| Infrastructure | MOGON NHR (A100-SXM4-40GB, Slurm, partition `a100dl`) |
| Judge | `yzha/AlignScore`, mode `nli` (locked final config) |
| Reward model | Bradley-Terry, `FacebookAI/roberta-base` backbone |
| Early-prototype only (superseded) | DPO + LoRA (TRL/PEFT), SummaC, QAFactEval, FineSurE, ROUGE-1/2/L, BERTScore F1 |

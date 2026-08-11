# Master Thesis: Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)

**University of Koblenz** | Master's Thesis | 2025–2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner  
**Mentor:** Lingxiao Kong  
**GitHub:** [@hasnat23](https://github.com/hasnat23)

---

## Overview

> Status (as of 10 August 2026): experimentation is complete. The core empirical question has been answered via a Bradley-Terry reward-model comparison, now validated with 20 independent runs at all three dataset scales (see [Final Results](#final-results) below). Remaining work is thesis write-up, theoretical framing, and related work. Early-phase DPO fine-tuning experiments (Mar–May 2026) are preserved in `src/dpo/` and earlier `progress-updates/` entries but are not part of the final reported pipeline — DPO was dropped on 2 June 2026 in favour of a pure reward-model comparison, per supervisor feedback.

This thesis investigates a controlled question: for long news summaries, are holistic A/B preferences sufficient to produce a learnable factual-reliability signal for reward-model training, or does supervision become more effective when the judge evaluates aligned sentence pairs whose local decisions are then aggregated into a summary-level preference via Granular Credit Assignment (GCA)?

The base candidate pool, judge, and reward-model training recipe are held constant across conditions. The only manipulated factor is the granularity of AI-generated preference labels used to construct the pairwise training data.

- **Dataset:** CNN/DailyMail (final runs at 1,000 / 5,000 / 10,000 samples; note these subsets are *nested*, see [Final Results](#final-results))
- **Candidate generation model:** Mistral-7B-Instruct-v0.3 (instruction-tuned 7B, Apache 2.0) — two-temperature sampling (T=0.7 / T=1.0)
- **Judge:** `yzha/AlignScore`, `nli` mode — a fixed factual-consistency metric used as the primary automatic judge (no OpenAI API, no generative LLM calls). Locked as the final judge configuration.
- **Reward model:** Bradley-Terry pairwise reward model, `FacebookAI/roberta-base` backbone, trained separately on holistic vs GCA preference sets, evaluated via 5-fold cross-validation pairwise accuracy
- **Primary metric:** RM pairwise validation accuracy (Holistic vs GCA), with bootstrap 95% CIs and Wilcoxon significance tests
- **Superseded/early-phase only:** ROUGE-1/2/L, BERTScore F1, SummaC, QAFactEval, FineSurE were used during the earlier DPO-based prototype (see `progress-updates/19-03-2026/` through `21-04-2026/`) but are not part of the final reward-model comparison

---

## Research Questions

**RQ1:** Does sentence-level AI feedback aggregated via GCA produce a more learnable reward-model preference signal than holistic AI feedback, as measured by pairwise validation accuracy?

**RQ2:** How sensitive are the observed effects to the AlignScore judge backend/mode and to dataset scale (1,000 / 5,000 / 10,000 samples)?

**RQ3:** What categories of factual errors are most affected by sentence-level supervision (entities, numbers, relations, temporal claims)? *(qualitative/error-analysis work, planned for the write-up phase)*

**RQ4:** Do gains persist across independent random seeds and larger sample sizes, or are they an artefact of one run? Answered — see H4 below.

### Hypotheses

- **H1:** Sentence-level aggregated preferences produce a more learnable reward-model signal than holistic preferences because they localise the supervision signal on long outputs.
- **H2:** Judge backend/mode configuration (e.g. AlignScore `nli` vs `nli_sp`/`bin`) materially affects whether the GCA advantage appears.
- **H3:** Improvements are largest for localised errors (entity/relation mistakes) rather than global attributes such as style.
- **H4:** If improvements reflect a generalisable signal, they should replicate across repeated runs — confirmed at n=1,000 (20/20 runs favour GCA, mean +3.95pp, run-level Wilcoxon p<0.001, significant) but confirmed absent at n=5,000 and n=10,000 (20/20-run campaigns at each scale, mean gaps of −0.22pp and −0.11pp, both statistically indistinguishable from zero) — see [Final Results](#final-results).

**Status:** H1 and H4 are supported at n=1,000, at the conventional significance threshold (p<0.001), and the effect is confirmed absent, not merely unreproduced, at n=5,000/10,000 (also at 20-run resolution, p=0.34 and p=0.59). H3 was not evaluated. H2 is supported by an exploratory sweep: the AlignScore `nli` mode is associated with the GCA advantage, but each mode was run once, so the mode comparison is suggestive rather than established.

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

## Final Results

Reward models were trained on holistic vs GCA preference pairs and compared by 5-fold cross-validation pairwise accuracy, at three dataset scales:

| Dataset size | Runs | Holistic mean acc | GCA mean acc | Mean gap | 95% CI | Wilcoxon p |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 20 | 0.5297 | 0.5691 | +3.95 pp | [+3.19, +4.70] pp | 8.8×10⁻⁵ |
| 5,000 | 20 | 0.5842 | 0.5820 | −0.22 pp | [−0.66, +0.21] pp | 0.3410 |
| 10,000 | 20 | 0.5859 | 0.5848 | −0.11 pp | [−0.39, +0.18] pp | 0.5882 |

Interpretation. An initial six-run campaign at n=1,000 gave GCA ahead in 5 of 6 runs, mean advantage +3.08 pp, but a two-sided Wilcoxon test over the six run-level differences returned p = 0.0625 — the smallest two-sided value attainable with only six paired observations, so the campaign was structurally unable to reach significance regardless of the true effect size. After fixing a determinism gap in the training code (PyTorch's generator was not seeded in the cross-validation path, so two runs at the same `--seed` could still differ), the campaign was extended to 20 independently seeded runs. GCA led in all 20, mean advantage +3.95 pp, bootstrap 95% CI [+3.19, +4.70] pp (entirely positive), run-level Wilcoxon p < 0.001 — now statistically significant.

The n=5,000 and n=10,000 rows were originally single runs (−0.42 pp and +0.35 pp respectively), which could not distinguish a genuine scale effect from ordinary run-to-run noise. Both were extended to the same 20-run resolution as n=1,000, under the same corrected procedure. The result is decisive: mean gaps of −0.22 pp and −0.11 pp, both statistically indistinguishable from zero, with confidence intervals that do not overlap the n=1,000 result at all (nearest edges over 2.5 pp apart). This resolves the question definitively: GCA's advantage is confirmed real and significant at n=1,000, and confirmed absent at n=5,000 and n=10,000, not merely smaller — and all three conclusions now rest on equally powered evidence. Full per-run numbers and the aggregation script are in `thesis/chapters/06_results.tex` §6.6–6.8 and `analysis/aggregate_campaigns.py`.

Two further caveats bound the result:

- Both models are weak in absolute terms at n=1,000. 0.5297 and 0.5691 sit only 3.0 and 6.9 pp above the 0.50 chance level, even though the difference between them is significant. Both conditions rise to roughly 0.58–0.59 at the larger scales.
- The three subsets are nested, not disjoint. All use subset seed 200 and differ only in sample count: n=5,000 is entirely contained in n=10,000, and 999 of the 1,000 n=1,000 samples appear in n=10,000. The larger-scale runs therefore measure behaviour as data is added to the same pool; they are not out-of-sample replication, independent of how many times each was repeated.

Higher reward-model accuracy means the preference relation is more consistently recoverable by the chosen architecture. It does not establish better agreement with human factuality judgments, more accurate labels, or better generated summaries.

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
│   ├── 01_prepare_subset.py  #   Select 200 CNN/DM articles
│   ├── 02_generate_candidates.py  # Generate summary pairs (two temperatures)
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
│   ├── generate_candidates.sh #  Candidate generation (A100)
│   ├── build_reward_preferences.sh / build_reward_preferences_rm500.sh
│   ├── train_reward_models.sh / train_rm_1000.sh / train_rm_scale.sh  #  Bradley-Terry RM training (holistic + GCA)
│   ├── submit_gca_hpsearch.sh #  RM hyperparameter search
│   └── mode_nli_seed_confirm.sh #  Seed-validation reruns for the locked `nli` judge mode
├── analysis/                 # GCA formula optimisation & disagreement analysis scripts
├── progress-updates/         # Biweekly meeting reports (chronological — see 11-08-2026/ for final results)
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

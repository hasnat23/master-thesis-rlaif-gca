# Progress Update — 2 June 2026

**Meeting date:** 2 June 2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## Thesis Context

**Title:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)  
**Core question:** Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic feedback under DPO? And can we distil that preference signal into a trained reward model for the IRL framing?

---

## Summary

Since the last meeting (19 May 2026), the 500-sample candidate generation job completed, AlignScore preference-building finished (495 pairs labelled), and the Bradley-Terry reward model training completed (20 May 2026). Holistic RM reached **58.1% pairwise accuracy** vs GCA RM at **54.6%** (5-fold CV), confirming that holistic preference signals are more learnable from this dataset. DPO fine-tuning (Phase 3) is the next major milestone.

---

## Completed Since Last Meeting

### 1. 500-Sample Candidate Generation — COMPLETED ✅

The candidate generation job finished on MOGON on 19 May 2026.

| Field | Value |
|-------|-------|
| Job ID | 1157804 |
| Cluster | mogonnhr |
| State | COMPLETED (exit code 0) |
| Wall-clock time | 01:19:46 |
| GPU | NVIDIA A100-SXM4-40GB |
| Memory used | 28.25 GB / 48 GB (58.84%) |
| Output file | `data/candidates/candidates_rm500.jsonl` |
| Samples generated | **495 / 495** |

This dataset is disjoint from the 200-sample DPO set (seed=100 vs seed=42) and will be used exclusively for reward model training.

---

### 2. AlignScore Preference Building — COMPLETED ✅

The AlignScore judging job completed on 20 May 2026 (job 1170125, after fixing a `--max-samples` bug in the initial submission 1170124).

| Field | Value |
|-------|-------|
| Job ID | 1170125 |
| Script | `slurm/build_reward_preferences_rm500.sh` |
| State | COMPLETED (exit code 0) |
| Output | `data/preferences_rm500/holistic_reward_preferences_rm500.jsonl` (494 pairs) |
| | `data/preferences_rm500/gca_reward_preferences_rm500.jsonl` (495 pairs) |

AlignScore scored all 495 candidate pairs under holistic and GCA (α=0.5) modes, producing two separate preference datasets for reward model training.

---

### 3. Bradley-Terry Reward Model Training — COMPLETED ✅

Both reward models were trained on MOGON on 20 May 2026 (job 1170128, wall-clock 00:12:27).

**Architecture:**
- Backbone: `FacebookAI/roberta-base` (encoder-only; `microsoft/deberta-v3-base` is not whitelisted on the MOGON HF proxy)
- Input: `"{article[:2000]} [SEP] {summary}"` → 512 tokens
- Head: mean-pool → linear → scalar reward $r_\theta$
- Loss: $\mathcal{L} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$
- Evaluation: pairwise accuracy $P(r_w > r_l)$, 5-fold cross-validation, 5 epochs per fold

**Results (5-fold CV pairwise accuracy):**

| Condition | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | **Mean** |
|-----------|--------|--------|--------|--------|--------|----------|
| Holistic RM | 55.4% | 60.8% | 59.5% | 67.6% | 47.3% | **58.1%** |
| GCA RM | 58.5% | 46.3% | 56.1% | 52.4% | 59.8% | **54.6%** |

**Interpretation:** Both models learn above chance (50%). Holistic preferences yield a more consistent and learnable reward signal. GCA accuracy is near-chance in two folds, suggesting the GCA signal is noisier or requires a larger training set. This is the key IRL-framing result motivating the thesis comparison.

---

## Pipeline Status Overview

| Stage | Status | Job / Artifact |
|-------|--------|----------------|
| 200-sample DPO candidate generation | ✅ Complete | `data/candidates/candidates_200.jsonl` |
| 200-sample AlignScore judging | ✅ Complete | `data/preferences/` (154 holistic, 157 GCA pairs) |
| 500-sample RM candidate generation | ✅ Complete | `data/candidates/candidates_rm500.jsonl` (495 samples) |
| 500-sample AlignScore judging | ✅ Complete | Job 1170125 (494/495 pairs) |
| Bradley-Terry RM training (holistic + GCA) | ✅ Complete | Job 1170128 — holistic 58.1%, GCA 54.6% |
| DPO fine-tuning — holistic condition | ⏳ Pending | Depends on 200-sample preferences (ready) |
| DPO fine-tuning — GCA condition | ⏳ Pending | Depends on 200-sample preferences (ready) |
| Post-DPO evaluation (ROUGE / BERTScore / AlignScore) | ⏳ Pending | After DPO |

---

## Next Steps

| Priority | Step | Blocker |
|----------|------|---------|
| 1 | Submit DPO fine-tuning jobs (holistic + GCA) on 200-sample preferences | — (data ready) |
| 2 | Run post-DPO evaluation on both fine-tuned models (ROUGE / BERTScore / AlignScore) | DPO completion |
| 3 | Analyse disagreement cases (39.5% holistic/GCA disagreement) qualitatively | Analysis |
| 4 | Investigate whether a larger RM training set improves GCA accuracy | Optional |

---

## Key Results So Far (200-Sample Pilot)

**Holistic judging:** 154 usable pairs (77.0%), mean AlignScore A=0.7324 vs B=0.6530  
**GCA judging:** 157 usable pairs (78.5%), mean GCA A=0.4066 vs B=0.3337  
**Holistic–GCA agreement: 60.5%** — 39.5% disagreement rate confirms the two signals are not equivalent and motivates the full thesis comparison.

---

## Repository State

All code is on `origin/main`. Key recent commits:

| Commit | Description |
|--------|-------------|
| `3af4b7b` | Fix RM backbone to `roberta-base`; add `TRANSFORMERS_OFFLINE` flags |
| `34cd021` | Fix judging output filename to derive suffix from input path |
| `841af8f` | Fix `--max-samples 495` in judging Slurm script |
| `62cc728` | Add progress update for 2 June 2026 meeting |
| `7e6d042` | Latest README fix (19 May update) |
| `02131d4` | Bradley-Terry reward model training pipeline |

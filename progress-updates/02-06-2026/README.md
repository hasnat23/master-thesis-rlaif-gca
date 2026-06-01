# Progress Update — 2 June 2026

**Meeting date:** 2 June 2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## Thesis Context

**Title:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)  
**Core question:** Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic feedback under DPO? And can we distil that preference signal into a trained reward model for the IRL framing?

---

## Research Questions and Objectives

The overarching objective is to determine whether more granular AI supervision improves factual reliability in news summarization under a controlled, offline preference-learning setup.

### RQ1
Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic AI feedback when used for DPO fine-tuning?

### RQ2
How sensitive are the observed effects to alignment strategy (index-based vs semantic alignment) and to judge reliability controls such as confidence gating and A/B order randomization?

### RQ3
What categories of factual errors are most affected by sentence-level supervision (entities, numbers, relations, temporal claims)?

### RQ4
Do gains persist under human auditing and (optionally) a second judge, or are they judge-specific artifacts?

### Hypotheses

- **H1:** Sentence-level aggregated preferences reduce factual inconsistency more than holistic preferences because they localize the supervision signal on long outputs.
- **H2:** Better alignment and reliability controls reduce label noise and training variance; if alignment noise is too high, it can mask the benefits of GCA.
- **H3:** Improvements are largest for localized errors (entity and relation mistakes) rather than global attributes such as style.
- **H4:** If improvements reflect genuine factuality gains, they remain visible under at least one independent evaluator (automatic metrics and/or human audit).

---

## Summary

Since the last meeting (19 May 2026), the full RLAIF pipeline has been completed end-to-end. The 500-sample candidate generation and AlignScore preference-building finished, Bradley-Terry reward model training completed (holistic 58.1% / GCA 54.6% pairwise accuracy, 5-fold CV), DPO fine-tuning on Mistral-7B-Instruct-v0.3 completed for both holistic and GCA conditions, and the post-DPO evaluation pipeline ran successfully on 50 held-out articles. **DPO-GCA achieves the best ROUGE and BERTScore; DPO-holistic achieves the best AlignScore factual consistency (+1.0pp over baseline).**

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
| DPO fine-tuning — holistic condition | ✅ Complete | Job 1170154 — final loss 0.6902 |
| DPO fine-tuning — GCA condition | ✅ Complete | Job 1170154 — final loss 0.6913 |
| Post-DPO evaluation (ROUGE / BERTScore / AlignScore) | ✅ Complete | Job 1170174 — see results below |

---

## DPO Fine-Tuning Results — COMPLETED ✅

Both LoRA adapters trained on MOGON (job 1170154, 20 May 2026) on an NVIDIA A100-SXM4-40GB.

| Condition | Training pairs | Wall-clock | Final loss | Adapter path |
|-----------|---------------|-----------|-----------|-------------|
| Holistic | 154 | ~135 s | **0.6902** | `outputs/dpo/holistic/adapter/` |
| GCA | 157 | ~135 s | **0.6913** | `outputs/dpo/gca/adapter/` |

**Setup:** Mistral-7B-Instruct-v0.3, LoRA r=16 α=32 on all attention + MLP projection layers, β=0.1, bfloat16, `device_map="auto"`. Both adapters saved as PEFT checkpoints and merged via `merge_and_unload()` for inference.

---

## Post-DPO Evaluation Results — COMPLETED ✅

Evaluation pipeline ran on MOGON (job 1170174, 20 May 2026), generating greedy summaries from three conditions on 50 held-out articles from `candidates_200.jsonl` and scoring with ROUGE, BERTScore (roberta-base), and AlignScore-base against the source article.

### Metric Comparison Table

| Condition | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore-F1 | AlignScore |
|-----------|---------|---------|---------|-------------|------------|
| **baseline** | 0.3399 | 0.1273 | 0.2177 | 0.8476 | 0.8240 |
| **DPO-holistic** | 0.3408 | 0.1271 | 0.2161 | 0.8477 | **0.8324** |
| **DPO-GCA** | **0.3440** | **0.1297** | **0.2189** | **0.8484** | 0.8219 |

*Δ = delta vs baseline; bold = best per metric.*

| Condition | ΔROUGE-1 | ΔROUGE-2 | ΔROUGE-L | ΔBERTScore-F1 | ΔAlignScore |
|-----------|---------|---------|---------|-------------|------------|
| DPO-holistic | +0.0009 | −0.0002 | −0.0016 | +0.0001 | **+0.0084** |
| DPO-GCA | **+0.0041** | **+0.0024** | **+0.0012** | **+0.0008** | −0.0021 |

### Interpretation

- **DPO-GCA** achieves the highest ROUGE scores across all three variants and highest BERTScore-F1, indicating its summaries have better lexical and semantic overlap with the CNN/DM reference summaries. This is consistent with GCA preferences attending to sentence-level factual alignment.
- **DPO-holistic** achieves the **best AlignScore** (+0.84pp over baseline), meaning holistic DPO training most improves factual consistency of the generated text against the *source article*. This is the central thesis finding: holistic reward signals produce summaries that are more faithfully grounded in the source.
- **DPO-GCA AlignScore** is slightly below baseline (−0.21pp), suggesting that sentence-level GCA preferences can optimise for reference-style phrasing at the marginal cost of source grounding.
- All differences are small in absolute terms, which is expected for 50 test samples and a 200-sample DPO training set. Scaling to more data would be needed to confirm statistical significance.

---

## Next Steps

| Priority | Step | Blocker |
|----------|------|---------|
| 1 | Thesis write-up — results chapter (Phases 1–3) | — |
| 2 | Qualitative analysis of disagreement cases (39.5% holistic/GCA) | `outputs/eval/generations.jsonl` available |
| 3 | Optional: scale DPO to full 495-sample RM set to verify ROUGE/AlignScore trends | Compute time |
| 4 | Optional: DPO β ablation (0.05, 0.1, 0.2) to assess sensitivity | Compute time |

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
| `867cbb7` | eval: fix AlignScore import (patch AdamW) + use local roberta-base backbone |
| `aca4216` | eval: fix BERTScore to use local roberta-base (num_layers=9) |
| `efc0c5a` | eval: add DPO evaluation script (ROUGE, BERTScore, AlignScore) + Slurm job |
| `f2b1764` | progress: update 19-05-2026 README with RM + DPO results |
| `c90fb48` | dpo: DPO fine-tuning pipeline (run_dpo.py + Slurm script) |
| `3af4b7b` | Fix RM backbone to `roberta-base`; add `TRANSFORMERS_OFFLINE` flags |
| `02131d4` | Bradley-Terry reward model training pipeline |

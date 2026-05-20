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

Since the last meeting (19 May 2026), the 500-sample candidate generation job completed successfully and the AlignScore preference-building job has been submitted. The full reward model training pipeline (Bradley-Terry) is ready and will be submitted once preferences are available. DPO fine-tuning (Phase 3) is the next major milestone.

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

### 2. AlignScore Preference Building — SUBMITTED ✅ ⏳

Immediately after confirming the generation output, the AlignScore judging job was submitted (20 May 2026).

| Field | Value |
|-------|-------|
| Job ID | 1170124 |
| Script | `slurm/build_reward_preferences_rm500.sh` |
| State | SUBMITTED / PENDING |
| Expected outputs | `data/preferences_rm500/holistic_reward_preferences_rm500.jsonl` |
| | `data/preferences_rm500/gca_reward_preferences_rm500.jsonl` |
| Expected runtime | ~1h on A100 |

AlignScore judges each of the 495 candidate pairs under both holistic scoring and GCA (α=0.5), producing two separate preference datasets for reward model training.

---

### 3. Bradley-Terry Reward Model Pipeline — READY ✅

The full reward model training code was implemented and committed last week (`src/reward_model/`, commit `02131d4`). It is ready to submit once job 1170124 completes.

**Architecture recap:**
- Backbone: `microsoft/deberta-v3-base` (encoder-only)
- Input: `"{article[:2000]} [SEP] {summary}"` → 512 tokens
- Head: mean-pool → linear → scalar reward $r_\theta$
- Loss: $\mathcal{L} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$
- Evaluation: pairwise accuracy $P(r_w > r_l)$, 5-fold cross-validation

**Two models trained separately:**
- `rm_holistic` — trained on holistic AlignScore preferences
- `rm_gca` — trained on GCA preferences

The comparison of their pairwise validation accuracies is the IRL framing result.

---

## Pipeline Status Overview

| Stage | Status | Job / Artifact |
|-------|--------|----------------|
| 200-sample DPO candidate generation | ✅ Complete | `data/candidates/candidates_200.jsonl` |
| 200-sample AlignScore judging | ✅ Complete | `data/preferences/` (154 holistic, 157 GCA pairs) |
| 500-sample RM candidate generation | ✅ Complete | `data/candidates/candidates_rm500.jsonl` (495 samples) |
| 500-sample AlignScore judging | ⏳ Running | Job 1170124 |
| Bradley-Terry RM training (holistic + GCA) | ⏳ Pending | Depends on job 1170124 |
| DPO fine-tuning — holistic condition | ⏳ Pending | Depends on 200-sample preferences (ready) |
| DPO fine-tuning — GCA condition | ⏳ Pending | Depends on 200-sample preferences (ready) |
| Post-DPO evaluation (ROUGE / BERTScore / AlignScore) | ⏳ Pending | After DPO |

---

## Next Steps

| Priority | Step | Blocker |
|----------|------|---------|
| 1 | Monitor job 1170124; submit `train_reward_models.sh` on completion | AlignScore job |
| 2 | Submit DPO fine-tuning jobs (holistic + GCA) on 200-sample preferences | Compute scheduling |
| 3 | Compare BT RM pairwise accuracies (holistic vs GCA) — IRL result | RM training |
| 4 | Run post-DPO evaluation on both fine-tuned models | DPO completion |
| 5 | Analyse disagreement cases (39.5% holistic/GCA disagreement) qualitatively | Analysis |

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
| `7e6d042` | Latest README fix (19 May update) |
| `02131d4` | Bradley-Terry reward model training pipeline |
| `2423487` | Remove `device_map`, use `torch_dtype` + `.to(cuda)` |
| `bb3f7c1` | `local_files_only=True` + `TRANSFORMERS_OFFLINE=1` |

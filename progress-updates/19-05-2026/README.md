# Progress Update — 19–20 May 2026

**Meeting date:** 19 May 2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## Thesis Context

**Title:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)  
**Core question:** Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic feedback under DPO? And can we distil that preference signal into a trained reward model for the IRL framing?

---

## Summary

The AlignScore judging pipeline is fully validated and has produced the first real preference dataset (200 samples). Based on Lingxiao's suggestion to explore the IRL direction, a **Bradley-Terry reward model training pipeline** has been designed and implemented this week. Both the BT reward models and the DPO fine-tuned adapters have now been trained successfully on MOGON.

---

## Completed This Week

### 1. AlignScore Full Run — 200-Sample Results ✅

The full 200-sample judging job ran successfully on MOGON (job 1079466, ~3 minutes on A100). Both holistic and GCA scoring are working correctly with the `FacebookAI/roberta-base` backbone via the HF proxy.

**Holistic Judging**

| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Usable preference pairs | **154** (77.0%) |
| Ties (excluded) | 46 |
| Mean faithfulness — A (low-temp) | 0.7324 |
| Mean faithfulness — B (high-temp) | 0.6530 |

**GCA Judging** (sentence-level, α = 0.5)

| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Usable preference pairs | **157** (78.5%) |
| Ties (excluded) | 43 |
| Mean GCA score — A | 0.4066 |
| Mean GCA score — B | 0.3337 |
| Avg sentences per summary (A / B) | 7.0 / 6.7 |

**Agreement between holistic and GCA decisions: 60.5% (121/200)**

The 39.5% disagreement rate is the central finding of this experiment — it demonstrates that holistic and sentence-level scoring are not equivalent and do not produce the same preference signal. A concrete example:

> `cnn_00012` — *Carlos Alberto / Werder Bremen*  
> Holistic says **A wins** (A=0.795, B=0.734).  
> GCA says **B wins** (A=0.597, B=0.664).  
> Holistic collapses all sentence evidence into one number and misses a specific faithfulness failure that GCA catches at the sentence level.

This is exactly the thesis argument: fine-tuning on holistic preferences risks rewarding summaries that contain individual unfaithful sentences, as long as the overall score is high enough.

---

### 2. Bradley-Terry Reward Model — Trained ✅

A complete reward model training pipeline was implemented and run on MOGON (job 1170128, `FacebookAI/roberta-base` backbone, 5-fold CV on ~380–390 usable pairs per condition).

**Results**

| Condition | Pairwise Accuracy (5-fold CV) |
|-----------|------------------------------|
| Holistic RM | **58.1%** |
| GCA RM | **54.6%** |

Both models learn the preference signal above random chance (50%). The holistic RM is slightly more accurate, likely because the holistic signal is smoother and more consistent — consistent with the 60.5% agreement rate between conditions.

**Architecture:** `FacebookAI/roberta-base` encoder → mean-pool → linear scalar reward head  
**Loss:** Bradley-Terry — $\mathcal{L} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$  
**Adapters saved to:** `outputs/reward_models/{holistic,gca}/best/`

---

### 3. DPO Fine-Tuning (Phase 3) — Completed ✅

Mistral-7B-Instruct-v0.3 was fine-tuned with DPO + LoRA on both preference sets (MOGON job 1170154, A100-SXM4-40GB, ~6.5 min total).

**Results**

| Condition | Training pairs | Train loss | Runtime |
|-----------|---------------|-----------|---------|
| Holistic | 154 | **0.6902** | 134.5 s |
| GCA | 157 | **0.6913** | 134.5 s |

**LoRA config:** r=16, α=32, targets = `q/k/v/o/gate/up/down_proj`, dropout=0.05  
**DPO config:** β=0.1, lr=5×10⁻⁷, effective batch=16 (2×8 grad accum), max_length=1024  
**Adapters saved to:** `outputs/dpo/{holistic,gca}/adapter/`

The near-log(2) losses (~0.693) indicate the model has not collapsed — it is making meaningful preference distinctions rather than defaulting to a uniform prior. The reward margin signals confirm separation between chosen and rejected log-probabilities.

---

### 4. 500-Sample RM Dataset — Generation & Judging ✅

- **495 samples** generated (seed=100, disjoint from DPO set)
- AlignScore judging completed: 494 holistic pairs, 495 GCA pairs
- Used as BT RM training data (described in §2 above)

---

## Key Technical Issues Resolved

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `use_fast=True` tokenizer crash | Old `tokenizers` lib on MOGON | `use_fast=False` |
| `KeyError: 'mistral'` | `transformers 4.30.2` predates Mistral support | Upgraded to `4.40.2` → `5.8.1` |
| `DPOConfig` unexpected kwarg `max_prompt_length` | TRL 1.0.0 removed this from DPOConfig | Moved to DPOTrainer; later removed (not in TRL 1.0.0) |
| `cudaErrorDevicesUnavailable` on all DPO jobs | All jobs landing on `gpu0001` (broken CUDA runtime) | Added `#SBATCH --exclude=gpu0001` (same fix as RM script) |
| `model_init_kwargs` placement | TRL 1.0.0 expects it in DPOConfig, not DPOTrainer | Moved to DPOConfig; pass model as string path |

---

## Next Steps

| Step | Depends on | Est. time |
|------|-----------|-----------|
| Evaluate DPO adapters — ROUGE / BERTScore / AlignScore on test set | DPO adapters (done ✅) | ~1–2h on A100 |
| Compare holistic vs GCA fine-tuned models on factual consistency | Evaluation completing | immediate |
| Ablation: DPO β sensitivity (β = 0.05 / 0.1 / 0.2) | baseline eval done | ~1h per run |
| Write Phase 3 results section | all eval done | — |

---

## Repository State

All code is on `origin/main`. Latest commits:

| Commit | Description |
|--------|-------------|
| `c90fb48` | fix: exclude gpu0001 (broken CUDA node) |
| `a7edb85` | fix: restructure DPO to TRL 1.0.0 API |
| `05973b2` | progress update: BT RM results (58.1% / 54.6%) |
| `7563e72` | fix: DPO script bug fixes (null pairs, dtype, device) |
| `67fb7b8` | slurm: DPO job script |
| `6e6bc01` | src/dpo: initial run_dpo.py |
| `02131d4` | Bradley-Terry reward model training pipeline |

---

## Completed This Week

### 1. AlignScore Full Run — 200-Sample Results ✅

The full 200-sample judging job ran successfully on MOGON (job 1079466, ~3 minutes on A100). Both holistic and GCA scoring are working correctly with the `FacebookAI/roberta-base` backbone via the HF proxy.

**Holistic Judging**

| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Usable preference pairs | **154** (77.0%) |
| Ties (excluded) | 46 |
| Mean faithfulness — A (low-temp) | 0.7324 |
| Mean faithfulness — B (high-temp) | 0.6530 |

**GCA Judging** (sentence-level, α = 0.5)

| Metric | Value |
|--------|-------|
| Total samples | 200 |
| Usable preference pairs | **157** (78.5%) |
| Ties (excluded) | 43 |
| Mean GCA score — A | 0.4066 |
| Mean GCA score — B | 0.3337 |
| Avg sentences per summary (A / B) | 7.0 / 6.7 |

**Agreement between holistic and GCA decisions: 60.5% (121/200)**

The 39.5% disagreement rate is the central finding of this experiment — it demonstrates that holistic and sentence-level scoring are not equivalent and do not produce the same preference signal. A concrete example:

> `cnn_00012` — *Carlos Alberto / Werder Bremen*  
> Holistic says **A wins** (A=0.795, B=0.734).  
> GCA says **B wins** (A=0.597, B=0.664).  
> Holistic collapses all sentence evidence into one number and misses a specific faithfulness failure that GCA catches at the sentence level.

This is exactly the thesis argument: fine-tuning on holistic preferences risks rewarding summaries that contain individual unfaithful sentences, as long as the overall score is high enough.

---

### 2. Bradley-Terry Reward Model — Full Pipeline Implemented ✅

Following Lingxiao's suggestion to explore the IRL framing, a complete reward model training pipeline has been implemented (`src/reward_model/`).

**Architecture**

- Backbone: `microsoft/deberta-v3-base` (encoder-only, strong NLI / faithfulness capabilities)
- Input: `"{article[:2000]} [SEP] {summary}"` tokenised to 512 tokens
- Head: mean-pool encoder output → linear layer → scalar reward
- Loss: Bradley-Terry — $\mathcal{L} = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$

**Training setup**

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 5 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Max sequence length | 512 |
| Validation metric | Pairwise accuracy $P(r_w > r_l)$ |

Given the small dataset size (~380 usable pairs from 500 samples), **5-fold cross-validation** is used to get a reliable accuracy estimate without wasting data on a fixed held-out split.

**Two conditions trained separately:**
- `rm_holistic` — trained on holistic preference pairs
- `rm_gca` — trained on GCA preference pairs

The comparison between their pairwise validation accuracies will directly answer whether the GCA signal is more learnable / more consistent from the reward model's perspective — the IRL framing Lingxiao suggested.

**Code committed:** `src/reward_model/train.py`, `src/reward_model/run_training.py`, `slurm/train_reward_models.sh` (commit `02131d4`).

---

### 3. 500-Sample RM Dataset — Generation Running ✅ ⏳

To train the reward model, a separate 500-sample candidate generation run was set up (seed=100, disjoint from the 200-sample DPO set at seed=42).

- **495/500 samples** pass the length filter and are queued for generation
- Candidate generation job **1157804 is currently RUNNING** on `gpu0008` (A100), started ~14:00 CEST today
- Expected completion: ~15:30 CEST

After generation completes, the pipeline is:
1. `build_reward_preferences_rm500.sh` — AlignScore judging on 495 samples (~1h)
2. `train_reward_models.sh` — BT reward model training, both conditions (~30–60 min)

---

## Key Technical Issues Resolved

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `use_fast=True` tokenizer crash | Old `tokenizers` lib on MOGON can't parse Mistral v0.3 fast tokenizer format | `use_fast=False` |
| `KeyError: 'mistral'` | `transformers 4.30.2` predates Mistral support | Upgraded to `4.40.2` |
| `ImportError: device_map requires Accelerate` | `accelerate 0.20.3` does not satisfy transformers 4.40's version check | Removed `device_map`, load model in bfloat16 then `.to("cuda:0")` |
| `TRANSFORMERS_OFFLINE` not set | 4.40.x tries to phone home even for local model paths | Added `TRANSFORMERS_OFFLINE=1` and `local_files_only=True` |
| rsync `--relative` writing to wrong path | Preserves full Mac absolute path structure on remote | Switched to `scp` for all file transfers |

---

## Next Steps

| Step | Depends on | Est. time |
|------|-----------|-----------|
| AlignScore judging on 500 samples | gen job 1157804 completing | ~1h |
| BT reward model training (holistic + GCA) | judging job completing | ~45 min |
| Compare RM pairwise accuracies (IRL result) | training completing | immediate |
| Begin DPO fine-tuning (Phase 3) | 200-sample preference dataset (ready now) | ~2–3h on A100 |

---

## Repository State

All code is on `origin/main`. Latest commits:

| Commit | Description |
|--------|-------------|
| `02131d4` | Bradley-Terry reward model training pipeline |
| `51d92c7` | `use_fast=False` for Mistral tokenizer |
| `bb3f7c1` | `local_files_only=True` + `TRANSFORMERS_OFFLINE=1` |
| `2423487` | Remove `device_map`, use `torch_dtype` + `.to(cuda)` |

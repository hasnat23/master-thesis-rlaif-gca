# Progress Update — 19 May 2026

**Meeting date:** 19 May 2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## Thesis Context

**Title:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)  
**Core question:** Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic feedback under DPO? And can we distil that preference signal into a trained reward model for the IRL framing?

---

## Summary

The AlignScore judging pipeline is fully validated and has produced the first real preference dataset (200 samples). Based on Lingxiao's suggestion to explore the IRL direction, a **Bradley-Terry reward model training pipeline** has been designed and implemented this week. A 500-sample candidate generation job is currently running on MOGON to produce the training data for that reward model.

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

**What is the Bradley-Terry model?**

Consider ranking the quality of summaries generated for a news article. A naive approach is to score each summary directly (e.g., with AlignScore) and sort by absolute value. But absolute scores vary across articles — a score of 0.75 on a dense wire report is not the same as 0.75 on a short blog post. Two summaries may be indistinguishable by absolute score yet one is clearly better when placed side-by-side.

A model-based alternative is to learn from *pairwise comparisons* instead. Let $\beta_i \in \mathbb{R}$ represent the latent "quality strength" of summary $i$. The Bradley-Terry model treats each comparison as an independent Bernoulli outcome where summary $i$ is preferred over summary $j$ with probability $p_{ij}$, and the log-odds of that preference equals the difference of the two strengths:

$$\log \frac{p_{ij}}{1 - p_{ij}} = \beta_i - \beta_j$$

Solving for $p_{ij}$ directly:

$$p_{ij} = \frac{e^{\beta_i}}{e^{\beta_i} + e^{\beta_j}} = \sigma(\beta_i - \beta_j)$$

This is structurally identical to logistic regression on the score difference: only the *relative* strength matters, not the absolute values. (The model is invariant to a global constant shift — adding a constant $c$ to every $\beta_i$ leaves all $p_{ij}$ unchanged, a fact that surfaces as an identifiability issue for NBA power ratings and is resolved in the same way, by fixing one reference point.)

In our reward model $\beta_i = r_\theta(\text{article},\, \text{summary}_i)$ is the scalar output of a neural network with parameters $\theta$. Training maximises the log-likelihood of the pairwise preferences produced by AlignScore — equivalently, minimising the **Bradley-Terry loss**:

$$\mathcal{L}_{\text{BT}} = -\log \sigma\!\bigl(r_\theta(x,\, y_w) - r_\theta(x,\, y_l)\bigr)$$

where $y_w$ (chosen) is the summary AlignScore preferred and $y_l$ (rejected) is the one it ranked lower for article $x$. This is the same loss used to train reward models in InstructGPT and Llama-2-chat, grounding the IRL framing in established RLHF practice.

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

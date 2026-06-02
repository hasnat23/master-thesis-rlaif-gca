# Meeting Notes — 2 June 2026

**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## 1. Recap of Previous Meeting (19 May 2026)

Last time we agreed to:
1. Train Bradley-Terry reward models on a dedicated 500-sample set
2. Run DPO fine-tuning on Mistral-7B for both holistic and GCA conditions
3. Evaluate the fine-tuned models against baseline on held-out articles

All three are done. ✅

---

## 2. Introduction

### Motivation

Summarization models often produce fluent but factually unreliable outputs. Standard RLHF uses holistic preference labels ("which summary is better overall?"), which collapse all evidence into a single score and can miss sentence-level errors. This thesis asks: does applying feedback at the sentence level via **Granular Credit Assignment (GCA)** produce more factually consistent summaries than holistic feedback under DPO?

The core comparison is **holistic AlignScore-based preferences** (full-summary factuality scoring) versus **GCA-based preferences** (sentence-level factuality scoring aggregated into a summary-level score). The judge in both cases is a fixed factuality/reward model — currently AlignScore. No open-ended LLM judge or GPT-4o is used as the main preference source.

### Architecture

```mermaid
flowchart TD
    A["1. Data Loader
    Input: CNN/DailyMail articles
    Output: article samples"] --> B["2. Candidate Generation
    Input: article + base policy model
    Output: candidate summaries A and B"]

    B --> C["3. Candidate Pairing
    Input: candidate summaries
    Output: paired summaries (A, B)"]

    C --> H1["4A. Holistic Full-Summary Scoring
    Input: article + full Summary A/B
    Scorer: AlignScore (fixed factuality model)
    Output: score_A_full, score_B_full"]

    H1 --> H2["5A. Holistic Preference Construction
    Input: score_A_full, score_B_full + margin threshold
    Output: P_hol — chosen/rejected pairs or no_preference"]

    C --> G1["4B. Sentence Segmentation
    Input: Summary A and Summary B
    Output: sentence lists SA and SB"]

    G1 --> G2["5B. Sentence-Level Factuality Scoring
    Input: article + each summary sentence
    Scorer: AlignScore (fixed factuality model)
    Output: local sentence-level scores"]

    G2 --> G3["6B. GCA Aggregation
    Input: local sentence scores
    Output: GCA_A, GCA_B — summary-level scores"]

    G3 --> G4["7B. GCA Preference Construction
    Input: GCA_A, GCA_B + margin threshold
    Output: P_gca — chosen/rejected pairs or no_preference"]

    H2 --> R1["8A. Bradley-Terry RM Training
    Input: P_hol
    Output: RM-Holistic"]

    G4 --> R2["8B. Bradley-Terry RM Training
    Input: P_gca
    Output: RM-GCA"]

    H2 --> D1["9A. DPO Fine-Tuning
    Input: P_hol + base policy model
    Output: DPO-Holistic adapter"]

    G4 --> D2["9B. DPO Fine-Tuning
    Input: P_gca + base policy model
    Output: DPO-GCA adapter"]

    D1 --> E["10. Final Evaluation
    Input: baseline, DPO-Holistic, DPO-GCA outputs
    Metrics: AlignScore, ROUGE, BERTScore
    Output: comparative factuality and quality results"]

    D2 --> E
    R1 --> E
    R2 --> E
```

The architecture has two parallel preference-construction branches from the same candidate pool, and two parallel training paths from each branch. If the score difference between two summaries is below the margin threshold, the pair is marked as `no_preference` and excluded from training.

### Key method points

- **GCA:** each sentence is scored against the article with AlignScore, and the sentence-level scores are aggregated into a summary-level preference score. The current aggregation is treated as the first operational version of GCA; the disagreement analysis will check whether low-faithfulness sentences are penalized strongly enough.
- **DPO:** trained with LoRA (r=16, α=32) on Mistral-7B-Instruct-v0.3, β=0.1.
- **Bradley-Terry RM:** RoBERTa-base with a mean-pool scalar head, pairwise ranking loss, 5-fold CV.

### Why sentence-level scoring is still useful

Full-summary AlignScore scoring is retained as the holistic baseline. Sentence-level scoring is still useful because factual errors in summaries are often local — a full-summary score can show which summary is better overall, but it does not identify which sentence caused the factuality problem. Sentence-level scoring provides that local evidence, and GCA turns it into a summary-level preference. This makes the preference construction more interpretable, even if the current results show that the first GCA aggregation rule is noisier than holistic scoring.

---

## 3. Progress Since Last Meeting

### What was completed

| Step | Result |
|------|--------|
| 500-sample RM candidate generation | 495 articles, disjoint from DPO set |
| AlignScore preference building (RM set) | 494 holistic pairs, 495 GCA pairs |
| Bradley-Terry RM training | Holistic: **58.1%**, GCA: **54.6%** pairwise accuracy (5-fold CV) |
| DPO fine-tuning | Holistic loss 0.6902, GCA loss 0.6913 |
| Evaluation on 200 held-out articles | See results below |

The RM accuracies are both above chance (50%), but holistic is more consistent. GCA drops near chance in two folds, suggesting that the sentence-level signal is noisier under the current aggregation setup. This is an important emerging finding and will be checked through disagreement-case analysis.

### Evaluation results (n=200, bootstrap 95% CI)

| Condition | ROUGE-1 | ROUGE-L | AlignScore |
|-----------|---------|---------|------------|
| Baseline | 0.3281 | 0.2132 | 0.7965 |
| DPO-Holistic | 0.3255 | 0.2111 | **0.8017 (+0.5%)** |
| DPO-GCA | 0.3255 | 0.2108 * | 0.7996 (+0.3%) |

\* DPO-GCA ROUGE-L is significantly different from baseline (Wilcoxon p=0.015). All other differences are non-significant.

**Key takeaways:**
- DPO-Holistic currently gives the best factual consistency (AlignScore +0.5pp) — the metric most directly tied to the thesis goal.
- DPO-GCA shows a statistically significant ROUGE-L change compared with baseline, suggesting that the GCA preference signal changes generation behavior. However, this should not be interpreted as a factuality improvement, since the AlignScore gain remains smaller than DPO-Holistic.
- The 39.5% holistic/GCA disagreement rate confirms the two signals are genuinely different and not interchangeable.

### Challenges

- AlignScore and BERTScore needed patches to work offline on the HPC cluster (AdamW removed in transformers v5, HF proxy unreachable from compute nodes).
- One GPU node had a broken CUDA runtime — excluded permanently via Slurm.

---

## 4. Next Steps

**Immediate:**
1. Thesis write-up — results chapter is ready to write
2. Qualitative error analysis on the 39.5% disagreement cases (entity / relation / temporal errors)
3. Finalize related work section

**If time permits:**
- DPO β ablation
- Scale DPO training to the full 495-sample set

**Open questions for today:**
- Is +0.5pp AlignScore sufficient, or should we add a small human evaluation?
- How to frame the ROUGE-L significance result — positive finding or distribution shift?
- Is the experimental scope sufficient for submission?

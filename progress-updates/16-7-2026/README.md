# Final Experimental Results Summary — 16 July 2026

**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## 1. Purpose of This Final Update

The main experimental campaign is complete. This final update summarizes the comparison between two preference-construction strategies for reward-model supervision:

- Holistic full-summary scoring
- Sentence-level GCA scoring with aggregation

I kept the scorer, RM backbone, and training setup fixed across the campaign, so the main variable is the preference-construction strategy.

---

## 2. Experimental Design

| Item | Setting |
|---|---|
| Judge | `yzha/AlignScore` |
| Judge mode | `nli` |
| RM backbone | `FacebookAI/roberta-base` |
| RM training | `epochs=5`, `lr=2e-5`, `batch=8`, `kfold=5` |
| Nested subset seed | `200` |
| Dataset sizes tested | `1000`, `5000`, `10000` |

Definitions:

- Holistic = full-summary AlignScore-based preference construction
- GCA = sentence-level AlignScore scoring followed by aggregation into a summary-level preference

---

## 3. Main Results Across Scales

| Dataset size | Holistic mean | GCA mean | Gap (GCA - Holistic) | Interpretation |
|---|---:|---:|---:|---|
| 1000 | 0.5295 | 0.5603 | +0.0308 | GCA clearly better in validated campaign |
| 5000 | 0.5788 | 0.5746 | -0.0042 | Holistic slightly better |
| 10000 | 0.5827 | 0.5862 | +0.0035 | GCA slightly better |

The 1000-sample setup gives the strongest evidence for GCA. The larger reruns show smaller effects, and the direction changes with scale. That pattern indicates that the gain is limited and not uniformly stable at larger sample sizes.

---

## 4. Validated 1000-Sample Campaign

| Run | Seed | Holistic | GCA | Gap |
|---|---:|---:|---:|---:|
| Original sweep | 42 | 0.523 | 0.583 | +0.060 |
| Confirmation | 42 | 0.543 | 0.556 | +0.013 |
| Seed validation 1 | 7 | 0.556 | 0.546 | -0.010 |
| Seed validation 2 | 100 | 0.510 | 0.561 | +0.051 |
| Seed validation 3 | 314 | 0.520 | 0.552 | +0.032 |
| Seed validation 4 | 2026 | 0.525 | 0.564 | +0.039 |

| Setting | Holistic mean | GCA mean | Gap | 95% CI | Wilcoxon p-value |
|---|---:|---:|---:|---:|---:|
| 6 runs / 30 folds | 0.5295 | 0.5603 | +0.0308 | [+0.013, +0.047] | 0.0034 |

- GCA wins on 22 of 30 folds.
- This is the strongest and most statistically supported result in the thesis.

---

## 5. Robustness Checks at Larger Scale

### 5.1 5000-Item Rerun

| Holistic mean | GCA mean | Gap |
|---|---:|---:|
| 0.5788 | 0.5746 | -0.0042 |

- The 5000-item rerun does not confirm the 1000-sample GCA advantage.
- The gap is very small.
- The result suggests that the GCA effect depends on the sample and does not scale uniformly.

### 5.2 10000-Item Rerun

| Holistic mean | GCA mean | Gap |
|---|---:|---:|
| 0.5827 | 0.5862 | +0.0035 |

- The 10000-item rerun moves slightly back in favor of GCA.
- The margin is positive but small.
- The effect does not grow monotonically with dataset size.

---

## 6. Final Interpretation

The experiments support a nuanced conclusion. GCA is not universally superior to holistic scoring. Under the controlled 1000-sample validation campaign, GCA produced a statistically significant and reproducible improvement in reward-model learnability. At larger scale, the effect became small and dataset-dependent. Sentence-level credit assignment can help, but its effectiveness depends on aggregation design and dataset composition.

---

## 7. Novelty and Positioning

This thesis does not claim to introduce a new factuality scorer. AlignScore serves as a fixed factuality judge. The contribution lies in the controlled comparison of two ways of converting the same factuality signal into reward-model supervision: holistic full-summary preference construction versus sentence-level GCA-based preference construction.

The RM tests whether the constructed preferences are learnable. The main research object is the preference-construction strategy, not the scorer itself.

---

## 8. Decision: Closing Main Experiments

Based on the completed 1000-sample validation campaign and the 5000/10000-item robustness checks, I consider the main experimentation phase complete. The remaining work should focus on thesis writing, theoretical grounding of GCA, related work, qualitative/error analysis, limitations, and final discussion.

---

## 9. Recorded Outputs

Completed jobs on MOGON:

- `1411306` — generate the 5000-item candidate set, completed
- `1411307` — build holistic/GCA preferences for the 5000-item set, completed
- `1411308` — train the 5000-item Bradley-Terry reward models, completed
- `1411412` — generate the 10000-item candidate set, completed
- `1411413` — build holistic/GCA preferences for the 10000-item set, completed
- `1411415` — train the 10000-item Bradley-Terry reward models, completed

Recorded outputs:

- Candidate sets: `~/thesis/data/candidates/candidates_5000.jsonl`, `~/thesis/data/candidates/candidates_10000.jsonl`
- Preference files: `~/thesis/data/preferences_5000/`, `~/thesis/data/preferences_10000/`
- RM summaries: `~/thesis/outputs/reward_models_5000/rm_training_summary.json`, `~/thesis/outputs/reward_models_10000/rm_training_summary.json`

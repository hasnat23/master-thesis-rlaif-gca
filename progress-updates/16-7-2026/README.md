# Final Experimental Results Summary — 16 July 2026

**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner  
**Mentor:** Lingxiao Kong  

---

## 1. Purpose of This Final Update

This update summarizes the final experimental campaign for comparing two preference-construction strategies for reward-model supervision in factual summarization:

- **Holistic scoring:** full-summary factuality scoring
- **GCA scoring:** sentence-level factuality scoring followed by aggregation

The factuality judge, reward-model backbone, and training setup were kept fixed across the experiments. Therefore, the main experimental variable is the way factuality signals are converted into pairwise preference labels.

The goal of this update is to close the main experimentation phase and use the current results as the final empirical basis for thesis writing, theoretical framing, and discussion.

---

## 2. Experimental Design

| Item | Setting |
|---|---|
| Judge | `yzha/AlignScore` |
| Judge mode | `nli` |
| Reward-model backbone | `FacebookAI/roberta-base` |
| RM training | `epochs=5`, `lr=2e-5`, `batch=8`, `kfold=5` |
| Nested subset seed | `200` |
| Dataset sizes tested | `1000`, `5000`, `10000` |

### Preference-construction strategies

**Holistic:**  
Each candidate summary is scored as a complete output against the source article. The resulting full-summary scores are used to construct pairwise preferences.

**GCA:**  
Each candidate summary is decomposed into sentence-level units. Each sentence is scored against the source article, and the local scores are aggregated into a summary-level preference signal.

---

## 3. Main Results Across Scales

| Dataset size | Holistic mean | GCA mean | Gap (GCA - Holistic) | Interpretation |
|---:|---:|---:|---:|---|
| 1000 | 0.5295 | 0.5603 | +0.0308 | GCA clearly better in the validated campaign |
| 5000 | 0.5788 | 0.5746 | -0.0042 | Holistic slightly better |
| 10000 | 0.5827 | 0.5862 | +0.0035 | GCA slightly better |

The 1000-sample validation campaign provides the strongest evidence in favor of GCA. At larger scale, the effect becomes much smaller and changes direction between the 5000-item and 10000-item reruns.

This suggests that GCA can improve reward-model learnability under controlled conditions, but the advantage is not uniformly stable across larger dataset sizes.

---

## 4. Validated 1000-Sample Campaign

The 1000-sample campaign was evaluated across six independent runs and 30 cross-validation folds.

| Run | Seed | Holistic | GCA | Gap (GCA - Holistic) |
|---|---:|---:|---:|---:|
| Original sweep | 42 | 0.523 | 0.583 | +0.060 |
| Confirmation | 42 | 0.543 | 0.556 | +0.013 |
| Seed validation 1 | 7 | 0.556 | 0.546 | -0.010 |
| Seed validation 2 | 100 | 0.510 | 0.561 | +0.051 |
| Seed validation 3 | 314 | 0.520 | 0.552 | +0.032 |
| Seed validation 4 | 2026 | 0.525 | 0.564 | +0.039 |

### Pooled result

| Setting | Holistic mean | GCA mean | Gap | 95% CI | Wilcoxon p-value |
|---|---:|---:|---:|---:|---:|
| 6 runs / 30 folds | 0.5295 | 0.5603 | +0.0308 | [+0.013, +0.047] | 0.0034 |

Additional observations:

- GCA wins on 22 of 30 folds.
- The average improvement is +3.08 percentage points.
- The confidence interval is fully positive.
- The Wilcoxon test indicates a statistically significant difference.

This is the strongest and most statistically supported result of the experimental campaign.

---

## 5. Robustness Checks at Larger Scale

To test whether the 1000-sample result generalizes to larger preference datasets, additional reruns were performed with 5000 and 10000 candidate items.

### 5.1 5000-Item Rerun

| Dataset size | Holistic mean | GCA mean | Gap (GCA - Holistic) |
|---:|---:|---:|---:|
| 5000 | 0.5788 | 0.5746 | -0.0042 |

The 5000-item rerun does not confirm the GCA advantage observed in the 1000-sample validation campaign. Holistic scoring performs slightly better, but the gap is small.

This result suggests that the GCA effect is sample-dependent and does not scale uniformly.

### 5.2 10000-Item Rerun

| Dataset size | Holistic mean | GCA mean | Gap (GCA - Holistic) |
|---:|---:|---:|---:|
| 10000 | 0.5827 | 0.5862 | +0.0035 |

The 10000-item rerun moves slightly back in favor of GCA. However, the margin is small, and the improvement is much weaker than in the validated 1000-sample campaign.

This indicates that increasing dataset size does not lead to a monotonic increase in the GCA advantage.

---

## 6. Final Interpretation

The experiments support a nuanced conclusion.

GCA is not universally superior to holistic scoring. However, in the controlled 1000-sample validation campaign, GCA produced a statistically significant and reproducible improvement in reward-model learnability.

At larger scale, the effect became smaller and dataset-dependent:

- The 5000-item rerun slightly favored Holistic.
- The 10000-item rerun slightly favored GCA.
- The larger-scale differences were much smaller than the 1000-sample validation result.

The final interpretation is therefore that sentence-level credit assignment can provide a useful preference signal, but its effectiveness depends on aggregation design, sample composition, and scale.

This is an important finding for the thesis: granular feedback is promising, but it does not automatically outperform holistic feedback in all settings.

---

## 7. Novelty and Positioning

This thesis does not claim to introduce a new factuality scorer.

AlignScore is used as a fixed factuality judge. The contribution lies in the controlled comparison of two ways of converting the same factuality signal into reward-model supervision:

1. **Holistic preference construction** from full-summary scores
2. **GCA-based preference construction** from sentence-level scores and aggregation

The reward model is used to test whether the resulting preference labels are learnable.

Therefore, the main research object is not AlignScore itself, but the preference-construction strategy. The thesis studies whether factuality supervision becomes more useful when it is applied at summary level or decomposed into sentence-level evidence before constructing pairwise preferences.

---

## 8. Theoretical Direction for Thesis Writing

The theoretical framing of GCA will be strengthened in the thesis through the following directions:

- **Fine-grained factuality evaluation:** factual errors often occur locally, at sentence or claim level.
- **Credit assignment:** local factuality evidence can help identify which parts of a summary contribute to the final preference.
- **Preference learning:** pairwise reward-model supervision depends strongly on how preference labels are constructed.
- **Aggregation design:** sentence-level scores must be aggregated carefully, because simple aggregation may dilute important factual errors or introduce noise.

This positions GCA as a method for studying feedback granularity in factual summarization, rather than as a replacement for existing factuality metrics.

---

## 9. Decision: Closing Main Experiments

Based on the completed 1000-sample validation campaign and the 5000/10000-item robustness checks, the main experimentation phase is considered complete.

The remaining work should focus on:

- thesis writing
- theoretical grounding of GCA
- related work
- qualitative/error analysis
- limitations
- final discussion

No further large-scale experiments are planned unless a specific additional ablation is required.

---

## 10. Recorded Outputs

### Completed MOGON jobs

| Job ID | Purpose | Status |
|---:|---|---|
| `1411306` | Generate 5000-item candidate set | Completed |
| `1411307` | Build holistic/GCA preferences for 5000-item set | Completed |
| `1411308` | Train 5000-item Bradley-Terry reward models | Completed |
| `1411412` | Generate 10000-item candidate set | Completed |
| `1411413` | Build holistic/GCA preferences for 10000-item set | Completed |
| `1411415` | Train 10000-item Bradley-Terry reward models | Completed |

### Candidate sets

```text
~/thesis/data/candidates/candidates_5000.jsonl
~/thesis/data/candidates/candidates_10000.jsonl
```

### Preference files

```text
~/thesis/data/preferences_5000/
~/thesis/data/preferences_10000/
```

### Reward-model summaries

```text
~/thesis/outputs/reward_models_5000/rm_training_summary.json
~/thesis/outputs/reward_models_10000/rm_training_summary.json
```

---

## 11. Final Takeaway

The final experimental evidence supports a careful thesis claim:

GCA-based preference construction produced a statistically significant improvement in the controlled 1000-sample validation campaign, but the advantage became small and dataset-dependent at larger scale.

This means the thesis should not claim that GCA is universally better than holistic feedback. Instead, it should argue that sentence-level credit assignment is a promising but sensitive approach for constructing factuality-based preference data for reward-model training.
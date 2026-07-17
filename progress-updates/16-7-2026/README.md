# Final Results Summary — 16 July 2026

**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## 1. Final Outcome

The thesis experiments are complete. The evaluation compared Holistic vs GCA reward-model supervision under the same judge, the same RM backbone, and the same training setup, while varying only the preference-construction strategy and the dataset scale.

The results are mixed across scales:
- On the 1000-sample setup, GCA is consistently better across six independent runs.
- On the 5000-item rerun, Holistic is slightly better.
- On the 10000-item rerun, GCA is slightly better again.

The final conclusion is that the GCA advantage is real in the validated 1000-sample campaign, but it is small and sensitive to dataset composition at larger scale.

---

## 2. Experimental Setup

The same experimental settings were kept fixed across the campaign:

- Judge: `yzha/AlignScore`
- Judge mode: `nli`
- RM backbone: `FacebookAI/roberta-base`
- RM training: `epochs=5`, `lr=2e-5`, `batch=8`, `kfold=5`
- Nested subset seed: `200`
- Dataset sizes tested: `1000`, `5000`, `10000`

The comparison is between two preference-construction strategies:

- Holistic: full-summary AlignScore scoring
- GCA: sentence-level AlignScore scoring with aggregation

---

## 3. 1000-Sample Validation Campaign

The 1000-sample setup was tested across six independent seed runs and produced the clearest positive result for GCA.

### Per-run results

| Run | Seed | Holistic | GCA | Gap (GCA - Holistic) |
|-----|------|----------|-----|----------------------|
| Original sweep | 42 | 0.523 | 0.583 | +0.060 |
| Confirmation | 42 | 0.543 | 0.556 | +0.013 |
| Seed validation 1 | 7 | 0.556 | 0.546 | -0.010 |
| Seed validation 2 | 100 | 0.510 | 0.561 | +0.051 |
| Seed validation 3 | 314 | 0.520 | 0.552 | +0.032 |
| Seed validation 4 | 2026 | 0.525 | 0.564 | +0.039 |

### Pooled result

| Aggregate | Holistic mean | GCA mean | Gap | 95% CI | Wilcoxon p (two-sided) |
|-----------|---------------|----------|-----|--------|------------------------|
| Pooled (6 runs, 30 folds) | 0.5295 | 0.5603 | +0.0308 | [+0.013, +0.047] | 0.0034 |

Additional summary:
- GCA wins on 22 of 30 folds.
- The pooled result is statistically significant and reproducible across seeds.

---

## 4. 5000-Item Rerun

The 5000-item rerun did not preserve the 1000-sample GCA advantage.

### Result

| Holistic mean | GCA mean | Gap (GCA - Holistic) |
|---------------|----------|----------------------|
| 0.5788 | 0.5746 | -0.0042 |

### Interpretation

- The 5k gap is very small, so the earlier advantage was not strong enough to remain stable at this larger scale.
- This suggests the effect is sensitive to the exact dataset composition and is not yet a clearly robust gain at 5000 items.

---

## 5. 10000-Item Rerun

The 10000-item rerun moved back slightly in favor of GCA.

### Result

| Holistic mean | GCA mean | Gap (GCA - Holistic) |
|---------------|----------|----------------------|
| 0.5827 | 0.5862 | +0.0035 |

### Interpretation

- The 10k result is again positive for GCA, but the margin is very small.
- The overall pattern shows that the effect is not monotonic with dataset size.
- The most defensible conclusion is that GCA can help, but the gain is narrow and depends on the sample drawn.

---

## 6. Final Conclusion

The campaign supports the following conclusion:

1. GCA is the better method on the validated 1000-sample campaign.
2. At larger scale, the advantage is not consistently large.
3. The 5000- and 10000-item reruns show that the effect is small and sensitive to dataset composition.

In short, the thesis demonstrates a reproducible GCA benefit on the controlled 1000-sample setup, but not a strong or uniform large-scale advantage.

---

## 7. Recorded Outputs

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

# Meeting Notes — 30 June 2026

**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## 1. Scope of This Update

This update carries forward the full reward-model optimization progress up to 23 June 2026 and includes the current continuation steps launched for additional `mode_nli` seed validation runs.

The main objective remains unchanged:
- Compare RM-Holistic vs RM-GCA fairly on the 1000-sample setup.
- Continue optimization until a reproducible GCA advantage is observed.

---

## 2. Pipeline State (Current)

### Final simplified pipeline used

1. Generate 1000 candidate summary pairs.
2. Build two preference sets per run:
   - Holistic (full-summary AlignScore)
   - GCA (sentence-level AlignScore + aggregation)
3. Train Bradley-Terry reward models with identical training settings.
4. Compare 5-fold validation mean accuracy:
   - RM-Holistic vs RM-GCA

Common fixed settings in current campaign:
- Judge backend: `yzha/AlignScore`
- Backbone: `FacebookAI/roberta-base`
- Margin: `0`
- Max samples: `1000`
- RM training: `epochs=5`, `lr=2e-5`, `batch=8`, `kfold=5`, `seed` varied per validation run

---

## 3. Completed Results Summary (Through 23 June)

### Baseline 1000-sample run (`margin=0`, default mode)

| Condition | Mean Val Accuracy |
|-----------|-------------------|
| Holistic RM | 0.572 |
| GCA RM | 0.560 |
| Gap (GCA - Holistic) | -0.012 |

Observation:
- GCA was competitive but below holistic.

### Alpha ablation (`alpha=0.0`)

| Condition | Mean Val Accuracy |
|-----------|-------------------|
| Holistic RM | 0.586 |
| GCA RM | 0.561 |
| Gap (GCA - Holistic) | -0.025 |

Observation:
- Removing aggregation penalty did not solve the gap.

### Hyperparameter search on alpha=0.0 preferences

Best one-off hpsearch config:
- `lr1e-5_ep7`: Holistic `0.572`, GCA `0.586` (gap `+0.014`)

Confirmation rerun:
- Holistic `0.577`, GCA `0.560` (gap `-0.017`)

Observation:
- Hyperparameter-only uplift was not stable.

---

## 4. AlignScore Mode Sweep (Completed)

### Mode sweep settings
- Modes tested: `nli_sp`, `nli`, `bin_sp`, `bin`
- Shared settings: `alpha=0.0`, full preference rebuild, full RM retraining

### Final mode-sweep results

| Mode | Holistic mean acc | GCA mean acc | Gap (GCA - Holistic) |
|------|-------------------|--------------|-----------------------|
| `nli` | 0.523 | 0.583 | +0.060 |
| `bin` | 0.510 | 0.564 | +0.054 |
| `bin_sp` | 0.554 | 0.562 | +0.008 |
| `nli_sp` | 0.578 | 0.557 | -0.021 |

Observation:
- `nli` was the strongest mode for GCA among the four tested modes.

### `mode_nli` confirmation rerun (completed)

| Run | Holistic | GCA | Gap (GCA - Holistic) |
|-----|----------|-----|----------------------|
| Original `mode_nli` sweep | 0.523 | 0.583 | +0.060 |
| Confirmation rerun (`1336456_1`) | 0.543 | 0.556 | +0.013 |
| Mean (2 runs) | 0.533 | 0.570 | +0.037 |

Observation:
- Direction remained GCA > Holistic in both runs.
- Magnitude varied noticeably across runs.

### Statistical summary from pooled two-run analysis

| Analysis | GCA - Holistic Mean | 95% CI |
|----------|----------------------|--------|
| Pooled (10 folds total) | +0.034 | [+0.003, +0.064] |

Additional test:
- Wilcoxon signed-rank p-value: `0.084` (borderline)

---

## 5. Continuation Work (Completed)

To continue the planned robustness check, additional independent seed validations for `mode_nli` were launched and completed.

### New remote script added on MOGON workspace
- Script path (remote): `~/thesis/slurm/mode_nli_seed_confirm.sh`
- Purpose: rerun `mode_nli` end-to-end with configurable `SEED` and isolated output folders.
- Output pattern:
  - Preferences: `data/preferences_1000_alpha0_mode_nli_seed<SEED>/`
  - RM summaries: `outputs/mode_sweep_alpha0/mode_nli_seed<SEED>/rm_training_summary.json`

### Seed validation jobs (final status)

| Job ID | Name | Seed | Status |
|--------|------|------|--------|
| 1337534 | mode_nli_seed | 7 | COMPLETED (ExitCode 0, 00:30:54) |
| 1337535 | mode_nli_seed | 100 | COMPLETED (ExitCode 0, 00:30:54) |
| 1337590 | mode_nli_seed | 314 | COMPLETED (ExitCode 0, 00:30:52) |

Submission notes:
- Both jobs were submitted with `--exclude=gpu0001` to avoid the prior CUDA-availability issue.
- Both completed on `gpu0002`.

### New results from additional seeds

| Run | Holistic mean acc | GCA mean acc | Gap (GCA - Holistic) |
|-----|-------------------|--------------|----------------------|
| `mode_nli_seed7` | 0.556 | 0.546 | -0.010 |
| `mode_nli_seed100` | 0.510 | 0.561 | +0.051 |
| `mode_nli_seed314` | 0.520 | 0.552 | +0.032 |

Observation:
- Seed-specific variance remains visible: one seed favors Holistic (`seed7`), one favors GCA (`seed100`).
- This confirms that single-run conclusions are unstable without pooled analysis.

### Five-run combined summary (`mode_nli` family)

Combined runs:
1. Original `mode_nli` sweep
2. `mode_nli` confirmation rerun
3. `mode_nli_seed7`
4. `mode_nli_seed100`
5. `mode_nli_seed314`

| Aggregate | Holistic mean | GCA mean | Gap (GCA - Holistic) |
|-----------|----------------|----------|-----------------------|
| Pooled over 25 folds | 0.530 | 0.560 | +0.029 |

Statistical summary (25 folds):
- Bootstrap 95% CI for gap: `[+0.009, +0.048]`
- Wilcoxon signed-rank: `p=0.0123`

Interpretation:
- After adding two extra seed runs, the pooled result remains in favor of GCA for `mode_nli`.
- Effect size is smaller than early two-run estimate but remains positive and statistically supported in pooled analysis.

---

## 6. Immediate Next Steps

1. Keep `mode_nli` as the current best backend direction for GCA-focused comparisons.
2. Prepare the thesis reporting table with both views:
   - Per-run results (to show variance)
   - Pooled five-run result (to show central tendency and significance)
3. Freeze experiment configuration for write-up (same candidate set, same margin, same training hyperparameters) to avoid moving-target comparisons.
4. If needed for final confidence, run one last independent seed and check whether pooled gap remains in the +2 to +3 pp range.

---

## 8. Short Summary (Reusable)

`mode_nli` remains the strongest backend mode for GCA in the current campaign.

Across five completed runs (25 folds total), pooled results are:
- Holistic mean: `0.530`
- GCA mean: `0.560`
- Gap (GCA - Holistic): `+0.029`
- Bootstrap 95% CI: `[+0.009, +0.048]`
- Wilcoxon p-value: `0.0123`

Interpretation:
- The pooled effect is positive for GCA, with visible run-to-run variance.
- The additional validation seed (`1337590`, seed `314`) completed and kept the pooled result in favor of GCA.

---

## 7. Artifacts and Paths

Primary MOGON workspace used for reliable execution:
- `~/thesis`

Key output roots:
- Baseline runs: `~/thesis/outputs/reward_models_1000/`
- Alpha ablation: `~/thesis/outputs/reward_models_1000_alpha_0.0/`
- HP search: `~/thesis/outputs/hpsearch_alpha_0.0/`
- Mode sweep: `~/thesis/outputs/mode_sweep_alpha0/`

This 30-06 update file now includes the completed five-run robustness summary for `mode_nli`.

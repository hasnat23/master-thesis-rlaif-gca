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

## 5. Continuation Work Launched for 30 June

To continue the planned robustness check, additional independent seed validations for `mode_nli` were launched.

### New remote script added on MOGON workspace
- Script path (remote): `~/thesis/slurm/mode_nli_seed_confirm.sh`
- Purpose: rerun `mode_nli` end-to-end with configurable `SEED` and isolated output folders.
- Output pattern:
  - Preferences: `data/preferences_1000_alpha0_mode_nli_seed<SEED>/`
  - RM summaries: `outputs/mode_sweep_alpha0/mode_nli_seed<SEED>/rm_training_summary.json`

### New jobs submitted (current status)

| Job ID | Name | Seed | Status |
|--------|------|------|--------|
| 1337534 | mode_nli_seed | 7 | RUNNING |
| 1337535 | mode_nli_seed | 100 | RUNNING |

Submission notes:
- Both jobs were submitted with `--exclude=gpu0001` to avoid the prior CUDA-availability issue.
- Current node assignment: `gpu0002`.

### Live execution health check (latest)

- Both jobs are currently running without CUDA/device errors.
- Holistic judging completed successfully for both jobs (`1000/1000` processed, `0` ties).
- GCA judging is actively progressing for both jobs (log checkpoints observed beyond `450/1000`).
- Current stage indicates normal pipeline flow before RM training starts.

---

## 6. Immediate Next Steps

1. Wait for jobs `1337534` and `1337535` to finish.
2. Extract metrics from:
   - `outputs/mode_sweep_alpha0/mode_nli_seed7/rm_training_summary.json`
   - `outputs/mode_sweep_alpha0/mode_nli_seed100/rm_training_summary.json`
3. Combine with prior two `mode_nli` runs to form a four-run summary.
4. Recompute pooled CI and paired significance tests with the expanded set.
5. Decide final reporting direction after reviewing four-run stability.

---

## 7. Artifacts and Paths

Primary MOGON workspace used for reliable execution:
- `~/thesis`

Key output roots:
- Baseline runs: `~/thesis/outputs/reward_models_1000/`
- Alpha ablation: `~/thesis/outputs/reward_models_1000_alpha_0.0/`
- HP search: `~/thesis/outputs/hpsearch_alpha_0.0/`
- Mode sweep: `~/thesis/outputs/mode_sweep_alpha0/`

This 30-06 update file is focused on continuity and active execution state. It will be extended after seed jobs `1337534` and `1337535` complete.

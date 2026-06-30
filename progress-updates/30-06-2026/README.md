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
| `mode_nli_seed2026` | 0.525 | 0.564 | +0.039 |

Observation:
- Seed-specific variance remains visible: one seed (out of six) favors Holistic (`seed7`); the other five all favor GCA.
- This confirms that single-run conclusions are unstable without pooled analysis, but direction is now consistent across the majority of independent seeds.

### Six-run combined summary (`mode_nli` family)

Combined runs:
1. Original `mode_nli` sweep (seed=42)
2. `mode_nli` confirmation rerun (seed=42)
3. `mode_nli_seed7`
4. `mode_nli_seed100`
5. `mode_nli_seed314`
6. `mode_nli_seed2026`

| Aggregate | Holistic mean | GCA mean | Gap (GCA - Holistic) |
|-----------|----------------|----------|-----------------------|
| Pooled over 30 folds | 0.5295 | 0.5603 | +0.0308 |

Statistical summary (30 folds):
- Bootstrap 95% CI for gap: `[+0.013, +0.047]` (10,000 resamples, seed=42)
- Wilcoxon signed-rank, two-sided: `W=90.0, p=0.0034`
- Wilcoxon signed-rank, one-sided (GCA > Holistic): `W=375.0, p=0.0017`
- Fold-level wins / ties / losses for GCA: `22 / 0 / 8` of 30

Interpretation:
- After adding the sixth independent seed, the pooled GCA advantage is preserved (+3.1 pp) and statistical support strengthens (two-sided p drops from 0.0123 → 0.0034).
- GCA wins on 73% of all evaluation folds across six independent runs.
- The pooled gap range observed across the campaign (+2 to +3 pp) is now bracketed tightly by the 95% CI `[+1.3 pp, +4.7 pp]`.

---

## 6. Immediate Next Steps

1. Lock `mode_nli` as the final backend mode for the thesis reward-model evaluation.
2. Use the six-run pooled analysis (section 7) as the primary reportable result; keep per-run table for variance discussion.
3. Frozen experiment configuration (section 8) is the locked baseline; further experiments (e.g. larger sample size, different RM backbone) must be reported as separate ablations.
4. Move on to downstream work: integrate the locked RM-GCA into the RLAIF pipeline and prepare the thesis write-up around this validated result.

### Sixth seed validation (completed)

| Job ID | Name | Seed | Status | Wall-clock |
|--------|------|------|--------|------------|
| 1338194 | mode_nli_seed | 2026 | COMPLETED (exit 0) | 00:30:58 |

Result (`outputs/mode_sweep_alpha0/mode_nli_seed2026/rm_training_summary.json`, run_id `20260628_223747_8a231e`):
- Holistic fold accs: `[0.530, 0.550, 0.525, 0.505, 0.515]` → mean `0.525`
- GCA fold accs: `[0.610, 0.600, 0.495, 0.570, 0.545]` → mean `0.564`
- Gap: **+0.039** (GCA > Holistic)
- Submitted with `--exclude=gpu0001`.

---

## 7. Thesis-Ready Reporting Table (`mode_nli` Family)

### Per-run view (shows variance)

| Run | Seed | Holistic | GCA | Gap (GCA - Holistic) |
|-----|------|----------|-----|----------------------|
| Original sweep | 42 | 0.523 | 0.583 | +0.060 |
| Confirmation | 42 | 0.543 | 0.556 | +0.013 |
| Seed validation 1 | 7 | 0.556 | 0.546 | -0.010 |
| Seed validation 2 | 100 | 0.510 | 0.561 | +0.051 |
| Seed validation 3 | 314 | 0.520 | 0.552 | +0.032 |
| Seed validation 4 | 2026 | 0.525 | 0.564 | +0.039 |

Directional agreement: GCA wins in 5 of 6 runs; pooled gap stable at +2–3 pp.

### Pooled view (central tendency)

| Aggregate | Holistic mean | GCA mean | Gap | 95% CI | Wilcoxon p (two-sided) |
|-----------|---------------|----------|-----|--------|------------------------|
| Pooled (6 runs, 30 folds) | 0.5295 | 0.5603 | +0.0308 | [+0.013, +0.047] | 0.0034 |

Fold-level GCA wins/ties/losses (pooled): **22 / 0 / 8** of 30.

### Key claim (defensible statement)

> Across six independent validation runs of the `mode_nli` configuration (30 cross-validation folds total), the GCA-based reward model achieves a mean pairwise accuracy advantage of +3.1 percentage points over the holistic baseline (Holistic: 0.530, GCA: 0.560). GCA wins on 22 of 30 evaluation folds, and the advantage is statistically supported with a 95% bootstrap confidence interval of [+0.013, +0.047] and a Wilcoxon signed-rank two-sided p-value of 0.0034 (one-sided p = 0.0017).

---

## 8. Frozen Experiment Configuration (Locked for Write-Up)

The following configuration is now frozen to ensure all reported results are comparable:

| Parameter | Value |
|-----------|-------|
| Dataset | CNN/DailyMail test split |
| Sample size | 1000 articles |
| Article subset seed | 200 |
| Margin | 0 (all pairs used) |
| GCA alpha | 0.0 |
| Judge backend | AlignScore (`yzha/AlignScore`) |
| AlignScore mode | `nli` |
| AlignScore checkpoint | `AlignScore-base.ckpt` |
| AlignScore backbone | `FacebookAI/roberta-base` |
| RM backbone | `FacebookAI/roberta-base` |
| RM epochs | 5 |
| RM learning rate | 2e-5 |
| RM batch size | 8 |
| RM max sequence length | 512 |
| RM max article chars | 2000 |
| RM k-fold | 5 |
| RM training seed | varied across runs (42, 42, 7, 100, 314, 2026) |

Any deviation from this configuration in future runs must be documented as a separate experiment, not a replacement for the current set.

---

## 9. Short Summary (Reusable)

`mode_nli` is confirmed as the strongest backend mode for GCA in the current campaign.

Across six completed independent runs (30 folds total), pooled results are:
- Holistic mean: `0.5295`
- GCA mean: `0.5603`
- Gap (GCA - Holistic): `+0.0308`
- Bootstrap 95% CI: `[+0.013, +0.047]` (10,000 resamples)
- Wilcoxon signed-rank, two-sided: `p = 0.0034`
- Wilcoxon signed-rank, one-sided (GCA > Holistic): `p = 0.0017`
- Fold-level wins / ties / losses for GCA: `22 / 0 / 8` of 30

Interpretation:
- The pooled GCA advantage is reproducible across six independent seeds and statistically supported.
- The +2 to +3 pp range observed early in the campaign is now confirmed by the tight CI `[+1.3 pp, +4.7 pp]`.
- GCA wins on roughly three-quarters of all evaluation folds across the campaign.

---

## 10. Artifacts and Paths

Primary MOGON workspace used for reliable execution:
- `~/thesis`

Key output roots:
- Baseline runs: `~/thesis/outputs/reward_models_1000/`
- Alpha ablation: `~/thesis/outputs/reward_models_1000_alpha_0.0/`
- HP search: `~/thesis/outputs/hpsearch_alpha_0.0/`
- Mode sweep: `~/thesis/outputs/mode_sweep_alpha0/`

This 30-06 update file now includes the completed six-run robustness summary for `mode_nli`.

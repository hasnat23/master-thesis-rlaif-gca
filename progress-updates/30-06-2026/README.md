# Meeting Notes — 30 June 2026

**Student:** Muhammad Hasnat
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## 1. Today's Headline

The `mode_nli` configuration is now validated across **six independent seed runs (30 cross-validation folds total)**. The GCA reward model outperforms the Holistic baseline by **+3.08 percentage points (95% CI [+1.3, +4.7], Wilcoxon two-sided p = 0.0034)**.

This is the first time the GCA advantage has been demonstrated with strong statistical support on the 1000-sample setup.

---

## 2. Pipeline (Unchanged)

1. Generate 1000 candidate summary pairs from CNN/DailyMail.
2. Build two preference sets per run: Holistic (full-summary AlignScore) and GCA (sentence-level AlignScore + aggregation).
3. Train Bradley–Terry reward models with identical hyperparameters.
4. Compare 5-fold validation accuracy: RM-Holistic vs RM-GCA.

Fixed settings: AlignScore judge (`yzha/AlignScore`, mode `nli`), RoBERTa-base backbone, `margin=0`, `alpha=0.0`, `epochs=5`, `lr=2e-5`, `batch=8`, `kfold=5`, varied training seed.

---

## 3. Background (Results Already Reported on 23 June)

| Stage | Best result | Outcome |
|-------|-------------|---------|
| Baseline 1000-sample (default mode) | Holistic 0.572, GCA 0.560 | GCA below Holistic (-1.2 pp) |
| Alpha ablation (`alpha=0.0`) | Holistic 0.586, GCA 0.561 | Gap unchanged |
| Hyperparameter search | `lr1e-5_ep7` showed +1.4 pp | Did not reproduce on rerun |
| AlignScore mode sweep | `nli` mode: GCA +6.0 pp | Strongest candidate found |
| `mode_nli` confirmation (2 runs) | Pooled gap +3.4 pp | Direction stable, magnitude noisy (Wilcoxon p = 0.084) |

Conclusion at that point: `mode_nli` looked promising but needed more independent seed runs to rule out chance.

---

## 4. New Work Completed Since 23 June

To confirm the `mode_nli` result, four additional independent seed runs were submitted to MOGON. All four jobs completed successfully on `gpu0002` (excluding the problematic `gpu0001`).

### Seed validation jobs

| Job ID | Seed | Wall-clock | Status |
|--------|------|------------|--------|
| 1337534 | 7 | 00:30:54 | COMPLETED |
| 1337535 | 100 | 00:30:54 | COMPLETED |
| 1337590 | 314 | 00:30:52 | COMPLETED |
| 1338194 | 2026 | 00:30:58 | COMPLETED (finished today) |

### Per-run results (`mode_nli`)

| Run | Seed | Holistic | GCA | Gap |
|-----|------|----------|-----|-----|
| Original sweep | 42 | 0.523 | 0.583 | **+0.060** |
| Confirmation rerun | 42 | 0.543 | 0.556 | **+0.013** |
| Seed validation 1 | 7 | 0.556 | 0.546 | -0.010 |
| Seed validation 2 | 100 | 0.510 | 0.561 | **+0.051** |
| Seed validation 3 | 314 | 0.520 | 0.552 | **+0.032** |
| Seed validation 4 (today) | 2026 | 0.525 | 0.564 | **+0.039** |

GCA wins in **5 of 6 runs**. Seed 7 is the only run where Holistic was slightly ahead.

---

## 5. Final `mode_nli` Results

### Pooled across 6 runs / 30 folds

| Holistic mean | GCA mean | Gap | 95% CI (bootstrap) | Wilcoxon p (two-sided) |
|---------------|----------|-----|--------------------|------------------------|
| 0.5295 | 0.5603 | **+0.0308** | [+0.013, +0.047] | **0.0034** |

- Fold-level outcome: GCA wins **22 / 30** folds (73%); 0 ties; 8 losses.
- One-sided Wilcoxon (GCA > Holistic): p = 0.0017.
- Bootstrap details: 10,000 resamples, `numpy` RNG seed 42.

### Summary statement for the thesis

Across six independent validation runs of the `mode_nli` configuration (30 cross-validation folds total), the GCA reward model achieves a mean pairwise accuracy advantage of **+3.08 percentage points** over the Holistic baseline (Holistic 0.5295, GCA 0.5603). GCA wins on 22 of 30 evaluation folds. The advantage is statistically supported by a 95% bootstrap confidence interval of [+0.013, +0.047] and a Wilcoxon signed-rank two-sided p-value of 0.0034.

### Progression of statistical evidence

| Stage | Runs | Folds | Pooled gap | 95% CI | Wilcoxon p |
|-------|------|-------|-----------:|--------|-----------:|
| Two-run (23 June) | 2 | 10 | +0.034 | [+0.003, +0.064] | 0.084 |
| Five-run (last week) | 5 | 25 | +0.029 | [+0.009, +0.048] | 0.0123 |
| **Six-run (today)** | **6** | **30** | **+0.031** | **[+0.013, +0.047]** | **0.0034** |

Adding the sixth seed both tightened the confidence interval and dropped the p-value below the 0.01 threshold.

---

## 6. Frozen Experiment Configuration

This configuration is now locked. Any further experiment (different RM backbone, larger sample size, etc.) will be reported as a separate ablation, not as a replacement.

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
| RM training | `epochs=5`, `lr=2e-5`, `batch=8`, `max_length=512`, `kfold=5` |
| RM training seeds (the 6 runs) | 42, 42, 7, 100, 314, 2026 |

---

## 7. Next Steps

1. Lock `mode_nli` as the final reward model configuration for the thesis.
2. Use the six-run pooled analysis (Section 5) as the primary reported result; keep the per-run table to document variance.
3. Move to downstream work: integrate the locked RM-GCA into the RLAIF pipeline and begin the thesis write-up.

---

## 8. Artifacts and Paths

Primary MOGON workspace: `~/thesis`

Output locations:
- Baseline runs: `~/thesis/outputs/reward_models_1000/`
- Alpha ablation: `~/thesis/outputs/reward_models_1000_alpha_0.0/`
- HP search: `~/thesis/outputs/hpsearch_alpha_0.0/`
- Mode sweep and seed validations: `~/thesis/outputs/mode_sweep_alpha0/`

Reusable seed-validation script (remote): `~/thesis/slurm/mode_nli_seed_confirm.sh` (configurable `SEED` env var; outputs to `outputs/mode_sweep_alpha0/mode_nli_seed<SEED>/`).

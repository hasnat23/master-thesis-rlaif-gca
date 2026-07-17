# Meeting Notes — 16 July 2026

**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## 1. Current Status

The `mode_nli` setting has already been validated on the 1000-sample setup across six independent seed runs. That result remains the strongest evidence so far: GCA is ahead of Holistic by **+3.08 percentage points** with a **95% bootstrap CI of [+1.3, +4.7]** and **Wilcoxon two-sided p = 0.0034**.

The larger 5000-item rerun has now also completed. On this larger dataset, Holistic slightly outperformed GCA: **0.5788 vs 0.5746**, a gap of **-0.0042** in favor of Holistic.

The 10000-item extension has now completed as well. Candidate generation, preference construction, and RM training are all finished.

The larger-scale rerun targeted a bigger dataset, using **5000 or 10000 articles/candidates** instead of 1000. The 5000-item run is the first large-scale check, and the 10000-item run is the final larger follow-up.

---

## 2. What Is Different Now

The goal is no longer only to confirm the 1000-sample result. The next experiment is to test whether the same GCA advantage still holds when the candidate pool is much larger.

Planned scale-up conditions:
- Keep the same judge: `yzha/AlignScore`
- Keep the same RM backbone: `FacebookAI/roberta-base`
- Keep the same RM training setup: `epochs=5`, `lr=2e-5`, `batch=8`, `kfold=5`
- Keep the same GCA mode: `nli`
- Increase the dataset size to **5000** first, then extend to **10000** if needed
- Use the same nested subset seed (`200`) so the larger set remains comparable to the earlier 1000-sample run

---

## 3. Why This Matters

The 1000-sample campaign showed that GCA can beat Holistic under controlled conditions, but the sample was still relatively small. A larger dataset is needed to check whether the advantage is stable when the number of candidate pairs increases substantially.

This is the next robustness step for the thesis, not a new method change.

---

## 4. Preparation Done Today

To support the larger rerun, I prepared scalable Slurm wrappers for the remote MOGON workspace so the same pipeline can be launched with a larger sample size.

The scale-up workflow now covers:
- candidate generation
- preference construction
- reward-model training

The current focus is the 10000-item dataset. The 5000-item run is already complete and serves as the first large-scale check.

Queued on MOGON:
- `1411306` — generate the 5000-item candidate set, **completed**
- `1411307` — build holistic/GCA preferences for the 5000-item set, **completed**
- `1411308` — train the 5000-item Bradley-Terry reward models, **completed**
- `1411412` — generate the 10000-item candidate set, **completed**
- `1411413` — build holistic/GCA preferences for the 10000-item set, **completed**
- `1411415` — train the 10000-item Bradley-Terry reward models, **completed**

Final outputs:
- Candidate set: `~/thesis/data/candidates/candidates_10000.jsonl`
- Preference files: `~/thesis/data/preferences_10000/`
- RM summary: `~/thesis/outputs/reward_models_10000/rm_training_summary.json`

5k summary:
- Holistic mean accuracy: `0.5788`
- GCA mean accuracy: `0.5746`
- Gap (GCA - Holistic): `-0.0042`

Interpretation of the 5k result:
- The 5k gap is very small, so the 1000-sample advantage was not strong enough to remain stable at this larger scale.
- This suggests the GCA effect is sensitive to the exact dataset composition and is not yet a clearly robust gain at 5000 items.

10k summary:
- Holistic mean accuracy: `0.5827`
- GCA mean accuracy: `0.5862`
- Gap (GCA - Holistic): `+0.0035`

---

## 5. Immediate Next Step

The 5000-item experiment is complete. The 10000-item extension is now also complete.

Staged on MOGON:
- `1411412` — generate the 10000-item candidate set, **completed**
- `1411413` — build holistic/GCA preferences for the 10000-item set, **completed**
- `1411415` — train the 10000-item Bradley-Terry reward models, **completed**

For the meeting tomorrow, the safe headline is:
- 1000-sample, six-run campaign: GCA wins reproducibly.
- 5000-sample rerun: Holistic is slightly better, so the larger dataset does **not** confirm the GCA advantage.
- 10000-sample follow-up: GCA is again slightly better, so the larger-dataset result is now back in favor of GCA.

# Biweekly Progress Update — 21 April 2026

**Period:** April 9 – April 21, 2026  
**Meeting date:** 21 April 2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner | **Mentor:** Lingxiao Kong

---

## Thesis Context

**Title:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)  
**Core question:** For long news summaries, are holistic A/B AI preferences sufficient to improve factual reliability under DPO, or does sentence-level supervision aggregated via GCA yield more factually consistent outputs?  
**Key invariants:** base model, candidate pool, DPO recipe, compute budget — only feedback granularity changes.

### Research Questions
- **RQ1:** Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic AI feedback under DPO?
- **RQ2:** How sensitive are results to alignment strategy (semantic vs index-based) and to judge reliability controls (confidence gating, A/B swap)?
- **RQ3:** Which factual error categories (entities, numbers, relations, temporal claims) benefit most from sentence-level supervision?
- **RQ4:** Do gains persist under human auditing / a second judge, or are they judge-specific artefacts?

---

## Summary

Phase 1 (setup & baselines) is complete. Phase 2 (judge prompts & candidate generation) infrastructure is fully implemented. The **200-sample candidate generation job is currently RUNNING on MOGON NHR A100** (job 1068994, started 14:03 CEST, 21 April 2026).

---

## Phase 1 — Setup & Baselines: COMPLETE ✅

### 1.1 Full Pipeline Codebase

All Phase 1–2 code implemented and committed to `origin/main`:

| Module | Description |
|--------|-------------|
| `src/data/schema.py` | Dataclass hierarchy: `SubsetSample → CandidatePair → Judgment → PreferencePair` |
| `src/data/subset.py` | Deterministic 200-sample selection from CNN/DailyMail with SHA256 IDs |
| `src/generation/candidates.py` | Model loading (bfloat16), left-padding fix, two-temperature generation (T=0.7, T=1.0), batched inference |
| `src/judging/holistic.py` | Holistic A/B GPT-4o judging with evidence-grounded rationales and confidence |
| `src/judging/sentence_level.py` | Per-sentence pair judging with alignment |
| `src/judging/gca.py` | GCA aggregation: `score = mean × (min/mean)^α`, α=0.5 |
| `src/judging/reliability.py` | Confidence gating (≥0.7), A/B swap consistency, rationale validation |
| `src/eval/metrics.py` | ROUGE-1/2/L, BERTScore F1 |
| `scripts/01–05` | End-to-end pipeline (subset → generation → judging → evaluation → plots) |
| `configs/` | YAML-driven config system; CLI `--override` for all stages |

### 1.2 Dataset Subset

- **Source:** CNN/DailyMail (Hermann et al., 2015)
- **Subset:** 200 articles, fixed random seed, SHA256 IDs for reproducibility
- **Confirmed on MOGON:** `data/subset/subset_200.jsonl` (779KB) ✅
- **Local dataset cache:** `data/cnn_dailymail_500/` (Arrow format) ✅

### 1.3 Baseline Reference Metrics (Phase 1, 20-sample pilot)

These are the pre-fine-tuning comparison baseline:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3278 |
| ROUGE-2 | 0.1201 |
| ROUGE-L | 0.2341 |
| BERTScore F1 | 0.8612 |

SummaC and QAFactEval baselines to be computed after `candidates_200.jsonl` is generated.

---

## Phase 2 — Candidate Generation: IN PROGRESS ⏳

### 2.1 Infrastructure

- **Cluster:** MOGON NHR (`mogon-nhr-01.zdv.uni-mainz.de`), partition `a100dl`, account `nhr-haloed`
- **GPU:** NVIDIA A100-SXM4-40GB (40,960 MiB)
- **Conda env `thesis_env`:** torch 2.4.0+cu118, transformers 5.5.0, datasets 4.8.4, peft, trl, bitsandbytes 0.49.2, accelerate, bert-score, rouge-score
- **Model:** `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0, instruction-tuned, 7B)

**Model confirmed on MOGON** at `~/thesis/models/Mistral-7B-Instruct-v0.3/`:
```
-rw-r--r--  14G  consolidated.safetensors
-rw-r--r-- 4.7G  model-00001-of-00003.safetensors
-rw-r--r-- 4.7G  model-00002-of-00003.safetensors
-rw-r--r-- 4.3G  model-00003-of-00003.safetensors
Total: ~28GB
```

### 2.2 Smoke Test — PASSED ✅

| Field | Value |
|-------|-------|
| Job ID | 1068925 |
| Partition | `a100dl` |
| GPU | NVIDIA A100-SXM4-40GB |
| Elapsed | 00:01:09 |
| Exit code | 0:0 |
| Model used | opt-125m (validation override) |
| Samples | 5 |
| Result | COMPLETED ✅ |

Full environment validated: conda activation, GPU detection, dataset loading, model loading, generation.

### 2.3 Bug Resolved: transformers 5.x / bitsandbytes Incompatibility

Jobs 1068953 and 1068993 crashed within 26–48 seconds with:
```
AttributeError: 'MistralForCausalLM' object has no attribute 'set_submodule'
```

**Root cause:** `transformers 5.5.0` calls `torch.nn.Module.set_submodule()` inside its bitsandbytes 4-bit integration, but this method is not available in the installed PyTorch build.

**Fix (commit `6ebd67c`):**
1. `torch_dtype=` → `dtype=` in `AutoModelForCausalLM.from_pretrained()` (transformers 5.x API change)
2. `load_in_4bit: false` in `configs/generation.yaml` — A100 has 40GB; Mistral-7B in bfloat16 uses ~14GB, quantisation not needed
3. rsync `--relative` flag — previous flat rsync sent files to wrong paths (`~/thesis/candidates.py` instead of `~/thesis/src/generation/candidates.py`)

**Verified on MOGON:**
```
~/thesis/configs/generation.yaml:         load_in_4bit: false
~/thesis/src/generation/candidates.py:    dtype=torch.bfloat16,
```

### 2.4 Job History

| Job ID | Status | Elapsed | Node | Notes |
|--------|--------|---------|------|-------|
| 1068925 | COMPLETED ✅ | 00:01:09 | a100dl | Smoke test — opt-125m, 5 samples |
| 1068953 | FAILED ❌ | 00:00:26 | gpu0003 | `torch_dtype` deprecated + bitsandbytes crash |
| 1068993 | FAILED ❌ | 00:00:48 | gpu0003 | Flat rsync — correct files not at right path |
| **1068994** | **RUNNING** ⏳ | Started 14:03 CEST | gpu0001 | **Fixed** — bfloat16, no 4-bit, correct paths |

### 2.5 Live Job Log (1068994, as of 14:08 CEST)

```
=== Candidate Generation Start ===
Job ID: 1068994
Node: gpu0001
GPU: NVIDIA A100-SXM4-40GB, 40960 MiB
Date: Tue Apr 21 14:03:46 CEST 2026

Working directory: /fshpc/muhhas01/thesis
Python: /home/muhhas01/.conda/envs/thesis_env/bin/python

--- Generating 200 candidate pairs ---
Config: load_in_4bit=False, model=models/Mistral-7B-Instruct-v0.3,
        batch_size=4, max_new_tokens=256, temps=[0.7, 1.0]
Starting candidate generation...
Loading model: models/Mistral-7B-Instruct-v0.3
[model loading in progress — no crash after 4+ min, consistent with ~14GB load time]
```

Expected output: `data/candidates/candidates_200.jsonl` — 200 articles × 2 temperatures = 400 candidate summaries.

---

## GCA Formula (implemented in `src/judging/gca.py`)

$$\text{score} = \bar{s} \cdot \left(\frac{\min(s)}{\bar{s}}\right)^\alpha, \quad \alpha = 0.5$$

- $\bar{s}$ = mean sentence score, $\min(s)$ = lowest sentence score
- Penalises summaries where one weak sentence undermines otherwise good content
- Addresses the credit assignment problem for long outputs: holistic labels may miss localised factual errors
- `tie_margin = 0.05`; pairs with ambiguous aggregated outcome are discarded

---

## Upcoming: Phase 2 → Phase 3

| Step | Action | Blocker |
|------|--------|---------|
| ⏳ | Await job 1068994 → `candidates_200.jsonl` | Running |
| ⏳ | Compute ROUGE/BERTScore/SummaC baseline (`scripts/04_evaluate_baseline.py`) | Needs candidates |
| ⏳ | Design holistic + sentence-level GPT-4o judge prompts | OpenAI API key needed on MOGON |
| ⏳ | Run 20-pair judge test (`scripts/03_run_judge_test.py`) | OpenAI API key |
| ⏳ | Human audit: 100 pairs (Cohen's Kappa, judge vs human agreement) | Post-judging |
| ⏳ | Phase 4: DPO fine-tuning on both preference datasets | Post-judging |

---

## Blockers / Risks

| Issue | Status | Mitigation |
|-------|--------|-----------|
| OpenAI API key on MOGON | Not yet set | Needed for Phase 3; not blocking current generation |
| transformers 5.x + bitsandbytes | ✅ Fixed | bfloat16, no 4-bit quantisation |
| rsync path bug | ✅ Fixed | `--relative` flag verified |
| Sentence alignment noise | To be quantified | Ablation: semantic vs index-based alignment (RQ2) |

---

## Codebase & Artifacts

| Artifact | Location | Status |
|----------|----------|--------|
| Full codebase | `origin/main`, commit `6ebd67c` | ✅ Pushed |
| Smoke test log | `slurm/smoke_test_1068925.out` (MOGON) | ✅ |
| Generation log (live) | `slurm/gen_candidates_1068994.out` (MOGON) | ⏳ Running |
| Candidates (pending) | `data/candidates/candidates_200.jsonl` | ⏳ |
| Subset | `data/subset/subset_200.jsonl` (779KB, 200 articles) | ✅ MOGON |
| Dataset cache | `data/cnn_dailymail_500/` (Arrow) | ✅ MOGON |


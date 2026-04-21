# Biweekly Progress Update — 21 April 2026

**Period:** April 9 – April 21, 2026  
**Meeting date:** 21 April 2026  

---

## Summary

Completed all infrastructure and codebase work for Phase 2 (Milestone 1). Successfully transferred Mistral-7B-Instruct-v0.3 (28GB) to MOGON NHR A100 cluster and resolved a transformers 5.x / bitsandbytes API incompatibility. Full 200-sample candidate generation is running as of this meeting (job 1068993).

---

## Completed This Period

### 1. Full Pipeline Codebase (src/, scripts/, configs/)
- `src/data/`: `schema.py` (dataclass hierarchy), `subset.py` (deterministic 200-sample selection with SHA256 IDs)
- `src/generation/candidates.py`: model loading, 4-bit quantization support, left-padding fix, two-temperature generation
- `src/judging/`: holistic GPT-4o judging, sentence-level scoring, GCA aggregation, reliability controls
- `src/eval/metrics.py`: ROUGE-1/2/L and BERTScore F1
- `scripts/01–05`: full pipeline from subset prep → generation → judging → evaluation → plots
- `configs/`: YAML-driven config system for all stages

### 2. MOGON NHR Setup & Infrastructure
- Migrated from MOGON II (GTX 1080 Ti) to **MOGON NHR** (A100-SXM4-40GB, partition `a100dl`)
- SSH ControlMaster setup via `mogon-nhr-01.zdv.uni-mainz.de` with 2FA
- Conda environment `thesis_env` with torch 2.4.0+cu118, transformers 5.5.0, all dependencies
- Slurm scripts: `smoke_test.sh`, `generate_candidates.sh`, `judge_test.sh`

### 3. Smoke Test — PASSED ✅
- Job **1068925** on `a100dl`, elapsed: **1m09s**, exit code 0
- GPU: NVIDIA A100-SXM4-40GB
- 5 samples generated successfully with opt-125m validation model

### 4. Model: Mistral-7B-Instruct-v0.3
- Selected: `mistralai/Mistral-7B-Instruct-v0.3` (Apache 2.0, instruction-tuned, ungated)
- Downloaded locally (29GB) → rsynced to MOGON NHR: `~/thesis/models/Mistral-7B-Instruct-v0.3/`
- Transfer completed: all 4 safetensors confirmed (consolidated: 14G, shards: 4.7G + 4.7G + 4.3G)

### 5. Bug Fix: transformers 5.x / bitsandbytes Incompatibility
**Error in job 1068953:**
```
AttributeError: 'MistralForCausalLM' object has no attribute 'set_submodule'
```
**Root cause:** `transformers 5.5.0` uses `set_submodule()` from `torch.nn.Module` in its bitsandbytes integration, which conflicts with the installed PyTorch build.

**Fix applied:**
- Changed `torch_dtype=` → `dtype=` in `from_pretrained()` (transformers 5.x API)
- Disabled `load_in_4bit` — not needed since A100 has 40GB; Mistral-7B in bfloat16 uses ~14GB

---

## Current Job Status (as of meeting)

| Job ID | Name | Status | Elapsed | Node |
|--------|------|--------|---------|------|
| 1068925 | smoke_test | COMPLETED ✅ | 01:09 | gpu node |
| 1068953 | gen_candidates | COMPLETED (crashed) ❌ | 00:26 | gpu0003 |
| **1068993** | **gen_candidates** | **RUNNING / PENDING** ⏳ | — | a100dl |

Job **1068993** is the fixed resubmission — uses `dtype=torch.bfloat16`, no 4-bit quantization.  
Expected runtime: **~1–2 hours** on A100.

---

## Smoke Test Output (Job 1068925)

```
GPU: NVIDIA A100-SXM4-40GB, 40960 MiB
Model: opt-125m (smoke test override)
Samples: 5
Status: COMPLETED
Elapsed: 00:01:09
Exit code: 0:0
```

---

## Phase 1–2 Baseline Reference Metrics (20 samples, previous work)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3278 |
| ROUGE-2 | 0.1201 |
| ROUGE-L | 0.2341 |
| BERTScore F1 | 0.8612 |

These are the comparison baseline for the upcoming 200-sample run.

---

## GCA Formula (implemented in `src/judging/gca.py`)

$$\text{score} = \bar{s} \cdot \left(\frac{\min(s)}{\bar{s}}\right)^\alpha, \quad \alpha = 0.5$$

- Penalises inconsistent sentence quality within a summary
- `tie_margin = 0.05` for winner determination
- Reliability controls: confidence gating (0.7), A/B swap consistency, rationale validation

---

## Next Steps

1. **Await job 1068993** — `candidates_200.jsonl` expected (~1–2h on A100)
2. **Run GPT-4o judging** on 20-pair test set — requires OpenAI API key on MOGON
3. **Compute baseline ROUGE/BERTScore** on 200 candidates (`scripts/04_evaluate_baseline.py`)
4. **Set up DPO training** pipeline (Phase 3 / Milestone 2)

---

## Blockers / Risks

- OpenAI API key still needed on MOGON for Phase 3 judging (not blocking generation)
- bitsandbytes compatibility confirmed issue with transformers 5.x — resolved by using float16 on A100

---

## Artifacts

- Full codebase: committed and pushed to `origin/main` (commit `cba6fc0`)
- Smoke test log: `slurm/smoke_test_1068925.out` (on MOGON)
- Generation job log: `slurm/gen_candidates_1068993.out` (in progress, on MOGON)
- Candidates output (pending): `data/candidates/candidates_200.jsonl`

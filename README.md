# Master Thesis: Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)

**University of Koblenz** | Master's Thesis | 2025–2026  
**Student:** Muhammad Hasnat  
**Supervisors:** Dr. Zeyd Boukhers, Prof. Dr. Frank Hopfgartner  
**Mentor:** Lingxiao Kong  
**GitHub:** [@hasnat23](https://github.com/hasnat23)

---

## Overview

This thesis investigates a controlled question: for long news summaries, are **holistic A/B preferences** sufficient to improve factual reliability under DPO fine-tuning, or does supervision become more effective when the judge evaluates **aligned sentence pairs** whose local decisions are then aggregated into a summary-level preference via **Granular Credit Assignment (GCA)**?

The base model, candidate pool, training objective (DPO), and compute budget are held constant across conditions. The only manipulated factor is the granularity of AI-generated preference labels.

- **Dataset:** CNN/DailyMail (200-sample compute-controlled subset)
- **Base model:** Mistral-7B-Instruct-v0.3 (instruction-tuned 7B, Apache 2.0)
- **Judge:** GPT-4o with confidence gating, A/B order randomization, evidence-grounded rationales
- **Fine-tuning:** QLoRA + DPO (offline preference optimisation via TRL)
- **Primary metrics:** ROUGE-1/2/L, BERTScore F1, SummaC (NLI-based), QAFactEval
- **Diagnostic evaluator:** FineSurE (fine-grained sentence/key-fact level analysis)

---

## Research Questions

**RQ1:** Does sentence-level AI feedback aggregated via GCA produce more factually consistent summaries than holistic AI feedback when used for DPO fine-tuning?

**RQ2:** How sensitive are the observed effects to alignment strategy (index-based vs semantic alignment) and to judge reliability controls such as confidence gating and A/B order randomisation?

**RQ3:** What categories of factual errors are most affected by sentence-level supervision (entities, numbers, relations, temporal claims)?

**RQ4:** Do gains persist under human auditing and (optionally) a second judge, or are they judge-specific artefacts?

### Hypotheses

- **H1:** Sentence-level aggregated preferences reduce factual inconsistency more than holistic preferences because they localise the supervision signal on long outputs.
- **H2:** Better alignment and reliability controls reduce label noise; if alignment noise is too high it can mask GCA benefits.
- **H3:** Improvements are largest for localised errors (entity/relation mistakes) rather than global attributes such as style.
- **H4:** If improvements reflect genuine factuality gains they remain visible under at least one independent evaluator.

---

---

## GCA: Granular Credit Assignment

GCA is an aggregation layer that converts sentence-level A/B judgments into a single summary-level preference pair usable by DPO — no separate reward model required.

$$\text{score} = \bar{s} \cdot \left(\frac{\min(s)}{\bar{s}}\right)^\alpha, \quad \alpha = 0.5$$

- $\bar{s}$ = mean sentence score, $\min(s)$ = lowest sentence score
- Penalises summaries with inconsistent sentence quality (one bad sentence drags the aggregate down)
- `tie_margin = 0.05` for winner determination; ambiguous pairs are discarded

**Reliability controls on the judge:**
- A/B position swap — inconsistent pairs are filtered or flagged for audit
- Confidence gating (≥ 0.7) — low-confidence labels discarded
- Evidence-grounded rationales — non-evidential outputs discarded
- Human audit: 100 pairs stratified across confidence / disagreement strata; Cohen's Kappa reported

---

## Methodology Summary

| Factor | Value |
|--------|-------|
| Task | Single-document abstractive summarisation (news) |
| Dataset | CNN/DailyMail — 200-sample subset (fixed seed) |
| Base model | Mistral-7B-Instruct-v0.3 |
| Fine-tuning | QLoRA + DPO (offline; TRL) |
| Judge | GPT-4o (fixed prompt; bias mitigations applied) |
| Main variable | Feedback granularity: holistic A/B vs sentence-level + GCA |
| Alignment ablation | Semantic similarity (primary) vs index-based |
| Optional extension | MoDPO with objective-tagged sentence labels |
| Primary metrics | SummaC, QAFactEval, ROUGE-1/2/L, BERTScore F1 |
| Diagnostic | FineSurE (sentence/fact-level analysis; not optimisation target) |
| Stats | Paired bootstrap resampling (10,000), 95% CI, p < 0.05, effect sizes |

---

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── subset.yaml           #   Data subset selection params
│   ├── generation.yaml       #   Candidate generation params
│   └── judging.yaml          #   AI judging params
├── scripts/                  # Pipeline entry points (run in order)
│   ├── 01_prepare_subset.py  #   Select 200 CNN/DM articles
│   ├── 02_generate_candidates.py  # Generate summary pairs (two temperatures)
│   ├── 03_run_judge_test.py  #   Test judging prompts (mock or live)
│   ├── 04_evaluate_baseline.py    # Compute ROUGE/BERTScore baseline
│   └── 05_plot_results.py    #   Generate comparison plots
├── src/                      # Core library
│   ├── data/                 #   Schema (SubsetSample→CandidatePair→Judgment→PreferencePair), subset selection
│   ├── generation/           #   Model loading, two-temperature candidate generation
│   ├── judging/              #   Holistic, sentence-level, GCA aggregation, reliability controls
│   ├── eval/                 #   ROUGE, BERTScore, SummaC, QAFactEval
│   └── utils/                #   Config loader, run-metadata logging
├── slurm/                    # MOGON NHR job scripts
│   ├── smoke_test.sh         #   5-sample validation (A100, ~1 min)
│   ├── generate_candidates.sh #  Full 200-sample generation (A100, ~1–2h)
│   └── judge_test.sh         #   20-pair judge test
├── progress-updates/         # Biweekly meeting reports
├── proposal/                 # Thesis proposal
├── data/                     # Generated data artifacts (gitignored)
├── outputs/                  # Metrics, plots, run metadata (gitignored)
└── requirements.txt
```

---

## Quick Start (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Prepare 200-sample subset
python scripts/01_prepare_subset.py --config configs/subset.yaml

# 2. (Requires GPU + model) Generate candidates
python scripts/02_generate_candidates.py --config configs/generation.yaml

# 3. Test judge prompts in mock mode
python scripts/03_run_judge_test.py --config configs/judging.yaml --n 5

# 4. Evaluate baseline metrics
python scripts/04_evaluate_baseline.py --candidates data/candidates/candidates_200.jsonl

# 5. Generate plots
python scripts/05_plot_results.py --metrics outputs/metrics/baseline_metrics_*.json
```

---

## MOGON NHR Execution (HPC)

```bash
# Sync code (preserving directory structure)
rsync -avz --relative src/ scripts/ configs/ slurm/ \
  -e "ssh -o 'ControlPath=/tmp/mogon-nhr-cm.sock'" mogon:~/thesis/

# SSH in (requires 2FA via freeOTP → hpcgate.zdv.uni-mainz.de ProxyJump)
ssh mogon

# Activate conda environment
module load lang/Anaconda3/2024.06-1
source $(conda info --base)/etc/profile.d/conda.sh && conda activate thesis_env

# Run smoke test, then full generation
sbatch slurm/smoke_test.sh
sbatch slurm/generate_candidates.sh

# Monitor
squeue -u muhhas01
```

---

## Key Design Decisions

- **Controlled experiment:** Only feedback granularity varies — model, data, and DPO recipe are identical across conditions
- **Config-driven:** All parameters in YAML files, overridable via CLI `--override key=value`
- **Reproducible:** Seeded randomness, SHA256-based sample IDs, run metadata JSON per execution
- **No reward model:** GCA converts sentence judgments directly into DPO preference pairs
- **Reliability-first:** Judge treated as a fallible measurement device with explicit quality controls and human audit

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Base model | Mistral-7B-Instruct-v0.3 |
| Dataset | CNN/DailyMail |
| Frameworks | Hugging Face Transformers 5.x, TRL, PEFT, BitsAndBytes, PyTorch 2.4 |
| Infrastructure | MOGON NHR (A100-SXM4-40GB, Slurm, partition `a100dl`) |
| Judge | GPT-4o (OpenAI API) |
| Alignment objective | DPO + QLoRA |
| Factuality metrics | SummaC, QAFactEval, FineSurE |
| Similarity metrics | ROUGE-1/2/L, BERTScore F1 |

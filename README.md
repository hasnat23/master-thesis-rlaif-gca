# Master Thesis: Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)

**University of Koblenz** | Master's Thesis | 2025–2026
**Student:** Muhammad Hasnat
**GitHub:** [@hasnat23](https://github.com/hasnat23)

---

## Overview

This thesis investigates whether **sentence-level RLAIF** (Reinforcement Learning from AI Feedback) with Granular Credit Assignment (GCA) produces better-calibrated reward signals and higher-quality factual summaries compared to **holistic (document-level) RLAIF**. Everything else is held constant: same base model, same dataset, same judge — only the feedback granularity differs.

- **Dataset:** CNN/DailyMail (200-sample compute-controlled subset)
- **Base model:** Mistral-7B-Instruct-v0.3 (instruction-tuned 7B, Apache 2.0)
- **Judge:** GPT-4o with confidence gating, A/B order randomization, evidence-grounded rationales
- **Fine-tuning:** QLoRA + DPO via TRL
- **Metrics:** ROUGE-1/2/L, BERTScore F1, FactScore

---

## Research Questions

1. Does sentence-level RLAIF with GCA outperform holistic RLAIF for factual summarization?
2. Does GCA reduce reward hacking and improve training stability?
3. How do the two approaches compare on ROUGE, BERTScore, and FactScore metrics?

---

## Project Structure

```
├── configs/                  # YAML configuration files
│   ├── subset.yaml           #   Data subset selection params
│   ├── generation.yaml       #   Candidate generation params
│   └── judging.yaml          #   AI judging params
├── scripts/                  # Pipeline entry points (run in order)
│   ├── 01_prepare_subset.py  #   Select 200 CNN/DM articles
│   ├── 02_generate_candidates.py  # Generate summary pairs
│   ├── 03_run_judge_test.py  #   Test judging prompts (mock or live)
│   ├── 04_evaluate_baseline.py    # Compute ROUGE/BERTScore
│   └── 05_plot_results.py    #   Generate comparison plots
├── src/                      # Core library
│   ├── data/                 #   Data loading, schema, subset selection
│   ├── generation/           #   Model loading, candidate generation
│   ├── judging/              #   Holistic, sentence-level, GCA, reliability
│   ├── eval/                 #   Metrics computation
│   └── utils/                #   Config, logging utilities
├── slurm/                    # MOGON HPC job scripts
│   ├── smoke_test.sh         #   5-sample validation (30 min)
│   ├── generate_candidates.sh #  Full 200-sample generation (2h, A100)
│   └── judge_test.sh         #   20-pair judge test (CPU)
├── reports/                  # Biweekly progress reports
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
# First update configs/generation.yaml with actual model_name
python scripts/02_generate_candidates.py --config configs/generation.yaml

# 3. Test judge prompts in mock mode
python scripts/03_run_judge_test.py --config configs/judging.yaml --n 5

# 4. Evaluate baseline metrics
python scripts/04_evaluate_baseline.py --candidates data/candidates/candidates_200.jsonl

# 5. Generate plots
python scripts/05_plot_results.py --metrics outputs/metrics/baseline_metrics_*.json
```

---

## MOGON Execution (HPC)

```bash
# Upload to MOGON
rsync -avz --exclude '.git' --exclude '__pycache__' . muhhas01@hpcgate.zdv.uni-mainz.de:~/thesis/

# SSH in
ssh mogon

# First time: set up venv
module load lang/Python/3.11.5-GCCcore-13.2.0
module load lib/CUDA/12.1.1
python -m venv ~/venvs/thesis
source ~/venvs/thesis/bin/activate
pip install -r requirements.txt

# Run smoke test first
sbatch slurm/smoke_test.sh

# Then full generation
sbatch slurm/generate_candidates.sh

# Monitor
squeue -u muhhas01
```

---

## Key Design Decisions

- **Config-driven:** All parameters in YAML files, overridable via CLI `--override key=value`
- **Reproducible:** Seeded randomness, SHA256-based sample IDs, run metadata JSON for every execution
- **GCA formula:** `score = mean(sentence_scores) × (min / mean)^α` where α=0.5
- **Reliability controls:** Confidence gating (≥0.7), A/B swap consistency, evidence rationale check

---

## Tech Stack

- **Model:** 7B instruction-tuned (TBD — NOT FLAN-T5)
- **Dataset:** CNN/DailyMail
- **Frameworks:** Hugging Face Transformers, TRL, PEFT, BitsAndBytes, PyTorch
- **Infrastructure:** MOGON HPC (A100 GPUs, Slurm)
- **Judge:** GPT-4o (OpenAI API)
- **Alignment:** DPO (Direct Preference Optimization) via TRL

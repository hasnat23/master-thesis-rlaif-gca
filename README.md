# Master Thesis: Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via Granular Credit Assignment (GCA)

**University of Koblenz** | Master's Thesis | 2025–2026  
**Student:** Muhammad Hasnat  
**GitHub:** [@hasnat23](https://github.com/hasnat23)

---

## Overview

This thesis investigates whether **sentence-level RLAIF** (Reinforcement Learning from AI Feedback) with Granular Credit Assignment (GCA) produces better-calibrated reward signals and higher-quality factual summaries compared to **holistic (document-level) RLAIF**, using the CNN/DailyMail dataset and FLAN-T5 as the base model.

---

## Research Questions

1. Does sentence-level RLAIF with GCA outperform holistic RLAIF for factual summarization?
2. Does GCA reduce reward hacking and improve training stability?
3. How do the two approaches compare on ROUGE, BERTScore, and FactScore metrics?

---

## Repository Structure

```
master-thesis-rlaif-gca/
├── progress-updates/
│   ├── 2026-03-19_Phase1-Phase2-Progress/   # Phase 1 & 2 progress update
│   └── 2026-04-09_Biweekly-Update-1/        # Biweekly update #1
├── code/
│   ├── phase1_setup/                        # Environment & baseline setup
│   ├── phase2_judging/                      # AI judging pipeline
│   ├── phase3_preferences/                  # Preference dataset generation
│   ├── phase4_dpo/                          # DPO / RLAIF training
│   └── phase5_evaluation/                   # Final evaluation & metrics
├── docs/                                    # Thesis proposal & reference docs
└── README.md
```

---

## Thesis Phases

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Environment setup, dataset prep, FLAN-T5 baseline | Completed |
| Phase 2 | AI judging pipeline (holistic & sentence-level) | Completed |
| Phase 3 | Preference dataset generation | In Progress |
| Phase 4 | DPO / RLAIF training (holistic vs GCA) | Pending |
| Phase 5 | Evaluation (ROUGE, BERTScore, FactScore) | Pending |

---

## Baseline Results (FLAN-T5 on CNN/DailyMail)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3278 |
| ROUGE-2 | 0.1201 |
| ROUGE-L | 0.2341 |
| BERTScore F1 | 0.8612 |

---

## Progress Updates

- **2026-03-19** — [Phase 1 & 2 Progress Update](./progress-updates/2026-03-19_Phase1-Phase2-Progress/)
- **2026-04-09** — [Biweekly Update #1](./progress-updates/2026-04-09_Biweekly-Update-1/)

---

## Tech Stack

- **Model:** FLAN-T5 (base/large)
- **Dataset:** CNN/DailyMail
- **Frameworks:** Hugging Face Transformers, TRL, PyTorch
- **Training:** Google Colab / GPU
- **Reward Model:** GPT-4o as AI judge
- **Alignment:** DPO (Direct Preference Optimization)

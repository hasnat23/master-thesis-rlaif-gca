# Progress Update — 2026-03-19: Phase 1 & Phase 2

**Date:** March 19, 2026  
**Thesis:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via GCA

---

## Phase 1: Environment Setup & Baseline (Completed)

### Tasks Completed
- [x] Set up Python environment with Hugging Face Transformers, TRL, PyTorch
- [x] Loaded and preprocessed CNN/DailyMail dataset
- [x] Fine-tuned FLAN-T5 baseline model for summarization
- [x] Evaluated baseline with ROUGE and BERTScore metrics

### Baseline Results (FLAN-T5 on CNN/DailyMail)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.3278 |
| ROUGE-2 | 0.1201 |
| ROUGE-L | 0.2341 |
| BERTScore F1 | 0.8612 |

---

## Phase 2: AI Judging Pipeline (Completed)

Note: Entries below describe the March 2026 prototype state. The current thesis implementation uses a fixed reward/factuality model as the main judge.

### Tasks Completed
- [x] Implemented holistic (document-level) AI judging (prototype initially used GPT-4o)
- [x] Implemented sentence-level AI judging pipeline
- [x] Generated preference pairs for CNN/DailyMail samples
- [x] Validated judging quality and consistency

### Notes
- Holistic judging: single preference score per summary
- Sentence-level judging: per-sentence scores aggregated via GCA
- Prototype used GPT-4o at that stage; current main pipeline uses a fixed reward/factuality model (`CogComp/bart-faithful-summary-detector`) and margin-gated `no_preference` handling

---

## Next Steps
- Phase 3: Generate full preference dataset
- Phase 4: Train DPO models (holistic vs GCA)

---

*Upload the progress update PDF document to this folder.*

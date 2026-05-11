# Biweekly Progress Update #1 — 2026-04-09

**Date:** April 9, 2026  
**Thesis:** Benchmarking Holistic vs Sentence-Level RLAIF for Factual Summarization via GCA

---

## Summary

This is the first formal biweekly progress update submitted to the supervising professor. It covers the status of all phases as of April 9, 2026.

Note: This report reflects the implementation status at that date. The current main judging pipeline has since been migrated to a fixed reward/factuality model (`CogComp/bart-faithful-summary-detector`) and no longer relies on OpenAI/GPT-4o for primary preference construction.

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Environment setup, dataset prep, FLAN-T5 baseline | Completed |
| Phase 2 | AI judging pipeline (holistic & sentence-level) | Completed |
| Phase 3 | Preference dataset generation | In Progress |
| Phase 4 | DPO / RLAIF training (holistic vs GCA) | Pending |
| Phase 5 | Evaluation (ROUGE, BERTScore, FactScore) | Pending |

---

## Work Done Since Last Update

- [x] Completed Phase 1 baseline evaluation (ROUGE + BERTScore)
- [x] Built and validated holistic AI judging pipeline (early prototype used GPT-4o; now superseded in the main pipeline by a fixed reward model)
- [x] Built sentence-level AI judging pipeline with per-sentence scoring
- [x] Started generating preference dataset (Phase 3)
- [x] Verified judging consistency across multiple samples

---

## Challenges & Blockers

- Early GPT-4o API rate limits affected the prototype; current main pipeline uses a local fixed reward model judge
- Sentence-level scoring aggregation strategy (GCA) requires further calibration

---

## Plan for Next 2 Weeks

- Complete preference dataset generation (Phase 3)
- Begin DPO training for holistic model (Phase 4)
- Set up training infrastructure on GPU

---

*Upload the biweekly progress update PDF document to this folder.*

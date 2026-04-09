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

## Tech Stack

- **Model:** FLAN-T5 (base/large)
- **Dataset:** CNN/DailyMail
- **Frameworks:** Hugging Face Transformers, TRL, PyTorch
- **Training:** Google Colab / GPU
- **Reward Model:** GPT-4o as AI judge
- **Alignment:** DPO (Direct Preference Optimization)

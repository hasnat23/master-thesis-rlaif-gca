# Code

This folder contains all code organized by thesis phase.

| Folder | Description |
|--------|-------------|
| `phase1_setup/` | Environment setup, dataset preprocessing, FLAN-T5 baseline training & evaluation |
| `phase2_judging/` | AI judging pipeline — holistic and sentence-level using GPT-4o |
| `phase3_preferences/` | Preference dataset generation and formatting for DPO |
| `phase4_dpo/` | DPO training for both holistic and GCA-based RLAIF models |
| `phase5_evaluation/` | Final evaluation: ROUGE, BERTScore, FactScore, comparison plots |

---

## Setup

```bash
pip install transformers trl datasets torch evaluate bert-score rouge-score openai
```

## Notes
- All notebooks are designed to run on Google Colab with GPU
- Hugging Face model checkpoints saved to `./checkpoints/`
- Dataset: CNN/DailyMail (via `datasets` library)

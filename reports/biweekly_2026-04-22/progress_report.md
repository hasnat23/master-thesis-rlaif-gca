# Biweekly Progress Report — April 22, 2026

## Period: April 9 – April 22, 2026

## Summary

Built the complete Milestone 1 codebase: reproducible pipeline scaffold for subset preparation, candidate generation, AI judging (holistic + sentence-level with GCA), evaluation metrics, and plotting. All code is modular, config-driven, and ready for MOGON execution.

## Completed This Period

### 1. Project Structure & Infrastructure
- Set up clean Python project with `src/` package layout
- Created YAML-based configuration system (configs/ directory)
- Built logging utilities: timestamped run IDs, metadata JSON tracking
- Defined data schema with dataclasses for all pipeline stages
- Created `.gitignore` and `requirements.txt`

### 2. Data Pipeline (scripts/01)
- `src/data/subset.py`: Deterministic selection of 200 CNN/DailyMail articles
- SHA256-based sample IDs for reproducibility
- Configurable length filters, seeded shuffling

### 3. Candidate Generation (scripts/02)
- `src/generation/candidates.py`: 4-bit quantized model loading with BitsAndBytes
- Left-padding tokenizer fix (known bug from Phase 1-2)
- Two temperature settings per article (low=0.7, high=1.0)
- Placeholder model name — actual 7B model TBD

### 4. AI Judging Pipeline (scripts/03)
- **Holistic judging** (`src/judging/holistic.py`): GPT-4o prompt with structured JSON response
- **Sentence-level judging** (`src/judging/sentence_level.py`): Per-sentence factual scoring
- **GCA aggregation** (`src/judging/gca.py`): `score = mean * (min/mean)^alpha` formula
- **Reliability controls** (`src/judging/reliability.py`): Confidence gating, swap consistency, rationale checks
- Mock test harness for validating prompts without API calls

### 5. Evaluation & Visualization (scripts/04-05)
- ROUGE-1/2/L and BERTScore F1 computation
- Bar chart generation comparing Summary A vs B
- Metrics saved as JSON for reproducibility

### 6. MOGON Slurm Jobs
- `slurm/smoke_test.sh`: 5-sample sanity check (30 min, 1 GPU)
- `slurm/generate_candidates.sh`: Full 200-sample generation (2h, A100)
- `slurm/judge_test.sh`: 20-pair judge validation (CPU-only)

## Next Steps (April 22 – May 6)

1. **Select final 7B model** — evaluate Mistral-7B-Instruct, Llama-2-7B-Chat, or similar on MOGON
2. **Run smoke test on MOGON** — validate environment, GPU access, model loading
3. **Generate full 200-sample candidates** — submit A100 job
4. **Run GPT-4o judging on 20-pair test set** — validate API integration and reliability filters
5. **Compute baseline metrics** — establish reference ROUGE/BERTScore numbers

## Blockers / Risks

- Model choice not yet finalized — need to test which 7B model fits MOGON's GPU memory with 4-bit quantization
- OpenAI API key needed for live judge testing
- MOGON queue times may delay GPU jobs

## Artifacts

- Full codebase: `src/`, `scripts/`, `configs/`, `slurm/`
- Data schema documentation: `src/data/schema.py`
- Run this locally: `python scripts/01_prepare_subset.py --config configs/subset.yaml`

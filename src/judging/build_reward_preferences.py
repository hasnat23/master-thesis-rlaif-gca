#!/usr/bin/env python3
"""
Build reward-model preference pairs from candidate summaries.

Primary judge: AlignScore (deterministic factual-consistency metric).
Fallback judge: sequence-classification model such as
CogComp/bart-faithful-summary-detector.

Two modes:
  holistic  — Score full summaries A and B; compare with margin.
  gca       — Split into sentences; score each; aggregate via GCA formula; compare.
  both      — Run both (default).

Outputs:
  data/preferences/holistic_reward_preferences_200.jsonl
  data/preferences/gca_reward_preferences_200.jsonl

Each output line is a JSON object with:
  - sample_id, article, reference_summary, summary_a, summary_b
  - scoring metadata (scores, labels, sentence details)
  - decision: "A" | "B" | "no_preference"
  - chosen / rejected: actual summary text (null when no_preference)

Usage:
    python src/judging/build_reward_preferences.py \\
        --candidates data/candidates/candidates_200.jsonl \\
        --output-dir data/preferences \\
        --model-name CogComp/bart-faithful-summary-detector \\
        --margin 0.05 \\
        --max-samples 200 \\
        --mode both
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# allow running as `python src/judging/build_reward_preferences.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.schema import load_jsonl
from src.judging.gca import aggregate_sentence_scores
from src.judging.reward_model_judge import RewardModelJudge
from src.judging.sentence_level import split_into_sentences
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_run_id, save_run_metadata, setup_logger


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def make_decision(score_a: float, score_b: float, margin: float) -> str:
    """
    Compare two scores with a margin guard.

    Returns "A" if A wins clearly, "B" if B wins clearly, "no_preference" otherwise.
    The margin prevents forcing a choice when the difference is noise-level.
    NOTE: Decision is based purely on faithfulness scores, NOT sentence count.
    """
    diff = score_a - score_b
    if diff > margin:
        return "A"
    if diff < -margin:
        return "B"
    return "no_preference"


# ---------------------------------------------------------------------------
# Holistic judging
# ---------------------------------------------------------------------------

def judge_holistic_pair(pair: dict, judge: RewardModelJudge, margin: float) -> dict:
    """
    Score full summaries A and B against the article.

    Returns a preference record ready for DPO consumption.
    """
    article = pair["article"]
    sa = pair.get("summary_a", "")
    sb = pair.get("summary_b", "")

    result_a = judge.score(article, sa)
    result_b = judge.score(article, sb)

    score_a = result_a["score"]
    score_b = result_b["score"]
    decision = make_decision(score_a, score_b, margin)

    chosen = rejected = None
    if decision == "A":
        chosen, rejected = sa, sb
    elif decision == "B":
        chosen, rejected = sb, sa

    return {
        "sample_id": pair.get("sample_id", ""),
        "article": article,
        "reference_summary": pair.get("reference_summary", ""),
        "summary_a": sa,
        "summary_b": sb,
        "score_a": round(score_a, 6),
        "score_b": round(score_b, 6),
        "score_diff": round(abs(score_a - score_b), 6),
        "decision": decision,
        "chosen": chosen,
        "rejected": rejected,
        "judge_method": "holistic",
        "judge_model": judge.judge_name,
        "margin": margin,
        "label_a": result_a["label"],
        "label_b": result_b["label"],
        "metadata_a": result_a["metadata"],
        "metadata_b": result_b["metadata"],
    }


# ---------------------------------------------------------------------------
# GCA judging
# ---------------------------------------------------------------------------

_LOW_FAITH_THRESHOLD = 0.5  # sentence score below this counts as low-faithfulness


def judge_gca_pair(
    pair: dict,
    judge: RewardModelJudge,
    margin: float,
    alpha: float = 0.5,
) -> dict:
    """
    Split summaries into sentences; score each sentence; aggregate via GCA.

    GCA formula (from thesis proposal):
        gca = mean(scores) * (min(scores) / mean(scores)) ** alpha

    alpha=0.5 penalises summaries with at least one low-faithfulness sentence
    without ignoring the overall quality.  A longer summary only wins if it
    contains more *faithful* content — length alone does not determine the winner.

    Returns a preference record ready for DPO consumption.
    """
    article = pair["article"]
    sa = pair.get("summary_a", "")
    sb = pair.get("summary_b", "")

    sentences_a = split_into_sentences(sa)
    sentences_b = split_into_sentences(sb)

    results_a = judge.score_sentences(article, sentences_a) if sentences_a else []
    results_b = judge.score_sentences(article, sentences_b) if sentences_b else []

    scores_a = [r["score"] for r in results_a]
    scores_b = [r["score"] for r in results_b]

    gca_a = aggregate_sentence_scores(scores_a, alpha=alpha) if scores_a else 0.0
    gca_b = aggregate_sentence_scores(scores_b, alpha=alpha) if scores_b else 0.0

    decision = make_decision(gca_a, gca_b, margin)

    chosen = rejected = None
    if decision == "A":
        chosen, rejected = sa, sb
    elif decision == "B":
        chosen, rejected = sb, sa

    def sent_detail(sentences: list[str], results: list[dict]) -> list[dict]:
        return [
            {
                "index": i,
                "text": s,
                "score": r["score"],
                "label": r["label"],
            }
            for i, (s, r) in enumerate(zip(sentences, results))
        ]

    return {
        "sample_id": pair.get("sample_id", ""),
        "article": article,
        "reference_summary": pair.get("reference_summary", ""),
        "summary_a": sa,
        "summary_b": sb,
        "gca_score_a": round(gca_a, 6),
        "gca_score_b": round(gca_b, 6),
        "gca_score_diff": round(abs(gca_a - gca_b), 6),
        "mean_score_a": round(sum(scores_a) / len(scores_a), 6) if scores_a else 0.0,
        "mean_score_b": round(sum(scores_b) / len(scores_b), 6) if scores_b else 0.0,
        "min_score_a": round(min(scores_a), 6) if scores_a else 0.0,
        "min_score_b": round(min(scores_b), 6) if scores_b else 0.0,
        "n_sentences_a": len(sentences_a),
        "n_sentences_b": len(sentences_b),
        "n_low_faithfulness_a": sum(1 for s in scores_a if s < _LOW_FAITH_THRESHOLD),
        "n_low_faithfulness_b": sum(1 for s in scores_b if s < _LOW_FAITH_THRESHOLD),
        "decision": decision,
        "chosen": chosen,
        "rejected": rejected,
        "judge_method": "gca",
        "judge_model": judge.judge_name,
        "margin": margin,
        "alpha": alpha,
        "sentence_details_a": sent_detail(sentences_a, results_a),
        "sentence_details_b": sent_detail(sentences_b, results_b),
    }


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    pairs: list[dict],
    judge: RewardModelJudge,
    mode: str,
    output_dir: Path,
    margin: float,
    alpha: float,
    logger,
) -> dict[str, str]:
    """Run holistic and/or GCA judging and write output files. Returns artifact paths."""
    artifacts: dict[str, str] = {}

    # ---- Holistic --------------------------------------------------------
    if mode in ("holistic", "both"):
        out_path = output_dir / "holistic_reward_preferences_200.jsonl"
        logger.info("=== Holistic judging  →  %s ===", out_path)

        records: list[dict] = []
        for i, pair in enumerate(pairs):
            pid = pair.get("sample_id", f"pair_{i}")
            try:
                rec = judge_holistic_pair(pair, judge, margin)
                records.append(rec)
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        "  [%d/%d] %s  A=%.4f  B=%.4f  → %s",
                        i + 1, len(pairs), pid,
                        rec["score_a"], rec["score_b"], rec["decision"],
                    )
            except Exception as exc:
                logger.warning("  SKIP %s: %s", pid, exc)
                logger.debug(traceback.format_exc())

        _write_jsonl(records, out_path)
        n_pref = sum(1 for r in records if r["decision"] != "no_preference")
        n_tie = sum(1 for r in records if r["decision"] == "no_preference")
        logger.info(
            "Holistic done: %d/%d processed | %d preferences | %d ties → %s",
            len(records), len(pairs), n_pref, n_tie, out_path,
        )
        artifacts["holistic_preferences"] = str(out_path)

    # ---- GCA -------------------------------------------------------------
    if mode in ("gca", "both"):
        out_path = output_dir / "gca_reward_preferences_200.jsonl"
        logger.info("=== GCA judging  →  %s ===", out_path)

        records = []
        for i, pair in enumerate(pairs):
            pid = pair.get("sample_id", f"pair_{i}")
            try:
                rec = judge_gca_pair(pair, judge, margin, alpha=alpha)
                records.append(rec)
                if (i + 1) % 10 == 0 or i == 0:
                    logger.info(
                        "  [%d/%d] %s  GCA_A=%.4f  GCA_B=%.4f  sents=(%d,%d)  → %s",
                        i + 1, len(pairs), pid,
                        rec["gca_score_a"], rec["gca_score_b"],
                        rec["n_sentences_a"], rec["n_sentences_b"],
                        rec["decision"],
                    )
            except Exception as exc:
                logger.warning("  SKIP %s: %s", pid, exc)
                logger.debug(traceback.format_exc())

        _write_jsonl(records, out_path)
        n_pref = sum(1 for r in records if r["decision"] != "no_preference")
        n_tie = sum(1 for r in records if r["decision"] == "no_preference")
        logger.info(
            "GCA done: %d/%d processed | %d preferences | %d ties → %s",
            len(records), len(pairs), n_pref, n_tie, out_path,
        )
        artifacts["gca_preferences"] = str(out_path)

    return artifacts


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reward-model preference pairs (no OpenAI/LLM dependency).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        default="data/candidates/candidates_200.jsonl",
        help="Path to candidates JSONL (relative to project root).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/preferences",
        help="Directory for preference output files.",
    )
    parser.add_argument(
        "--model-name",
        default="yzha/AlignScore",
        help="Judge model identifier. For AlignScore, use the checkpoint repo ID; for the fallback classifier, use a Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--judge-backend",
        choices=["auto", "alignscore", "sequence_classification"],
        default="alignscore",
        help="Judge backend. AlignScore is the primary thesis judge; sequence_classification keeps the BART-style fallback.",
    )
    parser.add_argument(
        "--alignscore-backbone",
        default="roberta-base",
        help="Backbone model used by AlignScore.",
    )
    parser.add_argument(
        "--alignscore-ckpt",
        default=None,
        help="Optional local path to an AlignScore .ckpt file. If omitted, the checkpoint is downloaded from --model-name / --alignscore-filename.",
    )
    parser.add_argument(
        "--alignscore-filename",
        default="AlignScore-base.ckpt",
        help="Checkpoint filename inside the AlignScore repo when auto-downloading.",
    )
    parser.add_argument(
        "--alignscore-evaluation-mode",
        choices=["nli_sp", "nli", "bin_sp", "bin"],
        default="nli_sp",
        help="AlignScore evaluation mode.",
    )
    parser.add_argument(
        "--alignscore-batch-size",
        type=int,
        default=8,
        help="AlignScore inference batch size.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Minimum score difference to declare a preference (avoids forcing noisy pairs).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of candidate pairs to process.",
    )
    parser.add_argument(
        "--mode",
        choices=["holistic", "gca", "both"],
        default="both",
        help="Judging mode.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="GCA penalty exponent for the minimum sentence score.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device override: cpu / cuda / mps. Auto-detected if not set.",
    )
    args = parser.parse_args()

    run_id = get_run_id()
    logger = setup_logger("build_reward_preferences", run_id=run_id)

    candidates_path = PROJECT_ROOT / args.candidates
    output_dir = PROJECT_ROOT / args.output_dir

    if not candidates_path.exists():
        logger.error("Candidates file not found: %s", candidates_path)
        sys.exit(1)

    pairs = load_jsonl(str(candidates_path))
    if args.max_samples and len(pairs) > args.max_samples:
        pairs = pairs[: args.max_samples]
    logger.info(
        "Loaded %d candidate pairs from %s", len(pairs), candidates_path
    )
    logger.info(
        "Config: backend=%s  model=%s  margin=%.3f  alpha=%.2f  mode=%s  device=%s",
        args.judge_backend, args.model_name, args.margin, args.alpha, args.mode,
        args.device or "auto",
    )

    judge = RewardModelJudge(
        model_name=args.model_name,
        device=args.device,
        backend=args.judge_backend,
        alignscore_backbone=args.alignscore_backbone,
        alignscore_ckpt=args.alignscore_ckpt,
        alignscore_filename=args.alignscore_filename,
        alignscore_evaluation_mode=args.alignscore_evaluation_mode,
        alignscore_batch_size=args.alignscore_batch_size,
    )

    artifacts = run_pipeline(
        pairs=pairs,
        judge=judge,
        mode=args.mode,
        output_dir=output_dir,
        margin=args.margin,
        alpha=args.alpha,
        logger=logger,
    )

    save_run_metadata(
        run_id=run_id,
        script_name="build_reward_preferences",
        config={
            "model_name": args.model_name,
            "margin": args.margin,
            "max_samples": args.max_samples,
            "mode": args.mode,
            "alpha": args.alpha,
            "device": args.device,
        },
        artifacts=artifacts,
    )
    logger.info("All done.")


if __name__ == "__main__":
    main()

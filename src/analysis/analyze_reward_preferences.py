#!/usr/bin/env python3
"""
Analyze reward-model preference construction results.

Reads holistic and GCA preference JSONL files and produces:
  - reports/reward_model_judging_results.md   (human-readable report)
  - reports/reward_model_judging_summary.csv  (summary table)

Usage:
    python src/analysis/analyze_reward_preferences.py
    python src/analysis/analyze_reward_preferences.py \\
        --holistic data/preferences/holistic_reward_preferences_200.jsonl \\
        --gca      data/preferences/gca_reward_preferences_200.jsonl \\
        --reports-dir reports
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.schema import load_jsonl
from src.utils.config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return load_jsonl(str(path))


def _counts(records: list[dict]) -> Counter:
    return Counter(r.get("decision", "unknown") for r in records)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _compute_agreement(holistic: list[dict], gca: list[dict]) -> dict:
    """
    Compare decisions between holistic and GCA on shared sample IDs.
    Returns agreement stats and up to 5 disagreement examples.
    """
    h_map = {r["sample_id"]: r for r in holistic}
    g_map = {r["sample_id"]: r for r in gca}
    common = set(h_map) & set(g_map)

    agree = disagree = 0
    examples: list[dict] = []

    for sid in sorted(common):
        hd = h_map[sid]["decision"]
        gd = g_map[sid]["decision"]
        if hd == gd:
            agree += 1
        else:
            disagree += 1
            if len(examples) < 5:
                examples.append({
                    "sample_id": sid,
                    "holistic_decision": hd,
                    "gca_decision": gd,
                    "holistic_score_a": h_map[sid].get("score_a"),
                    "holistic_score_b": h_map[sid].get("score_b"),
                    "gca_score_a": g_map[sid].get("gca_score_a"),
                    "gca_score_b": g_map[sid].get("gca_score_b"),
                    "article_snippet": h_map[sid].get("article", "")[:200],
                    "summary_a_snippet": h_map[sid].get("summary_a", "")[:150],
                    "summary_b_snippet": h_map[sid].get("summary_b", "")[:150],
                })

    total = len(common)
    return {
        "total_common": total,
        "agree": agree,
        "disagree": disagree,
        "agreement_rate": agree / total if total else 0.0,
        "examples": examples,
    }


def _find_weak_sentences(gca_records: list[dict], n: int = 5) -> list[dict]:
    """Return up to n examples of sentences with score < 0.3."""
    results: list[dict] = []
    for r in gca_records:
        for side in ("a", "b"):
            for sent in r.get(f"sentence_details_{side}", []):
                if sent.get("score", 1.0) < 0.3:
                    results.append({
                        "sample_id": r["sample_id"],
                        "summary_side": side.upper(),
                        "sentence": sent.get("text", ""),
                        "score": sent.get("score"),
                        "gca_decision": r.get("decision"),
                    })
                    if len(results) >= n:
                        return results
    return results


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(holistic: list[dict], gca: list[dict]) -> str:
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    ln("# Reward-Model Preference Construction — Results Report")
    ln()
    ln("**Judge:** `yzha/AlignScore` (AlignScore-base, RoBERTa backbone)")
    ln("**Method:** AlignScore NLI-based faithfulness scorer (`nli_sp` mode) — no OpenAI/GPT calls.")
    ln()

    # ------------------------------------------------------------------
    # 1. Holistic
    # ------------------------------------------------------------------
    ln("## 1. Holistic Scoring")
    if not holistic:
        ln("*No holistic preference file found.*")
    else:
        h = _counts(holistic)
        n_pref = h["A"] + h["B"]
        n_tie = h["no_preference"]
        usable_pct = n_pref / len(holistic) * 100
        sa = [r["score_a"] for r in holistic]
        sb = [r["score_b"] for r in holistic]
        margin = holistic[0].get("margin", "?")

        ln(f"| Metric | Value |")
        ln(f"|--------|-------|")
        ln(f"| Total samples | {len(holistic)} |")
        ln(f"| Preference pairs (A wins) | {h['A']} |")
        ln(f"| Preference pairs (B wins) | {h['B']} |")
        ln(f"| Total usable for DPO | **{n_pref}** ({usable_pct:.1f}%) |")
        ln(f"| Ties / no_preference | {n_tie} |")
        ln(f"| Mean faithfulness score (A) | {_mean(sa):.4f} |")
        ln(f"| Mean faithfulness score (B) | {_mean(sb):.4f} |")
        ln(f"| Margin threshold | {margin} |")
    ln()

    # ------------------------------------------------------------------
    # 2. GCA
    # ------------------------------------------------------------------
    ln("## 2. Granular (GCA) Scoring")
    if not gca:
        ln("*No GCA preference file found.*")
    else:
        g = _counts(gca)
        n_pref = g["A"] + g["B"]
        n_tie = g["no_preference"]
        usable_pct = n_pref / len(gca) * 100
        ga = [r["gca_score_a"] for r in gca]
        gb = [r["gca_score_b"] for r in gca]
        avg_sents_a = _mean([r["n_sentences_a"] for r in gca])
        avg_sents_b = _mean([r["n_sentences_b"] for r in gca])
        low_a = sum(r.get("n_low_faithfulness_a", 0) for r in gca)
        low_b = sum(r.get("n_low_faithfulness_b", 0) for r in gca)
        margin = gca[0].get("margin", "?")
        alpha = gca[0].get("alpha", "?")

        ln(f"| Metric | Value |")
        ln(f"|--------|-------|")
        ln(f"| Total samples | {len(gca)} |")
        ln(f"| Preference pairs (A wins) | {g['A']} |")
        ln(f"| Preference pairs (B wins) | {g['B']} |")
        ln(f"| Total usable for DPO | **{n_pref}** ({usable_pct:.1f}%) |")
        ln(f"| Ties / no_preference | {n_tie} |")
        ln(f"| Mean GCA score (A) | {_mean(ga):.4f} |")
        ln(f"| Mean GCA score (B) | {_mean(gb):.4f} |")
        ln(f"| Avg sentences per summary (A / B) | {avg_sents_a:.1f} / {avg_sents_b:.1f} |")
        ln(f"| Low-faithfulness sentences total (A / B) | {low_a} / {low_b} |")
        ln(f"| Margin threshold | {margin} |")
        ln(f"| GCA alpha | {alpha} |")
    ln()

    # ------------------------------------------------------------------
    # 3. Agreement
    # ------------------------------------------------------------------
    ln("## 3. Agreement Between Holistic and GCA Decisions")
    if not holistic or not gca:
        ln("*Cannot compute — one or both files are missing.*")
    else:
        ag = _compute_agreement(holistic, gca)
        rate_pct = ag["agreement_rate"] * 100
        ln(f"| Metric | Value |")
        ln(f"|--------|-------|")
        ln(f"| Shared samples | {ag['total_common']} |")
        ln(f"| Agree | {ag['agree']} ({rate_pct:.1f}%) |")
        ln(f"| Disagree | {ag['disagree']} |")
        ln()

        if ag["examples"]:
            ln("### Disagreement Examples")
            ln()
            for ex in ag["examples"]:
                ln(f"**`{ex['sample_id']}`**")
                ln(f"- Holistic: `{ex['holistic_decision']}`"
                   f"  (A={ex['holistic_score_a']:.4f}, B={ex['holistic_score_b']:.4f})")
                ln(f"- GCA:      `{ex['gca_decision']}`"
                   f"  (A={ex['gca_score_a']:.4f}, B={ex['gca_score_b']:.4f})")
                ln(f"- Article snippet: *{ex['article_snippet']}...*")
                ln(f"- Summary A: *{ex['summary_a_snippet']}...*")
                ln(f"- Summary B: *{ex['summary_b_snippet']}...*")
                ln()
    ln()

    # ------------------------------------------------------------------
    # 4. Weak sentences
    # ------------------------------------------------------------------
    ln("## 4. GCA: Detected Low-Faithfulness Sentences (score < 0.3)")
    if not gca:
        ln("*No GCA file available.*")
    else:
        weak = _find_weak_sentences(gca)
        if not weak:
            ln("*No sentences with score < 0.3 detected.*")
        else:
            for w in weak:
                ln(
                    f"- `{w['sample_id']}` | Summary **{w['summary_side']}** | "
                    f"score={w['score']:.4f} | gca_decision={w['gca_decision']}"
                )
                ln(f"  > {w['sentence']}")
    ln()

    # ------------------------------------------------------------------
    # 5. Interpretation
    # ------------------------------------------------------------------
    ln("## 5. Interpretation")
    if holistic and gca:
        h_pref = sum(1 for r in holistic if r["decision"] != "no_preference")
        g_pref = sum(1 for r in gca if r["decision"] != "no_preference")
        ag = _compute_agreement(holistic, gca)

        ln(f"- Holistic judging yielded **{h_pref}** usable DPO pairs out of {len(holistic)} samples.")
        ln(f"- GCA judging yielded **{g_pref}** usable DPO pairs out of {len(gca)} samples.")
        ln(
            f"- The two methods agree on **{ag['agreement_rate']*100:.1f}%** of shared decisions."
        )
        ln()
        ln(
            "Disagreements are expected and are the central interest of the thesis: "
            "holistic scoring collapses all sentence evidence into a single number, while "
            "GCA can detect a single low-faithfulness sentence that drags the aggregate score "
            "below the margin even when the mean score would be similar. "
            "These cases demonstrate the value of sentence-level localisation."
        )
    ln()
    ln("---")
    ln("*Generated by `src/analysis/analyze_reward_preferences.py`*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def build_csv_rows(holistic: list[dict], gca: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for label, records, score_key_a, score_key_b in [
        ("holistic", holistic, "score_a", "score_b"),
        ("gca", gca, "gca_score_a", "gca_score_b"),
    ]:
        if not records:
            continue
        c = _counts(records)
        n_pref = c["A"] + c["B"]
        rows.append({
            "method": label,
            "total": len(records),
            "pref_A": c["A"],
            "pref_B": c["B"],
            "total_preferences": n_pref,
            "no_preference": c["no_preference"],
            "usable_pct": round(n_pref / len(records) * 100, 1),
            "mean_score_a": round(_mean([r.get(score_key_a, 0) for r in records]), 4),
            "mean_score_b": round(_mean([r.get(score_key_b, 0) for r in records]), 4),
        })

    if holistic and gca:
        ag = _compute_agreement(holistic, gca)
        rows.append({
            "method": "agreement",
            "total": ag["total_common"],
            "pref_A": ag["agree"],
            "pref_B": "—",
            "total_preferences": ag["disagree"],
            "no_preference": "—",
            "usable_pct": round(ag["agreement_rate"] * 100, 1),
            "mean_score_a": "—",
            "mean_score_b": "—",
        })

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse reward-model preference files and produce a thesis-ready report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--holistic",
        default="data/preferences/holistic_reward_preferences_200.jsonl",
    )
    parser.add_argument(
        "--gca",
        default="data/preferences/gca_reward_preferences_200.jsonl",
    )
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    root = PROJECT_ROOT
    holistic = _load(root / args.holistic)
    gca = _load(root / args.gca)

    if not holistic and not gca:
        print(
            "ERROR: neither preference file found. "
            "Run build_reward_preferences.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loaded: {len(holistic)} holistic, {len(gca)} GCA records.")

    reports_dir = root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    report = build_report(holistic, gca)
    report_path = reports_dir / "reward_model_judging_results.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report  →  {report_path}")

    # CSV summary
    rows = build_csv_rows(holistic, gca)
    if rows:
        csv_path = reports_dir / "reward_model_judging_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV     →  {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Statistics for the ground-truth RM-precision evaluation.

Takes the per-seed success rates written by evaluate_ground_truth.py and
applies the same methodology used throughout the thesis for the seed
campaigns (analysis/aggregate_campaigns.py, analysis/effect_sizes_and_equivalence.py):
paired per-seed gap, run-level Wilcoxon signed-rank test, bootstrap 95% CI,
Cohen's d, rank-biserial correlation. The seed is the unit of replication
here exactly as it is there -- each seed is an independently and
deterministically trained pair of models (src/reward_model/train.py is
seeded the same way as the corrected _kfold_cv path).

Usage:
    python analysis/ground_truth_stats.py --in reports/campaigns/ground_truth_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parent.parent


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    import random
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return lo, hi


def cohens_d_paired(diffs):
    return statistics.mean(diffs) / statistics.stdev(diffs)


def rank_biserial(diffs):
    nonzero = [d for d in diffs if d != 0]
    ranks = stats.rankdata([abs(d) for d in nonzero])
    pos = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    neg = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    return (pos - neg) / (pos + neg) if (pos + neg) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default="reports/campaigns/ground_truth_eval.json")
    ap.add_argument("--out", default="reports/campaigns/ground_truth_stats.json")
    args = ap.parse_args()

    data = json.loads((REPO / args.infile).read_text())
    hol = data["results"]["holistic"]
    gca = data["results"]["gca"]

    common_seeds = sorted(set(int(s) for s in hol) & set(int(s) for s in gca))
    print(f"Seeds with both conditions: {len(common_seeds)} -> {common_seeds}")

    hol_rates = [hol[str(s)]["success_rate"] for s in common_seeds]
    gca_rates = [gca[str(s)]["success_rate"] for s in common_seeds]
    diffs_pp = [(g - h) * 100.0 for g, h in zip(gca_rates, hol_rates)]

    mean_hol = statistics.mean(hol_rates)
    mean_gca = statistics.mean(gca_rates)
    mean_gap = statistics.mean(diffs_pp)
    sd_gap = statistics.stdev(diffs_pp) if len(diffs_pp) > 1 else float("nan")

    lo, hi = bootstrap_ci(diffs_pp)

    if len(diffs_pp) >= 2 and any(d != 0 for d in diffs_pp):
        w = stats.wilcoxon(diffs_pp, alternative="two-sided")
        p_two_sided = float(w.pvalue)
        w_greater = stats.wilcoxon(diffs_pp, alternative="greater")
        p_greater = float(w_greater.pvalue)
    else:
        p_two_sided = p_greater = float("nan")

    d = cohens_d_paired(diffs_pp) if sd_gap == sd_gap and sd_gap != 0 else float("nan")
    r_rb = rank_biserial(diffs_pp)

    gca_ahead = sum(1 for x in diffs_pp if x > 0)

    print(f"\nGround-truth ranking success rate")
    print(f"  Holistic RM: mean = {mean_hol:.4f}")
    print(f"  GCA RM:      mean = {mean_gca:.4f}")
    print(f"  Mean gap (GCA - Holistic): {mean_gap:+.2f} pp  (SD {sd_gap:.2f} pp)")
    print(f"  GCA ahead: {gca_ahead}/{len(common_seeds)} seeds")
    print(f"  Bootstrap 95% CI: [{lo:+.2f}, {hi:+.2f}] pp")
    print(f"  Wilcoxon two-sided p: {p_two_sided:.4g}")
    print(f"  Wilcoxon one-sided (GCA > Holistic) p: {p_greater:.4g}")
    print(f"  Cohen's d: {d:+.2f}")
    print(f"  Rank-biserial r: {r_rb:+.2f}")

    out = {
        "n_seeds": len(common_seeds),
        "seeds": common_seeds,
        "mean_holistic_success_rate": mean_hol,
        "mean_gca_success_rate": mean_gca,
        "mean_gap_pp": mean_gap,
        "sd_gap_pp": sd_gap,
        "gca_ahead": gca_ahead,
        "bootstrap_ci_pp": [lo, hi],
        "wilcoxon_two_sided_p": p_two_sided,
        "wilcoxon_one_sided_greater_p": p_greater,
        "cohens_d": d,
        "rank_biserial": r_rb,
        "per_seed_gaps_pp": diffs_pp,
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

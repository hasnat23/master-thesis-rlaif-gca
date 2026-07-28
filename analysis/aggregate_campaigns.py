#!/usr/bin/env python3
"""
Aggregate the extended seed campaign and the truncation ablation.

Reads the per-run rm_training_summary.json files written by
src/reward_model/run_training.py and produces:

  * a run-level comparison of GCA against holistic preference construction,
    tested with the Wilcoxon signed-rank test on paired per-run means;
  * a bootstrap confidence interval for the mean run-level gap;
  * a per-configuration table for the truncation ablation;
  * LaTeX table bodies and pgfplots .dat files for direct inclusion in the thesis.

The unit of replication is the run, not the cross-validation fold. Folds within a
run share a preference set, a shuffle and a training recipe, so pooling folds
across runs would understate the variance and overstate the significance. Fold
numbers are reported here only as a descriptive spread.

Usage:
    python analysis/aggregate_campaigns.py \
        --seed-campaign outputs/seed_campaign \
        --truncation    outputs/truncation_ablation \
        --out-dir       reports/campaigns
"""

import argparse
import json
import math
import random
import statistics
from itertools import product
from pathlib import Path

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
except ImportError:                                    # pragma: no cover
    _scipy_wilcoxon = None


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def _exact_signed_rank_p(diffs):
    """
    Exact two-sided Wilcoxon signed-rank p-value, enumerating all 2^n sign
    assignments. Used when scipy is unavailable and as a cross-check for small n,
    where scipy's normal approximation would be inappropriate anyway.

    Zero differences are dropped, following the standard Wilcoxon convention.
    """
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return float("nan"), 0
    if n > 20:                     # 2^20 enumerations is already ~1M; stop there
        return None, n

    ranks = _ranks([abs(x) for x in d])
    w_obs = sum(r for x, r in zip(d, ranks) if x > 0)
    total = sum(ranks)

    count = 0
    for signs in product((0, 1), repeat=n):
        w = sum(r for s, r in zip(signs, ranks) if s)
        # two-sided: at least as extreme as the observed statistic in either tail
        if min(w, total - w) <= min(w_obs, total - w_obs):
            count += 1
    return count / 2 ** n, n


def _ranks(values):
    """Ranks with ties averaged."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean, resampling runs."""
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        means.append(sum(rng.choice(values) for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _load_summaries(root: Path):
    """Yield (path, parsed json) for every rm_training_summary.json under root."""
    if not root.exists():
        return
    for p in sorted(root.rglob("rm_training_summary.json")):
        try:
            yield p, json.loads(p.read_text())
        except json.JSONDecodeError as exc:
            print(f"  ! skipping unreadable {p}: {exc}")


def _run_record(path: Path, summary: dict):
    """Extract the paired holistic/GCA means from one run summary."""
    conds = summary.get("conditions", {})
    hol = conds.get("holistic", {})
    gca = conds.get("gca", {})
    if "mean_val_acc" not in hol or "mean_val_acc" not in gca:
        return None
    return {
        "path": str(path),
        "seed": summary.get("seed"),
        "backbone": summary.get("backbone"),
        "max_length": summary.get("max_length"),
        "max_article_chars": summary.get("max_article_chars"),
        "deterministic": summary.get("deterministic_seeding", False),
        "holistic": hol["mean_val_acc"],
        "gca": gca["mean_val_acc"],
        "gap": gca["mean_val_acc"] - hol["mean_val_acc"],
        "holistic_folds": hol.get("fold_accs", []),
        "gca_folds": gca.get("fold_accs", []),
    }


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

def analyse_seed_campaign(runs, out_dir: Path):
    runs = sorted(runs, key=lambda r: (r["seed"] is None, r["seed"]))
    if not runs:
        print("No seed-campaign runs found.")
        return None

    gaps = [r["gap"] for r in runs]
    n = len(runs)
    n_pos = sum(1 for g in gaps if g > 0)
    n_neg = sum(1 for g in gaps if g < 0)

    print(f"\n{'='*74}\nEXTENDED SEED CAMPAIGN  ({n} runs)\n{'='*74}")
    print(f"{'seed':>6} {'holistic':>10} {'gca':>10} {'gap (pp)':>10}  det")
    for r in runs:
        print(f"{str(r['seed']):>6} {r['holistic']:>10.4f} {r['gca']:>10.4f} "
              f"{100*r['gap']:>10.2f}  {'y' if r['deterministic'] else 'n'}")

    mean_gap = statistics.mean(gaps)
    med_gap = statistics.median(gaps)
    sd_gap = statistics.stdev(gaps) if n > 1 else float("nan")
    lo, hi = _bootstrap_ci(gaps)

    print(f"\nGCA ahead in {n_pos}/{n} runs (behind in {n_neg})")
    print(f"mean gap   {100*mean_gap:+.3f} pp   median {100*med_gap:+.3f} pp   "
          f"sd {100*sd_gap:.3f} pp")
    print(f"bootstrap 95% CI on mean gap  [{100*lo:+.3f}, {100*hi:+.3f}] pp")

    result = {
        "n_runs": n, "n_gca_ahead": n_pos, "n_holistic_ahead": n_neg,
        "mean_gap": mean_gap, "median_gap": med_gap, "sd_gap": sd_gap,
        "boot_ci_lo": lo, "boot_ci_hi": hi,
    }

    if _scipy_wilcoxon is not None and n >= 2 and any(g != 0 for g in gaps):
        w2 = _scipy_wilcoxon(gaps, alternative="two-sided")
        w1 = _scipy_wilcoxon(gaps, alternative="greater")
        print(f"Wilcoxon signed-rank (run level)  two-sided p = {w2.pvalue:.5f}"
              f"   one-sided p = {w1.pvalue:.5f}")
        result["wilcoxon_two_sided_p"] = float(w2.pvalue)
        result["wilcoxon_one_sided_p"] = float(w1.pvalue)

    exact_p, n_nonzero = _exact_signed_rank_p(gaps)
    if exact_p is not None:
        print(f"exact enumeration cross-check    two-sided p = {exact_p:.5f}"
              f"  (n non-zero = {n_nonzero})")
        result["exact_two_sided_p"] = exact_p

    # smallest attainable p, so an underpowered design is visible as such
    if n_nonzero and n_nonzero <= 20:
        floor = 2 / 2 ** n_nonzero
        print(f"smallest attainable two-sided p at n={n_nonzero}: {floor:.5f}")
        result["min_attainable_p"] = floor

    _write_seed_outputs(runs, result, out_dir)
    return result


def _write_seed_outputs(runs, result, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    dat = out_dir / "seed_campaign.dat"
    with dat.open("w") as fh:
        fh.write("seed holistic gca gap_pp\n")
        for r in runs:
            fh.write(f"{r['seed']} {r['holistic']:.6f} {r['gca']:.6f} "
                     f"{100*r['gap']:.4f}\n")

    tex = out_dir / "seed_campaign_rows.tex"
    with tex.open("w") as fh:
        for r in runs:
            fh.write(f"{r['seed']} & {r['holistic']:.4f} & {r['gca']:.4f} & "
                     f"{100*r['gap']:+.2f} \\\\\n")

    (out_dir / "seed_campaign_summary.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {dat}\n      {tex}\n      {out_dir/'seed_campaign_summary.json'}")


def analyse_truncation(runs, out_dir: Path):
    if not runs:
        print("\nNo truncation-ablation runs found.")
        return None

    groups = {}
    for r in runs:
        key = (r["backbone"], r["max_length"], r["max_article_chars"])
        groups.setdefault(key, []).append(r)

    print(f"\n{'='*74}\nTRUNCATION ABLATION\n{'='*74}")
    print(f"{'backbone':<28}{'len':>6}{'chars':>7}{'runs':>6}"
          f"{'holistic':>11}{'gca':>9}{'gap pp':>9}")

    rows = []
    for key in sorted(groups, key=lambda k: (str(k[0]), k[1] or 0, k[2] or 0)):
        rs = groups[key]
        hol = statistics.mean(r["holistic"] for r in rs)
        gca = statistics.mean(r["gca"] for r in rs)
        backbone, mlen, mchars = key
        short = (backbone or "?").split("/")[-1]
        print(f"{short:<28}{str(mlen):>6}{str(mchars):>7}{len(rs):>6}"
              f"{hol:>11.4f}{gca:>9.4f}{100*(gca-hol):>9.2f}")
        rows.append({
            "backbone": backbone, "max_length": mlen, "max_article_chars": mchars,
            "n_runs": len(rs), "holistic": hol, "gca": gca, "gap": gca - hol,
            "holistic_sd": statistics.stdev([r["holistic"] for r in rs]) if len(rs) > 1 else 0.0,
            "gca_sd": statistics.stdev([r["gca"] for r in rs]) if len(rs) > 1 else 0.0,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    tex = out_dir / "truncation_rows.tex"
    with tex.open("w") as fh:
        for r in rows:
            short = (r["backbone"] or "?").split("/")[-1]
            fh.write(f"\\texttt{{{short}}} & {r['max_length']} & "
                     f"{r['max_article_chars']} & {r['n_runs']} & "
                     f"{r['holistic']:.4f} & {r['gca']:.4f} & "
                     f"{100*r['gap']:+.2f} \\\\\n")

    (out_dir / "truncation_summary.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {tex}\n      {out_dir/'truncation_summary.json'}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-campaign", default="outputs/seed_campaign")
    ap.add_argument("--truncation", default="outputs/truncation_ablation")
    ap.add_argument("--out-dir", default="reports/campaigns")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    seed_runs = []
    for path, summary in _load_summaries(Path(args.seed_campaign)):
        rec = _run_record(path, summary)
        if rec:
            seed_runs.append(rec)
        else:
            print(f"  ! {path} has no paired holistic/GCA k-fold means; skipped")

    trunc_runs = []
    for path, summary in _load_summaries(Path(args.truncation)):
        rec = _run_record(path, summary)
        if rec:
            trunc_runs.append(rec)

    analyse_seed_campaign(seed_runs, out_dir)
    analyse_truncation(trunc_runs, out_dir)


if __name__ == "__main__":
    main()

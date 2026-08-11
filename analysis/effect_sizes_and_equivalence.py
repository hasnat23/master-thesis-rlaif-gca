#!/usr/bin/env python3
"""Effect sizes, equivalence tests, and power analysis for the seed campaigns.

The main aggregation script (aggregate_campaigns.py) reports the mean gap, a
bootstrap interval, and a Wilcoxon p-value for each campaign. Those answer
"is the gap distinguishable from zero?" but leave two questions open that the
thesis makes claims about:

  1. How large is the effect in standardised terms? A gap of +3.95pp means
     little without knowing the run-to-run spread it sits against.
  2. At n=5,000 and n=10,000 the Wilcoxon test does not reject the null. A
     non-significant result is not by itself evidence of absence, so the claim
     that the advantage is "confirmed absent" needs an equivalence test, which
     asks the opposite question: can effects larger than some bound be ruled
     out?

This script computes both, plus the minimum effect the 20-run design could
have detected, and writes a JSON summary alongside a LaTeX-ready table.

Usage:
    python analysis/effect_sizes_and_equivalence.py
"""

from __future__ import annotations

import glob
import json
import math
import statistics
from pathlib import Path

from scipy import stats

REPO = Path(__file__).resolve().parent.parent

CAMPAIGNS = {
    "1000": "outputs/seed_campaign",
    "5000": "outputs/seed_campaign_5000",
    "10000": "outputs/seed_campaign_10000",
}

# Smallest effect size of interest, in percentage points. The n=1,000 campaign
# found +3.95pp; a bound of 1pp is roughly a quarter of that and is well below
# the run-to-run spread, so an effect smaller than this would not be
# practically meaningful even if real.
SESOI_PP = 1.0


def load_gaps(campaign_dir: str) -> list[float]:
    """Return per-run (GCA - Holistic) gaps in percentage points, seed-ordered."""
    runs = []
    for path in glob.glob(str(REPO / campaign_dir / "seed_*" / "rm_training_summary.json")):
        with open(path) as fh:
            data = json.load(fh)
        hol = data["conditions"]["holistic"]["mean_val_acc"]
        gca = data["conditions"]["gca"]["mean_val_acc"]
        runs.append((data["seed"], (gca - hol) * 100.0))
    runs.sort()
    return [gap for _, gap in runs]


def cohens_d_paired(gaps: list[float]) -> float:
    """Paired Cohen's d: mean difference over the SD of the differences."""
    return statistics.mean(gaps) / statistics.stdev(gaps)


def rank_biserial(gaps: list[float]) -> float:
    """Matched-pairs rank-biserial correlation, the Wilcoxon effect size.

    Equals (sum of positive ranks - sum of negative ranks) / total rank sum,
    so it runs from -1 (every run favours holistic) to +1 (every run favours
    GCA), independent of the size of the gaps.
    """
    nonzero = [g for g in gaps if g != 0]
    ranks = stats.rankdata([abs(g) for g in nonzero])
    pos = sum(r for r, g in zip(ranks, nonzero) if g > 0)
    neg = sum(r for r, g in zip(ranks, nonzero) if g < 0)
    return float((pos - neg) / (pos + neg))


def tost(gaps: list[float], bound_pp: float) -> dict:
    """Two one-sided t-tests for equivalence within +/- bound_pp.

    Rejecting both one-sided nulls means the true effect lies inside the
    equivalence bounds: positive evidence of a negligible effect, rather than
    mere failure to detect one.
    """
    n = len(gaps)
    mean = statistics.mean(gaps)
    sd = statistics.stdev(gaps)
    se = sd / math.sqrt(n)
    df = n - 1

    # H0_lower: effect <= -bound   (want to reject, i.e. effect is above -bound)
    t_lower = (mean + bound_pp) / se
    p_lower = stats.t.sf(t_lower, df)
    # H0_upper: effect >= +bound   (want to reject, i.e. effect is below +bound)
    t_upper = (mean - bound_pp) / se
    p_upper = stats.t.cdf(t_upper, df)

    p_tost = float(max(p_lower, p_upper))
    return {
        "bound_pp": float(bound_pp),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": p_tost,
        "equivalent_at_05": bool(p_tost < 0.05),
    }


def tightest_equivalence_bound(gaps: list[float], alpha: float = 0.05) -> float:
    """Smallest symmetric bound at which equivalence still holds at alpha.

    This is the strongest statement the data support: "any effect larger than
    this can be ruled out." Found by bisection on the bound.
    """
    lo, hi = 0.0, 20.0
    if not tost(gaps, hi)["p_tost"] < alpha:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if tost(gaps, mid)["p_tost"] < alpha:
            hi = mid
        else:
            lo = mid
    return float(hi)


def min_detectable_effect(sd_pp: float, n: int, alpha: float = 0.05,
                          power: float = 0.80) -> float:
    """Smallest true effect a paired t-test at this n and SD would detect.

    Uses the normal approximation, which is close enough at n=20 for the
    purpose of stating what the design was capable of finding.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return float((z_alpha + z_beta) * sd_pp / math.sqrt(n))


def main() -> None:
    results = {}
    print(f"{'n':>7} {'runs':>5} {'mean':>8} {'SD':>7} {'d':>7} {'r_rb':>7} "
          f"{'TOST p':>9} {'tightest':>9} {'MDE':>7}")
    print("-" * 76)

    for label, campaign_dir in CAMPAIGNS.items():
        gaps = load_gaps(campaign_dir)
        if not gaps:
            print(f"{label:>7}  no runs found")
            continue

        mean = statistics.mean(gaps)
        sd = statistics.stdev(gaps)
        d = cohens_d_paired(gaps)
        r_rb = rank_biserial(gaps)
        eq = tost(gaps, SESOI_PP)
        tightest = tightest_equivalence_bound(gaps)
        mde = min_detectable_effect(sd, len(gaps))

        results[label] = {
            "n_runs": len(gaps),
            "gaps_pp": gaps,
            "mean_pp": mean,
            "sd_pp": sd,
            "cohens_d": d,
            "rank_biserial": r_rb,
            "tost_at_sesoi": eq,
            "tightest_equivalence_bound_pp": tightest,
            "min_detectable_effect_pp": mde,
        }

        print(f"{label:>7} {len(gaps):>5} {mean:>+8.3f} {sd:>7.3f} {d:>+7.2f} "
              f"{r_rb:>+7.2f} {eq['p_tost']:>9.4f} {tightest:>9.3f} {mde:>7.3f}")

    out = REPO / "reports" / "campaigns" / "effect_sizes_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"sesoi_pp": SESOI_PP, "campaigns": results}, fh, indent=2)
    print(f"\nwrote {out.relative_to(REPO)}")

    print("\nReadings:")
    print(f"  d      = paired Cohen's d (mean gap / SD of gaps)")
    print(f"  r_rb   = matched-pairs rank-biserial correlation")
    print(f"  TOST p = equivalence test at +/-{SESOI_PP}pp; <0.05 means the")
    print(f"           effect is positively shown to be negligible")
    print(f"  tightest = smallest bound at which equivalence still holds (p<0.05)")
    print(f"  MDE    = smallest effect detectable at 80% power, given this SD")


if __name__ == "__main__":
    main()

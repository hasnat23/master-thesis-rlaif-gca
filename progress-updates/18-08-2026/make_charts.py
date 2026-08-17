"""Generates the results charts for the 18 August progress update.

Reads directly from the committed evaluation JSON files under
reports/campaigns/, so the charts cannot drift out of sync with the numbers
written in the README.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = ROOT / "reports" / "campaigns"

HOL_COLOR = "#5B8FD4"
GCA_COLOR = "#E8A24C"


def load_eval(name):
    d = json.loads((CAMPAIGNS / name).read_text())
    hol = [v["success_rate"] for v in d["results"]["holistic"].values()]
    gca = [v["success_rate"] for v in d["results"]["gca"].values()]
    return hol, gca


conditions = [
    ("Same-article\n(gap ~0.04)", "ground_truth_eval.json"),
    ("Top/bottom 25%\n(gap >=0.36)", "biased_ground_truth_eval.json"),
    ("Top/bottom 10%\n(gap >=0.66)", "biased_top10_eval.json"),
    ("Top/bottom 5%,\nall-pairs (gap >=0.79)", "biased_top5_allpairs_eval.json"),
]

labels, hol_means, gca_means, hol_sd, gca_sd, gaps_pp, wilcoxon_p = [], [], [], [], [], [], []
for label, fname in conditions:
    hol, gca = load_eval(fname)
    labels.append(label)
    hol_means.append(statistics.mean(hol))
    gca_means.append(statistics.mean(gca))
    hol_sd.append(statistics.stdev(hol))
    gca_sd.append(statistics.stdev(gca))
    gaps_pp.append((statistics.mean(gca) - statistics.mean(hol)) * 100)

# ---------------------------------------------------------------------------
# Chart 1: grouped bars, mean success rate per condition
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
x = range(len(labels))
width = 0.32

bars1 = ax.bar([i - width / 2 for i in x], hol_means, width,
               yerr=hol_sd, capsize=4, label="Holistic", color=HOL_COLOR)
bars2 = ax.bar([i + width / 2 for i in x], gca_means, width,
               yerr=gca_sd, capsize=4, label="GCA", color=GCA_COLOR)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.text(3.35, 0.505, "random guessing", fontsize=8, color="gray", ha="right")

ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Mean success rate (20 seeds)")
ax.set_ylim(0.45, 1.0)
ax.set_title("Ground-truth ranking accuracy: Holistic vs GCA")
ax.legend(loc="upper left")

for bars, means in [(bars1, hol_means), (bars2, gca_means)]:
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.015,
                 f"{m*100:.1f}%", ha="center", fontsize=9)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
out1 = Path(__file__).parent / "ground_truth_results.png"
fig.savefig(out1)
print("wrote", out1)

# ---------------------------------------------------------------------------
# Chart 2: GCA advantage grows with how extreme the comparison is
# ---------------------------------------------------------------------------
score_gaps = [0.04, 0.36, 0.66, 0.79]
short_labels = ["Same-article", "Top/bottom 25%", "Top/bottom 10%", "Top/bottom 5%\n(all-pairs)"]

fig2, ax2 = plt.subplots(figsize=(7, 4.5), dpi=150)
ax2.plot(score_gaps, gaps_pp, marker="o", color=GCA_COLOR, linewidth=2, markersize=8)
ax2.axhline(0, color="gray", linestyle="--", linewidth=1)

for xg, yg, lab in zip(score_gaps, gaps_pp, short_labels):
    ax2.annotate(f"{lab}\n{yg:+.1f}pp", (xg, yg), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=8)

ax2.set_xlabel("Minimum AlignScore-GCA score gap between compared summaries")
ax2.set_ylabel("GCA minus Holistic (percentage points)")
ax2.set_title("GCA's precision advantage grows with the quality gap")
ax2.set_ylim(-2, 12)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
fig2.tight_layout()
out2 = Path(__file__).parent / "gap_vs_advantage.png"
fig2.savefig(out2)
print("wrote", out2)

print()
for label, hm, gm, sd_h, sd_g, gap in zip(labels, hol_means, gca_means, hol_sd, gca_sd, gaps_pp):
    print(f"{label.splitlines()[0]:<26s} holistic={hm*100:5.1f}% (sd {sd_h*100:.1f})  "
          f"gca={gm*100:5.1f}% (sd {sd_g*100:.1f})  gap={gap:+.2f}pp")

# ---------------------------------------------------------------------------
# Chart 3: does the GCA advantage hold as the training set grows?
# Compares the gap (GCA - Holistic) per condition at n=1,000 vs n=5,000.
# ---------------------------------------------------------------------------
conditions_5000 = [
    ("Same-article\n(gap ~0.04)", "ground_truth_eval_5000.json"),
    ("Top/bottom 25%\n(gap >=0.36)", "biased_ground_truth_eval_5000.json"),
    ("Top/bottom 10%\n(gap >=0.66)", "biased_top10_eval_5000.json"),
    ("Top/bottom 5%,\nall-pairs (gap >=0.79)", "biased_top5_allpairs_eval_5000.json"),
]
hol_means_5000, gca_means_5000, hol_sd_5000, gca_sd_5000, gaps_pp_5000 = [], [], [], [], []
for _, fname in conditions_5000:
    hol, gca = load_eval(fname)
    hol_means_5000.append(statistics.mean(hol))
    gca_means_5000.append(statistics.mean(gca))
    hol_sd_5000.append(statistics.stdev(hol))
    gca_sd_5000.append(statistics.stdev(gca))
    gaps_pp_5000.append((statistics.mean(gca) - statistics.mean(hol)) * 100)

# ---------------------------------------------------------------------------
# Chart 3b: grouped bars, mean success rate per condition, n=5,000
# (same layout as chart 1, so the two are directly comparable side by side)
# ---------------------------------------------------------------------------
fig3b, ax3b = plt.subplots(figsize=(8.5, 5), dpi=150)

bars1b = ax3b.bar([i - width / 2 for i in x], hol_means_5000, width,
                   yerr=hol_sd_5000, capsize=4, label="Holistic", color=HOL_COLOR)
bars2b = ax3b.bar([i + width / 2 for i in x], gca_means_5000, width,
                   yerr=gca_sd_5000, capsize=4, label="GCA", color=GCA_COLOR)

ax3b.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax3b.text(3.35, 0.505, "random guessing", fontsize=8, color="gray", ha="right")

ax3b.set_xticks(list(x))
ax3b.set_xticklabels(labels, fontsize=9)
ax3b.set_ylabel("Mean success rate (20 seeds)")
ax3b.set_ylim(0.45, 1.0)
ax3b.set_title("Ground-truth ranking accuracy at n=5,000: Holistic vs GCA")
ax3b.legend(loc="upper left")

for bars, means in [(bars1b, hol_means_5000), (bars2b, gca_means_5000)]:
    for bar, m in zip(bars, means):
        ax3b.text(bar.get_x() + bar.get_width() / 2, m + 0.015,
                   f"{m*100:.1f}%", ha="center", fontsize=9)

ax3b.spines["top"].set_visible(False)
ax3b.spines["right"].set_visible(False)
fig3b.tight_layout()
out3b = Path(__file__).parent / "ground_truth_results_5000.png"
fig3b.savefig(out3b)
print("wrote", out3b)

fig3, ax3 = plt.subplots(figsize=(7.5, 4.5), dpi=150)
ax3.plot(score_gaps, gaps_pp, marker="o", color=GCA_COLOR, linewidth=2,
         markersize=8, label="n=1,000")
ax3.plot(score_gaps, gaps_pp_5000, marker="s", color="#7a7a7a", linewidth=2,
         markersize=8, linestyle="--", label="n=5,000")
ax3.axhline(0, color="gray", linestyle=":", linewidth=1)

ax3.set_xlabel("Minimum AlignScore-GCA score gap between compared summaries")
ax3.set_ylabel("GCA minus Holistic (percentage points)")
ax3.set_title("GCA's precision advantage shrinks as the training set grows")
ax3.set_ylim(-3, 12)
ax3.legend(loc="upper left")
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
fig3.tight_layout()
out3 = Path(__file__).parent / "scale_comparison.png"
fig3.savefig(out3)
print("wrote", out3)

print()
for label, gap1k, gap5k in zip(labels, gaps_pp, gaps_pp_5000):
    print(f"{label.splitlines()[0]:<26s} n=1000 gap={gap1k:+.2f}pp   n=5000 gap={gap5k:+.2f}pp")

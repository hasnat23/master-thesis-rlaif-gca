"""Generates the results chart for the 11 August progress update.

Reads the same per-run data reported in thesis/chapters/06_results.tex
(Tables 6.8-6.10) directly from the committed campaign outputs, so the chart
cannot drift out of sync with the numbers written in the README or thesis.
"""
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def load_campaign(dirpath):
    holistic, gca = [], []
    for summary_path in sorted(Path(dirpath).glob("seed_*/rm_training_summary.json")):
        d = json.loads(summary_path.read_text())
        holistic.append(d["conditions"]["holistic"]["mean_val_acc"])
        gca.append(d["conditions"]["gca"]["mean_val_acc"])
    return holistic, gca


h1000, g1000 = load_campaign(ROOT / "outputs/seed_campaign")
h5000, g5000 = load_campaign(ROOT / "outputs/seed_campaign_5000")
h10000, g10000 = load_campaign(ROOT / "outputs/seed_campaign_10000")

sizes = ["1,000", "5,000", "10,000"]
hol_means = [statistics.mean(h1000), statistics.mean(h5000), statistics.mean(h10000)]
gca_means = [statistics.mean(g1000), statistics.mean(g5000), statistics.mean(g10000)]
hol_sd = [statistics.stdev(h1000), statistics.stdev(h5000), statistics.stdev(h10000)]
gca_sd = [statistics.stdev(g1000), statistics.stdev(g5000), statistics.stdev(g10000)]

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
x = range(len(sizes))
width = 0.32

bars1 = ax.bar(
    [i - width / 2 for i in x], hol_means, width,
    yerr=hol_sd, capsize=4, label="Holistic", color="#5B8FD4",
)
bars2 = ax.bar(
    [i + width / 2 for i in x], gca_means, width,
    yerr=gca_sd, capsize=4, label="GCA", color="#E8A24C",
)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
ax.text(2.35, 0.503, "random guessing", fontsize=8, color="gray")

ax.set_xticks(list(x))
ax.set_xticklabels([f"{s} examples" for s in sizes])
ax.set_ylabel("Reward model accuracy")
ax.set_ylim(0.48, 0.63)
ax.set_title("Reward model accuracy: Holistic vs GCA")
ax.legend(loc="upper left")

for bars, means in [(bars1, hol_means), (bars2, gca_means)]:
    for bar, m in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2, m + 0.006,
            f"{m*100:.1f}%", ha="center", fontsize=9,
        )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()

out = Path(__file__).parent / "reward_model_results.png"
fig.savefig(out)
print("wrote", out)
print("n=1000  holistic=%.4f (sd %.4f)  gca=%.4f (sd %.4f)" % (hol_means[0], hol_sd[0], gca_means[0], gca_sd[0]))
print("n=5000  holistic=%.4f (sd %.4f)  gca=%.4f (sd %.4f)" % (hol_means[1], hol_sd[1], gca_means[1], gca_sd[1]))
print("n=10000 holistic=%.4f (sd %.4f)  gca=%.4f (sd %.4f)" % (hol_means[2], hol_sd[2], gca_means[2], gca_sd[2]))

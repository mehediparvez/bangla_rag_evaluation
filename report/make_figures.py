"""Draws the figures of Chapter 4 into report/figures/.

The numbers are copied from the saved outputs of the Colab run, so the figures can be redrawn without a GPU:
  * training curve:   log.jsonl written by scripts/train_t5.py (one row per epoch)
  * retrieval:        retrieval_summary.json written by scripts/retrieve.py
  * end-to-end, hit/miss: analysis_main.csv and analysis_hitmiss.csv written by scripts/analyze_t5.py
Run:  python report/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
BLUE, ORANGE, GREEN, GREY = "#2f6fa3", "#d9822b", "#4f9a6a", "#8a8f98"

# ---- training log (validation set, 1,484 questions) --------------------------------------------
train_loss = [6.0802, 1.8227, 1.1478, 0.8132, 0.6788, 0.6053, 0.5408, 0.4990, 0.4634, 0.4538]
val_em = [24.12, 36.79, 50.07, 50.88, 53.91, 54.99, 57.82, 58.09, 59.10, 59.10]
val_f1 = [24.14, 45.65, 66.40, 68.27, 70.58, 72.54, 74.87, 74.74, 76.00, 75.66]
epochs = np.arange(1, 11)

fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.2))
a.plot(epochs, train_loss, marker="o", color=BLUE)
a.set_yscale("log")
a.set_xlabel("epoch")
a.set_ylabel("training loss (log scale)")
b.plot(epochs, val_f1, marker="s", color=ORANGE, label="F1")
b.plot(epochs, val_em, marker="o", color=GREEN, label="exact match")
b.axvline(9, color=GREY, ls="--", lw=1)
b.annotate("best F1\n(epoch 9)", (9, 40), ha="right", fontsize=9, color=GREY)
b.set_xlabel("epoch")
b.set_ylabel("validation score (%)")
b.set_ylim(15, 85)
b.legend(frameon=False, loc="lower right")
for ax in (a, b):
    ax.set_xticks(epochs)
fig.tight_layout()
fig.savefig(OUT / "training_curve.png", dpi=200)
plt.close(fig)

# ---- retrieval hit rates ---------------------------------------------------------------------------
names = ["BM25", "Dense (BGE-M3)", "Hybrid (RRF)"]
hits = {"hit@1": [77.9, 69.4, 74.8], "hit@3": [87.9, 82.5, 87.9], "hit@10": [91.3, 88.9, 94.1]}
fig, ax = plt.subplots(figsize=(6.4, 3.3))
w = 0.26
for i, (label, vals) in enumerate(hits.items()):
    bars = ax.bar(np.arange(3) + (i - 1) * w, vals, w, label=label, color=[BLUE, ORANGE, GREEN][i])
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
ax.set_xticks(range(3), names)
ax.set_ylabel("questions with the right passage (%)")
ax.set_ylim(0, 105)
ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
fig.tight_layout()
fig.savefig(OUT / "retrieval_hits.png", dpi=200)
plt.close(fig)

# ---- end-to-end accuracy on answerable questions, with 95% intervals ------------------------------
conds = ["Gold passage", "BM25", "Hybrid", "Dense"]
metric = {  # point, low, high
    "exact match": [(56.0, 52.9, 59.3), (52.0, 48.9, 55.2), (49.9, 46.9, 53.0), (45.5, 42.4, 48.5)],
    "F1": [(77.6, 75.5, 79.7), (71.6, 69.0, 74.1), (69.2, 66.8, 71.6), (63.6, 60.9, 66.3)],
    "F1 ≥ 0.5": [(82.4, 79.9, 84.8), (75.9, 73.0, 78.7), (73.4, 70.6, 76.1), (68.0, 64.9, 71.1)],
}
fig, ax = plt.subplots(figsize=(7.0, 3.5))
w = 0.26
for i, (label, vals) in enumerate(metric.items()):
    p = np.array([v[0] for v in vals])
    err = np.array([[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]])
    ax.bar(np.arange(4) + (i - 1) * w, p, w, yerr=err, capsize=2, label=label, color=[GREEN, ORANGE, BLUE][i],
           error_kw={"lw": 1})
ax.set_xticks(range(4), conds)
ax.set_ylabel("answerable questions (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=3, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "endtoend.png", dpi=200)
plt.close(fig)

# ---- what retrieval outcome does to answerable questions --------------------------------------------
right = [(57.5, 987), (56.6, 867), (57.1, 944)]   # exact match, number of questions
wrong = [(11.7, 137), (7.8, 257), (12.2, 180)]
fig, ax = plt.subplots(figsize=(6.0, 3.3))
w = 0.34
b1 = ax.bar(np.arange(3) - w / 2, [r[0] for r in right], w, color=GREEN, label="right passage retrieved")
b2 = ax.bar(np.arange(3) + w / 2, [r[0] for r in wrong], w, color=ORANGE, label="wrong passage retrieved")
for bars, data in ((b1, right), (b2, wrong)):
    for bar, (_, n) in zip(bars, data):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{bar.get_height():.1f}\n(n={n})",
                ha="center", fontsize=8)
ax.set_xticks(range(3), names)
ax.set_ylabel("exact match, answerable questions (%)")
ax.set_ylim(0, 88)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "hit_vs_miss.png", dpi=200)
plt.close(fig)
print("saved:", *sorted(p.name for p in OUT.glob("*.png")))

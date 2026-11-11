"""Draw the report figures into report/figures/.

The numbers are the ones printed by `bangla_rag.analyze` and `bangla_rag.human detect/analyze` in Colab
(result files live on Drive, not in the repo), typed in here and in the chapters. Run: python report/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
C = {"none": "#8c8c8c", "bm25": "#0072B2", "dense": "#009E73", "hybrid": "#E69F00", "en": "#0072B2", "bn": "#D55E00"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

SETTINGS = ["none", "bm25", "dense", "hybrid"]
LABELS = ["No retrieval", "BM25", "Dense (BGE-M3)", "Hybrid"]


def pipeline():
    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")
    boxes = [
        (1, "Question\n(BanglaRQA test,\n200 sampled)", "#F2F2F2"),
        (19, "Retriever\nnone / BM25 /\ndense / hybrid", "#CCE5F5"),
        (37, "Top-3 chunks\n(512 chars,\n3,000 passages)", "#F2F2F2"),
        (55, "Bangla prompt\n+ Gemini\n(same for all)", "#CCE5F5"),
        (73, "JSON answer\n→ EM, F1,\ncorrect (F1 ≥ 0.5)", "#F2F2F2"),
    ]
    for x, text, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 11), 16, 14, boxstyle="round,pad=0.4", fc=col, ec="#444444", lw=1))
        ax.text(x + 8, 18, text, ha="center", va="center", fontsize=9.5)
    for x in (17.6, 35.6, 53.6, 71.6):
        ax.add_patch(FancyArrowPatch((x, 18), (x + 1.4, 18), arrowstyle="-|>", mutation_scale=14, color="#444444"))
    ax.add_patch(FancyBboxPatch((71, 1), 20, 6, boxstyle="round,pad=0.4", fc="#FBE3D6", ec="#444444", lw=1))
    ax.text(81, 4, "RAGAS judge (Gemini):\nfaithfulness, relevancy", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((81, 10.4), (81, 8.2), arrowstyle="-|>", mutation_scale=14, color="#444444"))
    ax.text(94.5, 18, "RQ1", fontsize=11, weight="bold", ha="left", va="center", color="#0072B2")
    ax.text(94.5, 4, "RQ2", fontsize=11, weight="bold", ha="left", va="center", color="#D55E00")
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight")
    plt.close(fig)


def rq1_correct():
    mean = [0.280, 0.830, 0.855, 0.840]
    lo = [0.220, 0.775, 0.800, 0.790]
    hi = [0.345, 0.880, 0.900, 0.890]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    err = [[m - l for m, l in zip(mean, lo)], [h - m for m, h in zip(mean, hi)]]
    ax.bar(LABELS, mean, yerr=err, capsize=5, color=[C[s] for s in SETTINGS], ecolor="#222222")
    for i, m in enumerate(mean):
        ax.text(i, hi[i] + 0.02, f"{m*100:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Correct answers (F1 ≥ 0.5)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=12)
    fig.savefig(OUT / "rq1_correct.png", bbox_inches="tight")
    plt.close(fig)


def retrieval_gap():
    names = ["BM25", "Dense (BGE-M3)", "Hybrid"]
    hit = [0.980, 0.960, 0.990]
    aic = [0.745, 0.730, 0.750]
    x = range(3)
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    b1 = ax.bar([i - w / 2 for i in x], hit, w, label="Right passage in top 3", color="#0072B2")
    b2 = ax.bar([i + w / 2 for i in x], aic, w, label="Answer string in top-3 chunks", color="#E69F00")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012, f"{b.get_height()*100:.1f}%", ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.savefig(OUT / "retrieval_gap.png", bbox_inches="tight")
    plt.close(fig)


def rq2():
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    # A: detection AUC
    groups = ["All\n(n = 289)", "Synthetic\nunsupported", "Natural\nunsupported"]
    en = [0.909, 0.955, 0.875]
    bn = [0.919, 0.964, 0.887]
    en_err = [[0.909 - 0.866, 0, 0], [0.944 - 0.909, 0, 0]]
    bn_err = [[0.919 - 0.878, 0, 0], [0.955 - 0.919, 0, 0]]
    w = 0.36
    x = range(3)
    a.bar([i - w / 2 for i in x], en, w, label="English judge prompts", color=C["en"])
    a.bar([i + w / 2 for i in x], bn, w, label="Bengali judge prompts", color=C["bn"])
    a.errorbar([-w / 2], [0.909], yerr=[[0.043], [0.035]], fmt="none", ecolor="#222222", capsize=4)
    a.errorbar([w / 2], [0.919], yerr=[[0.041], [0.036]], fmt="none", ecolor="#222222", capsize=4)
    for i in x:
        a.text(i - w / 2, en[i] + 0.045 if i == 0 else en[i] + 0.012, f"{en[i]:.3f}", ha="center", fontsize=8.5)
        a.text(i + w / 2, bn[i] + 0.045 if i == 0 else bn[i] + 0.012, f"{bn[i]:.3f}", ha="center", fontsize=8.5)
    a.set_xticks(list(x))
    a.set_xticklabels(groups)
    a.set_ylim(0.5, 1.08)
    a.axhline(0.5, color="#888888", lw=0.8)
    a.set_ylabel("AUC: unsupported vs supported")
    a.set_title("Detecting unsupported answers\n(RAGAS faithfulness)", fontsize=11)
    a.yaxis.grid(True, alpha=0.3)
    a.set_axisbelow(True)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8.5)
    # B: Spearman
    names = ["Faithfulness", "Answer relevancy"]
    rho_en = [0.782, 0.294]
    rho_bn = [0.807, 0.246]
    ci_en = [(0.700, 0.853), (0.167, 0.398)]
    ci_bn = [(0.733, 0.874), (0.126, 0.346)]
    for i in range(2):
        b.bar(i - w / 2, rho_en[i], w, color=C["en"], yerr=[[rho_en[i] - ci_en[i][0]], [ci_en[i][1] - rho_en[i]]], capsize=4, ecolor="#222222")
        b.bar(i + w / 2, rho_bn[i], w, color=C["bn"], yerr=[[rho_bn[i] - ci_bn[i][0]], [ci_bn[i][1] - rho_bn[i]]], capsize=4, ecolor="#222222")
        b.text(i - w / 2, ci_en[i][1] + 0.02, f"{rho_en[i]:.2f}", ha="center", fontsize=9)
        b.text(i + w / 2, ci_bn[i][1] + 0.02, f"{rho_bn[i]:.2f}", ha="center", fontsize=9)
    b.set_xticks([0, 1])
    b.set_xticklabels(names)
    b.set_ylim(0, 1.0)
    b.set_ylabel("Spearman ρ with the grade")
    b.set_title("Correlation with reference grades", fontsize=11)
    b.yaxis.grid(True, alpha=0.3)
    b.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "rq2_ragas.png", bbox_inches="tight")
    plt.close(fig)


def grade_mix():
    # counts from the graded set (290 rows): faithfulness 54x1, 16x2, 220x3 ; relevance 11x1, 2x2, 277x3
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    cats = ["Faithfulness", "Relevance"]
    vals = {"1 (bad)": [54, 11], "2": [16, 2], "3 (good)": [220, 277]}
    cols = ["#D55E00", "#E69F00", "#009E73"]
    left = [0, 0]
    for (k, v), col in zip(vals.items(), cols):
        ax.barh(cats, v, left=left, label=k, color=col)
        for i, n in enumerate(v):
            if n >= 11:
                ax.text(left[i] + n / 2, i, str(n), ha="center", va="center", color="white", fontsize=10)
        left = [l + n for l, n in zip(left, v)]
    ax.set_xlabel("Number of answers (of 290)")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=9)
    ax.invert_yaxis()
    fig.savefig(OUT / "grade_mix.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    rq1_correct()
    retrieval_gap()
    rq2()
    grade_mix()
    print("figures written to", OUT)

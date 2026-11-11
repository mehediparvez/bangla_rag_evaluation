"""RQ2, step 2: RAGAS scores versus the reference grades (Spearman, detection AUC, kappa)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .stats import cohen_kappa, spearman_ci


def auc(pos, neg) -> float:
    """P(score_pos > score_neg), ties count half."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def detect(sheet_path, ragas_paths: dict[str, str], key_path=None, seed=0, n_boot=2000, rater="human"):
    """RAGAS faithfulness as a detector of unsupported answers.

    Unsupported = human_faithful < 3, or a synthetic row listed in the perturb key.
    Reports AUC (low faithfulness should rank unsupported answers) and the false-alarm rate
    (supported answers scored below 0.5).
    """
    import numpy as np
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig")
    sheet["hf"] = pd.to_numeric(sheet[f"{rater}_faithful"], errors="coerce")
    synth = set(pd.read_csv(key_path)["row_id"]) if key_path and Path(key_path).exists() else set()
    sheet["synthetic"] = sheet["row_id"].isin(synth)
    sheet = sheet[sheet["hf"].isin([1, 2, 3]) | sheet["synthetic"]]
    sheet["bad"] = sheet["synthetic"] | (sheet["hf"] < 3)
    rng = np.random.default_rng(seed)
    for tag, path in ragas_paths.items():
        m = sheet.merge(pd.read_csv(path, encoding="utf-8-sig"), on="row_id").dropna(subset=["faithfulness"])
        good, bad = m[~m["bad"]]["faithfulness"].to_numpy(), m[m["bad"]]["faithfulness"].to_numpy()
        print(f"[{tag}] reference rater: {rater}; supported n={len(good)}, unsupported n={len(bad)} (synthetic {int(m['synthetic'].sum())})")
        if len(good):
            print(f"  false-alarm rate (supported scored < 0.5): {(good < 0.5).mean():.3f}; mean faithfulness {good.mean():.3f}")
        if len(good) >= 5 and len(bad) >= 5:
            a = auc(-bad, -good)
            boots = [auc(-rng.choice(bad, len(bad)), -rng.choice(good, len(good))) for _ in range(n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  AUC unsupported vs supported: {a:.3f} [{lo:.3f}, {hi:.3f}]")
            print(f"  detection rate (unsupported scored < 0.5): {(bad < 0.5).mean():.3f}")
            for name, mask in [("synthetic", m["synthetic"]), ("natural (human-graded)", m["bad"] & ~m["synthetic"])]:
                sub = m[mask]["faithfulness"].to_numpy()
                if len(sub) >= 5:
                    print(f"    AUC {name} (n={len(sub)}): {auc(-sub, -good):.3f}")
        else:
            print("  too few unsupported rows for an AUC yet")


def _ci(x, y):
    rho, lo, hi = spearman_ci(x, y)
    return f"{rho:.3f} [{lo:.3f}, {hi:.3f}]"


def analyze(sheet_path, ragas_paths: dict[str, str], second_rater=None, rater="human"):
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig")
    if rater != "human":  # e.g. "ai": grade columns named ai_faithful / ai_relevant; use them as the reference
        sheet["human_faithful"], sheet["human_relevant"] = sheet[f"{rater}_faithful"], sheet[f"{rater}_relevant"]
        print(f"reference rater: {rater}")
    graded = sheet.dropna(subset=["human_faithful", "human_relevant"]).copy()
    for col in ["human_faithful", "human_relevant"]:
        num = pd.to_numeric(graded[col], errors="coerce")
        bad = graded[num.isna() | ~num.isin([1, 2, 3])]
        if len(bad):
            print(f"WARNING: {len(bad)} row(s) with an invalid {col} (must be 1, 2 or 3) are skipped. "
                  f"row_id / value: {list(zip(bad['row_id'], bad[col]))}")
        graded[col] = num.where(num.isin([1, 2, 3]))
    sheet = graded.dropna(subset=["human_faithful", "human_relevant"])
    print(f"human-graded rows: {len(sheet)}\n")
    lines = []
    for tag, path in ragas_paths.items():
        scores = pd.read_csv(path, encoding="utf-8-sig")
        m = sheet.merge(scores, on="row_id", how="inner")
        for human_col, ragas_col, name in [("human_faithful", "faithfulness", "faithfulness"),
                                           ("human_relevant", "answer_relevancy", "answer relevance")]:
            ok = m[[human_col, ragas_col]].dropna()
            if len(ok) >= 5:
                lines.append(f"RAGAS [{tag}] {name} vs human: Spearman rho = {_ci(ok[human_col], ok[ragas_col])}  (n={len(ok)})")
    if len(ragas_paths) == 2:  # difference between two judge prompt languages
        (t1, p1), (t2, p2) = list(ragas_paths.items())
        a = sheet.merge(pd.read_csv(p1, encoding="utf-8-sig"), on="row_id")
        b = sheet.merge(pd.read_csv(p2, encoding="utf-8-sig"), on="row_id")
        both = a.merge(b, on="row_id", suffixes=(f"_{t1}", f"_{t2}")).dropna(
            subset=[f"faithfulness_{t1}", f"faithfulness_{t2}", "human_faithful_" + t1])
        if len(both) >= 5:
            from .stats import spearman
            d = spearman(both["human_faithful_" + t1], both[f"faithfulness_{t1}"]) - \
                spearman(both["human_faithful_" + t1], both[f"faithfulness_{t2}"])
            lines.append(f"Faithfulness rho difference ({t1} - {t2}): {d:+.3f}")
    if second_rater:
        s2 = pd.read_csv(second_rater, encoding="utf-8-sig").dropna(subset=["human_faithful", "human_relevant"])
        m = sheet.merge(s2, on="row_id", suffixes=("", "_r2"))
        for col in ["human_faithful", "human_relevant"]:
            k = cohen_kappa(m[col], m[col + "_r2"], weights="quadratic")
            lines.append(f"Human vs second rater, {col}: quadratic-weighted kappa = {k:.3f}  (n={len(m)})")
    print("\n".join(lines) or "nothing to report yet (fill in the grades first)")

"""RQ2: human grading sheet and RAGAS-vs-human agreement.

Workflow
--------
1. ``python -m bangla_rag.human sample --runs bm25_c512_k3,hybrid-bge-m3_c512_k3 --n 100``
   writes ``human/grading_sheet.csv``. Open it in Google Sheets / Excel and fill the
   ``human_faithful`` and ``human_relevant`` columns (1-3, see RUBRIC below).
2. ``python -m bangla_rag.ragas_eval --sheet human/grading_sheet.csv --tag en``
3. ``python -m bangla_rag.human analyze --sheet human/grading_sheet.csv --ragas human/ragas_en.csv``
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from .analyze import load_runs
from .stats import cohen_kappa, spearman_ci

RUBRIC = """Grade each answer on two 1-3 scales.
human_faithful: is every claim in the answer supported by the retrieved passages?
  1 = mostly unsupported / contradicts the passages, 2 = partly supported, 3 = fully supported.
human_relevant: does the answer address the question that was asked?
  1 = off-topic, 2 = partly answers it, 3 = directly answers it.
Judge from the passages shown, not from your own knowledge. Do not look at RAGAS scores."""

SHEET_COLS = ["row_id", "run_id", "question_id", "question", "contexts", "reference", "response",
              "human_faithful", "human_relevant", "notes"]


def make_sheet(results_dir, run_ids, n, seed, out_path) -> pd.DataFrame:
    runs = load_runs(results_dir)
    run_ids = [r for r in run_ids if r in runs and r != "none" and "answer" in runs[r]]
    if not run_ids:
        raise SystemExit("no usable runs (need generation results from a retrieval run)")
    rng = random.Random(seed)
    per_run = max(1, n // len(run_ids))
    rows = []
    for run_id in run_ids:
        df = runs[run_id]
        for i in rng.sample(range(len(df)), min(per_run, len(df))):
            r = df.iloc[i]
            rows.append({
                "run_id": run_id, "question_id": r["question_id"], "question": r["question"],
                "contexts": "\n---\n".join(r["contexts"]), "reference": " | ".join(r["references"]),
                "response": r["answer"], "human_faithful": "", "human_relevant": "", "notes": "",
            })
    rng.shuffle(rows)  # hide the run order from the rater
    sheet = pd.DataFrame(rows)
    sheet.insert(0, "row_id", range(1, len(sheet) + 1))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet[SHEET_COLS].to_csv(out_path, index=False, encoding="utf-8-sig")
    (out_path.parent / "RUBRIC.txt").write_text(RUBRIC, encoding="utf-8")
    print(f"wrote {len(sheet)} rows to {out_path}\n\n{RUBRIC}")
    return sheet


def _ci(x, y):
    rho, lo, hi = spearman_ci(x, y)
    return f"{rho:.3f} [{lo:.3f}, {hi:.3f}]"


def analyze(sheet_path, ragas_paths: dict[str, str], second_rater=None):
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig")
    sheet = sheet.dropna(subset=["human_faithful", "human_relevant"])
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="create the human grading sheet")
    s.add_argument("--results", default="results")
    s.add_argument("--runs", required=True, help="comma-separated retrieval run ids to sample from")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="human/grading_sheet.csv")
    a = sub.add_parser("analyze", help="compare RAGAS scores with human grades")
    a.add_argument("--sheet", default="human/grading_sheet.csv")
    a.add_argument("--ragas", action="append", required=True,
                   help="RAGAS score CSV; repeat to compare prompt languages (en first, then bn)")
    a.add_argument("--second-rater", help="second rater's filled copy of the sheet")
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        make_sheet(args.results, args.runs.split(","), args.n, args.seed, args.out)
    else:
        tags = {Path(p).stem.replace("ragas_", ""): p for p in args.ragas}
        analyze(args.sheet, tags, args.second_rater)


if __name__ == "__main__":
    main()

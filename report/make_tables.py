"""Turn the RQ1 results into LaTeX tables so numbers are never typed by hand.

Run from the repository root, after ``python -m bangla_rag.analyze``:

    python report/make_tables.py --results results

Writes report/tables/rq1_metrics.tex and report/tables/rq1_mcnemar.tex.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bangla_rag.analyze import DEFAULT_RUN, compare, load_runs, summarise  # noqa: E402

_ESC = {"_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$"}


def esc(text: str) -> str:
    return "".join(_ESC.get(c, c) for c in str(text))


def latex_table(df: pd.DataFrame, colspec: str | None = None) -> str:
    cols = list(df.columns)
    colspec = colspec or "l" + "c" * (len(cols) - 1)
    lines = [r"\resizebox{\textwidth}{!}{%", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(rf"\textbf{{{esc(c)}}}" for c in cols) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for v in row:
            cells.append(esc(f"{v:.3f}") if isinstance(v, float) else esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}%", "}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", default="none")
    ap.add_argument("--out", default=str(ROOT / "report" / "tables"))
    args = ap.parse_args(argv)

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no result files in {args.results}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics = summarise(runs)
    (out / "rq1_metrics.tex").write_text(latex_table(metrics), encoding="utf-8")

    cmp_ = compare(runs, args.baseline)
    if cmp_.empty:
        text = r"\textit{No comparison available (baseline run missing).}" + "\n"
    else:
        text = latex_table(cmp_)
    (out / "rq1_mcnemar.tex").write_text(text, encoding="utf-8")
    print(f"wrote tables to {out} (runs: {', '.join(runs)}; default run {DEFAULT_RUN})")


if __name__ == "__main__":
    main()

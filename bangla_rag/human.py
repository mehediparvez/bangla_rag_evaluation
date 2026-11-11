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
from pathlib import Path

from .agreement import analyze, auc, detect  # noqa: F401  (re-exported: public API of this module)
from .grading import RUBRIC, SHEET_COLS, make_sheet  # noqa: F401


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="create the human grading sheet")
    s.add_argument("--results", default="results")
    s.add_argument("--runs", required=True, help="comma-separated retrieval run ids to sample from")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--out", default="human/grading_sheet.csv")
    s.add_argument("--hard-frac", type=float, default=0.0,
                   help="fraction of rows drawn from likely-bad answers (answer not in context, or F1 < 0.5)")
    s.add_argument("--closed-book", type=int, default=0,
                   help="add N no-retrieval answers shown with the retrieved passages (naturally unsupported cases)")
    s.add_argument("--ctx-run", default="bm25_c512_k3", help="run whose passages are shown for closed-book rows")
    s.add_argument("--perturb", type=int, default=0,
                   help="add N synthetic unsupported rows (answer + passages of another question); ids go to perturb_key.csv")
    s.add_argument("--extend", help="graded sheet to keep; adds --n NEW rows after it (write to a different --out)")
    d = sub.add_parser("detect", help="AUC / false-alarm analysis of RAGAS faithfulness")
    d.add_argument("--sheet", default="human/grading_sheet.csv")
    d.add_argument("--ragas", action="append", required=True)
    d.add_argument("--key", default="human/perturb_key.csv")
    d.add_argument("--rater", default="human", help="grade column prefix: human or ai")
    a = sub.add_parser("analyze", help="compare RAGAS scores with human grades")
    a.add_argument("--sheet", default="human/grading_sheet.csv")
    a.add_argument("--ragas", action="append", required=True,
                   help="RAGAS score CSV; repeat to compare prompt languages (en first, then bn)")
    a.add_argument("--rater", default="human", help="grade column prefix: human or ai")
    a.add_argument("--second-rater", help="second rater's filled copy of the sheet")
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        make_sheet(args.results, args.runs.split(","), args.n, args.seed, args.out, args.extend, args.hard_frac,
                   args.closed_book, args.ctx_run, args.perturb)
    elif args.cmd == "detect":
        detect(args.sheet, {Path(p).stem.replace("ragas_", ""): p for p in args.ragas}, args.key, rater=args.rater)
    else:
        tags = {Path(p).stem.replace("ragas_", ""): p for p in args.ragas}
        analyze(args.sheet, tags, args.second_rater, args.rater)


if __name__ == "__main__":
    main()

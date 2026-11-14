"""Summarise the hand-labelled error sample (error_sample.csv with the `cause` column filled in).

Prints the number of errors per cause for each condition, the share judged acceptable with a Wilson 95%
interval (strict: only `acceptable_wording`; lenient: also `partial_answer`), and writes a LaTeX table to
--tex-out. Rows whose `cause` is empty fall back to the automatic label in `auto_cause` when that label is
final (retrieval_miss, reader_gave_empty_answer -> false_abstention).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from bangla_rag.analysis import wilson_interval

ORDER = ["acceptable_wording", "partial_answer", "wrong_value", "unrelated_span", "yes_no_flipped",
         "false_abstention", "retrieval_miss", "dataset_issue"]
NAMES = {"acceptable_wording": "acceptable wording", "partial_answer": "partial answer", "wrong_value": "wrong value",
         "unrelated_span": "unrelated span", "yes_no_flipped": "yes/no flipped", "false_abstention": "false abstention",
         "retrieval_miss": "retrieval miss", "dataset_issue": "dataset issue"}
AUTO_FINAL = {"retrieval_miss": "retrieval_miss", "reader_gave_empty_answer": "false_abstention"}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["cause"] = df["cause"].fillna("").astype(str).str.strip()
    fallback = df["auto_cause"].map(AUTO_FINAL).fillna("")
    df["cause"] = df["cause"].where(df["cause"] != "", fallback)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--tex-out", default="report/tables/error_causes.tex")
    args = ap.parse_args()
    df = load(args.csv)
    missing = int((df["cause"] == "").sum())
    if missing:
        print(f"WARNING: {missing} rows still have no cause and are left out of the percentages")
    df = df[df["cause"] != ""]
    conds = list(dict.fromkeys(df["condition"]))
    counts = {c: df[df["condition"] == c]["cause"].value_counts().to_dict() for c in conds}
    labels = [l for l in ORDER if any(l in counts[c] for c in conds)] + sorted(
        {l for c in conds for l in counts[c]} - set(ORDER))
    ns = {c: int((df["condition"] == c).sum()) for c in conds}

    print(f"{'cause':22s}" + "".join(f"{c:>16s}" for c in conds))
    for l in labels:
        print(f"{l:22s}" + "".join(f"{counts[c].get(l, 0):>8d} ({100 * counts[c].get(l, 0) / ns[c]:4.1f}%)" for c in conds))
    print(f"{'total':22s}" + "".join(f"{ns[c]:>16d}" for c in conds))
    summary = {}
    for name, keep in (("strict (acceptable wording only)", {"acceptable_wording"}),
                       ("lenient (+ partial answer)", {"acceptable_wording", "partial_answer"})):
        print(f"\nShare of sampled wrong answers judged acceptable, {name}:")
        for c in conds:
            k = sum(v for l, v in counts[c].items() if l in keep)
            p, lo, hi = wilson_interval(k, ns[c])
            summary[(name, c)] = (k, ns[c], p, lo, hi)
            print(f"  {c:6s} {k}/{ns[c]} = {100 * k / ns[c]:.0f}%  95% CI [{100 * lo:.0f}%, {100 * hi:.0f}%]")

    out = Path(args.tex_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = "l" + "rr" * len(conds)
    head = " & ".join(rf"\multicolumn{{2}}{{c}}{{{c.upper() if c != 'gold' else 'Gold passage'} ($n={ns[c]}$)}}" for c in conds)
    lines = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", rf"Cause & {head} \\", r"\midrule"]
    for l in labels:
        cells = " & ".join(rf"{counts[c].get(l, 0)} & {100 * counts[c].get(l, 0) / ns[c]:.0f}\%" for c in conds)
        lines.append(rf"{NAMES.get(l, l.replace('_', ' '))} & {cells} \\")
    lines.append(r"\midrule")
    for name in ("strict (acceptable wording only)", "lenient (+ partial answer)"):
        cells = " & ".join(rf"\multicolumn{{2}}{{c}}{{{100 * summary[(name, c)][0] / summary[(name, c)][1]:.0f}\% [{100 * summary[(name, c)][3]:.0f}, {100 * summary[(name, c)][4]:.0f}]}}" for c in conds)
        lines.append(rf"Acceptable, {name} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

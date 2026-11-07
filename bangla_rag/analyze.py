"""Summarise runs: metrics with bootstrap CIs and paired McNemar tests (RQ1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .stats import bootstrap_ci, holm, mcnemar_exact

DEFAULT_RUN = "dense-bge-m3_c512_k3"


def load_runs(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load run files; a full run is preferred over a retrieval-only file for the same run id."""
    runs: dict[str, pd.DataFrame] = {}
    files = sorted(Path(results_dir).glob("*.jsonl"))
    for path in sorted(files, key=lambda p: p.name.endswith(".retrieval.jsonl")):
        if path.name == "gemini_cache.jsonl":
            continue
        run_id = path.name.removesuffix(".retrieval.jsonl").removesuffix(".jsonl")
        if run_id in runs:
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            runs[run_id] = pd.DataFrame(rows)
    return runs


def _fmt(vals) -> str:
    m, lo, hi = bootstrap_ci(vals)
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def summarise(runs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for run_id, df in runs.items():
        row = {"run": run_id, "n": len(df)}
        for col, label in [("em", "EM"), ("f1", "F1"), ("correct", "Correct (F1>=0.5)"),
                           ("passage_hit", "Passage hit@k"), ("answer_in_context", "Answer in context")]:
            row[label] = _fmt(df[col].astype(float)) if col in df and df[col].notna().all() else "-"
        rows.append(row)
    return pd.DataFrame(rows)


def compare(runs: dict[str, pd.DataFrame], reference: str, metric: str = "correct") -> pd.DataFrame:
    """Paired McNemar test of every run against ``reference`` on ``metric`` (Holm-adjusted)."""
    if reference not in runs or metric not in runs[reference]:
        return pd.DataFrame()
    ref = runs[reference].set_index("question_id")[metric].astype(bool)
    raw, rows = {}, {}
    for run_id, df in runs.items():
        if run_id == reference or metric not in df:
            continue
        other = df.set_index("question_id")[metric].astype(bool)
        common = ref.index.intersection(other.index)
        if len(common) == 0:
            continue
        a_only, b_only, p = mcnemar_exact(other[common], ref[common])
        raw[run_id] = p
        rows[run_id] = {"run": run_id, "paired n": len(common),
                        f"better than {reference} only": a_only, f"worse only": b_only,
                        "diff (acc)": float(other[common].mean() - ref[common].mean()), "p (exact)": p}
    adj = holm(raw) if raw else {}
    for run_id, row in rows.items():
        row["p (Holm)"] = adj[run_id]
    return pd.DataFrame(list(rows.values()))


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--default-run", default=DEFAULT_RUN)
    ap.add_argument("--baseline", default="none")
    args = ap.parse_args(argv)

    runs = load_runs(args.results)
    if not runs:
        raise SystemExit(f"no result files in {args.results}")
    parts = ["# RQ1 results\n", "## Per-run metrics (mean [95% bootstrap CI])\n", to_markdown(summarise(runs))]
    for ref, title in [(args.default_run, "default configuration"), (args.baseline, "no-retrieval baseline")]:
        table = compare(runs, ref)
        parts += [f"\n## Paired McNemar tests vs {title} (`{ref}`), correct = F1 >= 0.5\n", to_markdown(table)]
    text = "\n".join(parts)
    out = Path(args.results) / "summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nsaved to {out}")


if __name__ == "__main__":
    main()

"""Compare the evaluation conditions: confidence intervals, paired tests, hit/miss split, error sample.

Reads <eval-dir>/{gold,bm25,dense,hybrid,none}_preds.jsonl written by scripts/eval_t5.py and writes
analysis_main.csv, analysis_paired.csv, analysis_hitmiss.csv, error_sample.csv and analysis_summary.json
to --out-dir, and prints the tables.

Intervals are 95% percentile bootstrap intervals that resample whole passages (10,000 resamples).
Retriever pairs are compared with the mid-p McNemar test on exact match, Holm-adjusted over the three pairs.
The primary comparison is on answerable questions (HasAns): on unanswerable questions an empty answer is
correct whatever passage was retrieved, so that score mostly measures the model's tendency to abstain.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bangla_rag.analysis import ClusterBootstrap, mcnemar_midp
from bangla_rag.qa_eval import normalize_text
from bangla_rag.stats import holm

CONDITIONS = ["gold", "bm25", "dense", "hybrid", "none"]
RETRIEVERS = ["bm25", "dense", "hybrid"]
DANDA = "।"


def load(eval_dir: Path, name: str) -> list[dict]:
    path = eval_dir / f"{name}_preds.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(est, scale=100):
    p, lo, hi = est
    return f"{scale * p:5.1f} [{scale * lo:5.1f}, {scale * hi:5.1f}]"


def write_csv(path: Path, rows: list[dict]):
    if rows:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out-dir", help="defaults to --eval-dir")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--error-n", type=int, default=40, help="wrong answers to sample per condition")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    eval_dir = Path(args.eval_dir)
    out = Path(args.out_dir) if args.out_dir else eval_dir
    out.mkdir(parents=True, exist_ok=True)

    rows = {c: load(eval_dir, c) for c in CONDITIONS if (eval_dir / f"{c}_preds.jsonl").exists()}
    ref = rows["gold"]
    for c, r in rows.items():  # every condition must list the same questions in the same order
        assert [x["question_id"] for x in r] == [x["question_id"] for x in ref], f"{c}: question order differs"
    n = len(ref)
    answerable = np.array([x["answerable"] for x in ref])
    everyone = np.ones(n, dtype=bool)
    boot = ClusterBootstrap([x["gold_passage_id"] for x in ref], n_boot=args.n_boot, seed=args.seed)

    arr = {}
    for c, r in rows.items():
        arr[c] = {
            "em": np.array([x["em"] for x in r], dtype=float),
            "f1": np.array([x["f1"] for x in r], dtype=float),
            "empty": np.array([normalize_text(x["prediction"]) == "" for x in r], dtype=float),
            "hit": np.array([x["used_passage_id"] == x["gold_passage_id"] for x in r], dtype=float),
        }

    # ---- main table -------------------------------------------------------------------------
    main_rows = []
    print(f"\n{'condition':8s} {'HasAns EM':>20s} {'HasAns F1':>20s} {'NoAns EM':>20s} {'empty|ans':>20s} {'overall F1':>20s} {'hit@1':>8s}")
    for c in rows:
        a = arr[c]
        est = {
            "has_em": boot.mean_ci(a["em"], answerable), "has_f1": boot.mean_ci(a["f1"], answerable),
            "no_em": boot.mean_ci(a["em"], ~answerable), "empty_on_ans": boot.mean_ci(a["empty"], answerable),
            "overall_em": boot.mean_ci(a["em"], everyone), "overall_f1": boot.mean_ci(a["f1"], everyone),
        }
        hit = None if c in ("gold", "none") else float(a["hit"].mean())
        print(f"{c:8s} {fmt(est['has_em']):>20s} {fmt(est['has_f1']):>20s} {fmt(est['no_em']):>20s} "
              f"{fmt(est['empty_on_ans']):>20s} {fmt(est['overall_f1']):>20s} {'' if hit is None else f'{100 * hit:7.1f}%':>8s}")
        row = {"condition": c, "n": n, "hit_at_1": hit}
        for k, (p, lo, hi) in est.items():
            row.update({k: round(100 * p, 2), k + "_lo": round(100 * lo, 2), k + "_hi": round(100 * hi, 2)})
        main_rows.append(row)
    write_csv(out / "analysis_main.csv", main_rows)

    # ---- paired comparisons between retrievers ------------------------------------------------
    pair_rows, pairs = [], [(a, b) for i, a in enumerate(RETRIEVERS) for b in RETRIEVERS[i + 1:] if a in rows and b in rows]
    for subset_name, mask in (("HasAns (primary)", answerable), ("all questions", everyone)):
        raw = {}
        for a, b in pairs:
            ea, eb = arr[a]["em"][mask] == 1, arr[b]["em"][mask] == 1
            only_a, only_b = int((ea & ~eb).sum()), int((~ea & eb).sum())
            raw[f"{a} vs {b}"] = (a, b, only_a, only_b, mcnemar_midp(only_a, only_b))
        adj = holm({k: v[4] for k, v in raw.items()})
        print(f"\nPaired EM comparison, {subset_name}: only-first-correct / only-second-correct, Holm-adjusted mid-p")
        for key, (a, b, only_a, only_b, p) in raw.items():
            d = boot.diff_ci(arr[a]["em"], arr[b]["em"], mask)
            f = boot.diff_ci(arr[a]["f1"], arr[b]["f1"], mask)
            print(f"  {key:16s} {only_a:4d} / {only_b:4d}   p_adj = {adj[key]:.4f}   EM diff {fmt(d)}   F1 diff {fmt(f)}")
            pair_rows.append({"subset": subset_name, "comparison": key, "only_first": only_a, "only_second": only_b,
                              "p_raw": round(p, 5), "p_holm": round(adj[key], 5),
                              "em_diff": round(100 * d[0], 2), "em_diff_lo": round(100 * d[1], 2), "em_diff_hi": round(100 * d[2], 2),
                              "f1_diff": round(100 * f[0], 2), "f1_diff_lo": round(100 * f[1], 2), "f1_diff_hi": round(100 * f[2], 2)})
    # descriptive: how much does each retriever lose against the correct passage? (not a test)
    print("\nLoss against the gold passage, HasAns EM / F1 (percentage points, descriptive)")
    for r in RETRIEVERS:
        if r in rows:
            d = boot.diff_ci(arr["gold"]["em"], arr[r]["em"], answerable)
            f = boot.diff_ci(arr["gold"]["f1"], arr[r]["f1"], answerable)
            print(f"  gold - {r:7s} EM {fmt(d)}   F1 {fmt(f)}")
            pair_rows.append({"subset": "HasAns (primary)", "comparison": f"gold - {r}", "em_diff": round(100 * d[0], 2),
                              "em_diff_lo": round(100 * d[1], 2), "em_diff_hi": round(100 * d[2], 2),
                              "f1_diff": round(100 * f[0], 2), "f1_diff_lo": round(100 * f[1], 2), "f1_diff_hi": round(100 * f[2], 2)})
    write_csv(out / "analysis_paired.csv", pair_rows)

    # ---- split by whether retrieval found the right passage -----------------------------------
    hm_rows = []
    print("\nAnswerable questions split by retrieval outcome (n, EM, F1, empty-answer rate)")
    for r in RETRIEVERS:
        if r not in rows:
            continue
        for label, flag in (("right passage", 1.0), ("wrong passage", 0.0)):
            m = answerable & (arr[r]["hit"] == flag)
            if m.sum():
                row = {"retriever": r, "retrieved": label, "n": int(m.sum()), "em": round(100 * arr[r]["em"][m].mean(), 2),
                       "f1": round(100 * arr[r]["f1"][m].mean(), 2), "empty_rate": round(100 * arr[r]["empty"][m].mean(), 2)}
                hm_rows.append(row)
                print(f"  {r:7s} {label:14s} n={row['n']:4d}  EM {row['em']:5.1f}  F1 {row['f1']:5.1f}  empty {row['empty_rate']:5.1f}%")
    write_csv(out / "analysis_hitmiss.csv", hm_rows)

    # ---- error-analysis sample (to be labelled by hand) ----------------------------------------
    best = max((r for r in RETRIEVERS if r in rows), key=lambda r: arr[r]["em"][answerable].mean())
    rng = random.Random(args.seed)
    sample = []
    for c in ("gold", best):
        wrong = [i for i in range(n) if answerable[i] and arr[c]["em"][i] == 0]
        for i in sorted(rng.sample(wrong, min(args.error_n, len(wrong)))):
            x = rows[c][i]
            squash = lambda s: normalize_text(s.replace(DANDA, " "))
            if x["used_passage_id"] != x["gold_passage_id"]:
                auto = "retrieval_miss"
            elif normalize_text(x["prediction"]) == "":
                auto = "reader_gave_empty_answer"
            elif squash(x["prediction"]) == squash(x["reference"]):
                auto = "surface_mismatch_candidate"
            else:
                auto = "reader_error_or_surface"
            sample.append({"condition": c, "question_id": x["question_id"], "question_type": x["question_type"],
                           "answer_type": x["answer_type"], "question": x["question"], "reference": x["reference"],
                           "prediction": x["prediction"], "gold_passage_id": x["gold_passage_id"],
                           "used_passage_id": x["used_passage_id"], "auto_cause": auto, "cause": ""})
    write_csv(out / "error_sample.csv", sample)
    counts = {}
    for s in sample:
        counts.setdefault(s["condition"], {}).setdefault(s["auto_cause"], 0)
        counts[s["condition"]][s["auto_cause"]] += 1
    print(f"\nError sample ({args.error_n} wrong answerable questions from gold and from {best}), automatic pre-sort:")
    for c, d in counts.items():
        print(f"  {c}: {d}")

    (out / "analysis_summary.json").write_text(json.dumps(
        {"n": n, "n_answerable": int(answerable.sum()), "n_boot": args.n_boot, "main": main_rows,
         "paired": pair_rows, "hit_miss": hm_rows, "best_retriever": best, "error_sample_precounts": counts}, indent=2), encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()

"""RQ2, step 1: build the human grading sheet (random, hard-case, closed-book and synthetic rows)."""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

from .analyze import load_runs

RUBRIC = """Grade each answer on two 1-3 scales.
human_faithful: is every claim in the answer supported by the retrieved passages?
  1 = mostly unsupported / contradicts the passages, 2 = partly supported, 3 = fully supported.
human_relevant: does the answer address the question that was asked?
  1 = off-topic, 2 = partly answers it, 3 = directly answers it.
Judge from the passages shown, not from your own knowledge. Do not look at RAGAS scores."""

SHEET_COLS = ["row_id", "run_id", "question_id", "question", "contexts", "reference", "response",
              "human_faithful", "human_relevant", "notes"]


def _is_hard(r) -> bool:
    """Answers likely to be unfaithful or wrong: gold answer missing from the context, or low F1."""
    return (not bool(r.get("answer_in_context", True))) or float(r.get("f1", 1.0)) < 0.5


def make_sheet(results_dir, run_ids, n, seed, out_path, extend=None, hard_frac=0.0,
               closed_book=0, ctx_run="bm25_c512_k3", perturb=0) -> pd.DataFrame:
    """Sample ``n`` rows. ``hard_frac`` of them come from likely-bad answers (see ``_is_hard``).

    With ``extend`` (a previously graded sheet) its rows are kept unchanged, the new rows exclude
    the (run_id, question_id) pairs already in it and get row_ids that continue after its maximum.
    """
    runs = load_runs(results_dir)
    run_ids = [r for r in run_ids if r in runs and r != "none" and "answer" in runs[r]]
    if not run_ids:
        raise SystemExit("no usable runs (need generation results from a retrieval run)")
    old = pd.read_csv(extend, encoding="utf-8-sig") if extend else None
    taken = set(zip(old["run_id"], old["question_id"])) if old is not None else set()
    rng = random.Random(seed)
    per_run = n // len(run_ids) if n > 0 else 0
    rows = []
    for run_id in run_ids:
        if per_run == 0:
            break
        df = runs[run_id]
        pool = [i for i in range(len(df)) if (run_id, df.iloc[i]["question_id"]) not in taken]
        hard = [i for i in pool if _is_hard(df.iloc[i])]
        easy = [i for i in pool if i not in set(hard)]
        n_hard = min(len(hard), round(per_run * hard_frac))
        pick = rng.sample(hard, n_hard)
        rest = [i for i in pool if i not in set(pick)]
        pick += rng.sample(rest, min(per_run - n_hard, len(rest)))
        print(f"{run_id}: {len(pick)} rows ({n_hard} from {len(hard)} hard candidates, {len(easy)} easy)")
        for i in pick:
            r = df.iloc[i]
            rows.append({
                "run_id": run_id, "question_id": r["question_id"], "question": r["question"],
                "contexts": "\n---\n".join(r["contexts"]), "reference": " | ".join(r["references"]),
                "response": r["answer"], "human_faithful": "", "human_relevant": "", "notes": "",
            })
    if closed_book > 0:
        # Closed-book answers (no retrieval, so from the model's memory) shown with the passages that
        # retrieval found for the same question: many are unsupported, which gives real unfaithful cases.
        if "none" not in runs or ctx_run not in runs:
            raise SystemExit(f"--closed-book needs finished runs 'none' and '{ctx_run}'")
        ctx = {r["question_id"]: r for _, r in runs[ctx_run].iterrows()}
        label = f"none+ctx:{ctx_run}"
        pool = [i for i in range(len(runs["none"]))
                if runs["none"].iloc[i]["question_id"] in ctx and (label, runs["none"].iloc[i]["question_id"]) not in taken]
        picks = rng.sample(pool, min(closed_book, len(pool)))
        print(f"{label}: {len(picks)} closed-book rows from {len(pool)} available")
        for i in picks:
            r = runs["none"].iloc[i]
            c = ctx[r["question_id"]]
            rows.append({
                "run_id": label, "question_id": r["question_id"], "question": r["question"],
                "contexts": "\n---\n".join(c["contexts"]), "reference": " | ".join(r["references"]),
                "response": r["answer"], "human_faithful": "", "human_relevant": "", "notes": "",
            })
    perturbed = []
    if perturb > 0:
        # Synthetic unsupported answers: an answer paired with the passages retrieved for a DIFFERENT
        # question. The label is known by construction; row_ids are written to perturb_key.csv, and the
        # run_id shown to the rater is the source run, so the sheet does not reveal them.
        run_id = run_ids[0]
        df = runs[run_id]
        pool = [i for i in range(len(df)) if (run_id, df.iloc[i]["question_id"]) not in taken]
        picks = rng.sample(pool, min(perturb, len(pool)))
        for i in picks:
            r = df.iloc[i]
            others = [j for j in range(len(df)) if df.iloc[j]["gold_passage_id"] != r["gold_passage_id"]]
            o = df.iloc[rng.choice(others)]
            rows.append({
                "run_id": run_id, "question_id": r["question_id"], "question": r["question"],
                "contexts": "\n---\n".join(o["contexts"]), "reference": " | ".join(r["references"]),
                "response": r["answer"], "human_faithful": "", "human_relevant": "", "notes": "",
            })
            perturbed.append((run_id, r["question_id"]))
        print(f"perturb: {len(picks)} answers paired with passages from another question")
    rng.shuffle(rows)  # hide the run order and the hard/easy split from the rater
    sheet = pd.DataFrame(rows)
    first = int(old["row_id"].max()) + 1 if old is not None else 1
    sheet.insert(0, "row_id", range(first, first + len(sheet)))
    if old is not None:
        sheet = pd.concat([old, sheet], ignore_index=True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if perturbed:
        new_rows = sheet.iloc[len(old) if old is not None else 0:]
        ids = [int(x.row_id) for x in new_rows.itertuples()
               if any(x.run_id == a and x.question_id == b and x.contexts != _own(runs, a, b) for a, b in perturbed)]
        key = out_path.parent / "perturb_key.csv"
        prev = pd.read_csv(key)["row_id"].tolist() if key.exists() else []
        pd.DataFrame({"row_id": prev + ids}).to_csv(key, index=False)
        print(f"{len(ids)} synthetic row ids saved to {key} (keep this file; do not share it with the rater)")
    sheet[SHEET_COLS].to_csv(out_path, index=False, encoding="utf-8-sig")
    (out_path.parent / "RUBRIC.txt").write_text(RUBRIC, encoding="utf-8")
    new = len(sheet) - (len(old) if old is not None else 0)
    print(f"wrote {len(sheet)} rows to {out_path} ({new} new, ungraded)\n\n{RUBRIC}")
    return sheet


def _own(runs, run_id, qid) -> str:
    df = runs[run_id]
    return "\n---\n".join(df[df["question_id"] == qid].iloc[0]["contexts"])



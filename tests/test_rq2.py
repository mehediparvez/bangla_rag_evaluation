"""RQ2 tooling: grading-sheet sampling, detection AUC and agreement analysis (synthetic data only)."""
import json

import numpy as np
import pandas as pd

from bangla_rag import agreement, grading
from bangla_rag.human import main


def _write_run(path, n=20):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({
                "question_id": f"q{i}", "question": "প্র", "references": ["উ"], "answer": "উ",
                "contexts": [f"c{i}"], "gold_passage_id": f"p{i}", "f1": 1.0, "answer_in_context": True,
            }, ensure_ascii=False) + "\n")


def test_auc_basic():
    assert agreement.auc([1, 2], [0, 0.5]) == 1.0
    assert agreement.auc([0, 0], [1, 1]) == 0.0
    assert agreement.auc([1], [1]) == 0.5


def test_make_sheet_with_perturbed_rows(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    _write_run(res / "bm25_c512_k3.jsonl")
    out = tmp_path / "out" / "sheet.csv"
    sheet = grading.make_sheet(res, ["bm25_c512_k3"], 10, 1, out, perturb=5)
    assert len(sheet) == 15
    key = pd.read_csv(tmp_path / "out" / "perturb_key.csv")["row_id"].tolist()
    assert len(key) == 5 and set(key) <= set(sheet["row_id"])
    by_id = sheet.set_index("row_id")
    for rid in key:  # synthetic rows carry the passages of ANOTHER question
        assert by_id.loc[rid, "contexts"] != f"c{int(by_id.loc[rid, 'question_id'][1:])}"
    plain = sheet[~sheet["row_id"].isin(key)]
    assert all(r.contexts == f"c{r.question_id[1:]}" for r in plain.itertuples())


def test_extend_keeps_old_rows_and_ids(tmp_path):
    res = tmp_path / "res"
    res.mkdir()
    _write_run(res / "bm25_c512_k3.jsonl")
    first = grading.make_sheet(res, ["bm25_c512_k3"], 6, 1, tmp_path / "a.csv")
    more = grading.make_sheet(res, ["bm25_c512_k3"], 4, 2, tmp_path / "b.csv", extend=tmp_path / "a.csv")
    assert len(more) == 10 and more["row_id"].tolist() == list(range(1, 11))
    assert not set(first["question_id"]) & set(more.iloc[6:]["question_id"])


def _scores(tmp_path, human_col="human", n=40):
    sheet = pd.DataFrame({"row_id": range(1, n + 1), f"{human_col}_faithful": [3] * 20 + [1] * 20,
                          f"{human_col}_relevant": 3})
    sheet.to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"row_id": range(1, n + 1), "faithfulness": list(np.linspace(0.8, 1, 20)) + list(np.linspace(0, 0.6, 20)),
                  "answer_relevancy": 0.9}).to_csv(tmp_path / "ragas_en.csv", index=False)


def test_detect_reports_auc_and_false_alarms(tmp_path, capsys):
    _scores(tmp_path)
    agreement.detect(tmp_path / "s.csv", {"en": str(tmp_path / "ragas_en.csv")}, None, n_boot=50)
    out = capsys.readouterr().out
    assert "supported n=20, unsupported n=20" in out and "AUC unsupported vs supported: 1.000" in out
    assert "false-alarm rate (supported scored < 0.5): 0.000" in out


def test_detect_uses_ai_rater_columns(tmp_path, capsys):
    _scores(tmp_path, human_col="ai")
    agreement.detect(tmp_path / "s.csv", {"en": str(tmp_path / "ragas_en.csv")}, None, n_boot=20, rater="ai")
    assert "reference rater: ai" in capsys.readouterr().out


def test_analyze_skips_invalid_grades(tmp_path, capsys):
    _scores(tmp_path)
    s = pd.read_csv(tmp_path / "s.csv").astype({"human_faithful": object})
    s.loc[0, "human_faithful"] = "`2"
    s.to_csv(tmp_path / "s.csv", index=False)
    agreement.analyze(tmp_path / "s.csv", {"en": str(tmp_path / "ragas_en.csv")})
    assert "invalid human_faithful" in capsys.readouterr().out


def test_cli_entry_point_still_works(capsys):
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0


def test_cli_detect_and_analyze(tmp_path, capsys):
    _scores(tmp_path)
    main(["detect", "--sheet", str(tmp_path / "s.csv"), "--ragas", str(tmp_path / "ragas_en.csv"), "--key", "none"])
    main(["analyze", "--sheet", str(tmp_path / "s.csv"), "--ragas", str(tmp_path / "ragas_en.csv")])
    out = capsys.readouterr().out
    assert "AUC unsupported" in out and "Spearman" in out

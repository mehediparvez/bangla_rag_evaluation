import json

import numpy as np

from bangla_rag.chunking import build_chunks, chunk_text
from bangla_rag.data import sample_questions
from bangla_rag.metrics import answer_in_text, exact_match, f1_score
from bangla_rag.prompts import build_prompt, parse_response
from bangla_rag.retrieval import BM25Retriever, rrf, top_k
from bangla_rag.stats import bootstrap_ci, cohen_kappa, holm, mcnemar_exact, spearman
from bangla_rag.text import normalize, tokenize


def test_normalize_keeps_matras_and_drops_punctuation():
    assert tokenize("অক্ষয় তৃতীয়া কী ?") == ["অক্ষয়", "তৃতীয়া", "কী"]
    assert normalize("২০২৪ সালে।") == "2024 সালে"


def test_exact_match_and_f1():
    refs = ["বিষ্ণুর ষষ্ঠ অবতার পরশুরাম", " বিষ্ণুর ষষ্ঠ অবতার পরশুরাম"]
    assert exact_match("বিষ্ণুর ষষ্ঠ অবতার পরশুরাম।", refs) == 1.0
    assert exact_match("", refs) == 0.0
    assert 0 < f1_score("পরশুরাম", refs) < 1
    assert f1_score("কিছুই না", refs) == 0.0


def test_answer_in_text():
    assert answer_in_text(["ঢাকা"], "বাংলাদেশের রাজধানী ঢাকা।")
    assert not answer_in_text(["চট্টগ্রাম"], "বাংলাদেশের রাজধানী ঢাকা।")
    assert not answer_in_text([""], "যেকোনো লেখা")


def test_chunking_respects_size_and_covers_text():
    text = ("বাংলাদেশ একটি দেশ। " * 60).strip()
    chunks = chunk_text(text, 256, 25)
    assert len(chunks) > 1
    assert all(len(c) <= 256 for c in chunks)
    assert chunks[0].startswith("বাংলাদেশ")
    assert chunk_text("ছোট", 256) == ["ছোট"]
    assert chunk_text("", 256) == []


def test_build_chunks_keeps_passage_ids():
    corpus = [{"passage_id": "p1", "title": "", "text": "ক " * 400}, {"passage_id": "p2", "title": "", "text": "খ"}]
    chunks = build_chunks(corpus, 256, 0.1)
    assert {c["passage_id"] for c in chunks} == {"p1", "p2"}
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)


def test_rrf_and_topk():
    assert [i for i, _ in top_k(np.array([0.1, 0.9, 0.5]), 2)] == [1, 2]
    fused = rrf([[(1, 0), (2, 0), (3, 0)], [(2, 0), (1, 0), (4, 0)]], top=3)
    assert {fused[0][0], fused[1][0]} == {1, 2}  # items ranked high in both lists win
    assert len(fused) == 3


def test_bm25_finds_matching_chunk():
    chunks = [{"text": t} for t in [
        "ঢাকা বাংলাদেশের রাজধানী শহর", "কলকাতা পশ্চিমবঙ্গের শহর", "পদ্মা একটি নদী", "সুন্দরবন একটি বন",
        "ক্রিকেট একটি খেলা", "বাংলা একটি ভাষা"]]
    hits = BM25Retriever(chunks).search("পদ্মা নদী", 2)
    assert hits[0][0] == 2


def test_mcnemar_and_holm():
    a, b, p = mcnemar_exact([1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0])
    assert (a, b) == (4, 0) and abs(p - 0.125) < 1e-9
    assert mcnemar_exact([1, 0], [1, 0])[2] == 1.0
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == 0.03 and adj["b"] <= adj["c"]


def test_bootstrap_spearman_kappa():
    m, lo, hi = bootstrap_ci([1, 0, 1, 1, 0, 1, 1, 0], seed=1)
    assert lo <= m <= hi
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert cohen_kappa([1, 2, 3, 1], [1, 2, 3, 1]) == 1.0
    assert cohen_kappa([1, 2, 3, 1, 2, 3], [1, 2, 3, 1, 2, 3], "quadratic") == 1.0
    assert cohen_kappa([1, 1, 2, 2], [2, 2, 1, 1]) < 0


def test_prompt_and_parse():
    p = build_prompt("প্রশ্ন?", ["প্রথম", "দ্বিতীয়"])
    assert "অনুচ্ছেদ [১]: প্রথম" in p and "অনুচ্ছেদ [২]: দ্বিতীয়" in p and "প্রশ্ন: প্রশ্ন?" in p
    assert "অনুচ্ছেদ" not in build_prompt("প্রশ্ন?", None).split("প্রশ্ন:")[0].replace("নিজের", "")
    assert parse_response('{"answer": "ঢাকা", "explanation": "কারণ"}') == ("ঢাকা", "কারণ")
    assert parse_response('```json\n{"answer": "ঢাকা"}\n```')[0] == "ঢাকা"
    assert parse_response("শুধু টেক্সট") == ("শুধু টেক্সট", "")


def test_sample_questions_is_deterministic(tmp_path):
    qas = [{"question_id": f"p1_{i:02d}", "question_text": f"প্রশ্ন {i}?", "is_answerable": "1" if i % 3 else "0",
            "question_type": "factoid", "answers": {"answer_text": ["উত্তর", "উত্তর "] if i % 3 else ["", ""]}}
           for i in range(30)]
    data = {"data": [{"passage_id": "p1", "context": "অনুচ্ছেদ", "title": "t", "qas": qas}]}
    (tmp_path / "Test.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    a = sample_questions(tmp_path, n=5, seed=1)
    b = sample_questions(tmp_path, n=5, seed=1)
    assert a == b and len(a) == 5
    assert all(q["answers"] == ["উত্তর"] for q in a)  # deduplicated, unanswerable dropped
    assert len(sample_questions(tmp_path, n=0)) == 20

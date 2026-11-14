from pathlib import Path

import pytest

from bangla_rag.qa_eval import (compute_exact_match, compute_f1, multi_span_f1, normalize_text,
                                score_item, summarise)
from bangla_rag.t5_data import load_items

DATA = Path("data/banglarqa")


def test_normalize_text_matches_authors_behaviour():
    assert normalize_text("The  Dhaka, City!") == "dhaka city"
    # the Bangla danda is not ASCII punctuation, so the authors' scorer keeps it
    assert normalize_text("ঢাকা।") == "ঢাকা।"
    assert compute_exact_match("ঢাকা।", "ঢাকা") == 0


def test_f1_no_answer_rules():
    assert compute_f1("", "") == 1.0
    assert compute_f1("ঢাকা", "") == 0.0
    assert compute_f1("", "ঢাকা") == 0.0
    assert compute_f1("ঢাকা শহর", "ঢাকা") == pytest.approx(2 * 0.5 * 1 / 1.5)


def test_multi_span_f1_penalises_extra_and_missing_spans():
    assert multi_span_f1("a b; c", "a b; c") == 1.0
    assert multi_span_f1("a; c; b", "a; b") == pytest.approx(2 / 3)   # extra predicted span
    assert multi_span_f1("a", "a; b; c") == pytest.approx(1 / 3)      # missing spans
    assert multi_span_f1("b; a", "a; b") == 1.0                       # order does not matter


def test_score_item_uses_multi_span_only_for_that_type():
    assert score_item("x; y", "x; y", "multiple spans") == (1, 1.0)
    assert score_item("x", "x y", "single span")[1] == pytest.approx(2 * 1 * 0.5 / 1.5)


def test_summarise_splits_answerable_and_unanswerable():
    items = [
        {"question_id": "1", "answer": "ঢাকা", "answer_type": "single span", "question_type": "factoid"},
        {"question_id": "2", "answer": "", "answer_type": "", "question_type": "causal"},
        {"question_id": "3", "answer": "", "answer_type": "", "question_type": "causal"},
    ]
    out = summarise(items, ["ঢাকা", "", "কিছু"])["summary"]
    assert out["overall"]["n"] == 3 and out["has_answer"]["n"] == 1 and out["no_answer"]["n"] == 2
    assert out["has_answer"]["em"] == 1.0 and out["no_answer"]["em"] == 0.5
    assert out["empty_pred_rate_on_answerable"] == 0.0


@pytest.mark.skipif(not (DATA / "Test.json").exists(), reason="BanglaRQA not downloaded")
def test_load_items_counts_match_the_dataset():
    ident = lambda s: s
    expected = {"Train": 11912, "Validation": 1484, "Test": 1493}
    for split, n in expected.items():
        assert len(load_items(DATA, split, normalize=ident)) == n
    test = load_items(DATA, "Test", normalize=ident)
    assert sum(it["answer"] == "" for it in test) == 369          # unanswerable
    assert sum(it["answer"] != "" for it in test) == 1124         # answerable
    assert all((it["answer"] == "") == (not it["is_answerable"]) for it in test)


def test_length_grouped_batches_cover_everything_and_group_by_length():
    import random

    from bangla_rag.t5_data import length_grouped_batches

    lengths = [(i * 37) % 1000 for i in range(500)]
    a = length_grouped_batches(lengths, 8, random.Random(7))
    b = length_grouped_batches(lengths, 8, random.Random(7))
    assert a == b                                               # deterministic for a seed
    flat = [i for batch in a for i in batch]
    assert sorted(flat) == list(range(500))                     # each example exactly once
    assert all(len(batch) <= 8 for batch in a)
    spread = sum(max(lengths[i] for i in bt) - min(lengths[i] for i in bt) for bt in a) / len(a)
    assert spread < 150                                         # random batches would be ~700


@pytest.mark.skipif(not (DATA / "Test.json").exists(), reason="BanglaRQA not downloaded")
def test_load_passages_covers_all_3000_and_every_question_has_its_passage():
    from bangla_rag.t5_data import load_passages

    ident = lambda s: s
    passages = load_passages(DATA, normalize=ident)
    assert len(passages) == 3000
    assert all(it["passage_id"] in passages for it in load_items(DATA, "Test", normalize=ident))

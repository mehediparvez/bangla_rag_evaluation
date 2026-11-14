from bangla_rag.qa_system import NO_ANSWER_TEXT, BanglaRAGSystem


class FakeRetriever:
    def search(self, query, k):
        return [(1, 0.9), (0, 0.5), (2, 0.1)][:k]


CORPUS = [{"passage_id": "p0", "title": "ঢাকা", "text": "ঢাকা বাংলাদেশের রাজধানী।"},
          {"passage_id": "p1", "title": "পদ্মা", "text": "পদ্মা একটি নদী।"},
          {"passage_id": "p2", "title": "অন্য", "text": "অন্য লেখা।"}]


def make(reader):
    return BanglaRAGSystem(CORPUS, {"bm25": FakeRetriever()}, reader, normalize=lambda s: s.strip())


def test_answer_reads_only_the_top_passage_and_reports_the_evidence():
    seen = {}

    def reader(q, p):
        seen["passage"] = p
        return " একটি নদী "

    res = make(reader).answer("পদ্মা কী?", "bm25", show=3)
    assert seen["passage"] == "পদ্মা একটি নদী।"            # top-1 only
    assert res["answer"] == "একটি নদী" and not res["abstained"]
    assert [p["passage_id"] for p in res["passages"]] == ["p1", "p0", "p2"]
    assert res["passages"][0]["rank"] == 1 and res["seconds"] >= 0


def test_empty_model_output_is_reported_as_no_answer():
    res = make(lambda q, p: "").answer("প্রশ্ন", "bm25")
    assert res["abstained"] and res["answer"] == NO_ANSWER_TEXT
    res2 = make(lambda q, p: " ... ").answer("প্রশ্ন", "bm25")   # ASCII punctuation only scores as empty too
    assert res2["abstained"] and res2["answer"] == NO_ANSWER_TEXT

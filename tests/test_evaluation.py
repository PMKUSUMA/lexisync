import pytest

from lexisync.evaluation import (
    default_questions,
    run_ablation,
    run_configuration_evaluation,
    run_evaluation,
    run_system_evaluation,
)
from lexisync.answering import ABSTAIN, answer_question

def test_eval_set_has_required_unanswerables():
    qs=default_questions()
    assert len(qs)>=15
    assert sum(not q["answerable"] for q in qs)>=3


class FakeIndex:
    def __init__(self):
        self.chunks = [{"chunk_id": chunk_id} for chunk_id in ("a", "b", "c", "d")]
        self.results = {
            "partial": ["a", "x"],
            "miss": ["d"],
            "unanswerable": ["b"],
            "unlabelled": ["c"],
        }

    def search(self, question, top_k, mode, rerank=True):
        ids = self.results[question][:top_k]
        return [
            {"chunk_id": chunk_id, "text": f"evidence for {chunk_id}"}
            for chunk_id in ids
        ], 0, {}


def test_recall_is_fraction_of_gold_passages_and_excludes_unanswerable_questions():
    questions = [
        {"id": "Q1", "question": "partial", "type": "factual", "answerable": True,
         "gold_chunk_ids": ["a", "b"]},
        {"id": "Q2", "question": "miss", "type": "factual", "answerable": True,
         "gold_chunk_ids": ["c"]},
        {"id": "Q3", "question": "unanswerable", "type": "unanswerable", "answerable": False,
         "gold_chunk_ids": ["b"]},
        {"id": "Q4", "question": "unlabelled", "type": "factual", "answerable": True,
         "gold_chunk_ids": []},
    ]

    report = run_evaluation(questions, FakeIndex())

    assert report["recall_at_5"] == pytest.approx(0.25)
    assert report["recall_graded_questions"] == 2
    assert [row["recall_at_5"] for row in report["rows"]] == [0.5, 0.0, None, None]


def test_recall_rejects_gold_ids_from_a_different_index():
    questions = [{
        "id": "Q1", "question": "partial", "type": "factual", "answerable": True,
        "gold_chunk_ids": ["not-in-index"],
    }]

    with pytest.raises(ValueError, match="not in the current index"):
        run_evaluation(questions, FakeIndex())


def test_recall_stays_unavailable_until_answerable_questions_are_labelled():
    questions = [{
        "id": "Q1", "question": "partial", "type": "factual", "answerable": True,
        "gold_chunk_ids": [],
    }]

    report = run_evaluation(questions, FakeIndex())

    assert report["recall_at_5"] is None
    assert report["recall_graded_questions"] == 0


def test_ablation_uses_passage_recall_not_binary_question_hits():
    questions = [{
        "id": "Q1", "question": "partial", "type": "factual", "answerable": True,
        "gold_chunk_ids": ["a", "b"],
    }]

    result = run_ablation(questions, FakeIndex(), top_k=5)

    assert result["hybrid"]["recall_at_k"] == pytest.approx(0.5)


def test_configuration_evaluation_reports_each_retrieval_setup_separately():
    class ConfigurationIndex(FakeIndex):
        def search(self, question, top_k, mode, rerank=True):
            ids = {
                ("bm25", False): ["a"],
                ("dense", False): ["b"],
                ("hybrid", False): ["b"],
                ("hybrid", True): ["a"],
            }[(mode, rerank)]
            return [{"chunk_id": chunk_id, "text": "payment"} for chunk_id in ids], 0, {}

    questions = [{
        "id": "Q1", "question": "payment deadline", "type": "factual",
        "answerable": True, "gold_chunk_ids": ["a"],
    }]
    results = run_configuration_evaluation(questions, ConfigurationIndex())

    assert results["bm25"]["recall_at_1"] == 1
    assert results["dense"]["recall_at_1"] == 0
    assert results["hybrid"]["recall_at_1"] == 0
    assert results["hybrid_rerank"]["recall_at_1"] == 1


def test_abstention_calibration_uses_the_same_support_threshold_as_answer_gate():
    class CalibrationIndex(FakeIndex):
        def search(self, question, top_k, mode, rerank=True):
            text = "payment" if question == "payment deadline" else "unrelated"
            return [{"chunk_id": "a", "text": text}], 0, {}

    questions = [
        {"id": "Q1", "question": "payment deadline", "type": "factual",
         "answerable": True, "gold_chunk_ids": ["a"]},
        {"id": "Q2", "question": "weather forecast", "type": "unanswerable",
         "answerable": False, "gold_chunk_ids": []},
    ]

    report = run_evaluation(questions, CalibrationIndex())
    low_threshold = next(row for row in report["threshold_sweep"] if row["threshold"] == 0.05)
    high_threshold = next(row for row in report["threshold_sweep"] if row["threshold"] == 0.75)

    assert low_threshold["false_abstain_rate"] == 0
    assert high_threshold["false_abstain_rate"] == 1


def test_system_report_marks_missing_dense_model_instead_of_claiming_hybrid():
    class BM25OnlyIndex(FakeIndex):
        embedder = None
        vectors = None
        cross = None

        def search(self, question, top_k, mode, rerank=True):
            return [{
                "chunk_id": "a",
                "text": "payment deadline",
                "filename": "contract.pdf",
                "page": 1,
            }], 0, {}

    questions = [{
        "id": "Q1", "question": "payment deadline", "type": "factual",
        "answerable": True, "gold_chunk_ids": ["a"],
    }]
    report, configurations = run_system_evaluation(questions, BM25OnlyIndex())

    assert configurations["dense"]["status"].startswith("Dense embeddings are unavailable")
    assert configurations["hybrid"]["recall_at_5"] is None
    assert report["recall_at_5"] == 1
    assert report["pipeline_status"].startswith("Degraded: dense embeddings unavailable")


def test_system_report_uses_source_backed_quick_benchmark_when_unlabelled():
    class SourceIndex(FakeIndex):
        def __init__(self):
            self.chunks = [
                {"chunk_id": "a", "text": "payment terms require monthly invoice delivery"},
                {"chunk_id": "b", "text": "termination requires written notice thirty days earlier"},
            ]

        def search(self, question, top_k, mode, rerank=True):
            passage = self.chunks[0]
            return [passage], 0, {}

    report, configurations = run_system_evaluation(
        default_questions(), SourceIndex()
    )

    assert report["evaluation_source"] == "synthetic_source_queries"
    assert report["total"] == 2
    assert report["recall_at_5"] == pytest.approx(0.5)
    assert "not held-out question-answer recall" in report["evaluation_note"]
    assert configurations["bm25"]["recall_at_5"] == pytest.approx(0.5)


def test_answer_gate_applies_the_configured_threshold():
    class AnswerIndex:
        def search(self, question, top_k, mode):
            passage = {
                "chunk_id": "a",
                "text": "payment",
                "score": 0.1,
                "filename": "contract.pdf",
                "page": 1,
            }
            return [passage], 0, {}

    index = AnswerIndex()
    accepted = answer_question("payment deadline", index, "", "", threshold=0.4)
    rejected = answer_question("payment deadline", index, "", "", threshold=0.6)

    assert accepted["confidence"] == "LOW"
    assert rejected["answer"] == ABSTAIN
    assert rejected["evidence_support_score"] == pytest.approx(0.5)

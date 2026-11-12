import re
import time

from lexisync.evidence import lexical_support_score

_SYNTHETIC_STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been",
    "being", "but", "can", "could", "does", "during", "each", "for",
    "from", "had", "has", "have", "her", "here", "him", "his", "into",
    "its", "may", "more", "most", "not", "our", "out", "shall", "should",
    "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "under", "was",
    "were", "what", "when", "where", "which", "who", "will", "with",
    "would", "you",
}


def default_questions():
    # These are editable templates; annotate evidence against the currently indexed documents.
    return [
      {"id":"Q01","question":"What is the purpose of the agreement?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q02","question":"Who are the parties to the document?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q03","question":"What is the effective date?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q04","question":"What does clause 1.1 define?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q05","question":"What is the payment deadline?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q06","question":"What are the termination rights?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q07","question":"What limitation of liability is specified?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q08","question":"Which obligations survive termination?","answerable":True,"gold_chunk_ids":[],"type":"multi-hop"},
      {"id":"Q09","question":"How does the indemnity clause interact with the liability cap?","answerable":True,"gold_chunk_ids":[],"type":"multi-hop"},
      {"id":"Q10","question":"What notice period is required?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q11","question":"Which law governs this agreement?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q12","question":"What dispute resolution procedure applies?","answerable":True,"gold_chunk_ids":[],"type":"clause"},
      {"id":"Q13","question":"What is the contract's confidential information definition?","answerable":True,"gold_chunk_ids":[],"type":"factual"},
      {"id":"Q14","question":"What is the CEO's home address?","answerable":False,"gold_chunk_ids":[],"type":"unanswerable"},
      {"id":"Q15","question":"What is the private bank password of the supplier?","answerable":False,"gold_chunk_ids":[],"type":"unanswerable"},
      {"id":"Q16","question":"What was the weather on the signing date?","answerable":False,"gold_chunk_ids":[],"type":"unanswerable"},
    ]


def _synthetic_retrieval_questions(chunks, limit=16):
    eligible = []
    for chunk in chunks:
        terms = [
            term for term in re.findall(r"[a-z0-9]+", chunk.get("text", "").lower())
            if len(term) > 2 and term not in _SYNTHETIC_STOP_WORDS
        ]
        if len(terms) >= 4:
            eligible.append((chunk["chunk_id"], terms))
    if not eligible:
        return []

    if len(eligible) > limit:
        last_index = len(eligible) - 1
        selected_indices = {
            round(index * last_index / (limit - 1)) if limit > 1 else 0
            for index in range(limit)
        }
        eligible = [eligible[index] for index in sorted(selected_indices)]

    return [
        {
            "id": f"SYN{index:02d}",
            "question": "Locate the passage about " + " ".join(terms[:12]),
            "answerable": True,
            "gold_chunk_ids": [chunk_id],
            "type": "synthetic-source-query",
        }
        for index, (chunk_id, terms) in enumerate(eligible, start=1)
    ]


def _gold_chunk_ids(question, index):
    gold = set(question.get("gold_chunk_ids", []))
    available = {chunk["chunk_id"] for chunk in index.chunks}
    unknown = gold - available
    if unknown:
        raise ValueError(
            f"Question {question['id']} has gold chunk IDs that are not in the current index: "
            + ", ".join(sorted(unknown))
        )
    return gold


def _recall_at_k(retrieved_ids, gold_ids):
    if not gold_ids:
        return None
    return len(set(retrieved_ids) & gold_ids) / len(gold_ids)


def _unavailable_reason(index, requires_dense, requires_reranker):
    if requires_dense and hasattr(index, "embedder") and (
        index.embedder is None or getattr(index, "vectors", None) is None
    ):
        return "Dense embeddings are unavailable; verify the embedding model loaded."
    if requires_reranker and hasattr(index, "cross") and index.cross is None:
        return "The cross-encoder is unavailable; verify the reranker model loaded."
    return None


def _evaluate_configuration(questions, index, mode, rerank, top_k=5,
                            requires_dense=False, requires_reranker=False):
    unavailable = _unavailable_reason(index, requires_dense, requires_reranker)
    if unavailable:
        return {
            "status": unavailable,
            "rows": [],
            "recall_at_1": None,
            "recall_at_3": None,
            "recall_at_5": None,
            "recall_graded_questions": 0,
            "mean_latency_ms": None,
        }

    rows = []
    recalls = {1: [], 3: [], 5: []}
    latencies = []
    for question in questions:
        started = time.perf_counter()
        passages, _, _ = index.search(
            question["question"], top_k=top_k, mode=mode, rerank=rerank
        )
        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        retrieved_ids = [passage["chunk_id"] for passage in passages]
        gold = _gold_chunk_ids(question, index)
        answerable = question["answerable"]
        question_recalls = {
            k: _recall_at_k(retrieved_ids[:k], gold) if answerable else None
            for k in recalls
        }
        for k, recall in question_recalls.items():
            if recall is not None:
                recalls[k].append(recall)
        rows.append({
            "id": question["id"],
            "type": question["type"],
            "answerable": answerable,
            "question": question["question"],
            "gold_chunk_ids": sorted(gold),
            "retrieved_passages": len(passages),
            "gold_evidence_configured": bool(gold),
            "gold_passages": len(gold),
            "retrieved_gold_passages": len(set(retrieved_ids) & gold) if answerable else None,
            "retrieved_chunk_ids": retrieved_ids,
            "missed_gold_chunk_ids": sorted(gold - set(retrieved_ids)) if answerable else [],
            "recall_at_1": question_recalls[1],
            "recall_at_3": question_recalls[3],
            "recall_at_5": question_recalls[5],
            "evidence_support_score": lexical_support_score(question["question"], passages),
            "latency_ms": round(latency_ms, 1),
            "manual_answer_correctness": "NOT GRADED",
            "citation_precision": "NOT GRADED",
        })

    return {
        "status": "Ready",
        "rows": rows,
        "recall_at_1": sum(recalls[1]) / len(recalls[1]) if recalls[1] else None,
        "recall_at_3": sum(recalls[3]) / len(recalls[3]) if recalls[3] else None,
        "recall_at_5": sum(recalls[5]) / len(recalls[5]) if recalls[5] else None,
        "recall_graded_questions": len(recalls[5]),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
    }


def run_configuration_evaluation(questions, index):
    configurations = [
        ("bm25", "BM25 only", "bm25", False, False, False, "BM25"),
        ("dense", "Dense only", "dense", False, True, False, "all-MiniLM-L6-v2"),
        ("hybrid", "Hybrid (BM25 + dense, RRF)", "hybrid", False, True, False,
         "BM25 + all-MiniLM-L6-v2"),
        ("hybrid_rerank", "Hybrid + cross-encoder rerank", "hybrid", True, True, True,
         "BM25 + all-MiniLM-L6-v2 + ms-marco-MiniLM-L-6-v2"),
    ]
    results = {}
    for key, label, mode, rerank, requires_dense, requires_reranker, model_status in configurations:
        result = _evaluate_configuration(
            questions,
            index,
            mode,
            rerank,
            requires_dense=requires_dense,
            requires_reranker=requires_reranker,
        )
        result["label"] = label
        result["models_loaded"] = model_status if result["status"] == "Ready" else result["status"]
        results[key] = result
    return results


def _threshold_sweep(rows, operating_threshold):
    thresholds = sorted({round(step / 20, 2) for step in range(1, 20)} | {operating_threshold})
    labelled_answerable = [
        row for row in rows if row["answerable"] and row["gold_evidence_configured"]
    ]
    unanswerable = [row for row in rows if not row["answerable"]]
    calibration_rows = []
    for threshold in thresholds:
        true_abstentions = sum(
            row["evidence_support_score"] < threshold for row in unanswerable
        )
        false_abstentions = sum(
            row["evidence_support_score"] < threshold for row in labelled_answerable
        )
        denominator = len(labelled_answerable) + len(unanswerable)
        accuracy = (
            (true_abstentions + len(labelled_answerable) - false_abstentions) / denominator
            if labelled_answerable and unanswerable else None
        )
        calibration_rows.append({
            "threshold": threshold,
            "abstention_accuracy": accuracy,
            "false_abstain_rate": (
                false_abstentions / len(labelled_answerable) if labelled_answerable else None
            ),
            "unanswerable_abstain_rate": (
                true_abstentions / len(unanswerable) if unanswerable else None
            ),
            "answerable_questions": len(labelled_answerable),
            "unanswerable_questions": len(unanswerable),
            "is_operating_threshold": threshold == operating_threshold,
        })
    return calibration_rows


def _summarize_pipeline(result, total, operating_threshold):
    rows = result["rows"]
    threshold_sweep = _threshold_sweep(rows, operating_threshold)
    operating_point = next(
        row for row in threshold_sweep if row["is_operating_threshold"]
    )
    report_rows = []
    for row in rows:
        report_row = dict(row)
        report_row["recall_at_5"] = row["recall_at_5"]
        report_row["retrieval_only_abstention"] = (
            row["evidence_support_score"] < operating_threshold
        )
        report_rows.append(report_row)
    return {
        "total": total,
        "recall_at_5": result["recall_at_5"],
        "recall_graded_questions": result["recall_graded_questions"],
        "abstention_accuracy": operating_point["abstention_accuracy"],
        "false_abstain_rate": operating_point["false_abstain_rate"],
        "mean_latency_ms": result["mean_latency_ms"] or 0,
        "operating_threshold": operating_threshold,
        "threshold_sweep": threshold_sweep,
        "rows": report_rows,
        "note": (
            "Recall@k is mean per-question passage recall over answerable questions with "
            "manually verified gold_chunk_ids. Abstention calibration uses the same lexical "
            "support score and threshold as the answer gate; answer correctness and citation "
            "precision still require manual grading."
        ),
    }


def run_evaluation(questions, index, operating_threshold=0.42):
    pipeline = _evaluate_configuration(questions, index, "hybrid", True)
    if pipeline["status"] != "Ready":
        raise RuntimeError(f"Full-pipeline evaluation failed: {pipeline['status']}")
    return _summarize_pipeline(pipeline, len(questions), operating_threshold)


def run_system_evaluation(questions, index, operating_threshold=0.42):
    verified_question_count = sum(
        question["answerable"] and bool(question.get("gold_chunk_ids"))
        for question in questions
    )
    evaluation_questions = questions
    evaluation_source = "verified_gold"
    if not verified_question_count:
        synthetic_questions = _synthetic_retrieval_questions(index.chunks)
        if synthetic_questions:
            evaluation_questions = synthetic_questions
            evaluation_source = "synthetic_source_queries"

    configurations = run_configuration_evaluation(evaluation_questions, index)
    pipeline = configurations["hybrid_rerank"]
    pipeline_status = "Ready"
    if pipeline["status"] != "Ready":
        pipeline = _evaluate_configuration(evaluation_questions, index, "hybrid", True)
        if configurations["dense"]["status"] != "Ready":
            pipeline_status = (
                "Degraded: dense embeddings unavailable; full pipeline used BM25 fallback."
            )
        else:
            pipeline_status = (
                "Degraded: cross-encoder unavailable; full pipeline ran without reranking."
            )
    report = _summarize_pipeline(
        pipeline, len(evaluation_questions), operating_threshold
    )
    report["pipeline_status"] = pipeline_status
    report["evaluation_source"] = evaluation_source
    if evaluation_source == "synthetic_source_queries":
        report["evaluation_note"] = (
            "Quick diagnostic only: each query was generated from indexed text and its source "
            "chunk is the gold passage. This is not held-out question-answer recall. Replace it "
            "with manually verified gold evidence for a defensible benchmark."
        )
    return report, configurations


def run_ablation(questions, index, top_k=5):
    """Compare dense-only and hybrid retrieval on the same held-out questions."""
    results = {}
    for mode in ("dense", "hybrid"):
        result = _evaluate_configuration(
            questions, index, mode, False, top_k=top_k, requires_dense=(mode == "dense")
        )
        results[mode] = {
            "recall_at_k": result["recall_at_5"],
            "mean_latency_ms": result["mean_latency_ms"] or 0,
            "rows": result["rows"],
            "status": result["status"],
        }
    return results

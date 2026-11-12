import streamlit as st
from pathlib import Path
import tempfile, time, os, json
from dotenv import load_dotenv
from lexisync.ingestion import ingest_pdf
from lexisync.indexing import HybridIndex
from lexisync.answering import answer_question
from lexisync.evaluation import run_system_evaluation, default_questions

load_dotenv()
st.set_page_config(page_title="LexiSyncAI", page_icon="⚖️", layout="wide")
st.title("LexiSyncAI")
st.caption("Evidence-Grounded Legal Document Analyst | Hybrid Retrieval · Reranking · Cited Answers · Abstention")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Groq API key", value=os.getenv("GROQ_API_KEY", ""), type="password",
                            help="Create a key at https://console.groq.com/keys")
    model = st.selectbox("Groq model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant",
                                        "openai/gpt-oss-20b", "openai/gpt-oss-120b"])
    st.caption("The key is used only for this session unless you add it to .env.")
    st.divider()
    st.markdown("**Suggested demo:** upload 2–5 public legal PDFs, index, ask a clause question, then ask something absent.")

if "documents" not in st.session_state: st.session_state.documents = []
if "index" not in st.session_state: st.session_state.index = None
if "chunks" not in st.session_state: st.session_state.chunks = []
if "last_result" not in st.session_state: st.session_state.last_result = None
if "eval_questions" not in st.session_state: st.session_state.eval_questions = default_questions()
if "eval_version" not in st.session_state: st.session_state.eval_version = 0
if "eval_candidates" not in st.session_state: st.session_state.eval_candidates = {}

tabs = st.tabs(["1 · Case & documents", "2 · Ask LexiSyncAI", "3 · Evaluation", "4 · Architecture & methods"])

with tabs[0]:
    st.subheader("Create a case and upload public legal documents")
    case_name = st.text_input("Case name", value="Legal document review")
    uploads = st.file_uploader("Upload 2–5 public legal PDF files", type=["pdf"], accept_multiple_files=True)
    if uploads:
        st.info(f"{len(uploads)} PDF(s) selected. The assignment specifies a bundle of 2–5 public PDFs.")
        if len(uploads) > 5:
            st.error("Please upload no more than five PDFs.")
        if st.button("Ingest documents", type="primary", disabled=len(uploads) > 5):
            all_chunks, summaries = [], []
            progress = st.progress(0)
            for i, up in enumerate(uploads):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(up.getvalue()); path = tmp.name
                try:
                    result = ingest_pdf(path, up.name)
                    all_chunks.extend(result["chunks"])
                    summaries.append(result["summary"])
                finally:
                    os.unlink(path)
                progress.progress((i+1)/len(uploads))
            st.session_state.documents = summaries
            st.session_state.chunks = all_chunks
            st.session_state.index = None
            st.session_state.eval_questions = default_questions()
            st.session_state.eval_candidates = {}
            st.session_state.eval_version += 1
            st.session_state.pop("eval_report", None)
            st.session_state.pop("ablation", None)
            st.success(f"Ingestion complete: {len(summaries)} document(s), {len(all_chunks)} evidence chunks.")
    if st.session_state.documents:
        st.markdown("#### Detected document structure")
        for d in st.session_state.documents:
            with st.expander(f"{d['filename']} · {d['pages']} pages"):
                st.write(f"Digital text pages: {d['digital_pages']} · Scanned/image-only pages: {d['scanned_pages']}")
                st.write(f"Detected sections/headings: {d['sections']}")
                st.write(f"Tables detected: {d['tables']}")
                if d.get("warnings"): st.warning(" · ".join(d["warnings"]))
        if st.button("Build hybrid search index", type="primary"):
            with st.spinner("Building BM25 and dense embedding indexes; loading cross-encoder..."):
                st.session_state.index = HybridIndex(st.session_state.chunks)
            st.success("Index ready. BM25 + dense vectors + cross-encoder reranker initialized.")
    st.caption("Scanned pages are detected. OCR fallback uses Tesseract when installed; otherwise they are clearly flagged.")

with tabs[1]:
    st.subheader("Ask a question grounded in the uploaded documents")
    if not st.session_state.index:
        st.warning("Upload and index documents first.")
    else:
        q = st.text_area("Legal question", placeholder='e.g. Does clause 7.2 limit liability for the indemnities in Section 9?', height=90)
        c1,c2 = st.columns([1,1])
        with c1: top_k = st.slider("Evidence passages", 3, 12, 6)
        with c2: threshold = st.slider("Abstention threshold", 0.10, 0.95, 0.42, 0.01)
        if st.button("Ask LexiSyncAI", type="primary", disabled=not q.strip()):
            start = time.perf_counter()
            with st.spinner("Retrieving, reranking and verifying evidence..."):
                result = answer_question(q, st.session_state.index, api_key, model, top_k=top_k, threshold=threshold)
            result["latency_seconds"] = round(time.perf_counter()-start, 2)
            st.session_state.last_result = result
        r = st.session_state.last_result
        if r:
            status = r["confidence"]
            if status == "INSUFFICIENT EVIDENCE": st.error("INSUFFICIENT EVIDENCE · Abstained")
            elif status == "HIGH": st.success("HIGH confidence")
            elif status == "MEDIUM": st.info("MEDIUM confidence")
            else: st.warning("LOW confidence")
            st.markdown("### Answer")
            st.write(r["answer"])
            st.caption(f"Latency: {r.get('latency_seconds','—')} seconds · Retrieval score: {r.get('retrieval_score',0):.3f}")
            if r.get("explanation"): st.markdown("**Grounding explanation:** " + r["explanation"])
            st.markdown("### Citations and evidence")
            if not r.get("citations"):
                st.write("No sufficiently reliable supporting passage was found.")
            for cit in r.get("citations", []):
                with st.expander(f"{cit['filename']} · Page {cit['page']} · {cit.get('section') or 'Section not detected'}"):
                    st.markdown(cit["text"])
                    st.caption(f"Chunk: {cit['chunk_id']} · Source region: {cit.get('bbox','not available')}")
            if r.get("conflicts"):
                st.warning("Potential cross-document conflict detected")
                for conflict in r["conflicts"]: st.write("• " + conflict)
            with st.expander("Retrieval trace (technical demo)"):
                st.json(r.get("trace", {}))

with tabs[2]:
    st.subheader("Evaluation suite")
    st.write("The assignment requires ≥15 held-out questions, including factual, multi-hop and ≥3 unanswerable questions.")
    st.warning("The 16 starter questions are templates, not document-specific ground truth. Edit them for your PDFs and manually label supporting passages before using recall as a result.")
    if st.session_state.index:
        questions = st.session_state.eval_questions
        version = st.session_state.eval_version
        with st.expander("Prepare questions and gold evidence", expanded=not any(
                q["answerable"] and q.get("gold_chunk_ids") for q in questions)):
            qids = [q["id"] for q in questions]
            selected_id = st.selectbox("Question to edit", qids, key=f"eval_selected_{version}")
            question_row = next(q for q in questions if q["id"] == selected_id)
            question_text = st.text_area(
                "Document-specific question",
                value=question_row["question"],
                key=f"eval_question_{version}_{selected_id}",
            )
            if question_text != question_row["question"]:
                question_row["question"] = question_text
                question_row["gold_chunk_ids"] = []
                st.session_state.eval_candidates.pop(selected_id, None)
                st.session_state.pop(f"eval_gold_{version}_{selected_id}", None)
                st.session_state.pop("eval_report", None)
                st.session_state.pop("ablation", None)
            answerable = st.checkbox(
                "Answerable from the indexed documents",
                value=question_row["answerable"],
                key=f"eval_answerable_{version}_{selected_id}",
            )
            if answerable != question_row["answerable"]:
                question_row["answerable"] = answerable
                if not answerable:
                    question_row["gold_chunk_ids"] = []
                    st.session_state.pop(f"eval_gold_{version}_{selected_id}", None)
                st.session_state.pop("eval_report", None)
                st.session_state.pop("ablation", None)
            st.caption("Replace the template with a held-out question about these PDFs. Mark a question unanswerable only after checking the relevant documents.")

            if st.button("Find candidate evidence", key=f"eval_find_{version}_{selected_id}",
                         disabled=not question_text.strip()):
                candidates, _, _ = st.session_state.index.search(
                    question_text, top_k=min(20, len(st.session_state.chunks)), mode="hybrid"
                )
                st.session_state.eval_candidates[selected_id] = {
                    "question": question_text,
                    "chunk_ids": [p["chunk_id"] for p in candidates],
                }

            candidate_set = st.session_state.eval_candidates.get(selected_id, {})
            candidate_ids = (
                candidate_set.get("chunk_ids", [])
                if candidate_set.get("question") == question_text else []
            )
            chunk_by_id = {chunk["chunk_id"]: chunk for chunk in st.session_state.chunks}
            filenames = sorted({chunk["filename"] for chunk in st.session_state.chunks})
            selected_filename = st.selectbox(
                "Browse indexed document",
                filenames,
                key=f"eval_document_{version}_{selected_id}",
            )
            document_pages = sorted({
                chunk["page"] for chunk in st.session_state.chunks
                if chunk["filename"] == selected_filename
            })
            selected_page = st.selectbox(
                "Browse page",
                document_pages,
                key=f"eval_page_{version}_{selected_id}",
            )
            page_chunk_ids = [
                chunk["chunk_id"] for chunk in st.session_state.chunks
                if chunk["filename"] == selected_filename and chunk["page"] == selected_page
            ]
            existing_gold = [
                chunk_id for chunk_id in question_row.get("gold_chunk_ids", [])
                if chunk_id in chunk_by_id
            ]
            options = list(dict.fromkeys(candidate_ids + page_chunk_ids + existing_gold))
            def passage_label(chunk_id):
                passage = chunk_by_id[chunk_id]
                section = f" · {passage['section']}" if passage.get("section") else ""
                excerpt = " ".join(passage["text"].split())[:140]
                return f"{passage['filename']} · p. {passage['page']}{section} · {excerpt}"

            if candidate_ids:
                st.caption("Search candidates are suggestions. Browse the full indexed document and verify gold passages against the source PDFs.")
            selected_gold = st.multiselect(
                "Gold supporting passages",
                options=options,
                default=existing_gold,
                format_func=passage_label,
                key=f"eval_gold_{version}_{selected_id}",
                disabled=not answerable,
            )
            if selected_gold != question_row.get("gold_chunk_ids", []):
                question_row["gold_chunk_ids"] = selected_gold
                st.session_state.pop("eval_report", None)
                st.session_state.pop("ablation", None)
            st.markdown(f"**Indexed passages: {selected_filename} · Page {selected_page}**")
            with st.container(border=True):
                for chunk_id in page_chunk_ids:
                    passage = chunk_by_id[chunk_id]
                    st.markdown(f"**{passage.get('section') or 'Section not detected'}**")
                    st.write(passage["text"])

            st.caption(
                f"Gold passages configured for {sum(bool(q.get('gold_chunk_ids')) for q in questions)} "
                f"of {sum(q['answerable'] for q in questions)} answerable questions."
            )

        if st.button("Run retrieval and abstention evaluation"):
            with st.spinner("Running evaluation set..."):
                report, configurations = run_system_evaluation(questions, st.session_state.index)
                report["configurations"] = configurations
            st.session_state.eval_report = report
        if "eval_report" in st.session_state:
            rep = st.session_state.eval_report
            synthetic_evaluation = rep.get("evaluation_source") == "synthetic_source_queries"
            recall_label = (
                "Recall@5 (quick diagnostic)"
                if synthetic_evaluation else "Recall@5 (verified gold)"
            )
            a,b,c,d = st.columns(4)
            a.metric("Questions used", rep["total"])
            b.metric(recall_label, f"{rep['recall_at_5']:.1%}" if rep["recall_at_5"] is not None else "N/A")
            c.metric("Abstention accuracy", f"{rep['abstention_accuracy']:.1%}" if rep["abstention_accuracy"] is not None else "N/A")
            d.metric("Mean latency (full)", f"{rep['mean_latency_ms']:.0f} ms")
            if synthetic_evaluation:
                st.warning(rep["evaluation_note"])
            else:
                st.caption(
                    f"Recall@5 is mean per-question passage recall over "
                    f"{rep['recall_graded_questions']} answerable questions with verified gold evidence."
                )
            if rep["recall_at_5"] is None:
                st.info(
                    "No usable source passages were available for the quick benchmark. Select "
                    "verified gold passages in “Prepare questions and gold evidence” to calculate recall."
                )
            configuration_title = (
                "Recall@k per configuration (source-derived quick check)"
                if synthetic_evaluation else "Recall@k per configuration"
            )
            st.markdown(f"#### {configuration_title}")
            configuration_rows = [
                {
                    "Configuration": result["label"],
                    "Models loaded": result["models_loaded"],
                    "Recall@1": result["recall_at_1"],
                    "Recall@3": result["recall_at_3"],
                    "Recall@5": result["recall_at_5"],
                    "Latency (ms)": (
                        round(result["mean_latency_ms"], 1)
                        if result["mean_latency_ms"] is not None else None
                    ),
                    "Status": result["status"],
                }
                for result in rep["configurations"].values()
            ]
            st.dataframe(configuration_rows, use_container_width=True, hide_index=True)
            if any(result["status"] != "Ready" for result in rep["configurations"].values()):
                st.warning(
                    "Some configurations were not measured because their models did not load. "
                    "They are marked unavailable rather than being reported as BM25 results."
                )
            if rep.get("pipeline_status") != "Ready":
                st.warning(rep["pipeline_status"])
            st.markdown("#### Abstention-gate calibration (threshold sweep)")
            st.dataframe(rep["threshold_sweep"], use_container_width=True, hide_index=True)
            st.caption(
                f"The answer gate uses the same lexical evidence-support score and threshold "
                f"({rep['operating_threshold']:.2f}) shown here. False-abstain rate is the share "
                "of answerable questions with verified gold evidence that the gate rejects."
            )
            st.markdown("#### Per-question results")
            st.dataframe(rep["rows"], use_container_width=True)
            zero_recall_rows = [
                row for row in rep["rows"]
                if row["answerable"] and row["gold_evidence_configured"] and row["recall_at_5"] == 0
            ]
            if zero_recall_rows:
                st.warning(
                    "A 0% result means the labeled evidence was not in that question's top five. "
                    "Inspect the gold and retrieved passages below; verify the question and labels against the PDF."
                )
                with st.expander("Diagnose zero-recall questions"):
                    missed_id = st.selectbox(
                        "Question with zero recall",
                        [row["id"] for row in zero_recall_rows],
                        key=f"eval_zero_recall_{version}",
                    )
                    missed_row = next(row for row in zero_recall_rows if row["id"] == missed_id)
                    st.write(missed_row["question"])
                    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in st.session_state.chunks}
                    gold_col, retrieved_col = st.columns(2)
                    with gold_col:
                        st.markdown("**Labeled gold evidence**")
                        for chunk_id in missed_row["gold_chunk_ids"]:
                            passage = chunk_by_id.get(chunk_id)
                            if passage:
                                st.caption(f"{passage['filename']} · Page {passage['page']} · {chunk_id}")
                                st.write(passage["text"])
                    with retrieved_col:
                        st.markdown("**Actually retrieved top five**")
                        for chunk_id in missed_row["retrieved_chunk_ids"]:
                            passage = chunk_by_id.get(chunk_id)
                            if passage:
                                st.caption(f"{passage['filename']} · Page {passage['page']} · {chunk_id}")
                                st.write(passage["text"])
            st.download_button("Download evaluation report (JSON)", json.dumps(rep, indent=2), "evaluation_report.json")
    else:
        st.info("Index documents to enable evaluation.")

with tabs[3]:
    st.subheader("System architecture")
    st.code("""Public legal PDFs (2–5)
        │
        ▼
PyMuPDF extraction ── reading order / page metadata / scanned-page detection
        │                         └── Tesseract OCR fallback
        ▼
Clause- and section-aware chunking (+ table extraction)
        │
        ├── BM25 lexical index (exact clause numbers such as 7.2)
        └── SentenceTransformer dense embeddings
                    │
                    ▼
          Reciprocal Rank Fusion (hybrid merge)
                    │
                    ▼
          Cross-encoder reranking
                    │
                    ▼
        Evidence sufficiency / abstention gate
                    │
                    ▼
           Groq-hosted LLM (context only)
                    │
                    ▼
    Claim-level page + section citations / conflict warning
                    │
                    ▼
      UI + evaluation (recall@k, citation precision,
             abstention, latency, ablations)""", language="text")
    st.markdown("**Core implementation:** `lexisync/ingestion.py`, `chunking.py`, `indexing.py`, `answering.py`, `evaluation.py`.")
    st.markdown("**Important:** Groq is an API-hosted LLM, not a local LLM. The assignment explicitly permits an API if a local model cannot be used.")

# LexiSyncAI — Evidence-Grounded Legal Document Analyst

A Streamlit web application for ingesting public legal PDFs, building a hybrid BM25 + dense retrieval index, rank-fusing cross-encoder reranking with retrieval results, and generating grounded answers through the Groq API.

## 1. Requirements

- Python 3.10 or 3.11 recommended.
- Internet access on first run to download SentenceTransformers models and call Groq.
- A Groq API key: https://console.groq.com/keys
- Optional OCR: install the Tesseract OCR system binary. `pytesseract` alone is not sufficient.

> **API/local-model clarification:** Groq is an API-hosted inference service, not a local LLM. The assignment explicitly allows an API if a local model cannot be used. Do not describe this implementation as fully local.

## 2. Run

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
streamlit run app.py
```

### macOS/Linux
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_...
streamlit run app.py
```

The API key can alternatively be pasted into the sidebar at runtime. Never commit `.env` to GitHub.

## 3. Demo flow

1. Open the app and create a case name.
2. Upload 2–5 public legal PDFs (no confidential/client material).
3. Click **Ingest documents**. Review page counts, scanned-page flags, detected headings and tables.
4. Click **Build hybrid search index**. First run downloads the embedding and cross-encoder models.
5. Ask a factual clause question and inspect the cited source pages.
6. Ask a question whose answer is absent from the PDFs; demonstrate the insufficient-evidence response.
7. In the evaluation tab, edit the starter questions to match the indexed PDFs, find candidate passages, and manually select every passage that supports each answerable question. Then run the evaluation and explain which values are measured and which require manual grading.

## 4. Architecture

PDF → PyMuPDF page/block extraction → scanned-page detection / optional Tesseract OCR → section-aware chunks → BM25 + transformer embeddings → reciprocal-rank fusion + cross-encoder rank fusion → evidence sufficiency gate → Groq LLM → cited answer and confidence.

## 5. Requirement traceability

| Brief requirement | Implementation |
|---|---|
| Digital PDF parsing / reading order | `ingestion.py`, PyMuPDF block extraction with sorted reading order |
| Scanned page flag + OCR fallback | `ingestion.py`, optional Tesseract |
| Clause/section-aware chunks | `chunking.py` |
| BM25 exact clause retrieval | `indexing.py` |
| Dense transformer embeddings | `all-MiniLM-L6-v2` |
| Hybrid merge | Reciprocal Rank Fusion |
| Cross-encoder reranker | `ms-marco-MiniLM-L-6-v2` |
| Recall@k | Evaluation scaffold; requires manually labelled gold chunk IDs |
| Grounded generation | `answering.py`, retrieved passages only in prompt |
| Page/section citations | Citation evidence panels in UI |
| Abstention | Evidence gate and explicit abstention state |
| ≥15 held-out questions | 16-question starter scaffold; customize to the selected PDFs |
| Retrieval configurations | Evaluation compares BM25-only, dense-only, hybrid RRF, and hybrid + cross-encoder with the same labelled questions |
| Latency | Search and end-to-end timing |
| Multi-document conflict | Prompt instructions + conflict field from LLM |
| Table extraction | PyMuPDF table detection count; full merged-cell reconstruction is not implemented |
| Bounding-box citations | Source block bounding boxes retained as metadata; chunk-level region mapping is not yet implemented |
| Public deployment | Streamlit Community Cloud / Hugging Face Spaces compatible with a suitable requirements file |

## 6. Evaluation protocol — important

The starter set contains 16 question templates, including three unanswerable questions. They are **not** a valid final benchmark until adapted to the PDFs used in the experiment.

For a defensible report:
1. Write at least 15 held-out questions from the chosen legal document bundle.
2. Include factual questions, at least two-page multi-hop questions, and at least three unanswerable questions.
3. In **Evaluation → Prepare questions and gold evidence**, rewrite the templates for the uploaded PDFs, find candidate passages, and manually select all supporting passages for every answerable question. Verify them against the source PDFs; candidate ranking is not ground truth.
4. Record the expected answer and manually grade correctness.
5. Compare Recall@1, Recall@3 and Recall@5 for BM25-only, dense-only, hybrid RRF, and hybrid + cross-encoder using the same labelled questions. Recall is the mean per-question fraction of gold passages retrieved. If no gold labels are present, the dashboard runs a source-derived quick retrieval diagnostic; it is not held-out question-answer recall. A configuration is marked unavailable when its model did not load instead of silently substituting BM25.
6. Review the abstention threshold sweep. It uses the same lexical evidence-support score and threshold as the answer gate; false-abstain rate is measured only on answerable questions with gold labels. Citation precision (claims with correct supporting citations / all cited claims) and answer correctness still require manual grading.
7. Use the same question set and report the exact setup for every comparison.
8. Report failures and limitations honestly. Do not claim metrics that have not been measured.

## 7. Technical limitations to disclose

- Reading order is based on PyMuPDF's sorted text blocks. Complex multi-column layouts, footnotes and headers may need manual validation.
- OCR requires the Tesseract executable to be installed separately; OCR quality must be sampled and measured.
- Table detection currently reports table counts; complete merged-cell/spanning-header reconstruction and table-grounded QA are future work.
- Bounding boxes are retained at block level, but precise claim-to-region citations need additional mapping.
- The current abstention gate is a lexical-support heuristic, not a calibrated classifier. Use the threshold sweep on a validation set and report its limitations.
- Cross-document conflict detection depends on evidence retrieval and LLM interpretation; verify conflicts manually.
- The evaluation starter does not invent gold answers. When there are no manual gold labels, the dashboard creates a synthetic retrieval diagnostic whose query and known source chunk come from indexed text. This can smoke-test retrieval but is not held-out QA Recall@k; manually select verified gold evidence for a defensible result. Labels referencing chunks outside the current index are rejected. Recall is not a binary question-hit rate. Citation precision and answer correctness still require manual grading.
- A zero Recall@5 score means the labeled evidence did not appear in the top five. The evaluation view shows the gold and retrieved passages for diagnosis; verify the question and gold labels instead of changing labels to match retrieval.
- Groq model availability and rate limits may change. Select a model available to your account.

## 8. Suggested public legal sources

Use public, non-confidential PDFs from EUR-Lex, SEC EDGAR, legislation.gov.uk, or CourtListener. Keep the chosen bundle and question set fixed for reproducibility.

## 9. GitHub submission

```bash
git init
git add .
git commit -m "Initial LexiSyncAI implementation"
git add .
git commit -m "Add evaluation and documentation"
```

Push the repository and put its URL in the documentation submitted to eCampus. Do not upload `.env` or expose the Groq key.

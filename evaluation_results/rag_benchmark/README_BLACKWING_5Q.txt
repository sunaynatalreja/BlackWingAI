BLACKWING AI | Five-question source-level relevance review
2026-11-20 | ANALYST-ASSISTED DRAFT (human gold sign-off not yet complete)

Contents:
 - BLACKWING_5Q_Source_Relevance_Review.xlsx (5-question summary, 350 source judgments, limitations)
 - blackwing_5q_document_relevance_labels.csv (machine-readable labels for scoring)
 - blackwing_5q_label_summary.csv (five selected gold questions)
 - blackwing_5q_label_methodology.json (audit/limitations)
 - blackwing_5q_score_source_recall.py (READ-ONLY postprocessor)

Data reviewed: original BLACKWING_DATA_7_AGENT_RAG_READY.zip source manifest and
non-protected factual source texts. 70 non-protected source catalogue entries,
49 of which appear in the ZIP's separate FTS5 index. This is NOT assumed to
be identical to the user's separate BGE global index (50 non-protected sources).

Gold questions: BW-G001, BW-G002, BW-G005 (JFK), BW-G011, BW-G014 (Austin).
Relevant document labels, grade >=1:
 BW-G001 : BW-EV-001, BW-EV-002, BW-EV-003
 BW-G002 : BW-EV-001 (full), BW-EV-002 and BW-EV-003 (partial)
 BW-G005 : BW-EV-001, BW-EV-002, BW-EV-003
 BW-G011 : BW-EV-006
 BW-G014 : BW-EV-006

Windows PowerShell:
 1) Extract all files into your EXISTING repo's evaluation_results\rag_benchmark folder.
 2) From repo root, enter:

$bge = "$HOME\OneDrive\Documents\ChatGPT\Multi model RAG\BLACKWING_LOCAL_DELIVERY\04_GLOBAL_RAG_INDEX\global_v3_blind_20261118"
python .\evaluation_results\rag_benchmark\blackwing_5q_score_source_recall.py `
  --index "$bge" `
  --benchmark ".\evaluation_results\rag_benchmark\rag_benchmark_results.json" `
  --labels ".\evaluation_results\rag_benchmark\blackwing_5q_document_relevance_labels.csv" `
  --out ".\evaluation_results\rag_benchmark"

The script reads the existing global BGE chunks.jsonl and saved four-method
benchmark JSON. It does NOT rerun retrieval, use an LLM, call IIT API, or modify
the global index. It STOPS if any indexed source lacks a relevance judgment.

Outputs, only if index membership matches labels:
 blackwing_5q_source_recall_details.csv
 blackwing_5q_source_recall_summary.csv
 blackwing_5q_source_recall_results.json

Measured unit: fraction of all reviewed relevant SOURCE DOCUMENTS in the
actual global BGE index represented in the first K retrieved PASSAGE hits.
This is source-document coverage at passage cutoff K; NOT exhaustive chunk-
level Recall@K and not standard Recall@K on K uniquely ranked documents.

Submission caveat: technical trial only; no human mentor validation of draft
source labels; all other nonprotected documents still need eligibility approval.
Protected BW-HO-001..009 are excluded from label/scoring universe.

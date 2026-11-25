# Gold comparison score

The report view adds **Gold comparison score**, independent of Prompt Completeness,
Report Completeness and aviation risk. Existing scores, weights, risk calculations,
case-match labels and report content are unchanged.

After an investigation completes, choose **Compare with gold report** in its Report
tab. The server selects the matching case from nine preserved final reports. It
never averages unrelated incidents. Synthetic or ambiguous cases show Not applicable.
Use a real external incident identifier or an accurate airport and incident date.

Only asserted factual claims classified as documented facts are compared. Identical
claims are deduplicated. Hypotheses, unknowns, abstentions and risk ratings are not
treated as factual errors. Timeline entries marked unknown are not promoted to facts.

The hosted model compares each factual claim with two text-selected pages from the
matching gold PDF. Every supported, partly supported or contradicted decision must
include an exact quotation from a supplied page. Invalid quotations or malformed
model output withhold the numerical score. Transport/provider failures assign no
score. A completed evaluation uses equal claim weights: supported=1, partly supported=0.5,
contradicted/not verified=0; score=100 × awarded points / asserted factual claims.
Empty reports receive no score. Not verified means no support established in the
selected passages, not necessarily a false assertion. Quote checks cannot prove
semantic entailment. The score requires human review and is not full gold-report
coverage, factual accuracy certification or safety approval. The exact method,
claim count, quotations, rationales and improvement suggestions are visible in the UI.

Gold PDFs remain evaluation-only: no indexing, embeddings, generation prompts,
case evidence or ingestion endpoints. Their SHA-256 hashes are pinned in
`blackwing/gold_catalog.json`. The original PDFs and gold workbooks are read-only.
Set `BLACKWING_GOLD_DIR` to the protected root if it is not in the existing Downloads
location. No gold document text is packaged into the code repository.

Scores are stored using the existing SQLite repository's run metadata, separately
from the immutable generated report. Cache identity includes run, normalized report
hash, gold hash and rubric version. Display/refresh never triggers model calls.
Concurrent requests for the same evaluation are serialized and reuse its result.
All hosted calls use the existing serialized request helper, HTTPS verification,
bounded retries, a 600-token output limit and an 8192-token total-context guard.
Comparisons are blocked while any investigation is active in the API process.
Do not share the group key concurrently across other applications.

New endpoints:

- GET `/api/v1/cases/{case_id}/runs/{run_id}/gold-comparison`: availability/cached score.
- POST the same path: explicitly request a post-generation hosted evaluation.

Restart FastAPI in its existing credential-configured terminal to load the new routes.
Streamlit may reload automatically; refresh the report view afterward. The frontend
still needs no model credentials. No report regeneration or index rebuild is needed.

# BLACKWING AI Streamlit integration — 21 November 2026

Implemented exclusively on `feature/arun-rag-risk-compliance`. Source inspected: `origin/feature/blackwing-agentic-copilot` at `b0e24b40b473162f9ee32758ee58ec5b54f81410`. No branch switch, merge, commit or push was performed. Existing uncommitted backend work was preserved.

## Branch comparison and dependency findings

| Requested file | Existing Arun branch/worktree | Source branch | Integration |
| --- | --- | --- | --- |
| `blackwing/ui.py` | Branded hero, four report sections, agent JSON, synchronous API polling | Sidebar, intake, saved investigations, nine workspace tabs, file uploads, finding review, PDF export | Compared before editing. Retained Arun's hero/CSS and report renderer; adapted Sunayana's sidebar, intake, nine tabs, review and report controls in `copilot_ui.py` |
| `blackwing/copilot.py` | Absent | Separate question planner/model gateway/retrieval workflow | Added an HTTP-only facade for the existing FastAPI eight-agent workflow; did not introduce another model workflow |
| `blackwing/copilot_report.py` | Absent | Pure reviewed-findings compiler plus ReportLab PDF renderer | Copied byte-for-byte; caller supplies backend-derived findings and durable review decisions |
| `blackwing/copilot_state.py` | Absent | Separate SQLite cases/evidence/turns/findings plus session database override | Added pure backend-record projections; no SQLite or independent durable state |
| `blackwing/cloud/streamlit_app.py` | Untracked source-style wrapper importing missing modules and opening temporary session SQLite | Temporary per-browser database wrapper | Updated existing file to package-qualified integrated UI entrypoint; no temporary session database |
| `blackwing/cloud/requirements.txt` | Absent | Streamlit 1.67, Pydantic, pypdf, filelock, OpenAI, httpx, jsonschema, OpenTelemetry, ReportLab | Reuses root requirements, pins cloud Streamlit to installed 1.67.0, adds only needed PDF dependencies at root |

The legacy UI depended on `blackwing_app`, `inference`, `review_workflow`, `safety_register`, telemetry and token-budget modules. Importing that stack would have bypassed the completed backend and introduced a second case database and provider configuration. The integrated frontend imports package-qualified `blackwing.*` modules and sends actions to FastAPI instead. Provider selection, model credentials, token budgets, actual retrieval, the eight-agent graph and durable state remain backend responsibilities.

Installed versions checked: Streamlit 1.67.0, httpx 0.28.1, Pydantic 2.14.3, pypdf 6.21.4, ReportLab 5.1.0. `pip check` reported **No broken requirements found**. No package installation or upgrade was required. Root Streamlit minimum is now 1.67 because the integrated uploader/testing/accessibility APIs use that version; existing backend dependency bounds are retained.

## Components integrated

- Existing branded dashboard and preliminary report rendering.
- Start-investigation form with airport, jurisdiction, date, external incident ID and PDF/TXT evidence.
- Saved investigations from the canonical backend repository.
- Nine source-workstation tabs: Q&A, Overview and timeline, Historical comparison, Regulatory context, Hazards and risk, Mitigation plan, Evidence and verification, Report, Run history.
- Explicit eight-agent run creation, status refresh, interrupted-run recovery guidance and existing progress information. Reruns do not automatically start another investigation.
- Seven structured specialist outputs plus the eighth report role's generated report.
- Hazard-linked mitigation, backend Critic findings, required human verification, traces and durable audit history.
- Human finding review through existing backend review records. Overrides include the run ID, preventing approval of an older run from leaking into a new report.
- Backend JSON/Markdown exports and Sunayana's reviewed-finding PDF compiler/renderer.
- Both local and cloud entrypoints use exactly the same backend storage and API.

Q&A explicitly searches saved agent outputs. The existing backend has no separate conversational-turn endpoint, so this UI does not silently launch a second model pipeline or persist a second conversation/case history. The sidebar's historical-investigation explorer lists saved backend cases; indexed historical analogies remain available in the backend's Similar Case results rather than a copied pilot corpus.

## Aviation risk alignment

No severity-times-likelihood calculation is present in the integrated frontend. `risk_display` only formats the backend's categorical values:

- Severity labels: 1 catastrophic through 5 minimal.
- Likelihood labels: A–E.
- Backend-provided risk index, such as C3, and backend-provided classification.
- Backend evidence IDs, severity/likelihood bases, controls, gaps and assessment status.
- Missing or unassessed values remain **Unassessed**, including stale model-supplied numeric/category fields with an unassessed scoring status.

The displayed framework is **FAA AC 150/5200-37A, Figure 4-5 — provisional reference**. A visible warning says its jurisdictional applicability is unverified and it is not a binding FAA, DGCA, ICAO or local operational decision. Qualified human verification is required. Indian-case tests use this same explicit boundary. The reference framework is also included in PDF limitations.

The frontend does not alter `agents/risk/matrix.py`, `agents/risk/evidence.py` or `agents/risk/schema.py`. It does not infer missing inputs, assign operational acceptability or derive an aviation numerical score.

## Report-quality scorer: what was found and preserved

Searching the source branch's Python files found **no standalone numerical prompt/report-quality scorer**. The existing feature is `deterministic_tools.build_evidence_checklist`, displayed by the source UI's `show_gaps` and produced in its Copilot trace. It checks eight investigation topics and labels mentions as unverified; it is not a risk score or an entailment verifier.

That checklist is preserved with an identical function AST in `blackwing/report_quality.py` and is displayed under Evidence and verification. Its historical “investigation conclusions” topic is not treated as a requirement to ingest a protected final/gold report.

A separate, transparent completeness scorer now implements the requested prompt/report assessment:

- Prompt: narrative, airport/location, incident date, jurisdiction and source identity/provenance.
- Report: complete specialist outputs, current-case evidence, factual citation presence, generated sections, factual report references, human-review boundary, unresolved Critic issues, mitigation linkage and missing-information disclosure.
- Result: equal-weight checklist completion percentage, missing-information actions, explicit method and limitations.

The quality percentage measures document completeness only. It is not validated model-answer quality, factual correctness, source entailment, regulatory compliance, incident severity or aviation risk. Tests confirm changing severity/likelihood/classification without changing the report evidence does **not** change this score.

## Persistence and report semantics

FastAPI's existing SQLite repository is the only durable case history. The browser holds only navigation, draft widget values and an unsaved search answer. No `copilot.sqlite3`, session SQLite database or copied source-branch case/index data is introduced. Test databases were temporary and separate from real cases.

Review decisions go to `/reviews`; the PDF compiler includes only accepted findings with actual citation excerpts and reviewer records. Original-source review must be confirmed by a human. The report remains preliminary and backend `is_final` remains false. A newly completed run returns its findings to pending review instead of inheriting an earlier run's decisions.

For older completed reports without `generated_report`, the existing Markdown renderer remains available. If the newest run is running/failed/interrupted while an older completed report exists, the UI explicitly labels that earlier report; stale finding-review submission is refused.

## API endpoints used

| Method / route | UI use |
| --- | --- |
| `GET /health` | Backend reachability; no inference health/model call |
| `POST /api/v1/cases` | Intake and initially attached text evidence |
| `GET /api/v1/cases?limit=100&offset=…` | Paginated saved investigations |
| `GET /api/v1/cases/{case_id}` | Canonical case and evidence metadata/text |
| `POST /api/v1/cases/{case_id}/evidence` | Evidence additions while draft |
| `POST /api/v1/cases/{case_id}/runs` | Explicit eight-agent run/recovery, with `include_retrieval` |
| `GET /api/v1/cases/{case_id}/runs` | Run list, current status and progress during refresh |
| `GET /api/v1/runs/{run_id}` | Typed individual-run polling helper |
| `GET /api/v1/cases/{case_id}/report` | Latest completed report and all specialist outputs |
| `GET /api/v1/runs/{run_id}/trace` | Execution/tool traces |
| `GET /api/v1/runs/{run_id}/audit` | Planner, partial outputs, step counts and recovery semantics |
| `POST` / `GET /api/v1/cases/{case_id}/reviews` | Durable finding-review decisions |
| `GET /api/v1/cases/{case_id}/report/export?format=json\|md` | Backend report exports |

Case, run, report, review and trace responses are validated against the existing Pydantic response schemas. The UI never reads the repository's SQLite file directly. Uploaded PDFs/TXT are converted locally to the existing `EvidenceInput` text/metadata contract; no new multipart endpoint was added.

## Actual validation results

**128 passed, zero failures, zero skips; two existing dependency deprecation warnings; 12.83 seconds.** All original 110 tests remain in this complete run; 18 integration test cases were added.

| Check | Result |
| --- | --- |
| Imports | All seven UI integration modules import without starting an app, opening SQLite or making a provider call |
| Dependency compatibility | `pip check` passed |
| API schemas | Real FastAPI TestClient routes and existing response models passed |
| Risk rendering | Mock backend C3/medium displays unchanged; absent and explicitly unassessed values remain unassessed |
| Quality scorer | Missing intake, evidence checklist, consistency flags and independence from risk values passed |
| Uploads | Real text/PDF extraction, corrupt PDF rejection and size/text-limit checks passed |
| UI behavior | Streamlit AppTest covered intake, file upload, run, review, PDF availability, saved cases, cloud entrypoint, missing risk and stale/interrupted run handling |
| Reports | Backend JSON/Markdown export passed; source PDF output was parsed and checked for actual cited content/reviewer/framework |
| Running frontend/backend | Streamlit agent API drove the actual cloud entrypoint against a real localhost FastAPI process and temporary SQLite, with a mocked model runner |
| Running smoke outcome | Seven ready snapshots; one created case, one completed run, backend C3 display, one persisted human review, reviewed PDF control available |
| Server console | No new Streamlit warning or traceback in the final running-app log |
| Existing backend | All pre-existing backend Python hashes unchanged |
| Saved indexes | SHA-256 comparison of all 12 pre-existing RAG-index files: zero changes; no rebuild or embedding generation |

Evidence is in `docs/validation/streamlit-ui/`: `regression.xml`, `running-app-summary.json`, `preservation.json`, `streamlit.log` and `validation-summary.txt`.

Automated tests block outbound provider/network access, apart from Windows asyncio's internal socketpair. The running smoke used localhost and a temporary backend with mocked model IO. **No Ollama or IIT-hosted model requests were made.** Temporary validation servers were stopped. Protected evaluation data and the separate local-delivery copy were not changed.

## Remaining limits before live local end-to-end evaluation

1. Start the real FastAPI backend with its existing provider/environment configuration and one worker. Live Ollama/IIT connectivity and answer quality remain intentionally untested.
2. The current report API exposes RAG citation IDs but not every original retrieved excerpt. Such IDs remain visible; excerpt text is not fabricated. Reviewed PDF compilation requires actual source excerpts from uploaded evidence; RAG-only findings without exposed excerpts are omitted from that PDF. Full backend JSON/Markdown reports and the dashboard remain available.
3. Scanned/image-only PDFs need a factual text transcription; OCR was not added. Text must fit the existing 20,000-character-per-evidence API limit and each uploaded file is limited to 5 MB.
4. The Q&A tab searches saved results; arbitrary model-backed conversational turns require a future explicit backend API and are not implemented through the old parallel pipeline.
5. No browser-pixel/CSS verification was performed. Functional UI behavior was tested with AppTest and the running Streamlit agent API. Nonlocal/cloud deployment additionally needs a reachable secured backend and deployment-specific network/TLS configuration.

There are no outstanding import or mocked frontend/backend integration blockers.

## Exact launch commands

Run from the `BlackWingAI-Arun-Git` repository root. Keep the backend's existing configuration from `docs/local-agentic-delivery.md`; model credentials belong to the backend environment only.

Backend terminal, when its required settings are configured:

```powershell
.\.venv\Scripts\python.exe -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8000 --workers 1
```

Frontend terminal:

```powershell
$env:BLACKWING_UI_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run .\blackwing\cloud\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The equivalent local entrypoint is `blackwing/ui.py`. This launch does not enable the Streamlit agent API; that flag was used only for local validation. For a fresh environment install `requirements-dev.txt` (or the cloud requirements, which include root requirements) before launching.

Re-run validation without model calls:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Modified and added files for this task

Modified existing files:

- `blackwing/ui.py` — preserved brand/report renderer, integrated workstation entrypoint and risk labels.
- `blackwing/cloud/streamlit_app.py` — repaired the previously untracked cloud wrapper; removed session DB and legacy imports.
- `requirements.txt` — Streamlit compatibility floor and required PDF dependencies.
- `README.md` — integrated UI launch guide/link.

Added files:

- `blackwing/copilot_ui.py` — source-workstation layout, canonical API actions and separate quality/risk views.
- `blackwing/copilot.py` — typed HTTP bridge, bounded upload extraction, run/review/export helpers.
- `blackwing/copilot_state.py` — stateless display/finding projections; backend-only risk formatting.
- `blackwing/copilot_report.py` — unchanged source-branch compiler/PDF renderer.
- `blackwing/report_quality.py` — preserved evidence checklist plus independent completeness scorer.
- `blackwing/cloud/requirements.txt` — compatible cloud dependency entrypoint.
- `tests/test_streamlit_integration.py` — 18 integration test cases.
- `docs/streamlit-ui-integration.md` — this handoff.
- The five validation artifacts listed above in `docs/validation/streamlit-ui/`.

All earlier backend development files and their uncommitted changes remain as found. No Git staging, commit, branch merge or push was performed.

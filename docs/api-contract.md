# BlackWing AI API Contract

## Conventions

- Base path: `/api/v1`
- Media type: `application/json`
- IDs: opaque UUID strings
- Dates: ISO 8601 calendar dates (`YYYY-MM-DD`); timestamps use UTC RFC 3339.
- Additional JSON properties are rejected for request models unless noted.
- Run creation is asynchronous: clients receive `202 Accepted` and poll the run resource.
- `GET /api/v1/openapi.json` is the machine-readable contract; interactive docs are at `/api/v1/docs`.
- The current prototype has no authentication middleware. The target architecture requires JWT/OIDC and case-level RBAC before deployment beyond local development.

## Endpoint inventory

| Method and path | Purpose | Request | Success response |
| --- | --- | --- | --- |
| `GET /health` | Liveness | None | Service status |
| `GET /api/v1/capabilities` | Supported categories and feature flags | None | Capability document |
| `POST /api/v1/cases` | Create a runway-incursion case | `CaseCreate` | `201 CaseResponse` |
| `GET /api/v1/cases` | List cases; optional `status`, `limit`, `offset` | Query parameters | `200 CaseListResponse` |
| `GET /api/v1/cases/{case_id}` | Read case and attached evidence | None | `200 CaseResponse` |
| `PATCH /api/v1/cases/{case_id}` | Edit a draft case | Partial narrative/date/jurisdiction/location | `200 CaseResponse` |
| `POST /api/v1/cases/{case_id}/evidence` | Attach evidence excerpt and provenance to a draft case | `EvidenceInput` | `201 EvidenceResponse` |
| `POST /api/v1/cases/{case_id}/runs` | Start runway-incursion workflow | `RunCreate` | `202 RunResponse` |
| `GET /api/v1/runs/{run_id}` | Poll run status/progress | None | `200 RunResponse` |
| `GET /api/v1/cases/{case_id}/runs` | List runs for a case | None | `200 RunResponse[]` |
| `GET /api/v1/runs/{run_id}/trace` | Read ordered workflow trace | None | `200 TraceResponse` |
| `GET /api/v1/cases/{case_id}/report` | Read latest completed report and review history | None | `200 ReportResponse` |
| `GET /api/v1/cases/{case_id}/report/export?format=json` | Download typed report JSON | Query `format=json` | `200 application/json` attachment |
| `POST /api/v1/cases/{case_id}/reviews` | Record human decision and finding overrides | `ReviewCreate` | `201 ReviewResponse` |
| `GET /api/v1/cases/{case_id}/reviews` | List review decisions | None | `200 ReviewResponse[]` |
| `POST /api/v1/knowledge/documents` | Add a text source to the local knowledge library | `KnowledgeDocumentCreate` | `201 KnowledgeDocumentResponse` |
| `GET /api/v1/knowledge/documents` | List source metadata; optional `document_type`, `jurisdiction` filters | Query parameters | `200 KnowledgeDocumentResponse[]` |
| `GET /api/v1/knowledge/documents/{document_id}` | Read source metadata | None | `200 KnowledgeDocumentResponse` |
| `POST /api/v1/knowledge/search` | Search eligible sources | `KnowledgeSearchRequest` | `200 KnowledgeSearchResponse` |

## Request and response contracts

### Create a case

`POST /api/v1/cases`

```json
{
  "category": "runway_incursion",
  "narrative": "A vehicle entered the runway while an aircraft was taxiing for departure. The flight crew reported stopping before reaching the runway intersection.",
  "incident_date": "2026-10-30",
  "jurisdiction": "US",
  "location": "Example airport, runway 09",
  "evidence": [
    {
      "title": "Operator occurrence narrative",
      "document_type": "incident_narrative",
      "authority_type": "operator_report",
      "source_status": "reported_unverified",
      "version": null,
      "source_id": "OPS-2026-0042",
      "source_url": null,
      "jurisdiction": "US",
      "document_date": "2026-10-30",
      "effective_date": null,
      "page_number": null,
      "section_reference": null,
      "excerpt": "The vehicle entered the runway while the aircraft was taxiing."
    }
  ]
}
```

`201` response shape:

```json
{
  "case_id": "opaque-uuid",
  "category": "runway_incursion",
  "status": "draft",
  "narrative": "...",
  "incident_date": "2026-10-30",
  "jurisdiction": "US",
  "location": "Example airport, runway 09",
  "evidence": [],
  "created_at": "2026-11-10T10:00:00Z",
  "updated_at": "2026-11-10T10:00:00Z"
}
```

`GET` returns this same resource shape. `PATCH` accepts one or more of `narrative`, `incident_date`, `jurisdiction`, or `location`, and only works while the case is `draft`.

### Attach case evidence

`POST /api/v1/cases/{case_id}/evidence` accepts one evidence excerpt using the same fields as an entry in `CaseCreate.evidence`. It returns the assigned `evidence_id` and `created_at`. This endpoint stores a text excerpt and provenance only; it does not upload or preserve the original file.

### Add a knowledge document

`POST /api/v1/knowledge/documents`

```json
{
  "title": "Runway safety procedure excerpt",
  "document_type": "procedure",
  "authority_type": "procedure",
  "source_status": "official",
  "version": "2026-01",
  "source_id": "PROC-EXAMPLE-7",
  "source_url": "https://example.org/source",
  "jurisdiction": "US",
  "document_date": "2026-01-01",
  "effective_date": "2026-01-01",
  "page_number": 7,
  "section_reference": "4.2",
  "excerpt": "Relevant source metadata excerpt.",
  "content": "Full source text to index. Keep the original source and page references in the authoritative repository."
}
```

`201` returns metadata and `content_length`, but not the full content. User ingested sources are stored in memory. The approved dataset index is configured with `BLACKWING_RAG_INDEX_DIR`; protected final reports and unseen/golden prompt files are not loaded as index sources. The example URL is illustrative only.

### Search knowledge

`POST /api/v1/knowledge/search` accepts `query` (3–2000 chars), optional `document_types`, `jurisdiction`, `incident_date`, and `limit` (1–20). It returns ranked chunks with document and chunk IDs, metadata, source links and `ranking_score`, plus `retrieval_method` and limitations. Retrieval uses BM25 and hosted BGE dense embeddings fused with reciprocal rank; if the hosted embedding service is unavailable, it reports a BM25 fallback. A search result is a candidate passage, not a verified applicable requirement or a cause finding about a new case.

### Start a run

`POST /api/v1/cases/{case_id}/runs`

```json
{
  "workflow": "runway_incursion_investigation",
  "include_retrieval": true
}
```

`202` response:

```json
{
  "run_id": "opaque-uuid",
  "case_id": "opaque-uuid",
  "status": "queued",
  "workflow": "runway_incursion_investigation",
  "model": "qwen3:8b",
  "progress": [],
  "error": null,
  "created_at": "2026-11-10T10:00:00Z",
  "started_at": null,
  "completed_at": null,
  "links": {
    "self": "/api/v1/runs/opaque-uuid",
    "trace": "/api/v1/runs/opaque-uuid/trace",
    "report": "/api/v1/cases/opaque-uuid/report"
  }
}
```

The API retrieves at most five candidate passages and supplies case evidence plus retrieved excerpts to the workflow. A duplicate active run returns `409`.

### Run status and trace

`GET /api/v1/runs/{run_id}` returns the same envelope with status `queued`, `running`, `completed` or `failed`. `progress` lists completed steps. A failed run returns a safe machine-readable `error_code` and summary in `error`; current codes include `model_timeout`, `model_http_error`, `model_unreachable`, and `invalid_agent_output`. Logs record the run ID, agent and error code without logging prompts, response bodies, or credential values.

`GET /api/v1/runs/{run_id}/trace` returns `{"run_id": "...", "events": [...]}`. Each event has `sequence`, `step`, optional `agent`, `status`, optional `model`/`detail`, and `timestamp`.

### Report response

`GET /api/v1/cases/{case_id}/report` returns:

```json
{
  "case_id": "opaque-uuid",
  "run_id": "opaque-uuid",
  "case_status": "analysis_complete",
  "analysis_status": "completed",
  "report": {
    "case_id": "opaque-uuid",
    "status": "PRELIMINARY_REQUIRES_HUMAN_REVIEW",
    "workflow": "runway_incursion_investigation",
    "model": "qwen3:8b",
    "generated_report": {
      "executive_summary": "Preliminary report text assembled from the agent work products.",
      "sections": [
        {"task_id": "overview", "title": "Incident overview", "summary": "...", "key_points": [], "generation_method": "model"},
        {"task_id": "findings_and_causes", "title": "Findings and causal assessment", "summary": "...", "key_points": [], "generation_method": "model"},
        {"task_id": "comparisons_and_compliance", "title": "Historical comparisons and compliance", "summary": "...", "key_points": [], "generation_method": "model"},
        {"task_id": "risk_and_follow_up", "title": "Risk and recommended follow-up", "summary": "...", "key_points": [], "generation_method": "model"}
      ],
      "human_approval_required": true,
      "report_notice": "Preliminary AI-assisted analysis. Verify sources and obtain qualified human review."
    },
    "agent_outputs": {
      "investigation": {
        "incident_summary": "...",
        "timeline_events": [{"event": "...", "time_reference": null, "evidence_ids": [], "evidence_state": "reported_observation"}],
        "documented_facts": [], "reported_observations": [], "hypotheses": [], "contributing_factors": [], "unknowns": []
      },
      "similar_case": {"matches": [], "assessment": "No historical match found.", "limitations": []},
      "compliance": {"jurisdiction": "US", "overall_applicability": "requires_human_verification", "checks": [], "missing_context": []},
      "risk": {"hazards": [], "risk_notes": "...", "assessment_status": "not_scored_missing_severity_or_likelihood", "numeric_score": null, "risk_rating": null, "scoring_method": "provisional_5x5_severity_x_likelihood_max_hazard", "missing_inputs": []},
      "mitigation": {"recommendations": [], "assumptions": []},
      "critic": {"issues": [], "missing_analysis": [], "overall_assessment": "requires_human_review"},
      "verification": {"claims_needing_source": [], "unsupported_conclusions": [], "conflicts": [], "completeness_status": "incomplete", "human_approval_required": true}
    },
    "trace": [], "limitations": []
  },
  "reviews": [],
  "is_final": false,
  "notice": "AI output is preliminary until a qualified human reviewer records a decision."
}
```

Each agent output has a versioned Pydantic schema. Investigation claims and timeline entries carry an evidence state and evidence IDs. Compliance checks carry source type, jurisdiction, effective date, applicability status and rationale. Risk output always has a null numeric score until a deterministic matrix is configured. Mitigation options carry hazard links and an effectiveness indicator. Critic and Verification outputs identify issues and claims that still require review. These are typed model proposals, not verified conclusions.

### Human review

`POST /api/v1/cases/{case_id}/reviews`

```json
{
  "reviewer_id": "reviewer-identifier",
  "decision": "request_changes",
  "rationale": "The occurrence date and source for the reported runway entry need confirmation.",
  "overrides": [
    {
      "path": "investigation.hypotheses[0]",
      "action": "edit",
      "replacement_value": "Possible contributing factor; pending corroboration.",
      "rationale": "The supplied narrative does not independently establish this point."
    }
  ]
}
```

`edit` and `reject` overrides require a rationale. Reviews are append-only in the intended contract. The prototype records them in memory and does not mutate the model's original proposal.

## Common error statuses

| Status | Meaning |
| --- | --- |
| `404` | Case, run or document ID does not exist |
| `409` | Invalid lifecycle transition, active duplicate run, or report/review requested too early |
| `422` | Field or schema validation failed |
| `503` | Required application configuration is missing |

FastAPI validation errors use its standard `detail` response shape. Other endpoint errors currently use the same standard shape. Once a run has been accepted, model failures are reported as `status: failed` on the run resource with a safe error summary instead of changing the original `202` response.

## Contract versioning

Changes that remove or rename fields require a new API major version. Additive optional fields may be added within v1. Category IDs, run status values, agent role names and evidence-state values are API enums once formally frozen. The current Pydantic models reject unknown request properties to surface client/server mismatches early.

The report export currently supports JSON and Markdown. The Streamlit UI displays the Report Agent's readable report on screen; it does not generate a PDF. Resumable graph checkpoints, authentication, and tenant/case authorization remain target-architecture work and are not exposed as working v1 endpoints yet.


## Local agentic additions (21 November 2026)

Existing request fields and run status enum values remain compatible. Trace events add optional `tool`, `invocation_id`, `case_id`, `input_identifiers`, `result_summary`, `duration_ms` and sanitized `error` fields. The trace endpoint exposes executed tools for active/failed runs as well as completed traces.

`GET /api/v1/runs/{run_id}/audit` adds durable audit events, the plan, partial outputs, execution step count, `resumable=false` and controlled-recovery guidance. Interrupted queued/running runs retain status `failed` with `error_code=interrupted`; post a new case run to recover. Reports remain preliminary and contain all existing agent-output keys. See [the delivery guide](local-agentic-delivery.md) for validation and UI requirements.


## Q&A evaluation (local, human reviewed)

The investigation UI retains Sunayana's single-form layout and three report tabs.
`?view=qa` opens a separate page where users type or paste their own questions. No questions are created or automatically run. Generation and persistence remain in FastAPI.
Swagger documentation is available at `/api/v1/docs`; its live OpenAPI contract
is `/api/v1/openapi.json`.

| Method and path | Purpose |
| --- | --- |
| `GET /api/v1/qa/queries` | List the 80 question IDs, prompts and grouping metadata; no approved answer fields. |
| `POST /api/v1/qa/answers` | Generate one answer from the existing index and persist it. Supply `gold_query_id`, or a free `question`; optional `case_id` scopes case evidence. |
| `GET /api/v1/qa/answers` | Read saved Q&A tests and human verdicts. |
| `GET /api/v1/qa/answers/{answer_id}` | Read a frozen answer, citations and source excerpts. |
| `GET /api/v1/qa/answers/{answer_id}/reference` | Reveal the preserved approved answer only after a generated answer exists. Refuse if the workbook hash changed. |
| `POST /api/v1/qa/answers/{answer_id}/review` | Record `verdict`, `reviewer` and `rationale`; retain review history. |

Entered questions are matched uniquely by case-insensitive whitespace-normalized text against the existing gold evaluation workbook. A nonmatching question can still use RAG, but no approved answer or automatic score is fabricated. Configure `BLACKWING_GOLD_QUERY_FILE` if the protected workbook is elsewhere.
The default is the preserved `GOLD_80_QUESTIONS/BW-GQ-001-GOLD80-WORKBOOK.xlsx`
under `~/Downloads/BLACKWING_DATA_EXTRACTED/BLACKWING_DATA/02_GOLD_EVALUATION_PROTECTED`.
The workbook is opened read-only. Approved answers, expected source IDs and
review notes never enter answer generation, retrieval queries or the RAG index.
The same SQLite repository stores Q&A under a separate record kind; investigation
case/run histories and existing repository methods retain their contracts.

Opening the page performs no inference. Each Generate click sends one question.
The shared hosted request lock, timeout, token preflight and bounded transient
retries apply. An active investigation returns HTTP 409. Provider/schema failures
return a sanitized HTTP 502 and never count as successful answers. Missing
workbook/index configuration returns HTTP 503. There is no automatic gold
correctness score; human verdicts are separate from structural completeness,
gold report comparison and categorical aviation risk.

The fetched Sunayana `main` revision `bf7fda983c8b4e1c149860c8a457156fa9e75be2`
had no Q&A endpoint or Docker/Compose configuration. No Docker configuration
was introduced by this integration.

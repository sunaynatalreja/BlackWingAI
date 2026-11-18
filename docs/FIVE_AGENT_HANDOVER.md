# BLACKWING AI — Five-Agent Integration Handover

**Scope:** Investigation, Similar Case, Mitigation, Critic and Verification, integrated with Sunayana's existing seven-agent orchestration and the existing Arun RAG + Risk + Compliance contribution.

**Source basis:** The previously shared `BlackWingAI-Arun.zip` and `BLACKWING_AI_RAG_RISK_COMPLIANCE_FULL.zip` snapshots, plus the external-incident-ID/PDF-extraction fixes. Review the patch against the *current* `Arun` development branch before merging; the team's remote repository may have advanced since the snapshot.

## What is implemented

| Role | Key files | Behavior | Model calls |
|---|---|---|---|
| Investigation | `investigation/agent.py`, `investigation/grounding.py`, `investigation/prompt.md` | Model produces typed claims/timeline; deterministic guard retains current-case passage IDs, rejects cross-incident citations and demotes ungrounded statements. | 1, with one bounded schema-retry possible |
| Similar Case | `similar_case/agent.py` | Deterministic, source-deduplicated comparison; excludes current incident by external source ID; similarities are only analogies. | 0 |
| Mitigation | `mitigation/grounding.py`, prompt | Keeps proposed actions only when connected to existing hazard IDs or evidenced contributing factors. Unlinked recommendations are omitted with explanations. | 1 |
| Critic | `critic/grounding.py`, prompt | Adds deterministic checks for missing current-case citations, ungrounded hazards, unassessed compliance and unlinked mitigations. | 1 |
| Verification | `verification/grounding.py`, prompt | Cross-checks evidence gaps and provisional conclusions, never removes human-review requirement. | 1 |

**Unchanged boundaries:** Existing seven schemas (`schema.py`), Risk and Compliance evidence validators, RAG retrieval functions, API case/report JSON contracts and UI remain intact. The `similar_case` role stays deterministic, and Compliance can be deterministic `not_assessed` if no eligible rule. No new agent count or new UI is introduced.

**Shared changes requiring Sunayana review:** `blackwing/service.py`, `blackwing/api.py`, `blackwing/agents/runtime.py`. The latter offers in-process serialized IIT calls/retries, optional JSON response format, model TLS CA. The API now offers an optional **Markdown report export** without changing the JSON format.

## Integration flow

```text
Case narrative + source_incident_id
     |        (source_incident_id != internal UUID)
     v
Existing RAG: source ingestion -> BGE(384) + BM25 -> RRF -> CrossEncoder
     |
     v
agent_router: current facts / historical / rule / safety-guidance packets
     |
     v
Investigation (model -> grounded citations)
     v
Similar Case (deterministic, distinct historical incidents)
     v
Compliance (existing model + date/authority validation, or not_assessed)
     v
Risk (existing model + grounded hazards; numeric score unavailable)
     v
Mitigation (model -> hazard-linked proposals only)
     v
Critic (model + deterministic cross-agent audit)
     v
Verification (model + deterministic integrity check)
     v
Existing ReportResponse JSON; optional Markdown export; human review
```

**Citations:** A matching passage ID verifies that the source *was supplied* to the model. It does not prove the source's truth or entailment of the claim. The human reviewer must inspect cited source pages and event metadata.

## Install into current GitHub checkout (Windows PowerShell)

1. Open the team checkout in VS Code. Ensure `git status --short` has no unexpected modifications. Create a **new feature branch** for these five agents. Do not reuse or force-push Arun's already submitted RAG branch.
2. Extract the integration zip. Use `git apply --check <path-to-CHANGES.patch>` from the repository root; if it succeeds, run `git apply <path-to-CHANGES.patch>`. If the patch doesn't apply because Sunayana updated shared files, **do not force overwrite**; use the zip's `overlay/` files to manually compare and merge only changed sections.
3. Ensure Python >=3.11, and use the project venv: `pip install -r requirements.txt -r requirements-rag.txt -r requirements-dev.txt`, then `pip install -e .` as needed. Host tokenizer/model weights may require separate authorized downloads.
4. **No-IIT / no-model checks:** `python -m pytest -q tests/test_five_agent_grounding.py tests/test_five_agent_iit_gateway_mock.py tests/test_five_agent_service_mock.py` and `python scripts/demo_five_agents.py`. First two are pure offline; the service test needs LangGraph but stubs the LLM.
5. **Existing team tests:** `python -m pytest -q`. Review any compatibility failure with current teammates' changes.
6. **Real local test:** dot-source `scripts/use_local.ps1`, supply your local prebuilt BGE index under `BLACKWING_RAG_INDEX_DIR`, start Ollama and the FastAPI server, create a case with a correct `source_incident_id` matching the index, run the existing `POST /api/v1/cases/{id}/runs` flow. See sample PowerShell below.
7. Inspect JSON and the Markdown export. **Report completeness and factual correctness are not guaranteed by the mock tests.** Compare with approved gold answers and source pages.
8. Run the same incident through IIT-hosted Qwen after connecting the gateway and verifying secrets/tokenizer. Use a few requests at first and respect institute quotas.

### Local model profile

From the project root:

```powershell
. .\scripts\use_local.ps1
# If needed, override to an EXISTING prebuilt index:
# $env:BLACKWING_RAG_INDEX_DIR = "$PWD\rag_indexes\austin_local_bge_v1"
.\.venv\Scripts\python.exe -m pytest -q tests/test_five_agent_grounding.py tests/test_five_agent_iit_gateway_mock.py tests/test_five_agent_service_mock.py
.\.venv\Scripts\python.exe scripts\demo_five_agents.py
.\.venv\Scripts\python.exe -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8001
```

**Automatic local/API smoke run (API must already be running in the other terminal):**

```powershell
.\scripts\run_real_incident_smoke.ps1
```

This creates the Austin test case, starts the seven-agent workflow, polls the run, saves JSON and Markdown reports under ignored `work/`, and prints citation/linking gap counts. Running it with `use_iit.ps1` consumes IIT model quota; the same script works with either provider. The case narrative is a *starter description*, not ground-truth evidence.

Existing routes:

- `POST /api/v1/cases` (include `source_incident_id` to match RAG current case)
- `POST /api/v1/cases/{case_id}/runs` with `{"workflow":"runway_incursion_investigation","include_retrieval":true}`
- `GET /api/v1/runs/{run_id}`
- `GET /api/v1/cases/{case_id}/report`
- `GET /api/v1/cases/{case_id}/report/export?format=json`
- **NEW** `GET /api/v1/cases/{case_id}/report/export?format=md`

The Markdown export only displays the typed report with explicit uncertainties; it is **not** a new LLM report-generation step, so there is no extra model call or token cost.

### IIT model profile and embedding options

**Set these values in your private PowerShell session, NOT in Git:**

```powershell
$env:BLACKWING_LLM_API_BASE_URL = "https://<IIT-CHAT-ENDPOINT-BASE>/v1"
$env:BLACKWING_LLM_API_MODEL = "<institute-provided-model-id>"
$env:BLACKWING_LLM_API_KEY = "<key from IIT secure channel>"
$env:BLACKWING_LLM_CA_CERT = "C:\private\certificates\iit-ca.crt"
$env:BLACKWING_RAG_INDEX_DIR = "$PWD\rag_indexes\austin_local_bge_v1"
. .\scripts\use_iit.ps1 -Embeddings local
```

- Mode A: **IIT chat model + local BGE embeddings and local CrossEncoder**. Default `-Embeddings local`; recommended first to avoid rebuilding the local vector index or consuming IIT embedding quota unnecessarily.
- Mode B: **IIT chat model + hosted IIT BGE embeddings + local CrossEncoder**. Requires `BLACKWING_EMBEDDING_URL`, `BLACKWING_EMBEDDING_API_KEY`, `BLACKWING_EMBEDDING_CA_CERT`, and `BLACKWING_IIT_RAG_INDEX_DIR`, then `. .\scripts\use_iit.ps1 -Embeddings iit`.
- **Never reuse local vectors with IIT embeddings if backend identity or embedding outputs differ.** Build a new versioned IIT index from the *same approved source manifest* using existing `python -m blackwing.rag.build --backend iit ...`. The existing retriever rejects incompatible vector identities rather than corrupting similarity results.
- IIT's Qwen3:8B serves through the OpenAI-compatible `/chat/completions`; ensure the provided base URL ends just before `/chat/completions` (usually `/v1`). User's actual service endpoint should be confirmed in institute documentation.
- Hosted max combined input/output is **8,192** tokens; maximum output **1,024**. Profiles default output 768 and conservative input 2,600. **Exact Qwen3 tokenizer must be cached locally** for hosted preflight; failures are intentionally closed rather than bypassing token caps.
- `BLACKWING_API_JSON_RESPONSE_FORMAT=0` disables `response_format: {"type":"json_object"}` when the IIT gateway does not support JSON mode. Schema-specific prompts and Pydantic validation still apply, but malformed JSON can fail the run.
- Hosted client uses a verified CA bundle, never `verify=False`; calls are serialized with a lock **within one process**, paced >=1.05 seconds apart, and transient 429/503 errors retried. A shared API key across **multiple processes** still needs distributed rate/concurrency control; run one API worker for the pilot.
- The embedding endpoint must return **384-dimensional** BGE-small-compatible vectors. The CrossEncoder remains local (sentence-transformers); it is not silently replaced by an IIT endpoint.

### Test inventory and acceptance criteria

| Check | Expected outcome |
|---|---|
| Current source and historical source cite the same narrative | Current-case ID retained, historical rejected |
| Invalid / cross-incident passage ID | Removed; original claim demoted to unverified |
| User-uploaded evidence | Kept as reported, not official fact |
| Similar historical reports | Deduplicated, source ID retained, never used as current-case cause |
| Proposed mitigation with fake hazard ID | Omitted, with reason |
| Risk unsupported numeric scoring | Unavailable (same as original Risk validator) |
| Critic | Flags missing citations and unassessed compliance |
| Verification | Never auto-approves |
| Markdown export | Shows sections, passage IDs, uncertainties; masks common email/phone patterns |
| Provider switch | Ollama or IIT chosen by environment profile without code edits |
| Hosted error | Transient gateway status retried (bounded) |
| Local full workflow | Manually run with actual Ollama and Austin index; inspect true report |
| Hosted full workflow | Manually run with IIT credentials, CA, exact tokenizer, quotas |

### Deliberately outside this patch / remaining project work

1. No real local Ollama Qwen run was executed **in this packaging environment**. The prior Austin run demonstrated the earlier version; this five-agent patch requires a fresh real-model regression test on the user's laptop.
2. No IIT-hosted inference or hosted embedding requests were made. The IIT gateway, certificate, key, model ID and hosted embedding availability must be verified on the institute's network.
3. The actual Austin PDFs, private gold labels and vector index are excluded from the deliverable. Sunayana must obtain the approved corpus separately and create/reuse her own compatible index.
4. The 500–2,000-document acceptance requirement, true Recall@k, prompt-baseline comparisons, 80-question validated gold answers, formal agentic tool criteria, persistent state, Docker, exhaustive PII/prompt-injection red-team tests and end-to-end report accuracy are **not proven by these tests**.
5. Contact masking in Markdown is a narrow safeguard, **not full PII detection**. The JSON endpoint and repository still need a complete PII policy and access controls.
6. Model-backed outputs remain preliminary. A valid citation ID is not automatic semantic entailment of a claim. Human source verification is required.

### Recommended merge approach

Review/apply on a fresh `feature/five-agent-integration` branch. Compare shared `api.py`, `service.py`, and `runtime.py` with Sunayana's latest code. Do not force-push or rewrite the already-reviewed `feature/arun-rag-risk-compliance` branch. Ask agent owners to validate outputs and decide final prompts after the first real-model report is inspected.

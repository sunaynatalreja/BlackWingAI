# Risk and Compliance — Evidence-Grounded Agents

The existing report architecture is preserved:

`Incident narrative → RAG retrieval → Investigation → Similar Case → Compliance → Risk → Mitigation → Critic → Verification → structured preliminary report`

`blackwing/rag/agent_router.py` assembles compact, role-specific evidence packets. Both agents retain their original `RiskOutput` / `ComplianceOutput` schemas and integrate through `blackwing/service.py`. No new UI or interactive Q&A endpoint is created.

## Grounding contracts

**Risk (`blackwing/agents/risk/`)**

- Receives only current-case reports and safety guidance, never historical analogies as current-case facts.
- `evidence.py` removes citations not present in the supplied evidence packet.
- Hazards without direct case evidence stay explicitly provisional and trigger missing-evidence notices.
- A transparent provisional 5×5 estimate is computed as severity × likelihood; the overall score is the highest hazard score and always requires human review.
- Severity runs 1 (Negligible) through 5 (Catastrophic); likelihood runs 1 (Rare) through 5 (Almost certain). Score bands are 1–4 Low, 5–9 Moderate, 10–16 High, and 17–25 Extreme. Missing either dimension leaves that hazard unscored.
- Suggestions of existing barriers and gaps remain preliminary, not independently verified findings.

**Compliance (`blackwing/agents/compliance/`)**

- Receives incident facts **separately** from regulations/guidance, within the small-context budget.
- A regulation or procedure candidate must match the incident jurisdiction, have an eligible effective date and source ID, and appear in the supplied RAG packet.
- Advisory guidance is `advisory_only`, not binding regulation.
- A fabricated rule source, future-dated rule, invented clause or invented evidence ID cannot be treated as verified.
- `applicability_status` stays `requires_human_verification` for rule candidates. The code does **not** determine legal compliance.
- When no eligible rule is supplied, the current agent returns `not_assessed`.

## Local installation

From the extracted `BlackWingAI-Arun` project directory in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-rag.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Quick local retrieval and routing smoke test (BM25 only; **synthetic evidence**, no IIT quota):

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.smoke
```

Real **local** BGE + CrossEncoder test (downloads public weights if not cached; does not call IIT):

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.smoke --real-models
```

To run a full **local** seven-agent report in the existing FastAPI application, configure Ollama using the instructions in `README_RAG.md`, set `BLACKWING_AGENT_RAG=1`, and use the existing case/run/report endpoints. The user interface is not part of this contribution.

## IIT Track B verification

The source, chunking, hosted BGE embeddings, BM25, dense, RRF, CrossEncoder, citation fields, evaluation code are in `blackwing/rag/`. The 500–2,000-document criterion is a **real-corpus acceptance test**, not something the synthetic tests can prove. Chunking experiment results, true Recall@5/10 using complete relevance annotations, actual hosted BGE/CrossEncoder performance, answer/report quality and IIT-hosted endpoint/TLS validation require separate measured experiments.

The hosted inference guard caps output to 1,024 tokens and checks 8,192 combined input/output tokens. Retrieval packets normally contain no more than three passages; documents remain in the index, not in the Qwen context. Local-first embeddings avoid institute quota until the backend is deliberately set to `iit`.

## Code ownership and pull-request review

All existing seven agent keys, JSON report structure, source files and UI are retained. This contribution edits only the two owned agent prompts, adds two grounding modules and corresponding tests, and makes small RAG/service/API changes to pass case date/jurisdiction and route correct evidence. The separate PR archive also contains the previously prepared RAG components, making it a single reviewable contribution.

Do not commit cache files, model weights, source PDFs without redistribution permission, evaluation gold answers, API keys, or TLS certificate files.

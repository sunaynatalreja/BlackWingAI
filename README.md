# BlackWing AI

Runway-incursion API and local UI prototype with evidence-grounded agents, a bounded Planner/Executor/Critic workflow, durable SQLite history, typed reports and required human review.

## Current local implementation

See [local agentic delivery and validation](docs/local-agentic-delivery.md) for architecture, test evidence, modified files, limitations, IIT setup and Sunayana UI requirements. The API supports case creation, text evidence, investigation/status, seven agent outputs, provisional deterministic risk classification, hazard-linked mitigation, Critic findings, review and JSON/Markdown export.

The LangGraph Planner selects execution modes. The Executor runs the existing roles, the Critic can trigger one evidence-supported targeted revision, and the workflow stops within ten counted execution steps. The report role assembles four typed sections deterministically within this budget; standalone report drafting remains available through its existing function.

Cases, evidence, runs, partial/final outputs, tool traces, reviews and report references use SQLite. Interrupted runs are retained as failed with `error_code=interrupted`; recovery creates a new run rather than claiming checkpoint resumption. Run one API worker per database and hosted API key.

Retrieval keeps current-case evidence separate from historical analogies, including voluntary ASRS reports. The existing FAA reference matrix supplies provisional categorical risk only when supporting inputs exist; numeric scores remain null. Human approval is required. Existing saved indexes are reused without rebuilding.

## API contract

- Swagger: `http://127.0.0.1:8000/api/v1/docs`
- OpenAPI: `http://127.0.0.1:8000/api/v1/openapi.json`
- [Endpoint contract](docs/api-contract.md)
- [Architecture](docs/architecture.md)
- Detailed run audit: `GET /api/v1/runs/{run_id}/audit` (additive; existing routes remain).

Local Ollama and OpenAI-compatible IIT-hosted Qwen3-8B share the same agent code. The hosted gateway uses exact cached-tokenizer preflight, bounded output, serial paced requests and bounded transient retries. Credentials resolve from environment variables. Hosted testing remains unverified. The integrated Streamlit frontend has passed mocked FastAPI integration tests and a running-app smoke; see [Streamlit integration and launch instructions](docs/streamlit-ui-integration.md).

## Requirements

- Python 3.11 or newer
- For local mode only: Ollama running locally with `qwen3:8b` pulled
- FastEmbed downloads the BGE embedding weights on first dense search; allow network access for this one-time model download, or use BM25 until weights are available.

## Configuration

Application settings are declared in `config/blackwing.properties` and resolved from environment variables. The BGE model property has a non-secret default that can be overridden by `BLACKWING_RAG_EMBEDDING_MODEL`. No credentials are stored in code or in the properties file. Choose **one** of the configurations below. Run provider configuration only in the FastAPI terminal. The integrated Streamlit UI needs only BLACKWING_UI_API_URL and an optional timeout; it does not need model credentials. In hosted mode, the key is read only from `BLACKWING_LLM_API_KEY`.

### Option A: local Ollama

Pull the model once:

```bash
ollama pull qwen3:8b
```

From the repository root, export these variables in the terminal used to start the API. The frontend uses only the API URL:

```bash
export BLACKWING_CONFIG="$PWD/config/blackwing.properties"
export BLACKWING_API_HOST="127.0.0.1"
export BLACKWING_API_PORT="8000"
export BLACKWING_LLM_PROVIDER="ollama"
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="qwen3:8b"
export OLLAMA_TIMEOUT_SECONDS="600"
export OLLAMA_TEMPERATURE="0.1"
export OLLAMA_THINK="false"
export OLLAMA_MAX_OUTPUT_TOKENS="1024"
export BLACKWING_RAG_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export BLACKWING_UI_API_URL="http://127.0.0.1:8000"
export BLACKWING_UI_TIMEOUT_SECONDS="2400"
```

### Option B: hosted OpenAI-compatible API

Use this instead of Option A when connecting to a hosted provider. Replace the three angle-bracket values with your provider's endpoint, exact model ID, and API key. Do not commit the key or place it in `config/blackwing.properties`. Run this full block in the API terminal only:

```bash
export BLACKWING_CONFIG="$PWD/config/blackwing.properties"
export BLACKWING_API_HOST="127.0.0.1"
export BLACKWING_API_PORT="8000"
export BLACKWING_LLM_PROVIDER="openai_compatible"
export BLACKWING_LLM_API_BASE_URL="<provider OpenAI-compatible base URL>"
export BLACKWING_LLM_API_MODEL="<provider model ID>"
export BLACKWING_LLM_API_KEY="<API key>"
export BLACKWING_LLM_API_TIMEOUT_SECONDS="120"
export BLACKWING_LLM_API_TEMPERATURE="0.1"
export BLACKWING_LLM_API_MAX_OUTPUT_TOKENS="1024"
export BLACKWING_RAG_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export BLACKWING_UI_API_URL="http://127.0.0.1:8000"
export BLACKWING_UI_TIMEOUT_SECONDS="900"
```

For Alibaba Cloud Model Studio, use the OpenAI-compatible base URL from the same region as the API key (for example, the Singapore workspace URL ending in `/compatible-mode/v1`) and the exact model ID enabled for that workspace. Model availability and free quota vary by region and model. The provider must support OpenAI-compatible Chat Completions and JSON mode.

If the app says `BLACKWING_LLM_PROVIDER is not set` or reports an invalid provider, set it in the terminal that launches the process. For the local Ollama setup, the value must be exactly:

```bash
export BLACKWING_LLM_PROVIDER="ollama"
```

For the hosted API setup, use `export BLACKWING_LLM_PROVIDER="openai_compatible"`. Restart the backend after exporting the variable. The frontend does not select a model provider.

## Install and run

From the repository root, after setting the environment values above in the current terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
uvicorn blackwing.api:app --host "$BLACKWING_API_HOST" --port "$BLACKWING_API_PORT"
```

The API starts on `http://127.0.0.1:8000`. Open a second terminal at the repository root, set `BLACKWING_UI_API_URL` to the backend address, activate `.venv`, then run:

`BLACKWING_RAG_EMBEDDING_MODEL` is optional: the properties file defaults it to `BAAI/bge-small-en-v1.5`. Set the variable only to override that model.

```bash
source .venv/bin/activate
python -m streamlit run blackwing/ui.py
```

The UI creates a runway-incursion case, starts an analysis run, polls for completion, and displays a readable, task-built report with separate agent details and a trace view. The report includes the tagline “Every flight leaves a story. Every clue reveals the truth.” The eight-agent workflow and four section-writing calls can take several minutes with a local model. The report is shown in the UI; no PDF is generated.

When an agent response fails JSON or schema validation, the API service log records the agent, provider/model, retry number, elapsed time, validation errors, and a JSON-escaped preview of the raw model response. The preview defaults to 4,000 characters and is bounded by `logging.model_response_log_limit` in `config/blackwing.properties`, backed by `BLACKWING_MODEL_RESPONSE_LOG_LIMIT`. Set that environment variable to `0` to disable response previews. Logs stay local and are not returned in API errors. Critic retries once and normalizes common small-model JSON variations; Critic and Verification also have explicitly labeled deterministic fallbacks when their model pass is malformed or cannot fit the safe context budget. Context-budget failures are labeled separately because no model response exists to log. The UI shows the agent outputs and execution trace, including fallback status.

## Run the contract test

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

The vertical-slice test uses a mocked model runner. It exercises the runway-incursion contract from knowledge ingestion and search through case creation, run status, report, trace and human review without depending on Ollama latency.

## Safety boundaries

- Treat uploaded narratives, evidence and retrieved passages as untrusted data.
- The system does not assign blame, determine legal liability, issue operational instructions, or make binding regulatory decisions.
- Candidate retrieval is not proof of source authenticity or regulatory applicability.
- A qualified human must verify sources, applicability, findings and recommendations.


## Integrated Streamlit UI (PowerShell)

Keep the configured FastAPI backend running with one worker. From the repository root in a second terminal:

```powershell
$env:BLACKWING_UI_API_URL = "http://127.0.0.1:8000"
.\.venv\Scripts\python.exe -m streamlit run .\blackwing\cloud\streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. This UI uses backend SQLite history, backend categorical risk and a separate report-quality checklist. It never opens a session case database or calls a model directly. See [the UI delivery report](docs/streamlit-ui-integration.md) for file changes, source-branch comparison, endpoint usage, 128 passing tests and remaining live-testing limits.


## Live investigation diagnostics

Agent-start, physical model-wait, completion/failure and restart-interruption events are now persisted incrementally and exposed through run progress, trace and audit. Streamlit explains report HTTP 409 by running/failed/interrupted state. See [the diagnostics and recovery guide](docs/live-diagnostics.md) for event meanings, sanitized errors, server-failure boundaries and 141 passing tests. The historical JFK process-stop cause remains undetermined; no live model replay was performed.


## Independent completeness results

The integrated v2 rubric separately reports Prompt Completeness (supplied intake) and Report Completeness (investigation documentation), with weighted checks and disclosed integrity ceilings. Aviation risk remains in its independent backend-calculated view, and Sunayana's original evidence checklist is retained. See [the scorer integration and saved-report evaluation](docs/quality-scorer-v2.md): 158 regression tests passed; existing JFK/Austin reports scored 84/58 for report completeness without model calls or report changes.

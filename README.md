# BlackWing AI

Runway-incursion API and local UI prototype. The repository now has a versioned case API, typed request/response models, an OpenAPI document, knowledge source endpoints with chunked hybrid retrieval, seven sequential LangGraph agents, run polling, trace access, report delivery and human review recording.

## API contract

- Interactive Swagger UI: `http://127.0.0.1:8000/api/v1/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/api/v1/openapi.json`
- Full endpoint/request/response contract: [docs/api-contract.md](docs/api-contract.md)
- Component and runtime architecture: [docs/architecture.md](docs/architecture.md)

## Current runnable slice

The API supports only `runway_incursion` in v1. It lets a client create/edit a draft case, attach text evidence, ingest/list/search knowledge documents, start an analysis run, poll status, read the typed report and trace, and record/list a review. Investigation, Compliance, Risk, Mitigation, Critic and Verification call the selected model provider. Similar Case Analysis uses deterministic, deduplicated source metadata so it can reliably mark a case unseen in the current corpus and report sourced historical findings without making them findings about the new case. Each role has its own response schema.

Each role is maintained as an independent package under `blackwing/agents/<role>/`. A package owns `agent.py`, `schema.py`, and `prompt.md`; shared model-provider handling lives in `blackwing/agents/runtime.py`, and workflow order/upstream context lives in `blackwing/agents/registry.py`. Contributors can change one role's prompt and schema without editing the other agent packages. Changes to an output schema must remain compatible with the public report schema in `blackwing/schemas.py`.

Cases and user-ingested knowledge sources use an in-memory repository and are lost when the API stops. The bundled historical NTSB report is loaded at startup. Retrieval chunks documents, combines BM25 and local BGE dense embeddings with reciprocal-rank fusion, and applies an operational-term rerank. If model weights are not available yet, searches continue using BM25. No approved risk matrix is configured; risk scores remain null. The app records review decisions but does not authenticate reviewers or publish a final regulatory determination. See the architecture document before using it beyond local development.

The bundled reference is a distinct Austin-Bergstrom runway-incursion case (NTSB DCA23FA149). The JFK scenario is incoming query data and is not included in the knowledge corpus. The Austin event is an analogy with different circumstances, not a source of findings about the JFK event. Source: [NTSB investigation DCA23FA149](https://www.ntsb.gov/investigations/Pages/DCA23FA149.aspx).

The model provider is selected through `BLACKWING_LLM_PROVIDER`: `ollama` uses the local Ollama chat API, while `openai_compatible` uses an OpenAI-compatible hosted chat completions endpoint with a bearer API key. Agent packages and workflow stay the same. The API adapter requests JSON object output and validates every result against the role's Pydantic schema. Providers must support the OpenAI-compatible Chat Completions request format and JSON mode.

## Requirements

- Python 3.11 or newer
- For local mode only: Ollama running locally with `qwen3:8b` pulled
- FastEmbed downloads the BGE embedding weights on first dense search; allow network access for this one-time model download, or use BM25 until weights are available.

## Configuration

Application settings are declared in `config/blackwing.properties` and resolved from environment variables. The BGE model property has a non-secret default that can be overridden by `BLACKWING_RAG_EMBEDDING_MODEL`. No credentials are stored in code or in the properties file. Choose **one** of the configurations below. Run the selected block in every terminal that starts the API or Streamlit UI because environment variables are local to each terminal. In hosted mode, the key is read only from `BLACKWING_LLM_API_KEY`.

### Option A: local Ollama

Pull the model once:

```bash
ollama pull qwen3:8b
```

From the repository root, export these variables in the terminal used to start the API. Repeat the same exports in the UI terminal:

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

Use this instead of Option A when connecting to a hosted provider. Replace the three angle-bracket values with your provider's endpoint, exact model ID, and API key. Do not commit the key or place it in `config/blackwing.properties`. Run this full block in the API terminal and repeat it in the UI terminal:

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

For the hosted API setup, use `export BLACKWING_LLM_PROVIDER="openai_compatible"`. Restart the affected process after exporting the variable, and repeat the export in the other terminal because the API and UI have separate environments.

## Install and run

From the repository root, after setting the environment values above in the current terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
uvicorn blackwing.api:app --host "$BLACKWING_API_HOST" --port "$BLACKWING_API_PORT"
```

The API starts on `http://127.0.0.1:8000`. Open a second terminal at the repository root, repeat the **same selected provider configuration block** there, activate `.venv`, then run:

`BLACKWING_RAG_EMBEDDING_MODEL` is optional: the properties file defaults it to `BAAI/bge-small-en-v1.5`. Set the variable only to override that model.

```bash
source .venv/bin/activate
python -m streamlit run blackwing/ui.py
```

The UI creates a runway-incursion case, starts an analysis run, polls for completion, and displays each typed agent output plus the trace. The seven-step workflow can take several minutes with a local model.

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

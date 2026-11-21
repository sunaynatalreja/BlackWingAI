# BlackWing AI

Runway-incursion API and local UI prototype. The repository has a versioned case API, typed request/response models, an OpenAPI document, knowledge source endpoints with chunked hybrid retrieval, eight sequential LangGraph agents, run polling, trace access, report delivery and human review recording.

## API contract

- Interactive Swagger UI: `http://127.0.0.1:8000/api/v1/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/api/v1/openapi.json`
- Full endpoint/request/response contract: [docs/api-contract.md](docs/api-contract.md)
- Component and runtime architecture: [docs/architecture.md](docs/architecture.md)

## Current runnable slice

The API supports only `runway_incursion` in v1. It lets a client create/edit a draft case, attach text evidence, ingest/list/search knowledge documents, start an analysis run, poll status, read the typed report and trace, and record/list a review. Investigation, Compliance, Risk, Mitigation, Critic, Verification and Report call the selected model provider. Similar Case Analysis uses deterministic, deduplicated source metadata so it can reliably mark a case unseen in the current corpus and report sourced historical findings without making them findings about the new case. Each role has its own response schema.

Each role is maintained as an independent package under `blackwing/agents/<role>/`. A package owns its registration, schema, and prompt files; shared model-provider handling lives in `blackwing/agents/runtime.py`, and workflow order/upstream context lives in `blackwing/agents/registry.py`. The Report Agent receives the previous agents' structured work products, creates a constrained task plan, then drafts the report as four smaller section tasks before assembling a typed report. The UI presents that report first and keeps the source agent JSON in a separate details tab. Contributors can change one role's prompts and schema without editing the other agent packages. Changes to an output schema must remain compatible with the public report schema in `blackwing/schemas.py`.

Prompting strategies are role-specific: the Investigation Agent uses few-shot examples to teach the model how to distinguish sourced facts, reported observations, and unknowns. Other model-backed analysis agents use zero-shot role instructions in their individual prompt files. Similar Case Analysis is deterministic code and makes no model call. The Report Agent uses task breakdown prompting: it plans four fixed report sections, then drafts each section in a focused call using relevant upstream agent outputs.

Cases and user-ingested knowledge sources use an in-memory repository and are lost when the API stops. The RAG index is built from the approved source manifest in `datasets/BLACKWING_DATA/01_RAG_KNOWLEDGE_BASE`; protected final reports, gold evaluation files, and unseen case prompts are excluded. Retrieval combines BM25 and hosted BGE dense embeddings with reciprocal-rank fusion. The API uses the shared IIT BGE-small-en-v1.5 embedding endpoint for dense retrieval and falls back to BM25 only when dense retrieval is unavailable. Risk uses a provisional 5×5 severity × likelihood estimate, always subject to qualified human review; it is not an approved operational risk determination. The app records review decisions but does not authenticate reviewers or publish a final regulatory determination. See the architecture document before using it beyond local development.

The JFK scenario and the `unseen` and `golden` sample prompts are query inputs, never index sources. Historical factual records in the knowledge base may be retrieved as analogies, while final reports and evaluation answers remain held out.

### Build and use the dataset RAG index

The dataset-aware builder reads `BW-MF-001-SOURCE-MANIFEST.json` and indexes only approved, trial-eligible files under `01_RAG_KNOWLEDGE_BASE`. It refuses to overwrite an existing index. Install the RAG dependencies first with `pip install -r requirements-rag.txt`. Configure the shared embedding endpoint and keys below before building; this sends document batches to the hosted service. `02_GOLD_EVALUATION_PROTECTED` and `03_UNSEEN_INCIDENT_TEST_CASES` are evaluation/demo inputs only and are never embedded. In Streamlit, select **Load golden demo input** and attach the Delhi Ariana preliminary report as current-case evidence; the file is not added to the shared index.

```bash
python -m blackwing.rag.build \
  --dataset-root datasets/BLACKWING_DATA \
  --output rag_indexes/blackwing-kb-iit-bge-v1 \
  --backend iit \
  --cache rag_cache/embeddings_iit.sqlite \
  --strategy fixed
```

After setting the hosted API and embedding environment variables in the Configuration section below, build the fresh hosted-vector index with this command. When it completes, set `BLACKWING_RAG_INDEX_DIR` to the generated index in both API and UI environments. `config/blackwing.properties` reads this path from the environment:

```bash
export BLACKWING_RAG_INDEX_DIR="$PWD/rag_indexes/blackwing-kb-iit-bge-v1"
```

PowerShell:

```powershell
$env:BLACKWING_RAG_INDEX_DIR = "$PWD\rag_indexes\blackwing-kb-iit-bge-v1"
```

To run retrieval-only validation against unseen prompts (without exposing answer keys or evaluator notes):

```bash
python -m blackwing.rag.evaluate_datasets \
  --index rag_indexes/blackwing-kb-iit-bge-v1 \
  --split unseen \
  --output work/unseen-retrieval.json \
  --backend iit \
  --cache rag_cache/embeddings_iit.sqlite
```

Use `--split golden` to prepare the sanitized demo prompt. These retrieval outputs do not measure answer quality or recall@k until relevance labels are independently reviewed.

BlackWing uses the IIT Kharagpur hosted OpenAI-compatible chat-completions endpoint with `Qwen/Qwen3-8B`. The API key and CA certificate path are read only from environment variables referenced by `config/blackwing.properties`. Model replies are validated against each agent's Pydantic schema.

## Requirements

- Python 3.11 or newer
- Document chunks and search queries are embedded by the shared IIT endpoint. Requests batch up to 16 texts, each validated to fit within 512 BGE tokens; vectors are cached locally. Rebuild any previous local-BGE index with `--backend iit` because local and hosted vectors have different identities.

## Configuration

BlackWing uses the IIT Kharagpur hosted OpenAI-compatible endpoint with `Qwen/Qwen3-8B`. Settings are declared in `config/blackwing.properties` and resolved from environment variables. API keys are never stored in source code or the properties file. Set the assigned URL, group key, and CA certificate path in both the API and Streamlit terminals.

| Group | API base URL | CA certificate |
|---|---|---|
| 001–015 | `https://103.42.50.41:1031/v1` | `vm07-ca.crt` |
| 016–030 | `https://103.42.50.41:1032/v1` | `vm06-ca.crt` |
| 031–045 | `https://103.42.50.41:1033/v1` | `vm05-ca.crt` |
| 046–065 | `https://103.42.50.41:1034/v1` | `vm04-ca.crt` |

The attached `vm02-ca.crt` is the certificate for the shared embeddings endpoint. The inference endpoint uses its group certificate (`vm04-ca.crt` through `vm07-ca.crt`), so configure those two CA paths separately. The group number was not included in the request; enter your assigned inference URL and matching model CA below.

Run this configuration from the repository root in the API terminal and repeat it in the Streamlit terminal:

```bash
export BLACKWING_CONFIG="$PWD/config/blackwing.properties"
export BLACKWING_API_HOST="127.0.0.1"
export BLACKWING_API_PORT="8000"
export BLACKWING_LLM_PROVIDER="openai_compatible"
export BLACKWING_LLM_API_BASE_URL="<your assigned URL from the group table>"
export BLACKWING_LLM_API_MODEL="Qwen/Qwen3-8B"
read -s "BLACKWING_LLM_API_KEY?Enter your inference API key: "
echo
export BLACKWING_LLM_API_KEY
export BLACKWING_LLM_CA_CERT="<path to your assigned inference-group CA certificate>"
export BLACKWING_LLM_API_TIMEOUT_SECONDS="300"
export BLACKWING_LLM_API_TEMPERATURE="0.1"
export BLACKWING_LLM_API_MAX_OUTPUT_TOKENS="768"
export BLACKWING_AGENT_TOKEN_QUOTA_INVESTIGATION="768"
export BLACKWING_AGENT_TOKEN_QUOTA_COMPLIANCE="768"
export BLACKWING_AGENT_TOKEN_QUOTA_RISK="768"
export BLACKWING_AGENT_TOKEN_QUOTA_MITIGATION="512"
export BLACKWING_AGENT_TOKEN_QUOTA_CRITIC="512"
export BLACKWING_AGENT_TOKEN_QUOTA_VERIFICATION="512"
export BLACKWING_AGENT_TOKEN_QUOTA_REPORT_PLANNER="256"
export BLACKWING_AGENT_TOKEN_QUOTA_REPORT_SECTION="768"
export BLACKWING_API_JSON_RESPONSE_FORMAT="false"
export BLACKWING_EMBEDDING_URL="https://103.42.50.41:1035/v1"
export BLACKWING_EMBEDDING_MODEL="bge-small-en-v1.5"
read -s "BLACKWING_EMBEDDING_API_KEY?Enter the shared embeddings API key: "
echo
export BLACKWING_EMBEDDING_API_KEY
export BLACKWING_EMBEDDING_CA_CERT="$HOME/Downloads/VM02_Embeddings_Access_2/vm02-ca.crt"
export BLACKWING_EMBEDDING_TIMEOUT="60"
export BLACKWING_EMBEDDING_RETRIES="2"
export BLACKWING_RAG_BACKEND="iit"
export BLACKWING_RAG_CACHE="$PWD/rag_cache/embeddings_iit.sqlite"
export BLACKWING_RAG_INDEX_DIR="$PWD/rag_indexes/blackwing-kb-iit-bge-v1"
export BLACKWING_UI_API_URL="http://127.0.0.1:8000"
export BLACKWING_UI_TIMEOUT_SECONDS="1800"
```

The base URL already includes `/v1`; BlackWing appends `/chat/completions` once. TLS verification remains enabled and uses the configured CA certificate. Each agent has an output-token quota in `config/blackwing.properties`, overridable by its `BLACKWING_AGENT_TOKEN_QUOTA_<AGENT>` environment variable. The quota is capped by `BLACKWING_LLM_API_MAX_OUTPUT_TOKENS` (768 in this example; up to 1,024 configured here). Input and output share the model context window. Requests that exceed the context budget return an error rather than silently truncating input. The report workflow drafts sections separately and preserves details extracted by its source agents.

Hosted token preflight uses local tokenizer files for Qwen3 and BGE, not local model inference. Install dependencies with `pip install -r requirements.txt`; cache the tokenizer once with `python -c 'from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("Qwen/Qwen3-8B"); AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")'` before starting the API. This downloads tokenizer files only, not inference/embedding model weights or API credentials.

Environment variables are process-local, so set the same configuration in both terminals. Restart the API and UI after changing the values. On Windows PowerShell, use `$env:VARIABLE = 'value'` syntax and set the certificate path to its actual Windows location, for example `$env:BLACKWING_LLM_CA_CERT = "$env:USERPROFILE\Downloads\VM02_Embeddings_Access_2\vm02-ca.crt"`.

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

```bash
source .venv/bin/activate
python -m streamlit run blackwing/ui.py
```

The UI creates a runway-incursion case, starts an analysis run, polls for completion, and displays a readable, task-built report with separate agent details and a trace view. The report includes the tagline “Every flight leaves a story. Every clue reveals the truth.” The eight-agent workflow and four section-writing calls can take several minutes. The report is shown in the UI; no PDF is generated.

When an agent response fails JSON or schema validation, the API service log records the agent, provider/model, retry number, elapsed time, validation errors, and a JSON-escaped preview of the raw model response. The preview defaults to 4,000 characters and is bounded by `logging.model_response_log_limit` in `config/blackwing.properties`, backed by `BLACKWING_MODEL_RESPONSE_LOG_LIMIT`. Set that environment variable to `0` to disable response previews. Logs stay local and are not returned in API errors. Critic retries once and normalizes common small-model JSON variations; Critic and Verification also have explicitly labeled deterministic fallbacks when their model pass is malformed or cannot fit the safe context budget. Context-budget failures are labeled separately because no model response exists to log. The UI shows the agent outputs and execution trace, including fallback status.

## Run the contract test

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q
```

The vertical-slice test uses a mocked model runner. It exercises the runway-incursion contract from knowledge ingestion and search through case creation, run status, report, trace and human review without making hosted inference calls.

## Safety boundaries

- Treat uploaded narratives, evidence and retrieved passages as untrusted data.
- The system does not assign blame, determine legal liability, issue operational instructions, or make binding regulatory decisions.
- Candidate retrieval is not proof of source authenticity or regulatory applicability.
- A qualified human must verify sources, applicability, findings and recommendations.

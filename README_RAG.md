# Retrieval and Evaluation

Generic evidence retrieval and evaluation for a seven-agent runway-incursion investigation workflow. No standalone Q&A interface is required. The existing UI and agent schemas remain unchanged.

## Local setup

Extract the full project package, open a PowerShell terminal in its `BlackWingAI-Arun` directory, and run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-rag.txt -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

### Retrieval checks — no hosted quota

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.smoke
.\.venv\Scripts\python.exe -m blackwing.rag.smoke --real-models
```

The first is a synthetic lexical and routing check. The optional real-model smoke check uses a local BGE model on synthetic evidence and does not establish full-corpus retrieval accuracy. Normal indexing and runtime retrieval use the hosted IIT BGE endpoint configured below.

## Integration with the report workflow

The application uses Investigation → Similar Case → Compliance → Risk → Mitigation → Critic → Verification. `blackwing/rag/retriever.py` maintains the original `RAGRetriever.search` API. `agent_router.py` provides agent-specific evidence packets, separating current incident facts from historical analogies and jurisdiction/date-filtered requirements. `small_context.py` preflights complete input against the model context instead of clipping text.

Hosted Qwen inference and BGE embeddings for the API terminal:

```powershell
$env:BLACKWING_CONFIG = "$PWD\config\blackwing.properties"
$env:BLACKWING_LLM_PROVIDER = "openai_compatible"
$env:BLACKWING_LLM_API_BASE_URL = "<your assigned group URL ending in /v1>"
$env:BLACKWING_LLM_API_MODEL = "Qwen/Qwen3-8B"
$env:BLACKWING_LLM_API_KEY = "<set securely; do not save in repository>"
$env:BLACKWING_LLM_CA_CERT = "<inference-group CA certificate path>"
$env:BLACKWING_EMBEDDING_URL = "https://103.42.50.41:1035/v1"
$env:BLACKWING_EMBEDDING_MODEL = "bge-small-en-v1.5"
$env:BLACKWING_EMBEDDING_API_KEY = "<set securely; do not save in repository>"
$env:BLACKWING_EMBEDDING_CA_CERT = "$env:USERPROFILE\Downloads\VM02_Embeddings_Access_2\vm02-ca.crt"
$env:BLACKWING_LLM_API_MAX_OUTPUT_TOKENS = "768"
$env:BLACKWING_AGENT_TOKEN_QUOTA_INVESTIGATION = "768"
$env:BLACKWING_AGENT_TOKEN_QUOTA_COMPLIANCE = "768"
$env:BLACKWING_AGENT_TOKEN_QUOTA_RISK = "768"
$env:BLACKWING_AGENT_TOKEN_QUOTA_MITIGATION = "512"
$env:BLACKWING_AGENT_TOKEN_QUOTA_CRITIC = "512"
$env:BLACKWING_AGENT_TOKEN_QUOTA_VERIFICATION = "512"
$env:BLACKWING_AGENT_TOKEN_QUOTA_REPORT_PLANNER = "256"
$env:BLACKWING_AGENT_TOKEN_QUOTA_REPORT_SECTION = "768"
$env:BLACKWING_API_JSON_RESPONSE_FORMAT = "false"
$env:BLACKWING_RAG_BACKEND = "iit"
$env:BLACKWING_RAG_RERANKER = "crossencoder"
$env:BLACKWING_AGENT_RAG = "1"
$env:BLACKWING_AGENT_RAG_METHOD = "rerank"
$env:BLACKWING_RAG_CACHE = "$PWD\rag_cache\embeddings.sqlite"
$env:BLACKWING_RAG_INDEX_DIR = "$PWD\rag_indexes\blackwing-kb-iit-bge-v1"
.\.venv\Scripts\python.exe -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8000
```

Per-agent quotas are declared in `config/blackwing.properties` and can be overridden with `BLACKWING_AGENT_TOKEN_QUOTA_<AGENT>` environment variables. The `BLACKWING_LLM_API_MAX_OUTPUT_TOKENS` setting remains the provider-wide ceiling, so a per-agent value above it is capped. `similar_case` is deterministic in the current workflow; its quota is ready for model-backed use.

## Document indexing and retrieval experiments

Prepare approved authentic documents in `source_documents/`; create `source_manifest.json` using `examples/sources.example.json` as a schema example. Each manifest record needs stable identifiers, provenance, document category, jurisdiction and relevant dates. Generated reports and protected evaluation answers must not be indexed.

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.build --manifest .\source_manifest.json --root .\source_documents --output .\rag_indexes\iit_bge_v1 --backend iit --strategy overlap --cache .\rag_cache\embeddings_iit.sqlite
```

Run three chunking comparisons with approved *development* page judgments:

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.chunking_experiments --manifest .\source_manifest.json --root .\source_documents --development-gold .\gold_private\development_pages.csv --output .\work\chunking_development_v1.json --backend iit --method rerank --cache .\rag_cache\embeddings_iit.sqlite
```

The chunking output is expected-page Hit@k/MRR, **not** true Recall@k. For actual Recall@5/10, supply complete judged relevant chunk IDs (`question_id,query,split,relevant_chunk_ids,complete_relevance`) and run:

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.evaluate --index .\rag_indexes\iit_bge_v1 --gold .\gold_private\development_chunk_judgments.csv --output .\work\retrieval_eval_v1.json --backend iit --rerank --cache .\rag_cache\embeddings_iit.sqlite
```

BM25, dense, hybrid and reranked methods are measured separately. Report factual quality requires approved evidence-based reference judgments, distinct from retrieval metrics.

## Infrastructure boundaries

- Hosted embeddings use `bge-small-en-v1.5` (384 dimensions), the same model for indexing and search, a validated 512-token input limit, and batches of at most 16 texts. Vectors are cached locally in SQLite.
- The shared endpoint is `https://103.42.50.41:1035/v1/embeddings`. Its key and `vm02-ca.crt` path are environment-backed properties. TLS verification remains enabled; credentials stay out of source control.
- Hosted inference calls are sequential and respect a combined 8,192-token input/output cap, with output at most 1,024 tokens. The current launcher sets a 768-token response target. Never pass the full corpus to the model.
- Cache files, source document copies, secrets, private gold and model-generated outputs remain ignored by Git.
- The 500–2,000 authentic-document requirement is a future *data acceptance check*: 500–2,000 real unique eligible documents must be established using an approved corpus, not inferred from chunk counts or the synthetic examples.
- Hosted API connectivity, genuine 500–2,000-document performance, true Recall@k and independent report correctness remain **NOT VERIFIED** until the corresponding real-data tests are run.

## Pull-request review

The separate contribution-only archive contains additions and changed files, plus a unified diff for modifications to existing code. Apply it on a current authenticated checkout on a feature branch. Review changes to `blackwing/api.py`, `blackwing/service.py`, and `blackwing/agents/runtime.py` against the *current* branch before merging. The original UI, agent schemas and agent prompts are not replaced.

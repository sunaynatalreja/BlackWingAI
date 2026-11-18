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

The first is a synthetic lexical and routing check. The second runs local BGE and CrossEncoder on synthetic evidence. Neither establishes full-corpus retrieval accuracy. The real-model check can download public model weights.

## Integration with the report workflow

The application uses Investigation → Similar Case → Compliance → Risk → Mitigation → Critic → Verification. `blackwing/rag/retriever.py` maintains the original `RAGRetriever.search` API. `agent_router.py` provides small agent-specific evidence packets, separating current incident facts from historical analogies and jurisdiction/date-filtered requirements. `small_context.py` bounds each model call rather than attempting to send all retrieved documents to Qwen.

Opt-in local environment for the API terminal:

```powershell
$env:BLACKWING_CONFIG = "$PWD\config\blackwing.properties"
$env:BLACKWING_LLM_PROVIDER = "ollama"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_MODEL = "qwen3:8b"
$env:OLLAMA_THINK = "false"
$env:OLLAMA_MAX_OUTPUT_TOKENS = "512"
$env:BLACKWING_RAG_BACKEND = "local"
$env:BLACKWING_RAG_RERANKER = "crossencoder"
$env:BLACKWING_AGENT_RAG = "1"
$env:BLACKWING_AGENT_RAG_METHOD = "rerank"
$env:BLACKWING_RAG_CACHE = "$PWD\rag_cache\embeddings.sqlite"
.\.venv\Scripts\python.exe -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8000
```

## Document indexing and retrieval experiments

Prepare approved authentic documents in `source_documents/`; create `source_manifest.json` using `examples/sources.example.json` as a schema example. Each manifest record needs stable identifiers, provenance, document category, jurisdiction and relevant dates. Generated reports and protected evaluation answers must not be indexed.

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.build --manifest .\source_manifest.json --root .\source_documents --output .\rag_indexes\local_bge_v1 --backend local --strategy overlap --cache .\rag_cache\embeddings.sqlite
```

Run three chunking comparisons with approved *development* page judgments:

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.chunking_experiments --manifest .\source_manifest.json --root .\source_documents --development-gold .\gold_private\development_pages.csv --output .\work\chunking_development_v1.json --backend local --method rerank --cache .\rag_cache\embeddings.sqlite
```

The chunking output is expected-page Hit@k/MRR, **not** true Recall@k. For actual Recall@5/10, supply complete judged relevant chunk IDs (`question_id,query,split,relevant_chunk_ids,complete_relevance`) and run:

```powershell
.\.venv\Scripts\python.exe -m blackwing.rag.evaluate --index .\rag_indexes\local_bge_v1 --gold .\gold_private\development_chunk_judgments.csv --output .\work\retrieval_eval_v1.json --backend local --rerank --cache .\rag_cache\embeddings.sqlite
```

BM25, dense, hybrid and reranked methods are measured separately. Report factual quality requires approved evidence-based reference judgments, distinct from retrieval metrics.

## Infrastructure boundaries

- Local-first embeddings use `BAAI/bge-small-en-v1.5` (384 dimensions), with the same model for index and query, tokenizer-validated maximum 512 input tokens and maximum 16 texts per batch.
- Hosted embedding calls require explicit `BLACKWING_RAG_BACKEND=iit` together with `BLACKWING_EMBEDDING_URL`, `BLACKWING_EMBEDDING_API_KEY` and `BLACKWING_EMBEDDING_CA_CERT`. HTTPS/TLS verification must not be disabled; no credentials in source control.
- Hosted inference calls are sequential and respect a combined 8,192-token input/output cap, with output at most 1,024 tokens. Set conservative output target 512 by default. Never pass the full corpus to the model.
- Cache files, source document copies, secrets, private gold and model-generated outputs remain ignored by Git.
- The 500–2,000 authentic-document requirement is a future *data acceptance check*: 500–2,000 real unique eligible documents must be established using an approved corpus, not inferred from chunk counts or the synthetic examples.
- Hosted API connectivity, genuine 500–2,000-document performance, true Recall@k and independent report correctness remain **NOT VERIFIED** until the corresponding real-data tests are run.

## Pull-request review

The separate contribution-only archive contains additions and changed files, plus a unified diff for modifications to existing code. Apply it on a current authenticated checkout on a feature branch. Review changes to `blackwing/api.py`, `blackwing/service.py`, and `blackwing/agents/runtime.py` against the *current* branch before merging. The original UI, agent schemas and agent prompts are not replaced.

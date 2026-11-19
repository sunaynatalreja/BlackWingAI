# Run as: . .\scripts\use_local.ps1
# For local Ollama inference + local BGE & CrossEncoder; no IIT API calls.
$env:BLACKWING_CONFIG = (Join-Path (Get-Location) 'config\blackwing.properties')
$env:BLACKWING_API_HOST = '127.0.0.1'
$env:BLACKWING_API_PORT = '8001'
$env:BLACKWING_UI_API_URL = 'http://127.0.0.1:8001'
$env:BLACKWING_UI_TIMEOUT_SECONDS = '1800'
$env:BLACKWING_LLM_PROVIDER = 'ollama'
$env:OLLAMA_BASE_URL = 'http://127.0.0.1:11434'
$env:OLLAMA_MODEL = 'qwen3:8b'
$env:OLLAMA_TIMEOUT_SECONDS = '900'
$env:OLLAMA_TEMPERATURE = '0'
$env:OLLAMA_THINK = 'false'
$env:OLLAMA_MAX_OUTPUT_TOKENS = '768'
$env:BLACKWING_RAG_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
$env:BLACKWING_RAG_BACKEND = 'local'
$env:BLACKWING_RAG_RERANKER = 'crossencoder'
$env:BLACKWING_AGENT_RAG = '1'
$env:BLACKWING_AGENT_RAG_METHOD = 'rerank'
$env:BLACKWING_RAG_CACHE = (Join-Path (Get-Location) 'rag_cache\embeddings.sqlite')
# Use a prebuilt compatible index; override before starting if different.
if (-not $env:BLACKWING_RAG_INDEX_DIR) {
  $env:BLACKWING_RAG_INDEX_DIR = (Join-Path (Get-Location) 'rag_indexes\austin_local_bge_v1')
}
Remove-Item Env:BLACKWING_LLM_CA_CERT -ErrorAction SilentlyContinue
Write-Host 'Local configuration set (Ollama + BGE local). Index:' $env:BLACKWING_RAG_INDEX_DIR
Write-Host 'Start: .\.venv\Scripts\python.exe -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8001'

param()
# Dot-source: . .\scripts\use_iit.ps1
# Requires environment secrets set outside repository; does NOT store credentials.
$env:BLACKWING_EMBEDDING_URL = 'https://103.42.50.41:1035/v1'
if (-not $env:BLACKWING_EMBEDDING_CA_CERT) {
  $env:BLACKWING_EMBEDDING_CA_CERT = Join-Path $env:USERPROFILE 'Downloads\VM02_Embeddings_Access_2\vm02-ca.crt'
}
if (-not $env:BLACKWING_RAG_INDEX_DIR) {
  $env:BLACKWING_RAG_INDEX_DIR = Join-Path (Get-Location) 'rag_indexes\blackwing-kb-iit-bge-v1'
}
$required = @('BLACKWING_LLM_API_BASE_URL','BLACKWING_LLM_API_KEY','BLACKWING_LLM_CA_CERT',
              'BLACKWING_EMBEDDING_API_KEY','BLACKWING_EMBEDDING_CA_CERT',
              'BLACKWING_RAG_INDEX_DIR')
foreach ($key in $required) {
  $value = [Environment]::GetEnvironmentVariable($key, 'Process')
  if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing $key; set it securely outside Git." }
}
if (-not $env:BLACKWING_LLM_API_BASE_URL.StartsWith('https://')) {
  throw 'Hosted model URL must use HTTPS.'
}
if (-not (Test-Path $env:BLACKWING_LLM_CA_CERT)) { throw 'Model CA certificate path not found.' }
if (-not (Test-Path $env:BLACKWING_EMBEDDING_CA_CERT)) { throw 'Embeddings CA certificate path not found.' }
if (-not (Test-Path $env:BLACKWING_RAG_INDEX_DIR)) { throw 'Build a hosted-BGE index and set BLACKWING_RAG_INDEX_DIR first.' }
$env:BLACKWING_CONFIG = (Join-Path (Get-Location) 'config\blackwing.properties')
$env:BLACKWING_API_HOST = '127.0.0.1'
$env:BLACKWING_API_PORT = '8001'
$env:BLACKWING_UI_API_URL = 'http://127.0.0.1:8001'
$env:BLACKWING_UI_TIMEOUT_SECONDS = '1800'
$env:BLACKWING_LLM_PROVIDER = 'openai_compatible'
$env:BLACKWING_LLM_API_MODEL = 'Qwen/Qwen3-8B'
$env:BLACKWING_LLM_API_TIMEOUT_SECONDS = '300'
$env:BLACKWING_LLM_API_TEMPERATURE = '0.1'
$env:BLACKWING_LLM_API_MAX_OUTPUT_TOKENS = '768' # IIT gateway output budget; combined context max 8192
$env:BLACKWING_QWEN_TOKENIZER = 'Qwen/Qwen3-8B' # Exact model tokenizer must be cached locally
$env:BLACKWING_EMBEDDING_MODEL = 'bge-small-en-v1.5'
$env:BLACKWING_RAG_BACKEND = 'iit'
$env:BLACKWING_RAG_RERANKER = 'crossencoder' # Always runs locally; NOT on IIT endpoint
$env:BLACKWING_AGENT_RAG = '1'
$env:BLACKWING_AGENT_RAG_METHOD = 'rerank'
$env:BLACKWING_RAG_CACHE = (Join-Path (Get-Location) 'rag_cache\embeddings_iit.sqlite')
$env:BLACKWING_API_JSON_RESPONSE_FORMAT = 'false' # Match the documented chat-completions request
$env:BLACKWING_IIT_MIN_INTERVAL_SECONDS = '1.1' # Conservative max 60 req/min on one key
Write-Host 'IIT Qwen3-8B inference and hosted BGE embeddings configured.'
Write-Host 'No live request made. Verify tokenizer and credentials, then start the API.'

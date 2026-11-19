param([ValidateSet('local','iit')][string]$Embeddings = 'local')
# Dot-source: . .\scripts\use_iit.ps1 -Embeddings local
# Requires environment secrets set outside repository; does NOT store credentials.
$required = @('BLACKWING_LLM_API_BASE_URL','BLACKWING_LLM_API_MODEL',
              'BLACKWING_LLM_API_KEY','BLACKWING_LLM_CA_CERT')
if ($Embeddings -eq 'iit') {
  $required += @('BLACKWING_EMBEDDING_URL','BLACKWING_EMBEDDING_API_KEY',
                 'BLACKWING_EMBEDDING_CA_CERT','BLACKWING_IIT_RAG_INDEX_DIR')
}
foreach ($key in $required) {
  $value = [Environment]::GetEnvironmentVariable($key, 'Process')
  if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing $key; set it securely outside Git." }
}
if (-not $env:BLACKWING_LLM_API_BASE_URL.StartsWith('https://')) {
  throw 'Hosted model URL must use HTTPS.'
}
if (-not (Test-Path $env:BLACKWING_LLM_CA_CERT)) { throw 'Model CA certificate path not found.' }
if ($Embeddings -eq 'iit') {
  if (-not $env:BLACKWING_EMBEDDING_URL.StartsWith('https://')) { throw 'Embeddings URL must use HTTPS.' }
  if (-not (Test-Path $env:BLACKWING_EMBEDDING_CA_CERT)) { throw 'Embeddings CA certificate path not found.' }
  if (-not (Test-Path $env:BLACKWING_IIT_RAG_INDEX_DIR)) { throw 'Build a separate IIT-compatible vector index first.' }
  $env:BLACKWING_RAG_INDEX_DIR = $env:BLACKWING_IIT_RAG_INDEX_DIR
} elseif (-not $env:BLACKWING_RAG_INDEX_DIR) {
  throw 'Set BLACKWING_RAG_INDEX_DIR to an existing local-BGE index.'
}
$env:BLACKWING_CONFIG = (Join-Path (Get-Location) 'config\blackwing.properties')
$env:BLACKWING_API_HOST = '127.0.0.1'
$env:BLACKWING_API_PORT = '8001'
$env:BLACKWING_UI_API_URL = 'http://127.0.0.1:8001'
$env:BLACKWING_UI_TIMEOUT_SECONDS = '1800'
$env:BLACKWING_LLM_PROVIDER = 'openai_compatible'
$env:BLACKWING_LLM_API_TIMEOUT_SECONDS = '300'
$env:BLACKWING_LLM_API_TEMPERATURE = '0'
$env:BLACKWING_LLM_API_MAX_OUTPUT_TOKENS = '768' # Max allowed 1024; combined max 8192
$env:BLACKWING_QWEN_TOKENIZER = 'Qwen/Qwen3-8B' # Exact model tokenizer must be cached locally
$env:BLACKWING_RAG_EMBEDDING_MODEL = 'BAAI/bge-small-en-v1.5'
$env:BLACKWING_RAG_BACKEND = $Embeddings
$env:BLACKWING_RAG_RERANKER = 'crossencoder' # Always runs locally; NOT on IIT endpoint
$env:BLACKWING_AGENT_RAG = '1'
$env:BLACKWING_AGENT_RAG_METHOD = 'rerank'
$env:BLACKWING_RAG_CACHE = (Join-Path (Get-Location) 'rag_cache\embeddings_iit.sqlite')
$env:BLACKWING_API_JSON_RESPONSE_FORMAT = '1' # Change to 0 if IIT gateway rejects JSON mode
$env:BLACKWING_IIT_MIN_INTERVAL_SECONDS = '1.1' # Conservative max 60 req/min on one key
Write-Host 'IIT model selected; embeddings backend:' $Embeddings
Write-Host 'No live request made. Verify tokenizer and credentials, then start the API.'

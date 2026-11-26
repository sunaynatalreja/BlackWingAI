param([switch]$PromptKeys)
# Run in PowerShell: . .\scripts\use_iit.ps1
# Existing keys: . .\scripts\use_iit.ps1
# Enter both keys privately: . .\scripts\use_iit.ps1 -PromptKeys
# LM and embeddings use separate credentials. No keys are saved or displayed.
# This script performs local checks only; it does not call IIT or rebuild indexes.

$iitRepoRoot = Split-Path -Parent $PSScriptRoot
$iitDefaultIndex = Join-Path $iitRepoRoot 'rag_indexes\expanded_iit_bge_20261122_v1'
$iitLegacyIndex = Join-Path $iitRepoRoot 'rag_indexes\blackwing-kb-iit-bge-v1'
$iitDefaultChatCert = Join-Path $env:USERPROFILE 'Downloads\vm07-ca.crt'
$iitLegacyChatCert = Join-Path $env:USERPROFILE 'Downloads\VM02_Embeddings_Access_2\vm07-ca.crt'
$iitDefaultEmbeddingCert = Join-Path $env:USERPROFILE 'Downloads\VM02_Embeddings_Access_2\vm02-ca.crt'

# Prepare values before changing the environment, so a failed check does not
# leave a partially configured setup. Retain valid explicit endpoint overrides.
$iitChatUrl = $env:BLACKWING_LLM_API_BASE_URL
if ([string]::IsNullOrWhiteSpace($iitChatUrl)) { $iitChatUrl = 'https://103.42.50.41:1031/v1' }
$iitEmbeddingUrl = $env:BLACKWING_EMBEDDING_URL
if ([string]::IsNullOrWhiteSpace($iitEmbeddingUrl)) { $iitEmbeddingUrl = 'https://103.42.50.41:1035/v1' }
$iitChatCert = $env:BLACKWING_LLM_CA_CERT
if ([string]::IsNullOrWhiteSpace($iitChatCert) -or $iitChatCert -eq $iitLegacyChatCert) {
    $iitChatCert = $iitDefaultChatCert
}
$iitEmbeddingCert = $env:BLACKWING_EMBEDDING_CA_CERT
if ([string]::IsNullOrWhiteSpace($iitEmbeddingCert)) { $iitEmbeddingCert = $iitDefaultEmbeddingCert }
$iitIndex = $env:BLACKWING_RAG_INDEX_DIR
if ([string]::IsNullOrWhiteSpace($iitIndex)) { $iitIndex = $env:BLACKWING_IIT_RAG_INDEX_DIR }
if ([string]::IsNullOrWhiteSpace($iitIndex) -or ($iitIndex -eq $iitLegacyIndex -and -not (Test-Path -LiteralPath (Join-Path $iitIndex 'manifest.json') -PathType Leaf))) {
    $iitIndex = $iitDefaultIndex
}
$iitEmbeddingModel = $env:BLACKWING_EMBEDDING_MODEL
if ([string]::IsNullOrWhiteSpace($iitEmbeddingModel)) { $iitEmbeddingModel = 'bge-small-en-v1.5' }

# Keep each credential bound to its own service; never reuse the LM key as
# an embedding fallback. -PromptKeys replaces both values in this session only.
$iitCredentials = @{
    BLACKWING_LLM_API_KEY = $env:BLACKWING_LLM_API_KEY
    BLACKWING_EMBEDDING_API_KEY = $env:BLACKWING_EMBEDDING_API_KEY
}
if ($PromptKeys) {
    $iitLmSecret = Read-Host 'LM/chat API key for port 1031 (key only)' -AsSecureString
    try { $iitCredentials.BLACKWING_LLM_API_KEY = ([System.Net.NetworkCredential]::new('', $iitLmSecret).Password).Trim() }
    finally { $iitLmSecret.Dispose() }
    $iitEmbeddingSecret = Read-Host 'BGE embedding API key for port 1035 (key only)' -AsSecureString
    try { $iitCredentials.BLACKWING_EMBEDDING_API_KEY = ([System.Net.NetworkCredential]::new('', $iitEmbeddingSecret).Password).Trim() }
    finally { $iitEmbeddingSecret.Dispose() }
}

foreach ($iitName in @('BLACKWING_LLM_API_KEY','BLACKWING_EMBEDDING_API_KEY')) {
    $iitValue = $iitCredentials[$iitName]
    if ([string]::IsNullOrWhiteSpace($iitValue)) { throw "Missing $iitName. Set it securely in this PowerShell terminal, then rerun this script." }
    if ($iitValue -match '\s' -or $iitValue -match '[^\x21-\x7E]' -or $iitValue.StartsWith('Bearer ') -or $iitValue -match '^["'']|["'']$') {
        throw "Invalid $iitName formatting. Use only the key, without Bearer, whitespace or surrounding quotes."
    }
}
foreach ($iitUrl in @($iitChatUrl,$iitEmbeddingUrl)) {
    $iitUri = $null
    if (-not [Uri]::TryCreate($iitUrl, [UriKind]::Absolute, [ref]$iitUri) -or $iitUri.Scheme -ne 'https' -or $iitUri.UserInfo -or $iitUri.Query -or $iitUri.Fragment) {
        throw 'IIT endpoints must be absolute HTTPS URLs without embedded credentials, query parameters or fragments.'
    }
}
$iitChatUrl = $iitChatUrl.TrimEnd('/')
if (-not $iitChatUrl.EndsWith('/v1')) { throw 'BLACKWING_LLM_API_BASE_URL must be the assigned base URL ending in /v1, not /chat/completions.' }
if (-not (Test-Path -LiteralPath $iitChatCert -PathType Leaf)) { throw 'Chat CA file not found. Set BLACKWING_LLM_CA_CERT to the matching existing certificate.' }
if (-not (Test-Path -LiteralPath $iitEmbeddingCert -PathType Leaf)) { throw 'Embedding CA file not found. Set BLACKWING_EMBEDDING_CA_CERT to the matching existing certificate.' }
foreach ($iitFilename in @('manifest.json','chunks.jsonl','vectors.npy')) {
    if (-not (Test-Path -LiteralPath (Join-Path $iitIndex $iitFilename) -PathType Leaf)) { throw "IIT index is incomplete or missing $iitFilename. Set BLACKWING_RAG_INDEX_DIR to the existing IIT index; no rebuild was attempted." }
}
$iitManifest = Get-Content -LiteralPath (Join-Path $iitIndex 'manifest.json') -Raw | ConvertFrom-Json -ErrorAction Stop
if (-not ([string]$iitManifest.embedding_model).StartsWith('iit:')) { throw 'Selected index is not an IIT embedding index. A local-BGE index cannot be substituted.' }

# Match the same normalized endpoint identity used by blackwing.rag.embeddings.
$iitEmbeddingEndpoint = $iitEmbeddingUrl.TrimEnd('/')
if ($iitEmbeddingEndpoint.EndsWith('/v1')) { $iitEmbeddingEndpoint += '/embeddings' }
if (-not $iitEmbeddingEndpoint.EndsWith('/v1/embeddings')) { throw 'Embedding URL must end in /v1 or /v1/embeddings.' }
$iitHasher = [System.Security.Cryptography.SHA256]::Create()
try { $iitEndpointHash = ([BitConverter]::ToString($iitHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($iitEmbeddingEndpoint)))).Replace('-','').ToLowerInvariant().Substring(0,12) }
finally { $iitHasher.Dispose() }
$iitExpectedIdentity = 'iit:' + $iitEmbeddingModel + ':' + $iitEndpointHash
if ($iitManifest.embedding_model -ne $iitExpectedIdentity) { throw 'Embedding endpoint/model does not match the saved IIT index. Use the endpoint and model that created this index; its vectors were not changed.' }
$iitConfig = Join-Path $iitRepoRoot 'config\blackwing.properties'
if (-not (Test-Path -LiteralPath $iitConfig -PathType Leaf)) { throw 'BLACKWING properties file is missing from this checkout.' }

[Environment]::SetEnvironmentVariable('BLACKWING_LLM_API_KEY', $iitCredentials.BLACKWING_LLM_API_KEY, 'Process')
[Environment]::SetEnvironmentVariable('BLACKWING_EMBEDDING_API_KEY', $iitCredentials.BLACKWING_EMBEDDING_API_KEY, 'Process')
$iitCredentials.Clear()
Remove-Variable iitValue -ErrorAction SilentlyContinue
$env:BLACKWING_CONFIG = $iitConfig
$env:BLACKWING_LLM_API_BASE_URL = $iitChatUrl
$env:BLACKWING_LLM_CA_CERT = (Resolve-Path -LiteralPath $iitChatCert).Path
$env:BLACKWING_EMBEDDING_URL = $iitEmbeddingUrl.TrimEnd('/')
$env:BLACKWING_EMBEDDING_CA_CERT = (Resolve-Path -LiteralPath $iitEmbeddingCert).Path
$env:BLACKWING_RAG_INDEX_DIR = (Resolve-Path -LiteralPath $iitIndex).Path
$env:BLACKWING_IIT_RAG_INDEX_DIR = $env:BLACKWING_RAG_INDEX_DIR
$env:BLACKWING_API_HOST = '127.0.0.1'
$env:BLACKWING_API_PORT = '8001'
$env:BLACKWING_UI_API_URL = 'http://127.0.0.1:8001'
$env:BLACKWING_UI_TIMEOUT_SECONDS = '1800'
$env:BLACKWING_LLM_PROVIDER = 'openai_compatible'
$env:BLACKWING_LLM_API_MODEL = 'Qwen/Qwen3-8B'
$env:BLACKWING_LLM_API_TIMEOUT_SECONDS = '300'
$env:BLACKWING_LLM_API_TEMPERATURE = '0.0'
$env:BLACKWING_LLM_API_MAX_OUTPUT_TOKENS = '1024'
$env:BLACKWING_AGENT_TOKEN_QUOTA_INVESTIGATION = '1024'
$env:BLACKWING_AGENT_TOKEN_QUOTA_MITIGATION = '768'
$env:BLACKWING_QWEN_TOKENIZER = 'Qwen/Qwen3-8B'
$env:BLACKWING_EMBEDDING_MODEL = $iitEmbeddingModel
$env:BLACKWING_RAG_BACKEND = 'iit'
$env:BLACKWING_RAG_RERANKER = 'crossencoder' # Local reranking, not local embeddings.
$env:BLACKWING_AGENT_RAG = '1'
$env:BLACKWING_AGENT_RAG_METHOD = 'rerank'
$env:BLACKWING_RAG_CACHE = Join-Path $iitRepoRoot 'rag_cache\embeddings_iit.sqlite'
$iitFinalGoldDir = $env:BLACKWING_FINAL_REPORT_GOLD_DIR
if ([string]::IsNullOrWhiteSpace($iitFinalGoldDir)) {
    $iitFinalGoldDir = Join-Path $env:USERPROFILE 'Downloads\BLACKWING_REAL_UNSEEN_5_20261124_v2\EVALUATOR_ONLY'
}
$env:BLACKWING_FINAL_REPORT_GOLD_DIR = $iitFinalGoldDir
if (-not (Test-Path -LiteralPath $iitFinalGoldDir -PathType Container)) {
    Write-Warning 'Final report evaluation gold directory was not found; set BLACKWING_FINAL_REPORT_GOLD_DIR to its location.'
}
$env:BLACKWING_API_JSON_RESPONSE_FORMAT = 'true'
$env:BLACKWING_IIT_MIN_INTERVAL_SECONDS = '1.1'

Write-Host 'LM/chat and BGE embedding credentials are configured separately.'
Write-Host 'IIT settings configured. Local file and embedding-identity checks passed.'
Write-Host "Saved IIT index: $($iitManifest.document_count_indexed) sources; $($iitManifest.chunk_count) chunks."
Write-Host 'No live request, embedding generation or index rebuild was performed.'
Write-Host 'From the repository folder, start: python -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8001'

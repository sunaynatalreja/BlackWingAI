param()
# Start from this script so an old terminal's 768-token values are refreshed.
# Credentials remain separate and are handled by the existing use_iit.ps1.
$ErrorActionPreference = 'Stop'
$iitStartRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $iitStartRoot
try {
    . (Join-Path $PSScriptRoot 'use_iit.ps1')
    $iitStartPython = Join-Path $iitStartRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $iitStartPython -PathType Leaf)) {
        throw 'Repository virtual-environment Python was not found.'
    }
    Write-Host "Output token limits: gateway=$env:BLACKWING_LLM_API_MAX_OUTPUT_TOKENS; investigation=$env:BLACKWING_AGENT_TOKEN_QUOTA_INVESTIGATION; mitigation=$env:BLACKWING_AGENT_TOKEN_QUOTA_MITIGATION"
    & $iitStartPython -m uvicorn blackwing.api:app --host 127.0.0.1 --port 8001
    $iitStartExit = $LASTEXITCODE
    if ($iitStartExit -ne 0) { throw "FastAPI exited with code $iitStartExit." }
}
finally {
    Pop-Location
}

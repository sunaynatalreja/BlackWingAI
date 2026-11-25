param(
    [Parameter(Mandatory = $true)][string]$CaseId,
    [Parameter(Mandatory = $true)][string]$RunId,
    [string]$ApiUrl = 'http://127.0.0.1:8001',
    [switch]$Start
)
# GET is read-only and never calls the model. -Start explicitly starts a judge job.
$caseSegment = [Uri]::EscapeDataString($CaseId)
$runSegment = [Uri]::EscapeDataString($RunId)
$evaluationUri = $ApiUrl.TrimEnd('/') + '/api/v1/cases/' + $caseSegment + '/runs/' + $runSegment + '/report-evaluation'
if ($Start) {
    Invoke-RestMethod -Method Post -Uri $evaluationUri -TimeoutSec 60 | ConvertTo-Json -Depth 25
} else {
    Invoke-RestMethod -Method Get -Uri $evaluationUri -TimeoutSec 30 | ConvertTo-Json -Depth 25
}

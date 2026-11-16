param(
    [string]$BaseUrl = 'http://127.0.0.1:8001',
    [string]$IncidentId = 'DCA23FA149',
    [string]$IncidentDate = '2023-02-04',
    [string]$Jurisdiction = 'US',
    [string]$Location = 'Austin-Bergstrom International Airport',
    [string]$Narrative = 'On 4 February 2023, a runway-occupancy conflict occurred at Austin-Bergstrom International Airport involving FedEx and Southwest aircraft. Using current-incident NTSB factual evidence, review hazards, applicable rules, and potential mitigations. Do not claim unverified causes or regulatory violations.',
    [int]$MaxPolls = 120,
    [int]$PollSeconds = 8
)
# Run only AFTER starting FastAPI in another terminal with your selected env profile.
# Uses the configured local or IIT provider; IIT calls consume real hosted quota.
$ErrorActionPreference = 'Stop'
$start = [System.Diagnostics.Stopwatch]::StartNew()
$body = @{
  category = 'runway_incursion'
  source_incident_id = $IncidentId
  incident_date = $IncidentDate
  jurisdiction = $Jurisdiction
  location = $Location
  narrative = $Narrative
} | ConvertTo-Json -Depth 5

$case = Invoke-RestMethod -Uri "$BaseUrl/api/v1/cases" -Method Post -ContentType 'application/json' -Body $body
$run = Invoke-RestMethod -Uri "$BaseUrl/api/v1/cases/$($case.case_id)/runs" -Method Post -ContentType 'application/json' -Body '{"workflow":"runway_incursion_investigation","include_retrieval":true}'
Write-Host 'CASE:' $case.case_id 'RUN:' $run.run_id
for ($i=0; $i -lt $MaxPolls; $i++) {
  $run = Invoke-RestMethod -Uri "$BaseUrl/api/v1/runs/$($run.run_id)"
  if ($run.status -in @('completed', 'failed')) { break }
  Write-Host "Current status: $($run.status) (poll $($i+1)/$MaxPolls)"
  Start-Sleep -Seconds $PollSeconds
}
$start.Stop()
if ($run.status -eq 'failed') {
  throw "Investigation failed: $($run.error_code) : $($run.error)"
}
if ($run.status -ne 'completed') {
  Write-Warning "Run remains $($run.status); rerun GET /api/v1/runs/$($run.run_id) later."
  exit 1
}
$folder = Join-Path (Get-Location) 'work'
New-Item -ItemType Directory -Force $folder | Out-Null
$jsonPath = Join-Path $folder "report_$IncidentId.json"
$mdPath = Join-Path $folder "report_$IncidentId.md"
$report = Invoke-RestMethod -Uri "$BaseUrl/api/v1/cases/$($case.case_id)/report"
$report | ConvertTo-Json -Depth 60 | Set-Content -Path $jsonPath -Encoding UTF8
Invoke-WebRequest -Uri "$BaseUrl/api/v1/cases/$($case.case_id)/report/export?format=md" -OutFile $mdPath
$agents = $report.report.agent_outputs
$invUncited = @($agents.investigation.documented_facts | Where-Object { (-not $_.evidence_ids -or $_.evidence_ids.Count -eq 0) }).Count
$riskUncited = @($agents.risk.hazards | Where-Object { (-not $_.evidence_ids -or $_.evidence_ids.Count -eq 0) }).Count
$unlinked = @($agents.mitigation.recommendations | Where-Object { (-not $_.hazard_ids -or $_.hazard_ids.Count -eq 0) -and -not $_.contributing_factor }).Count
Write-Host 'REPORTS:' $jsonPath 'and' $mdPath
Write-Host "Elapsed seconds (API submission and analysis): $([math]::Round($start.Elapsed.TotalSeconds,1))"
Write-Host 'Current factual statements without citations:' $invUncited
Write-Host 'Potential hazards without citations:' $riskUncited
Write-Host 'Unlinked proposed mitigations:' $unlinked
Write-Host 'Compliance status:' $agents.compliance.overall_applicability
Write-Host 'Verification status:' $agents.verification.completeness_status
Write-Host 'Human approval required:' $agents.verification.human_approval_required
Write-Host 'Inspect source pages against the generated claims; no score proves factual accuracy.'

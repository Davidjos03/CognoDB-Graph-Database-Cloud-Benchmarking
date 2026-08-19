# One command, whole benchmark, for Windows hosts. See run_all.sh for the
# POSIX equivalent; both do the same thing so a result cannot depend on which
# shell produced it.
#
#   .\scripts\run_all.ps1                    # every platform in .env
#   .\scripts\run_all.ps1 cognodb neo4j      # a subset
#   $env:SKIP_DOCKER = 1; .\scripts\run_all.ps1   # cloud targets only

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Targets)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path ".env")) {
    Write-Error "'.env' not found. Copy .env.example to .env and fill in your credentials."
}

if ($env:SKIP_DOCKER -ne "1") {
    Write-Host "== starting the resource-capped comparators ==" -ForegroundColor Cyan
    docker compose up -d --wait
    Write-Host "== container resource usage (should show the caps) ==" -ForegroundColor Cyan
    docker stats --no-stream --format "{{.Name}}`t{{.CPUPerc}}`t{{.MemUsage}}"
}

Write-Host "== configuration ==" -ForegroundColor Cyan
gdbbench validate

Write-Host "== benchmark and report ==" -ForegroundColor Cyan
if ($Targets) { gdbbench run-all --targets @Targets } else { gdbbench run-all }

Write-Host ""
Write-Host "Done. Tables: results/tables.md - charts: results/charts/ - evidence: results/raw/"

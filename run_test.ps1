[CmdletBinding()]
param (
  [ValidateSet("offline", "live", "all")]
  [string]$Mode = "offline"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-AppTests {
  param ([string[]]$PytestArgs)
  if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    & uv run --project $PSScriptRoot --group dev pytest @PytestArgs
  } else {
    & python -m pytest @PytestArgs
  }
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# Which tests run is configured once in pyproject.toml: testpaths keeps collection inside
# tests/, and addopts deselects the live ones. This wrapper therefore only selects a mode.
switch ($Mode) {
  "offline" {
    Invoke-AppTests @("-q")
  }
  "live" {
    Invoke-AppTests @("-sv", "-m", "live")
  }
  "all" {
    Invoke-AppTests @("--override-ini", "addopts=-q")
  }
}

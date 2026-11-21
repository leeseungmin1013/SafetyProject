$ErrorActionPreference = 'Stop'
$repoDir = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repoDir 'src'
Set-Location -LiteralPath $repoDir
python -m unittest discover -s tests -v


param(
    [switch]$Run
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'examples\cpp_smoke\main.cpp'
$output = Join-Path $root 'target\release\cpp_smoke.exe'
$dll = Join-Path $root 'target\release\datamosh.dll'

if (-not (Test-Path -LiteralPath $dll)) {
    throw "Missing $dll. Run cargo build --release first."
}

$gpp = (Get-Command g++.exe -ErrorAction Stop).Source
& $gpp -std=c++17 -O2 -I (Join-Path $root 'include') $source -o $output
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Run) {
    & $output $dll
    exit $LASTEXITCODE
}

$output

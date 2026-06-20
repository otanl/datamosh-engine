param(
    # Skip the CUDA TOPs (use on machines without an NVIDIA GPU / CUDA toolkit).
    [switch]$SkipCuda
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$releaseDir = Join-Path $root 'target\release'
$pluginsDir = Join-Path $root 'touchdesigner\demo\Plugins'

Write-Host '== cargo build --release (datamosh.dll, datamosh-cli.exe) =='
Push-Location $root
try {
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw 'cargo build --release failed' }
}
finally { Pop-Location }

$cpuScripts = @('build-td-top.cmd', 'build-td-scanline-top.cmd', 'build-td-dct-top.cmd')
$cudaScripts = @('build-td-cuda-top.cmd', 'build-td-dct-cuda-top.cmd')
$buildScripts = if ($SkipCuda) { $cpuScripts } else { $cpuScripts + $cudaScripts }

foreach ($script in $buildScripts) {
    Write-Host "== $script =="
    & (Join-Path $PSScriptRoot $script)
    if ($LASTEXITCODE -ne 0) { throw "$script failed" }
}

# Stage the runtime DLLs TouchDesigner needs into the demo's Plugins folder.
New-Item -ItemType Directory -Path $pluginsDir -Force | Out-Null
$dlls = @('datamosh.dll', 'DatamoshTOP.dll', 'ScanlineSignalTOP.dll', 'DatamoshDctTOP.dll')
if (-not $SkipCuda) { $dlls += @('DatamoshCudaTOP.dll', 'DatamoshDctCudaTOP.dll') }

foreach ($dll in $dlls) {
    $src = Join-Path $releaseDir $dll
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination $pluginsDir -Force
        Write-Host "  staged $dll"
    }
    else {
        Write-Warning "  missing $src (not staged)"
    }
}

Write-Host "Plugins staged into $pluginsDir"

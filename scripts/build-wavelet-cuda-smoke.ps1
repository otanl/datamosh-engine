param(
    [string]$VisualStudioPath = ''
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = Join-Path $root 'touchdesigner\DatamoshWaveletCudaTOP'
$harness = Join-Path $root 'examples\wavelet_cuda_smoke\main.cu'
$kernels = Join-Path $sourceDir 'DatamoshWaveletCudaKernels.cu'
$outputDir = Join-Path $root 'target\release'
$output = Join-Path $outputDir 'wavelet_cuda_smoke.exe'

function Find-VsDevCmd {
    if ($VisualStudioPath) {
        $candidate = Join-Path $VisualStudioPath 'Common7\Tools\VsDevCmd.bat'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
        throw "Visual Studio path does not contain Common7\Tools\VsDevCmd.bat: $VisualStudioPath"
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $candidate = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
    }
    throw 'Could not find Visual Studio C++ tools.'
}

if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    throw 'nvcc.exe was not found on PATH. Install the CUDA Toolkit.'
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$vsDevCmd = Find-VsDevCmd
$nvccArgs = "-O3 -std=c++17 -Xcompiler=/utf-8 -I`"$sourceDir`" `"$harness`" `"$kernels`" -o `"$output`""
$command = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && nvcc.exe $nvccArgs"
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$output

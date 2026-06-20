param(
    [switch]$Run,
    [string]$VisualStudioPath = ''
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = Join-Path $root 'touchdesigner\DatamoshCudaTOP'
$main = Join-Path $root 'examples\cuda_smoke\main.cpp'
$outputDir = Join-Path $root 'target\release'
$mainObject = Join-Path $outputDir 'datamosh_cuda_smoke.obj'
$kernelObject = Join-Path $outputDir 'DatamoshCudaKernels.obj'
$output = Join-Path $root 'target\release\datamosh_cuda_smoke.exe'

function Find-VsDevCmd {
    if ($VisualStudioPath) {
        $candidate = Join-Path $VisualStudioPath 'Common7\Tools\VsDevCmd.bat'
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
        throw "Visual Studio path does not contain Common7\Tools\VsDevCmd.bat: $VisualStudioPath"
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (Test-Path -LiteralPath $vswhere) {
        $installPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $installPath) {
            $candidate = Join-Path $installPath 'Common7\Tools\VsDevCmd.bat'
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }
    throw 'Could not find Visual Studio C++ tools.'
}

if (-not (Test-Path -LiteralPath $kernelObject)) {
    throw "Missing $kernelObject. Run .\scripts\build-td-cuda-top.cmd first."
}

$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6' }
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLib = Join-Path $cudaPath 'lib\x64'
$vsDevCmd = Find-VsDevCmd
$command = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && cl.exe /nologo /std:c++17 /O2 /EHsc /MD /c /I`"$cudaInclude`" /I`"$sourceDir`" /Fo`"$mainObject`" `"$main`" && link.exe /nologo /OUT:`"$output`" `"$mainObject`" `"$kernelObject`" /LIBPATH:`"$cudaLib`" cudart.lib"
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Run) {
    & $output
    exit $LASTEXITCODE
}

$output

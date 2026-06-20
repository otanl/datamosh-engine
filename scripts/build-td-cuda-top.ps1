param(
    [string]$TouchDesignerRoot = 'C:\Program Files\Derivative\TouchDesigner',
    [string]$VisualStudioPath = ''
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$sourceDir = Join-Path $root 'touchdesigner\DatamoshCudaTOP'
$cppSource = Join-Path $sourceDir 'DatamoshCudaTOP.cpp'
$cudaSource = Join-Path $sourceDir 'DatamoshCudaKernels.cu'
$outputDir = Join-Path $root 'target\release'
$output = Join-Path $outputDir 'DatamoshCudaTOP.dll'
$cppObject = Join-Path $outputDir 'DatamoshCudaTOP.obj'
$cudaObject = Join-Path $outputDir 'DatamoshCudaKernels.obj'
$pdb = Join-Path $outputDir 'DatamoshCudaTOP.pdb'
$cppRsp = Join-Path $outputDir 'DatamoshCudaTOP.cl.rsp'
$cudaRsp = Join-Path $outputDir 'DatamoshCudaTOP.nvcc.rsp'
$linkRsp = Join-Path $outputDir 'DatamoshCudaTOP.link.rsp'
$tdSdk = Join-Path $TouchDesignerRoot 'Samples\CPlusPlus\CudaTOP'

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

if (-not (Get-Command nvcc.exe -ErrorAction SilentlyContinue)) {
    throw 'nvcc.exe was not found on PATH. Install the CUDA Toolkit.'
}
if (-not (Test-Path -LiteralPath (Join-Path $tdSdk 'TOP_CPlusPlusBase.h'))) {
    throw "Missing TouchDesigner CUDA TOP headers under $tdSdk."
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
$vsDevCmd = Find-VsDevCmd
$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { Split-Path (Split-Path (Get-Command nvcc.exe).Source) }
$cudaInclude = Join-Path $cudaPath 'include'
$cudaLib = Join-Path $cudaPath 'lib\x64'

$cppArgs = @(
    '/nologo',
    '/std:c++17',
    '/O2',
    '/EHsc',
    '/MD',
    '/c',
    '/DNDEBUG',
    '/DWIN32',
    '/D_WINDOWS',
    "/I$tdSdk",
    "/I$cudaInclude",
    "/I$sourceDir",
    "/Fo$cppObject",
    $cppSource
)
$cudaArgs = @(
    '-c',
    '-O3',
    '-std=c++17',
    '-Xcompiler=/MD,/EHsc,/O2',
    "-I$tdSdk",
    "-I$sourceDir",
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_86,code=compute_86',
    "-o=$cudaObject",
    $cudaSource
)
$linkArgs = @(
    '/nologo',
    '/DLL',
    "/OUT:$output",
    "/PDB:$pdb",
    $cppObject,
    $cudaObject,
    "/LIBPATH:$cudaLib",
    'cudart.lib'
)

$cppArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' } |
    Set-Content -LiteralPath $cppRsp -Encoding ASCII
$cudaArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' } |
    Set-Content -LiteralPath $cudaRsp -Encoding ASCII
$linkArgs | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' } |
    Set-Content -LiteralPath $linkRsp -Encoding ASCII

$command = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && cl.exe @$cppRsp && nvcc.exe --options-file `"$cudaRsp`" && link.exe @$linkRsp"
& cmd.exe /d /s /c $command
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$output

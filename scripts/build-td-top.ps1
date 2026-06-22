param(
    [string]$TouchDesignerRoot = 'C:\Program Files\Derivative\TouchDesigner',
    [string]$VisualStudioPath = '',
    [switch]$ReleaseRust,
    [switch]$Scanline,
    [switch]$Dct,
    [switch]$Wavelet
)

$ErrorActionPreference = 'Stop'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$source = Join-Path $root 'touchdesigner\DatamoshTOP\DatamoshTOP.cpp'
$outputDir = Join-Path $root 'target\release'
$selectedVariants = @($Scanline, $Dct, $Wavelet).Where({ $_ }).Count
if ($selectedVariants -gt 1) {
    throw 'Select at most one of -Scanline, -Dct, or -Wavelet.'
}
$targetName = if ($Scanline) {
    'DatamoshScanlineTOP'
} elseif ($Dct) {
    'DatamoshDctTOP'
} elseif ($Wavelet) {
    'DatamoshWaveletTOP'
} else {
    'DatamoshTOP'
}
$output = Join-Path $outputDir "$targetName.dll"
$obj = Join-Path $outputDir "$targetName.obj"
$pdb = Join-Path $outputDir "$targetName.pdb"
$rsp = Join-Path $outputDir "$targetName.cl.rsp"
$datamoshDll = Join-Path $outputDir 'datamosh.dll'
$tdSdk = Join-Path $TouchDesignerRoot 'Samples\CPlusPlus\CPUMemoryTOP'

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

    $fallback = Get-ChildItem -Path "$env:ProgramFiles\Microsoft Visual Studio" -Recurse -Filter VsDevCmd.bat -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if ($fallback) {
        return $fallback
    }

    throw "Could not find Visual Studio C++ tools. Install Desktop development with C++."
}

function Format-RspArg([string]$arg) {
    if ($arg -match '\s') {
        return '"' + ($arg -replace '"', '\"') + '"'
    }
    return $arg
}

if ($ReleaseRust) {
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

if (-not (Test-Path -LiteralPath $datamoshDll)) {
    throw "Missing $datamoshDll. Run cargo build --release first, or pass -ReleaseRust."
}

if (-not (Test-Path -LiteralPath (Join-Path $tdSdk 'TOP_CPlusPlusBase.h'))) {
    throw "Missing TouchDesigner C++ TOP headers under $tdSdk."
}

New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$vsDevCmd = Find-VsDevCmd
$includeDir = Join-Path $root 'include'
$clArgs = @(
    '/nologo',
    '/std:c++17',
    '/O2',
    '/EHsc',
    '/MD',
    '/LD',
    '/DNDEBUG',
    '/DWIN32',
    '/D_WINDOWS',
    '/D_USRDLL',
    "/I$includeDir",
    "/I$tdSdk",
    "/Fo$obj",
    "/Fd$pdb",
    "/Fe$output",
    $source
)
if ($Scanline) {
    $clArgs += '/DSCANLINE_SIGNAL_TOP'
}
if ($Dct) {
    $clArgs += '/DDCT_TRANSFORM_TOP'
}
if ($Wavelet) {
    $clArgs += '/DWAVELET_PYRAMID_TOP'
}

$clArgs | ForEach-Object { Format-RspArg $_ } | Set-Content -LiteralPath $rsp -Encoding ASCII

$cmd = "`"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && cl.exe @$rsp"
& cmd.exe /d /s /c $cmd
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$output

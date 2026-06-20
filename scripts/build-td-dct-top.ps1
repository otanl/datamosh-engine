param(
    [string]$TouchDesignerRoot = 'C:\Program Files\Derivative\TouchDesigner',
    [string]$VisualStudioPath = '',
    [switch]$ReleaseRust
)

$invokeArgs = @{
    TouchDesignerRoot = $TouchDesignerRoot
    Dct = $true
}
if ($VisualStudioPath) {
    $invokeArgs.VisualStudioPath = $VisualStudioPath
}
if ($ReleaseRust) {
    $invokeArgs.ReleaseRust = $true
}

& (Join-Path $PSScriptRoot 'build-td-top.ps1') @invokeArgs
exit $LASTEXITCODE

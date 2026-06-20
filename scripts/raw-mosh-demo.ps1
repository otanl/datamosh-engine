param(
    [switch]$ListPresets,
    [switch]$PrintDefaultCommand,
    [switch]$PrintPersistentCommands,
    [switch]$SmokeGui
)

$ErrorActionPreference = 'Stop'

$Script:Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Script:DatamoshExe = Join-Path $Script:Root 'target\release\datamosh-cli.exe'
$Script:CurrentProcess = $null
$Script:CurrentCmdFile = $null
$Script:PreviewProcess = $null
$Script:PreviewCmdFile = $null
$Script:CurrentPreviewCommand = $null
$Script:UpdatingRealtimeControls = $false

$Script:Presets = @(
    [pscustomobject]@{
        Name = 'clean'
        Title = 'Clean decode'
        Notes = 'Codec reconstruction without intentional state corruption; useful for A/B checks.'
    },
    [pscustomobject]@{
        Name = 'drift'
        Title = 'Temporal slice drift'
        Notes = 'Horizontal bands read different decoded reference ages; good first demo.'
    },
    [pscustomobject]@{
        Name = 'bank'
        Title = 'Residual bank swap'
        Notes = 'Residual samples are decoded from wrong cells while motion stays readable.'
    },
    [pscustomobject]@{
        Name = 'plane'
        Title = 'Channel plane desync'
        Notes = 'RGB planes read different channels and reference ages.'
    },
    [pscustomobject]@{
        Name = 'vector'
        Title = 'Motion-vector bank desync'
        Notes = 'Motion vectors come from wrong block banks. More datamosh-like, a bit heavier.'
    },
    [pscustomobject]@{
        Name = 'residue'
        Title = 'Residual stream desync'
        Notes = 'Strong color/detail breakage from residual address and channel corruption.'
    },
    [pscustomobject]@{
        Name = 'entropy'
        Title = 'Entropy byte-slip'
        Notes = 'MSH0 residual payload bytes slip before decode; harsher stream-state corruption.'
    },
    [pscustomobject]@{
        Name = 'coeff'
        Title = 'Transform coefficient drift'
        Notes = 'Residuals are transformed to coefficient tiles, damaged, then decoded back.'
    },
    [pscustomobject]@{
        Name = 'codebook'
        Title = 'Residual codebook leak'
        Notes = 'Residual tiles are decoded from older dictionary slots, producing texture intrusion.'
    },
    [pscustomobject]@{
        Name = 'scan'
        Title = 'Scanline history desync'
        Notes = 'Thin horizontal reference-history misreads.'
    },
    [pscustomobject]@{
        Name = 'pixel'
        Title = 'Pixel dirty reference tearing'
        Notes = 'Fine dirty-reference misreads, useful for dense texture.'
    },
    [pscustomobject]@{
        Name = 'grain'
        Title = 'Medium grain tearing'
        Notes = 'Between melt and pixel; less blocky than classic motion smear.'
    },
    [pscustomobject]@{
        Name = 'melt'
        Title = 'Motion melt'
        Notes = 'Classic active-area dirty reference smear.'
    },
    [pscustomobject]@{
        Name = 'classic'
        Title = 'Classic motion smear'
        Notes = 'Readable baseline for comparing custom-codec variants.'
    },
    [pscustomobject]@{
        Name = 'unstable'
        Title = 'Codec state collapse'
        Notes = 'More chaotic dirty-reference and predictor desync.'
    }
)

$Script:DemoPresetNames = @(
    'drift',
    'bank',
    'plane',
    'vector',
    'residue',
    'entropy',
    'coeff',
    'codebook',
    'melt',
    'unstable'
)
$Script:PresetGroups = @(
    'Curated',
    'Motion',
    'Reference',
    'Residual',
    'Bitstream',
    'Hybrid',
    'All'
)
$Script:UpdatingPresetList = $false

$Script:RealtimeParameters = @(
    [pscustomobject]@{ Id = 'mv_scale'; Group = 'Motion'; Label = 'Motion scale'; Min = 0.0; Max = 2.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'mv_jitter'; Group = 'Motion'; Label = 'Vector jitter'; Min = 0.0; Max = 16.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'mv_field_interpolation'; Group = 'Motion'; Label = 'Vector interpolation'; Min = 0.0; Max = 1.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'sample_address_desync'; Group = 'Motion'; Label = 'Sample address desync'; Min = 0.0; Max = 4.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'mv_bank_stride'; Group = 'Motion'; Label = 'Vector bank stride'; Min = -64.0; Max = 64.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'reference_lag'; Group = 'Reference'; Label = 'Reference lag'; Min = 1.0; Max = 32.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'reference_bleed'; Group = 'Reference'; Label = 'Reference bleed'; Min = 0.0; Max = 1.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'reference_latch_frames'; Group = 'Reference'; Label = 'Reference latch'; Min = 1.0; Max = 64.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'temporal_slice_drift'; Group = 'Reference'; Label = 'Temporal drift'; Min = -16.0; Max = 16.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'reference_channel_lag_span'; Group = 'Reference'; Label = 'Channel lag span'; Min = 0.0; Max = 32.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'residual_keep'; Group = 'Residual'; Label = 'Residual gain'; Min = -2.0; Max = 2.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'residual_address_jitter'; Group = 'Residual'; Label = 'Residual address jitter'; Min = 0.0; Max = 32.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'residual_channel_shift'; Group = 'Residual'; Label = 'Residual channel shift'; Min = -4.0; Max = 4.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'residual_bank_stride'; Group = 'Residual'; Label = 'Residual bank stride'; Min = -64.0; Max = 64.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'entropy_slip_every'; Group = 'Bitstream'; Label = 'Entropy slip period'; Min = 0.0; Max = 64.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'entropy_slip_windows'; Group = 'Bitstream'; Label = 'Entropy windows'; Min = 0.0; Max = 64.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'entropy_resync_bytes'; Group = 'Bitstream'; Label = 'Entropy resync bytes'; Min = 0.0; Max = 65536.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'coeff_shift'; Group = 'Bitstream'; Label = 'Coefficient shift'; Min = -32.0; Max = 32.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'coeff_quant'; Group = 'Bitstream'; Label = 'Coefficient quant'; Min = 1.0; Max = 64.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'codebook_replace_every'; Group = 'Bitstream'; Label = 'Codebook period'; Min = 0.0; Max = 64.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'codebook_stride'; Group = 'Bitstream'; Label = 'Codebook stride'; Min = -128.0; Max = 128.0; Default = 1.0 },
    [pscustomobject]@{ Id = 'codebook_shuffle_every'; Group = 'Bitstream'; Label = 'Codebook shuffle'; Min = 0.0; Max = 64.0; Default = 0.0 },
    [pscustomobject]@{ Id = 'bitstream_enabled'; Group = 'Hybrid'; Label = 'Bitstream enabled'; Min = 0.0; Max = 1.0; Default = 0.0 }
)

function Get-PresetGroup {
    param([Parameter(Mandatory = $true)]$Preset)

    switch ($Preset.Name) {
        { $_ -in @('melt', 'classic', 'vector') } { return 'Motion' }
        { $_ -in @('drift', 'plane', 'scan') } { return 'Reference' }
        { $_ -in @('residue', 'bank', 'pixel', 'grain') } { return 'Residual' }
        { $_ -in @('entropy', 'coeff', 'codebook') } { return 'Bitstream' }
        { $_ -in @('clean', 'unstable') } { return 'Hybrid' }
        default { return 'Other' }
    }
}

function Get-RealtimeParametersForPreset {
    param([string]$PresetName)

    $preset = $Script:Presets | Where-Object { $_.Name -eq $PresetName } | Select-Object -First 1
    $group = if ($preset) { Get-PresetGroup $preset } else { 'Hybrid' }

    if ($group -eq 'Hybrid') {
        return @($Script:RealtimeParameters)
    }

    @($Script:RealtimeParameters | Where-Object { $_.Group -eq $group -or $_.Group -eq 'Hybrid' })
}

function Get-ParameterValueFromSlider {
    param(
        [Parameter(Mandatory = $true)]$Parameter,
        [int]$SliderValue
    )

    $span = [double]$Parameter.Max - [double]$Parameter.Min
    if ($span -le 0) {
        return [double]$Parameter.Min
    }
    [double]$Parameter.Min + ($span * ([double]$SliderValue / 1000.0))
}

function Get-ParameterSliderValue {
    param(
        [Parameter(Mandatory = $true)]$Parameter,
        [double]$Value
    )

    $span = [double]$Parameter.Max - [double]$Parameter.Min
    if ($span -le 0) {
        return 0
    }
    $raw = [math]::Round((($Value - [double]$Parameter.Min) / $span) * 1000.0)
    [Math]::Max(0, [Math]::Min(1000, [int]$raw))
}

function Send-RawMoshControlMessage {
    param([Parameter(Mandatory = $true)][string]$Message)

    if (-not $realtimeCheck.Checked) {
        return
    }

    $port = Get-IntOrDefault $controlPortBox.Text 24000
    if ($port -le 0) {
        return
    }

    $client = [Net.Sockets.UdpClient]::new()
    try {
        $bytes = [Text.Encoding]::ASCII.GetBytes($Message)
        [void]$client.Send($bytes, $bytes.Length, '127.0.0.1', $port)
    } finally {
        $client.Close()
    }
}

function Get-VisiblePresets {
    param(
        [bool]$ShowAll = $false,
        [string]$Group = 'Curated'
    )

    if ($ShowAll -or $Group -eq 'All') {
        $visible = @($Script:Presets)
    } else {
        $visible = foreach ($name in $Script:DemoPresetNames) {
            $preset = $Script:Presets | Where-Object { $_.Name -eq $name } | Select-Object -First 1
            if ($preset) {
                $preset
            }
        }
    }

    if ($Group -ne 'Curated' -and $Group -ne 'All') {
        $visible = @($visible | Where-Object { (Get-PresetGroup $_) -eq $Group })
    }

    @($visible)
}

function Quote-CmdPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    '"' + $Path.Replace('"', '""') + '"'
}

function Get-IntOrDefault {
    param(
        [string]$Text,
        [int]$Default
    )

    $value = 0
    if ([int]::TryParse($Text, [ref]$value) -and $value -gt 0) {
        return $value
    }
    return $Default
}

function Build-RawMoshCommand {
    param(
        [string]$Preset = 'drift',
        [string]$Device = 'OBS Virtual Camera',
        [bool]$UseTestSource = $false,
        [string]$CaptureSize = '1280x720',
        [int]$ProcessWidth = 480,
        [int]$ProcessHeight = 270,
        [int]$FrameRate = 30,
        [int]$History = 16,
        [int]$Upscale = 2,
        [string]$ScaleMode = 'nearest',
        [string]$OutputWidth = '',
        [string]$OutputHeight = '',
        [string]$ExtraRawMoshArgs = '',
        [int]$ControlPort = 24000,
        [string]$LogLevel = 'fatal'
    )

    $safeDevice = $Device.Replace('"', '')
    $scaleFilter = "scale=$($ProcessWidth):$($ProcessHeight):flags=fast_bilinear,format=rgb24"

    if ($UseTestSource) {
        $ffmpeg = "ffmpeg -hide_banner -loglevel $LogLevel -re -f lavfi -i testsrc2=size=$($ProcessWidth)x$($ProcessHeight):rate=$FrameRate -vf format=rgb24 -f rawvideo -"
    } else {
        $ffmpeg = "ffmpeg -hide_banner -loglevel $LogLevel -f dshow -rtbufsize 256M -video_size $CaptureSize -framerate $FrameRate -pixel_format yuv420p -i video=""$safeDevice"" -an -vf $scaleFilter -f rawvideo -"
    }

    $mosh = "$(Quote-CmdPath $Script:DatamoshExe) raw-mosh --width $ProcessWidth --height $ProcessHeight --preset $Preset --history $History"

    $outW = 0
    $outH = 0
    if ([int]::TryParse($OutputWidth, [ref]$outW) -and [int]::TryParse($OutputHeight, [ref]$outH) -and $outW -gt 0 -and $outH -gt 0) {
        $mosh += " --output-width $outW --output-height $outH"
        $playW = $outW
        $playH = $outH
    } else {
        $mosh += " --upscale $Upscale"
        $playW = $ProcessWidth * $Upscale
        $playH = $ProcessHeight * $Upscale
    }

    $mosh += " --scale-mode $ScaleMode --quiet"
    if ($ControlPort -gt 0) {
        $mosh += " --control-port $ControlPort"
    }
    if (-not [string]::IsNullOrWhiteSpace($ExtraRawMoshArgs)) {
        $mosh += " $ExtraRawMoshArgs"
    }

    $ffplay = "ffplay -hide_banner -loglevel $LogLevel -fflags nobuffer -flags low_delay -framedrop -f rawvideo -pixel_format rgb24 -video_size $($playW)x$($playH) -framerate $FrameRate -"
    "$ffmpeg | $mosh | $ffplay"
}

function Build-RawMoshPersistentCommands {
    param(
        [string]$Preset = 'drift',
        [string]$Device = 'OBS Virtual Camera',
        [bool]$UseTestSource = $false,
        [string]$CaptureSize = '1280x720',
        [int]$ProcessWidth = 480,
        [int]$ProcessHeight = 270,
        [int]$FrameRate = 30,
        [int]$History = 16,
        [int]$Upscale = 2,
        [string]$ScaleMode = 'nearest',
        [string]$OutputWidth = '',
        [string]$OutputHeight = '',
        [string]$ExtraRawMoshArgs = '',
        [int]$UdpPort = 23000,
        [int]$ControlPort = 24000,
        [string]$LogLevel = 'fatal'
    )

    $safeDevice = $Device.Replace('"', '')
    $scaleFilter = "scale=$($ProcessWidth):$($ProcessHeight):flags=fast_bilinear,format=rgb24"

    if ($UseTestSource) {
        $ffmpeg = "ffmpeg -hide_banner -loglevel $LogLevel -re -f lavfi -i testsrc2=size=$($ProcessWidth)x$($ProcessHeight):rate=$FrameRate -vf format=rgb24 -f rawvideo -"
    } else {
        $ffmpeg = "ffmpeg -hide_banner -loglevel $LogLevel -f dshow -rtbufsize 256M -video_size $CaptureSize -framerate $FrameRate -pixel_format yuv420p -i video=""$safeDevice"" -an -vf $scaleFilter -f rawvideo -"
    }

    $mosh = "$(Quote-CmdPath $Script:DatamoshExe) raw-mosh --width $ProcessWidth --height $ProcessHeight --preset $Preset --history $History"

    $outW = 0
    $outH = 0
    if ([int]::TryParse($OutputWidth, [ref]$outW) -and [int]::TryParse($OutputHeight, [ref]$outH) -and $outW -gt 0 -and $outH -gt 0) {
        $mosh += " --output-width $outW --output-height $outH"
        $playW = $outW
        $playH = $outH
    } else {
        $mosh += " --upscale $Upscale"
        $playW = $ProcessWidth * $Upscale
        $playH = $ProcessHeight * $Upscale
    }

    $mosh += " --scale-mode $ScaleMode --quiet"
    if ($ControlPort -gt 0) {
        $mosh += " --control-port $ControlPort"
    }
    if (-not [string]::IsNullOrWhiteSpace($ExtraRawMoshArgs)) {
        $mosh += " $ExtraRawMoshArgs"
    }

    $udpUrl = "udp://127.0.0.1:$($UdpPort)?pkt_size=1316"
    $sender = "$ffmpeg | $mosh | ffmpeg -hide_banner -loglevel $LogLevel -f rawvideo -pixel_format rgb24 -video_size $($playW)x$($playH) -framerate $FrameRate -i - -an -c:v mpeg2video -q:v 3 -g 12 -bf 0 -f mpegts ""$udpUrl"""
    $previewUrl = "udp://127.0.0.1:$($UdpPort)?fifo_size=1000000&overrun_nonfatal=1"
    $preview = "ffplay -hide_banner -loglevel $LogLevel -fflags nobuffer -flags low_delay -framedrop -i ""$previewUrl"""

    [pscustomobject]@{
        Sender = $sender
        Preview = $preview
        OutputSize = "$($playW)x$($playH)"
    }
}

function Stop-ProcessTree {
    param([int]$ProcessId)

    Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcessId" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-ProcessTree -ProcessId $_.ProcessId }

    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Stop-SenderPipeline {
    if ($Script:CurrentProcess -and -not $Script:CurrentProcess.HasExited) {
        Stop-ProcessTree -ProcessId $Script:CurrentProcess.Id
    }
    $Script:CurrentProcess = $null

    if ($Script:CurrentCmdFile -and (Test-Path -LiteralPath $Script:CurrentCmdFile)) {
        Remove-Item -LiteralPath $Script:CurrentCmdFile -Force -ErrorAction SilentlyContinue
    }
    $Script:CurrentCmdFile = $null
}

function Stop-PreviewPipeline {
    if ($Script:PreviewProcess -and -not $Script:PreviewProcess.HasExited) {
        Stop-ProcessTree -ProcessId $Script:PreviewProcess.Id
    }
    $Script:PreviewProcess = $null
    $Script:CurrentPreviewCommand = $null

    if ($Script:PreviewCmdFile -and (Test-Path -LiteralPath $Script:PreviewCmdFile)) {
        Remove-Item -LiteralPath $Script:PreviewCmdFile -Force -ErrorAction SilentlyContinue
    }
    $Script:PreviewCmdFile = $null
}

function Stop-DemoPipeline {
    param([switch]$KeepPreview)

    Stop-SenderPipeline
    if (-not $KeepPreview) {
        Stop-PreviewPipeline
    }
}

function Start-DemoPipeline {
    param(
        [string]$Command,
        [switch]$KeepPreview
    )

    if (-not (Test-Path -LiteralPath $Script:DatamoshExe)) {
        throw "Release binary not found: $Script:DatamoshExe. Run cargo build --release first."
    }

    Stop-DemoPipeline -KeepPreview:$KeepPreview

    $Script:CurrentCmdFile = Join-Path ([IO.Path]::GetTempPath()) ("datamosh-demo-{0}.cmd" -f ([Guid]::NewGuid().ToString('N')))
    $cmdText = "@echo off`r`ncd /d $(Quote-CmdPath $Script:Root)`r`n$Command`r`n"
    Set-Content -LiteralPath $Script:CurrentCmdFile -Value $cmdText -Encoding ASCII

    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $env:ComSpec
    $psi.Arguments = "/d /c $(Quote-CmdPath $Script:CurrentCmdFile)"
    $psi.WorkingDirectory = $Script:Root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $Script:CurrentProcess = [Diagnostics.Process]::Start($psi)
}

function Start-PreviewPipeline {
    param([string]$Command)

    if ($Script:PreviewProcess -and -not $Script:PreviewProcess.HasExited -and $Script:CurrentPreviewCommand -eq $Command) {
        return
    }

    Stop-PreviewPipeline

    $Script:PreviewCmdFile = Join-Path ([IO.Path]::GetTempPath()) ("datamosh-preview-{0}.cmd" -f ([Guid]::NewGuid().ToString('N')))
    $cmdText = "@echo off`r`ncd /d $(Quote-CmdPath $Script:Root)`r`n$Command`r`n"
    Set-Content -LiteralPath $Script:PreviewCmdFile -Value $cmdText -Encoding ASCII

    $psi = [Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $env:ComSpec
    $psi.Arguments = "/d /c $(Quote-CmdPath $Script:PreviewCmdFile)"
    $psi.WorkingDirectory = $Script:Root
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $Script:PreviewProcess = [Diagnostics.Process]::Start($psi)
    $Script:CurrentPreviewCommand = $Command
}

function Test-DemoPipelineRunning {
    $Script:CurrentProcess -and -not $Script:CurrentProcess.HasExited
}

if ($ListPresets) {
    $Script:Presets |
        Select-Object Name, @{Name = 'Group'; Expression = { Get-PresetGroup $_ } }, Title, Notes |
        Format-Table Name, Group, Title, Notes -AutoSize
    return
}

if ($PrintDefaultCommand) {
    Build-RawMoshCommand
    return
}

if ($PrintPersistentCommands) {
    $commands = Build-RawMoshPersistentCommands
    "# persistent ffplay preview"
    $commands.Preview
    ""
    "# sender restarted on preset changes"
    $commands.Sender
    return
}

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$form = [Windows.Forms.Form]::new()
$form.Text = 'datamosh raw-mosh demo'
$form.StartPosition = 'CenterScreen'
$form.Size = [Drawing.Size]::new(980, 680)
$form.MinimumSize = [Drawing.Size]::new(920, 620)

$font = [Drawing.Font]::new('Segoe UI', 9)
$form.Font = $font

$presetList = [Windows.Forms.ListBox]::new()
$presetList.Location = [Drawing.Point]::new(14, 14)
$presetList.Size = [Drawing.Size]::new(220, 360)
$presetList.DisplayMember = 'Title'
$form.Controls.Add($presetList)

$description = [Windows.Forms.TextBox]::new()
$description.Location = [Drawing.Point]::new(14, 382)
$description.Size = [Drawing.Size]::new(220, 74)
$description.Multiline = $true
$description.ReadOnly = $true
$description.ScrollBars = 'Vertical'
$form.Controls.Add($description)

$showAllPresetsCheck = [Windows.Forms.CheckBox]::new()
$showAllPresetsCheck.Text = 'Show all presets'
$showAllPresetsCheck.Checked = $false
$showAllPresetsCheck.Location = [Drawing.Point]::new(14, 462)
$showAllPresetsCheck.Size = [Drawing.Size]::new(220, 24)
$form.Controls.Add($showAllPresetsCheck)

$groupLabel = [Windows.Forms.Label]::new()
$groupLabel.Text = 'Group'
$groupLabel.Location = [Drawing.Point]::new(14, 492)
$groupLabel.Size = [Drawing.Size]::new(52, 22)
$form.Controls.Add($groupLabel)

$groupBox = [Windows.Forms.ComboBox]::new()
$groupBox.DropDownStyle = 'DropDownList'
$groupBox.Items.AddRange($Script:PresetGroups)
$groupBox.SelectedIndex = 0
$groupBox.Location = [Drawing.Point]::new(70, 489)
$groupBox.Size = [Drawing.Size]::new(164, 24)
$form.Controls.Add($groupBox)

$inputGroup = [Windows.Forms.GroupBox]::new()
$inputGroup.Text = 'Input'
$inputGroup.Location = [Drawing.Point]::new(252, 14)
$inputGroup.Size = [Drawing.Size]::new(330, 148)
$form.Controls.Add($inputGroup)

$deviceLabel = [Windows.Forms.Label]::new()
$deviceLabel.Text = 'dshow device'
$deviceLabel.Location = [Drawing.Point]::new(12, 28)
$deviceLabel.Size = [Drawing.Size]::new(90, 22)
$inputGroup.Controls.Add($deviceLabel)

$deviceBox = [Windows.Forms.TextBox]::new()
$deviceBox.Text = 'OBS Virtual Camera'
$deviceBox.Location = [Drawing.Point]::new(112, 25)
$deviceBox.Size = [Drawing.Size]::new(200, 24)
$inputGroup.Controls.Add($deviceBox)

$testSourceCheck = [Windows.Forms.CheckBox]::new()
$testSourceCheck.Text = 'Use FFmpeg test pattern'
$testSourceCheck.Location = [Drawing.Point]::new(112, 54)
$testSourceCheck.Size = [Drawing.Size]::new(190, 24)
$inputGroup.Controls.Add($testSourceCheck)

$captureLabel = [Windows.Forms.Label]::new()
$captureLabel.Text = 'capture size'
$captureLabel.Location = [Drawing.Point]::new(12, 85)
$captureLabel.Size = [Drawing.Size]::new(90, 22)
$inputGroup.Controls.Add($captureLabel)

$captureBox = [Windows.Forms.ComboBox]::new()
$captureBox.DropDownStyle = 'DropDown'
$captureBox.Items.AddRange(@('1280x720', '1920x1080', '640x360'))
$captureBox.Text = '1280x720'
$captureBox.Location = [Drawing.Point]::new(112, 82)
$captureBox.Size = [Drawing.Size]::new(110, 24)
$inputGroup.Controls.Add($captureBox)

$fpsLabel = [Windows.Forms.Label]::new()
$fpsLabel.Text = 'fps'
$fpsLabel.Location = [Drawing.Point]::new(230, 85)
$fpsLabel.Size = [Drawing.Size]::new(28, 22)
$inputGroup.Controls.Add($fpsLabel)

$fpsBox = [Windows.Forms.TextBox]::new()
$fpsBox.Text = '30'
$fpsBox.Location = [Drawing.Point]::new(260, 82)
$fpsBox.Size = [Drawing.Size]::new(52, 24)
$inputGroup.Controls.Add($fpsBox)

$processGroup = [Windows.Forms.GroupBox]::new()
$processGroup.Text = 'Raw-mosh processing'
$processGroup.Location = [Drawing.Point]::new(252, 174)
$processGroup.Size = [Drawing.Size]::new(330, 190)
$form.Controls.Add($processGroup)

$procLabel = [Windows.Forms.Label]::new()
$procLabel.Text = 'process size'
$procLabel.Location = [Drawing.Point]::new(12, 30)
$procLabel.Size = [Drawing.Size]::new(90, 22)
$processGroup.Controls.Add($procLabel)

$procWBox = [Windows.Forms.TextBox]::new()
$procWBox.Text = '480'
$procWBox.Location = [Drawing.Point]::new(112, 27)
$procWBox.Size = [Drawing.Size]::new(54, 24)
$processGroup.Controls.Add($procWBox)

$procHBox = [Windows.Forms.TextBox]::new()
$procHBox.Text = '270'
$procHBox.Location = [Drawing.Point]::new(174, 27)
$procHBox.Size = [Drawing.Size]::new(54, 24)
$processGroup.Controls.Add($procHBox)

$historyLabel = [Windows.Forms.Label]::new()
$historyLabel.Text = 'history'
$historyLabel.Location = [Drawing.Point]::new(238, 30)
$historyLabel.Size = [Drawing.Size]::new(46, 22)
$processGroup.Controls.Add($historyLabel)

$historyBox = [Windows.Forms.TextBox]::new()
$historyBox.Text = '16'
$historyBox.Location = [Drawing.Point]::new(286, 27)
$historyBox.Size = [Drawing.Size]::new(32, 24)
$processGroup.Controls.Add($historyBox)

$upscaleLabel = [Windows.Forms.Label]::new()
$upscaleLabel.Text = 'upscale'
$upscaleLabel.Location = [Drawing.Point]::new(12, 64)
$upscaleLabel.Size = [Drawing.Size]::new(90, 22)
$processGroup.Controls.Add($upscaleLabel)

$upscaleBox = [Windows.Forms.TextBox]::new()
$upscaleBox.Text = '2'
$upscaleBox.Location = [Drawing.Point]::new(112, 61)
$upscaleBox.Size = [Drawing.Size]::new(54, 24)
$processGroup.Controls.Add($upscaleBox)

$scaleModeBox = [Windows.Forms.ComboBox]::new()
$scaleModeBox.DropDownStyle = 'DropDownList'
$scaleModeBox.Items.AddRange(@('nearest', 'linear'))
$scaleModeBox.SelectedIndex = 0
$scaleModeBox.Location = [Drawing.Point]::new(174, 61)
$scaleModeBox.Size = [Drawing.Size]::new(94, 24)
$processGroup.Controls.Add($scaleModeBox)

$outputLabel = [Windows.Forms.Label]::new()
$outputLabel.Text = 'output override'
$outputLabel.Location = [Drawing.Point]::new(12, 98)
$outputLabel.Size = [Drawing.Size]::new(96, 22)
$processGroup.Controls.Add($outputLabel)

$outWBox = [Windows.Forms.TextBox]::new()
$outWBox.Text = ''
$outWBox.Location = [Drawing.Point]::new(112, 95)
$outWBox.Size = [Drawing.Size]::new(54, 24)
$processGroup.Controls.Add($outWBox)

$outHBox = [Windows.Forms.TextBox]::new()
$outHBox.Text = ''
$outHBox.Location = [Drawing.Point]::new(174, 95)
$outHBox.Size = [Drawing.Size]::new(54, 24)
$processGroup.Controls.Add($outHBox)

$extraLabel = [Windows.Forms.Label]::new()
$extraLabel.Text = 'extra args'
$extraLabel.Location = [Drawing.Point]::new(12, 132)
$extraLabel.Size = [Drawing.Size]::new(90, 22)
$processGroup.Controls.Add($extraLabel)

$extraBox = [Windows.Forms.TextBox]::new()
$extraBox.Text = ''
$extraBox.Location = [Drawing.Point]::new(112, 129)
$extraBox.Size = [Drawing.Size]::new(206, 24)
$processGroup.Controls.Add($extraBox)

$realtimeGroup = [Windows.Forms.GroupBox]::new()
$realtimeGroup.Text = 'Realtime control'
$realtimeGroup.Location = [Drawing.Point]::new(600, 14)
$realtimeGroup.Size = [Drawing.Size]::new(352, 350)
$form.Controls.Add($realtimeGroup)

$realtimeCheck = [Windows.Forms.CheckBox]::new()
$realtimeCheck.Text = 'Enable UDP control'
$realtimeCheck.Checked = $true
$realtimeCheck.Location = [Drawing.Point]::new(12, 24)
$realtimeCheck.Size = [Drawing.Size]::new(150, 24)
$realtimeGroup.Controls.Add($realtimeCheck)

$controlPortLabel = [Windows.Forms.Label]::new()
$controlPortLabel.Text = 'control port'
$controlPortLabel.Location = [Drawing.Point]::new(176, 26)
$controlPortLabel.Size = [Drawing.Size]::new(74, 22)
$realtimeGroup.Controls.Add($controlPortLabel)

$controlPortBox = [Windows.Forms.TextBox]::new()
$controlPortBox.Text = '24000'
$controlPortBox.Location = [Drawing.Point]::new(258, 23)
$controlPortBox.Size = [Drawing.Size]::new(70, 24)
$realtimeGroup.Controls.Add($controlPortBox)

function Add-MacroTrack {
    param(
        [string]$Name,
        [int]$Y
    )

    $label = [Windows.Forms.Label]::new()
    $label.Text = $Name
    $label.Location = [Drawing.Point]::new(12, $Y)
    $label.Size = [Drawing.Size]::new(70, 22)
    $realtimeGroup.Controls.Add($label)

    $track = [Windows.Forms.TrackBar]::new()
    $track.Minimum = 0
    $track.Maximum = 100
    $track.TickFrequency = 25
    $track.Value = 100
    $track.Location = [Drawing.Point]::new(82, $Y - 6)
    $track.Size = [Drawing.Size]::new(198, 36)
    $realtimeGroup.Controls.Add($track)

    $value = [Windows.Forms.Label]::new()
    $value.Text = '1.00'
    $value.Location = [Drawing.Point]::new(286, $Y)
    $value.Size = [Drawing.Size]::new(48, 22)
    $realtimeGroup.Controls.Add($value)

    [pscustomobject]@{ Track = $track; Value = $value }
}

$intensityControl = Add-MacroTrack -Name 'Intensity' -Y 62
$motionControl = Add-MacroTrack -Name 'Motion' -Y 98
$residualControl = Add-MacroTrack -Name 'Residual' -Y 134
$temporalControl = Add-MacroTrack -Name 'Temporal' -Y 170
$bitstreamControl = Add-MacroTrack -Name 'Bitstream' -Y 206

$parameterLabel = [Windows.Forms.Label]::new()
$parameterLabel.Text = 'Parameter'
$parameterLabel.Location = [Drawing.Point]::new(12, 250)
$parameterLabel.Size = [Drawing.Size]::new(70, 22)
$realtimeGroup.Controls.Add($parameterLabel)

$parameterBox = [Windows.Forms.ComboBox]::new()
$parameterBox.DropDownStyle = 'DropDownList'
$parameterBox.DisplayMember = 'Label'
$parameterBox.Location = [Drawing.Point]::new(82, 247)
$parameterBox.Size = [Drawing.Size]::new(246, 24)
$realtimeGroup.Controls.Add($parameterBox)

$parameterTrack = [Windows.Forms.TrackBar]::new()
$parameterTrack.Minimum = 0
$parameterTrack.Maximum = 1000
$parameterTrack.TickFrequency = 250
$parameterTrack.Location = [Drawing.Point]::new(82, 280)
$parameterTrack.Size = [Drawing.Size]::new(198, 38)
$realtimeGroup.Controls.Add($parameterTrack)

$parameterValueLabel = [Windows.Forms.Label]::new()
$parameterValueLabel.Text = ''
$parameterValueLabel.Location = [Drawing.Point]::new(286, 287)
$parameterValueLabel.Size = [Drawing.Size]::new(58, 22)
$realtimeGroup.Controls.Add($parameterValueLabel)

$sendPresetButton = [Windows.Forms.Button]::new()
$sendPresetButton.Text = 'Send preset'
$sendPresetButton.Location = [Drawing.Point]::new(12, 314)
$sendPresetButton.Size = [Drawing.Size]::new(96, 26)
$realtimeGroup.Controls.Add($sendPresetButton)

$resetGlitchButton = [Windows.Forms.Button]::new()
$resetGlitchButton.Text = 'Reset glitch'
$resetGlitchButton.Location = [Drawing.Point]::new(118, 314)
$resetGlitchButton.Size = [Drawing.Size]::new(100, 26)
$realtimeGroup.Controls.Add($resetGlitchButton)

$resetControlsButton = [Windows.Forms.Button]::new()
$resetControlsButton.Text = 'Reset knobs'
$resetControlsButton.Location = [Drawing.Point]::new(228, 314)
$resetControlsButton.Size = [Drawing.Size]::new(100, 26)
$realtimeGroup.Controls.Add($resetControlsButton)

$commandBox = [Windows.Forms.TextBox]::new()
$commandBox.Location = [Drawing.Point]::new(252, 382)
$commandBox.Size = [Drawing.Size]::new(700, 178)
$commandBox.Anchor = 'Left,Right,Bottom'
$commandBox.Multiline = $true
$commandBox.ReadOnly = $true
$commandBox.ScrollBars = 'Both'
$commandBox.WordWrap = $false
$form.Controls.Add($commandBox)

$statusLabel = [Windows.Forms.Label]::new()
$statusLabel.Text = 'Ready'
$statusLabel.Location = [Drawing.Point]::new(252, 572)
$statusLabel.Size = [Drawing.Size]::new(700, 24)
$statusLabel.Anchor = 'Left,Right,Bottom'
$form.Controls.Add($statusLabel)

$startButton = [Windows.Forms.Button]::new()
$startButton.Text = 'Start'
$startButton.Location = [Drawing.Point]::new(252, 606)
$startButton.Size = [Drawing.Size]::new(72, 30)
$startButton.Anchor = 'Left,Bottom'
$form.Controls.Add($startButton)

$stopButton = [Windows.Forms.Button]::new()
$stopButton.Text = 'Stop'
$stopButton.Location = [Drawing.Point]::new(332, 606)
$stopButton.Size = [Drawing.Size]::new(72, 30)
$stopButton.Anchor = 'Left,Bottom'
$form.Controls.Add($stopButton)

$applyButton = [Windows.Forms.Button]::new()
$applyButton.Text = 'Apply'
$applyButton.Location = [Drawing.Point]::new(412, 606)
$applyButton.Size = [Drawing.Size]::new(72, 30)
$applyButton.Anchor = 'Left,Bottom'
$form.Controls.Add($applyButton)

$copyButton = [Windows.Forms.Button]::new()
$copyButton.Text = 'Copy command'
$copyButton.Location = [Drawing.Point]::new(494, 606)
$copyButton.Size = [Drawing.Size]::new(112, 30)
$copyButton.Anchor = 'Left,Bottom'
$form.Controls.Add($copyButton)

$buildButton = [Windows.Forms.Button]::new()
$buildButton.Text = 'Build release'
$buildButton.Location = [Drawing.Point]::new(616, 606)
$buildButton.Size = [Drawing.Size]::new(112, 30)
$buildButton.Anchor = 'Left,Bottom'
$form.Controls.Add($buildButton)

$autoApplyCheck = [Windows.Forms.CheckBox]::new()
$autoApplyCheck.Text = 'Auto apply preset'
$autoApplyCheck.Checked = $true
$autoApplyCheck.Location = [Drawing.Point]::new(742, 586)
$autoApplyCheck.Size = [Drawing.Size]::new(150, 24)
$autoApplyCheck.Anchor = 'Left,Bottom'
$form.Controls.Add($autoApplyCheck)

$keepPreviewCheck = [Windows.Forms.CheckBox]::new()
$keepPreviewCheck.Text = 'Keep ffplay'
$keepPreviewCheck.Checked = $true
$keepPreviewCheck.Location = [Drawing.Point]::new(742, 610)
$keepPreviewCheck.Size = [Drawing.Size]::new(110, 24)
$keepPreviewCheck.Anchor = 'Left,Bottom'
$form.Controls.Add($keepPreviewCheck)

$udpPortLabel = [Windows.Forms.Label]::new()
$udpPortLabel.Text = 'port'
$udpPortLabel.Location = [Drawing.Point]::new(858, 612)
$udpPortLabel.Size = [Drawing.Size]::new(30, 22)
$udpPortLabel.Anchor = 'Left,Bottom'
$form.Controls.Add($udpPortLabel)

$udpPortBox = [Windows.Forms.TextBox]::new()
$udpPortBox.Text = '23000'
$udpPortBox.Location = [Drawing.Point]::new(890, 609)
$udpPortBox.Size = [Drawing.Size]::new(58, 24)
$udpPortBox.Anchor = 'Left,Bottom'
$form.Controls.Add($udpPortBox)

function Get-CurrentOneShotCommand {
    $preset = $presetList.SelectedItem.Name
    $processWidth = Get-IntOrDefault $procWBox.Text 480
    $processHeight = Get-IntOrDefault $procHBox.Text 270
    $fps = Get-IntOrDefault $fpsBox.Text 30
    $history = Get-IntOrDefault $historyBox.Text 16
    $upscale = Get-IntOrDefault $upscaleBox.Text 2
    $controlPort = if ($realtimeCheck.Checked) { Get-IntOrDefault $controlPortBox.Text 24000 } else { 0 }
    Build-RawMoshCommand `
        -Preset $preset `
        -Device $deviceBox.Text `
        -UseTestSource $testSourceCheck.Checked `
        -CaptureSize $captureBox.Text `
        -ProcessWidth $processWidth `
        -ProcessHeight $processHeight `
        -FrameRate $fps `
        -History $history `
        -Upscale $upscale `
        -ScaleMode $scaleModeBox.SelectedItem `
        -OutputWidth $outWBox.Text `
        -OutputHeight $outHBox.Text `
        -ExtraRawMoshArgs $extraBox.Text `
        -ControlPort $controlPort
}

function Get-CurrentPersistentCommands {
    $preset = $presetList.SelectedItem.Name
    $processWidth = Get-IntOrDefault $procWBox.Text 480
    $processHeight = Get-IntOrDefault $procHBox.Text 270
    $fps = Get-IntOrDefault $fpsBox.Text 30
    $history = Get-IntOrDefault $historyBox.Text 16
    $upscale = Get-IntOrDefault $upscaleBox.Text 2
    $udpPort = Get-IntOrDefault $udpPortBox.Text 23000
    $controlPort = if ($realtimeCheck.Checked) { Get-IntOrDefault $controlPortBox.Text 24000 } else { 0 }
    Build-RawMoshPersistentCommands `
        -Preset $preset `
        -Device $deviceBox.Text `
        -UseTestSource $testSourceCheck.Checked `
        -CaptureSize $captureBox.Text `
        -ProcessWidth $processWidth `
        -ProcessHeight $processHeight `
        -FrameRate $fps `
        -History $history `
        -Upscale $upscale `
        -ScaleMode $scaleModeBox.SelectedItem `
        -OutputWidth $outWBox.Text `
        -OutputHeight $outHBox.Text `
        -ExtraRawMoshArgs $extraBox.Text `
        -UdpPort $udpPort `
        -ControlPort $controlPort
}

function Get-CurrentCommand {
    if ($keepPreviewCheck.Checked) {
        $commands = Get-CurrentPersistentCommands
        return "# persistent ffplay preview`r`n$($commands.Preview)`r`n`r`n# sender; realtime UDP control can change preset/parameters without restart`r`n$($commands.Sender)"
    }

    Get-CurrentOneShotCommand
}

function Format-RealtimeNumber {
    param([double]$Value)
    $Value.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
}

function Get-MacroControlValue {
    param($Control)
    [double]$Control.Track.Value / 100.0
}

function Update-MacroControlLabels {
    foreach ($control in @($intensityControl, $motionControl, $residualControl, $temporalControl, $bitstreamControl)) {
        $control.Value.Text = (Format-RealtimeNumber (Get-MacroControlValue $control))
    }
}

function Send-RealtimeControls {
    Update-MacroControlLabels
    if (-not (Test-DemoPipelineRunning)) {
        return
    }

    $message = "controls {0} {1} {2} {3} {4}" -f `
        (Format-RealtimeNumber (Get-MacroControlValue $intensityControl)), `
        (Format-RealtimeNumber (Get-MacroControlValue $motionControl)), `
        (Format-RealtimeNumber (Get-MacroControlValue $residualControl)), `
        (Format-RealtimeNumber (Get-MacroControlValue $temporalControl)), `
        (Format-RealtimeNumber (Get-MacroControlValue $bitstreamControl))
    Send-RawMoshControlMessage $message
    $statusLabel.Text = 'Realtime controls sent.'
}

function Send-RealtimePreset {
    if (-not (Test-DemoPipelineRunning) -or -not $realtimeCheck.Checked -or -not $presetList.SelectedItem) {
        return $false
    }

    $preset = $presetList.SelectedItem.Name
    Send-RawMoshControlMessage "preset $preset"
    Send-RealtimeControls
    $statusLabel.Text = "Realtime preset sent: $preset"
    return $true
}

function Send-RealtimeResetGlitch {
    if (-not (Test-DemoPipelineRunning) -or -not $realtimeCheck.Checked) {
        return $false
    }

    Send-RawMoshControlMessage 'reset-glitch'
    $statusLabel.Text = 'Realtime glitch state reset.'
    return $true
}

function Update-SelectedParameterValue {
    if (-not $parameterBox.SelectedItem) {
        $parameterValueLabel.Text = ''
        return
    }

    $parameter = $parameterBox.SelectedItem
    $value = Get-ParameterValueFromSlider -Parameter $parameter -SliderValue $parameterTrack.Value
    $parameterValueLabel.Text = Format-RealtimeNumber $value
}

function Send-SelectedRealtimeParameter {
    if ($Script:UpdatingRealtimeControls -or -not $parameterBox.SelectedItem) {
        return
    }

    Update-SelectedParameterValue
    if (-not (Test-DemoPipelineRunning)) {
        return
    }

    $parameter = $parameterBox.SelectedItem
    $value = Get-ParameterValueFromSlider -Parameter $parameter -SliderValue $parameterTrack.Value
    Send-RawMoshControlMessage ("set {0} {1}" -f $parameter.Id, (Format-RealtimeNumber $value))
    $statusLabel.Text = "Realtime parameter sent: $($parameter.Id)=$(Format-RealtimeNumber $value)"
}

function Update-RealtimeParameterList {
    param([string]$PreferredId = '')

    if ([string]::IsNullOrWhiteSpace($PreferredId) -and $parameterBox.SelectedItem) {
        $PreferredId = $parameterBox.SelectedItem.Id
    }
    $presetName = if ($presetList.SelectedItem) { $presetList.SelectedItem.Name } else { 'drift' }
    $parameters = @(Get-RealtimeParametersForPreset -PresetName $presetName)

    $Script:UpdatingRealtimeControls = $true
    try {
        $parameterBox.Items.Clear()
        foreach ($parameter in $parameters) {
            [void]$parameterBox.Items.Add($parameter)
        }

        if ($parameterBox.Items.Count -gt 0) {
            $index = 0
            if (-not [string]::IsNullOrWhiteSpace($PreferredId)) {
                for ($i = 0; $i -lt $parameterBox.Items.Count; $i++) {
                    if ($parameterBox.Items[$i].Id -eq $PreferredId) {
                        $index = $i
                        break
                    }
                }
            }
            $parameterBox.SelectedIndex = $index
            $parameterTrack.Value = Get-ParameterSliderValue -Parameter $parameterBox.SelectedItem -Value ([double]$parameterBox.SelectedItem.Default)
        }
        Update-SelectedParameterValue
    } finally {
        $Script:UpdatingRealtimeControls = $false
    }
}

function Update-Preview {
    $preset = $presetList.SelectedItem
    if ($preset) {
        $description.Text = "$($preset.Name) - $($preset.Title)`r`n$($preset.Notes)"
    }

    try {
        $commandBox.Text = Get-CurrentCommand
        $statusLabel.Text = if (Test-DemoPipelineRunning) {
            if ($realtimeCheck.Checked) {
                "Running. Presets and realtime controls are sent over UDP."
            } else {
                "Running. Click Apply to restart with current settings."
            }
        } elseif (Test-Path -LiteralPath $Script:DatamoshExe) {
            "Ready: $Script:DatamoshExe"
        } else {
            "Release binary missing. Click Build release or run cargo build --release."
        }
    } catch {
        $commandBox.Text = $_.Exception.Message
        $statusLabel.Text = 'Invalid settings'
    }
}

function Update-PresetList {
    param([string]$PreferredName = '')

    if ([string]::IsNullOrWhiteSpace($PreferredName) -and $presetList.SelectedItem) {
        $PreferredName = $presetList.SelectedItem.Name
    }

    $Script:UpdatingPresetList = $true
    $presetList.BeginUpdate()
    try {
        $presetList.Items.Clear()
        $group = if ($groupBox.SelectedItem) { [string]$groupBox.SelectedItem } else { 'Curated' }
        foreach ($preset in (Get-VisiblePresets -ShowAll:$showAllPresetsCheck.Checked -Group $group)) {
            [void]$presetList.Items.Add($preset)
        }

        if ($presetList.Items.Count -gt 0) {
            $index = 0
            if (-not [string]::IsNullOrWhiteSpace($PreferredName)) {
                for ($i = 0; $i -lt $presetList.Items.Count; $i++) {
                    if ($presetList.Items[$i].Name -eq $PreferredName) {
                        $index = $i
                        break
                    }
                }
            }
            $presetList.SelectedIndex = $index
        }
    } finally {
        $presetList.EndUpdate()
        $Script:UpdatingPresetList = $false
    }
}

function Invoke-ApplyCurrentSettings {
    param([bool]$RequireRunning = $false)

    if ($RequireRunning -and -not (Test-DemoPipelineRunning)) {
        return
    }

    try {
        $preset = $presetList.SelectedItem.Name
        if ($keepPreviewCheck.Checked) {
            $commands = Get-CurrentPersistentCommands
            Start-PreviewPipeline -Command $commands.Preview
            Start-DemoPipeline -Command $commands.Sender -KeepPreview
            if ($realtimeCheck.Checked) {
                Start-Sleep -Milliseconds 150
                Send-RealtimeControls
                Send-SelectedRealtimeParameter
            }
            $statusLabel.Text = "Running $preset via persistent ffplay, sender pid $($Script:CurrentProcess.Id)."
        } else {
            $command = Get-CurrentOneShotCommand
            Start-DemoPipeline -Command $command
            if ($realtimeCheck.Checked) {
                Start-Sleep -Milliseconds 150
                Send-RealtimeControls
                Send-SelectedRealtimeParameter
            }
            $statusLabel.Text = "Running $preset, cmd pid $($Script:CurrentProcess.Id). ffplay preview opens separately."
        }
    } catch {
        [Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Apply failed', 'OK', 'Error') | Out-Null
        $statusLabel.Text = 'Apply failed'
    }
}

$changeControls = @(
    $deviceBox, $testSourceCheck, $captureBox, $fpsBox, $procWBox, $procHBox,
    $historyBox, $upscaleBox, $scaleModeBox, $outWBox, $outHBox, $extraBox,
    $keepPreviewCheck, $udpPortBox, $realtimeCheck, $controlPortBox
)

foreach ($control in $changeControls) {
    if ($control -is [Windows.Forms.TextBox]) {
        $control.Add_TextChanged({ Update-Preview })
    } elseif ($control -is [Windows.Forms.ComboBox]) {
        $control.Add_SelectedIndexChanged({ Update-Preview })
        $control.Add_TextChanged({ Update-Preview })
    } elseif ($control -is [Windows.Forms.CheckBox]) {
        $control.Add_CheckedChanged({ Update-Preview })
    }
}

foreach ($control in @($intensityControl, $motionControl, $residualControl, $temporalControl, $bitstreamControl)) {
    $control.Track.Add_Scroll({ Send-RealtimeControls })
    $control.Track.Add_ValueChanged({ Update-MacroControlLabels })
}

$parameterBox.Add_SelectedIndexChanged({
    if ($parameterBox.SelectedItem) {
        $Script:UpdatingRealtimeControls = $true
        try {
            $parameterTrack.Value = Get-ParameterSliderValue -Parameter $parameterBox.SelectedItem -Value ([double]$parameterBox.SelectedItem.Default)
            Update-SelectedParameterValue
        } finally {
            $Script:UpdatingRealtimeControls = $false
        }
    }
})

$parameterTrack.Add_Scroll({ Send-SelectedRealtimeParameter })
$parameterTrack.Add_ValueChanged({ Update-SelectedParameterValue })

$sendPresetButton.Add_Click({
    if (-not (Send-RealtimePreset)) {
        $statusLabel.Text = 'Start the pipeline before sending realtime preset.'
    }
})

$resetGlitchButton.Add_Click({
    if (-not (Send-RealtimeResetGlitch)) {
        $statusLabel.Text = 'Start the pipeline before resetting glitch state.'
    }
})

$resetControlsButton.Add_Click({
    foreach ($control in @($intensityControl, $motionControl, $residualControl, $temporalControl, $bitstreamControl)) {
        $control.Track.Value = 100
    }
    Send-RawMoshControlMessage 'reset-controls'
    Send-RealtimeControls
})

$presetList.Add_SelectedIndexChanged({
    Update-Preview
    Update-RealtimeParameterList
    if ($Script:UpdatingPresetList) {
        return
    }
    if ($autoApplyCheck.Checked) {
        if (Send-RealtimePreset) {
            return
        }
        Invoke-ApplyCurrentSettings -RequireRunning $true
    }
})

$showAllPresetsCheck.Add_CheckedChanged({
    $currentName = ''
    if ($presetList.SelectedItem) {
        $currentName = $presetList.SelectedItem.Name
    }
    Update-PresetList -PreferredName $currentName
    Update-Preview
})

$groupBox.Add_SelectedIndexChanged({
    $currentName = ''
    if ($presetList.SelectedItem) {
        $currentName = $presetList.SelectedItem.Name
    }
    Update-PresetList -PreferredName $currentName
    Update-Preview
})

$startButton.Add_Click({
    Invoke-ApplyCurrentSettings
})

$stopButton.Add_Click({
    Stop-DemoPipeline
    $statusLabel.Text = 'Stopped'
})

$applyButton.Add_Click({ Invoke-ApplyCurrentSettings })

$copyButton.Add_Click({
    [Windows.Forms.Clipboard]::SetText($commandBox.Text)
    $statusLabel.Text = 'Command copied'
})

$buildButton.Add_Click({
    try {
        $psi = [Diagnostics.ProcessStartInfo]::new()
        $psi.FileName = $env:ComSpec
        $psi.Arguments = '/k cargo build --release'
        $psi.WorkingDirectory = $Script:Root
        $psi.UseShellExecute = $true
        [Diagnostics.Process]::Start($psi) | Out-Null
        $statusLabel.Text = 'Started cargo build --release in a separate window.'
    } catch {
        [Windows.Forms.MessageBox]::Show($_.Exception.Message, 'Build failed', 'OK', 'Error') | Out-Null
    }
})

$timer = [Windows.Forms.Timer]::new()
$timer.Interval = 1000
$timer.Add_Tick({
    if ($Script:CurrentProcess -and $Script:CurrentProcess.HasExited) {
        $statusLabel.Text = "Pipeline exited with code $($Script:CurrentProcess.ExitCode)"
        $Script:CurrentProcess = $null
    }
    if ($Script:PreviewProcess -and $Script:PreviewProcess.HasExited) {
        $Script:PreviewProcess = $null
        $Script:CurrentPreviewCommand = $null
    }
})
$timer.Start()

$form.Add_FormClosing({
    Stop-DemoPipeline
})

Update-PresetList -PreferredName 'drift'
Update-RealtimeParameterList
Update-MacroControlLabels
Update-Preview
if ($SmokeGui) {
    "GUI smoke ok: $($presetList.Items.Count) presets"
    return
}
[Windows.Forms.Application]::Run($form)

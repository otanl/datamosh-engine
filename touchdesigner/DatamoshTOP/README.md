# Datamosh Motion TOP

Experimental TouchDesigner C++ TOP bridge for the Rust `MoshEngine`.

This plugin is intentionally thin:

- TouchDesigner owns TOP input, parameters, cooking, and output upload.
- `datamosh.dll` owns the codec state through the C ABI.
- The TOP creates one `DatamoshMoshEngine` per node instance.
- This operator is fixed to backend `1`, `raw_mosh_v1`.
- Other codec families use separate TOP DLLs with their own parameter surfaces.

## Build

```powershell
cargo build --release
.\scripts\build-td-top.cmd
```

The build expects TouchDesigner to be installed at:

```text
C:\Program Files\Derivative\TouchDesigner
```

Pass another location if needed:

```powershell
.\scripts\build-td-top.cmd -TouchDesignerRoot "D:\Apps\TouchDesigner"
```

The script writes:

```text
target\release\DatamoshTOP.dll
target\release\datamosh.dll
```

Use `DatamoshTOP.dll` as the C++ TOP plugin DLL. Keep `datamosh.dll` in the same directory so the TOP can load the Rust engine.

The TOP DLL must be built with MSVC, not MinGW/g++. TouchDesigner passes C++ plugin objects across the DLL boundary, so using the wrong C++ ABI can make TouchDesigner hang or crash while loading the plugin.

## Parameters

- `Pattern`: numeric TouchDesigner menu. The UI displays names, while CHOP export and Python can control the same parameter by zero-based index. Index `0` is always `Clean`. The menu is curated to visually distinct directions:
  `Clean`, `Motion Melt`, `Temporal Slice Drift`, `Channel Plane Desync`, `Residual Stream Desync`,
  `Motion Vector Bank Desync`, `Entropy Byte Slip`, `Transform Coefficient Drift`,
  `Residual Codebook Leak`, and `Codec State Collapse`.
- `Intensity`, `Motion`, `Residual`, `Temporal`, `Bitstream`: realtime macro controls. `0..1` is the authored range and `1..2` is overdrive.
- `Use Overrides`: enables dedicated parameter overrides. Leave it off to see the preset exactly as authored.
- `Motion` page: `MV Scale`, `MV Jitter`, `Vector Interp`, `Sample Desync`.
- `Reference` page: `Reference Lag`, `Reference Bleed`, `Reference Latch`, `Temporal Drift`.
- `Residual` page: `Residual Keep`, `Residual Jitter`, `Residual Channel`.
- `Bitstream` page: entropy, coefficient, and codebook corruption.
- `Advanced` page: direct `Param ID` / `Param Value` override.
- `Audio Enable`, `Control CHOP`: enables CHOP-driven macro control. Channel fields accept either a channel name or a numeric channel index such as `0`.
- `Intensity Chan`, `Motion Chan`, `Residual Chan`, `Temporal Chan`, `Bitstream Chan`: map CHOP channels to macro controls.
- `Reset Chan`, `Reset Threshold`, `Reset Rearm`: rising-threshold audio trigger for `Reset Glitch`.
- `Reset Glitch`: clears codec history/codebook state without changing parameters.
- `Recreate Engine`: drops the current engine and creates a fresh one on the next frame.

Dedicated and advanced overrides are state-safe: disabling `Use Overrides`, or changing/clearing `Param ID`, restores the selected preset before applying the remaining overrides.

The current CPU TOP path downloads the input as `RGBA8Fixed`, processes the previous download result, and uploads `RGBA8Fixed` output. This keeps the node from stalling on the same-frame GPU readback and gives roughly one frame of plugin-side latency.

For a quick audio-reactive patch, set `Pattern` to a named pattern such as `Residual Codebook Leak`, connect an analyzed envelope CHOP to `Control CHOP`, turn on `Audio Enable`, and leave `Intensity Chan` at `0`. Add a pulse/beat channel to `Reset Chan` if you want beat-synced history clears.

Pattern indices, usable directly through the `Pattern` parameter:

| Index | Pattern |
| ---: | --- |
| 0 | Clean |
| 1 | Motion Melt |
| 2 | Temporal Slice Drift |
| 3 | Channel Plane Desync |
| 4 | Residual Stream Desync |
| 5 | Motion Vector Bank Desync |
| 6 | Entropy Byte Slip |
| 7 | Transform Coefficient Drift |
| 8 | Residual Codebook Leak |
| 9 | Codec State Collapse |

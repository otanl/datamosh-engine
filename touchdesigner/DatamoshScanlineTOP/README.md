# Datamosh Scanline TOP

TouchDesigner CPU TOP for the SCN0 predictive scanline codec.

This is a separate operator from `DatamoshTOP`. It is fixed to C ABI backend `2`, `scanline_signal_v1`, and exposes only parameters that exist in the SCN0 representation.

## Build

```powershell
cargo build --release
.\scripts\build-td-scanline-top.cmd
```

The output is:

```text
target\release\DatamoshScanlineTOP.dll
target\release\datamosh.dll
```

Keep both DLLs in the same directory and load `DatamoshScanlineTOP.dll` as the C++ TOP plugin.

## Patterns

| Index | Pattern |
| ---: | --- |
| 0 | Clean |
| 1 | Timebase Tear |
| 2 | Sample Clock Skew |
| 3 | Line Sync Dropout |
| 4 | Chroma Sequence Desync |
| 5 | Burst Seed Loss |
| 6 | Carrier Codeword XOR |
| 7 | Temporal Predictor Ghost |
| 8 | Luma RLE Runaway |
| 9 | Luma/Chroma Crosswire |
| 10 | Signal State Collapse |

These are codec failures, not post-process effects. SCN0 v7 transmits even and odd scan fields separately and reconstructs progressive output. Its receiver carries horizontal clock, burst phase, field parity, and expected line state. Field starts are strong anchors, while additional resync markers are placed adaptively from compressed payload size. `Sync Dropout` can now lose field anchors or flip field parity instead of merely deleting isolated scanlines.

## Controls

- `Intensity`: master amount.
- `Timebase`: line clock and line address corruption.
- `Carrier`: chroma carrier, sampling ratio, and quantizer corruption.
- `Prediction`: temporal predictor and history corruption.
- `Packet`: sync, RLE, length, and payload corruption.
- `Use Overrides`: enables direct SCN0 parameter overrides.
- `Timebase` page: line shift, line address, sync, field sync, and parity.
- `Carrier` page: burst phase, chroma sequence/codewords, carrier sign, and quantizer.
- `Prediction` page: temporal predictor and history weave.
- `Packet` page: luma/chroma payload slips, RLE, packet length, and plane swap.
- `Advanced` page: direct `Param ID` / `Param Value` override.
- `Reset Glitch`: clears predictor and concealment history.

The five macro controls use `0..1` as the authored range and `1..2` as overdrive. Small SCN0 integer fields saturate at their codec limits rather than wrapping during overdrive.

The `Pattern` menu can be controlled by its displayed name or by a zero-based CHOP/Python index.

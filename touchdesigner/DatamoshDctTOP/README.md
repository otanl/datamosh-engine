# Datamosh DCT TOP

TouchDesigner CPU TOP for the DCT0 transform-domain codec (JPEG/DV-style).

This is a separate operator from `DatamoshTOP` (motion) and `ScanlineSignalTOP` (analog
signal). It is fixed to C ABI backend `3`, `dct_transform_v1`, and exposes only the
parameters that exist in the DCT0 representation. It is built from the shared CPU TOP
source (`DatamoshTOP.cpp`) with `DCT_TRANSFORM_TOP` defined.

## Build

```powershell
cargo build --release
.\scripts\build-td-dct-top.cmd
```

Output:

```text
target\release\DatamoshDctTOP.dll
target\release\datamosh.dll
```

Keep both DLLs in the same directory and load `DatamoshDctTOP.dll` as the C++ TOP plugin.

## Nature

DCT0 is **intra-only**: each frame is an independent 8x8-block DCT + quantization (like
JPEG / Motion-JPEG / DV), so glitches read as quantization storms, blocking, ringing,
DC colour shifts and block displacement *within* a frame rather than temporal smear.
Chroma is **4:2:0-subsampled** (stored at half resolution), so colour blocking is twice as
coarse as luma and fine colour detail softens — the characteristic JPEG/DV look. A
`Persistence` control adds optional temporal feedback (the previous glitched output is
blended back into the encode and re-corrupted through the real transform pipeline), so
the codec stays purely intra at `Persistence = 0` and gains flow as it rises.

The codec also has a real **entropy/bitstream stage** (`DTE0`): the quantized coefficients
are losslessly coded (DC DPCM + AC run-length + per-frame canonical Huffman) into a byte
stream. The `Entropy Desync` / `Scan Shred` / `Entropy Truncate` patterns (and the `Entropy`
override page) damage *that byte stream*, so the decoder loses sync and the rest of the scan
slides into colour with the DC predictor running away — the cascading "broken JPEG" look that
coefficient glitches cannot produce. With no entropy damage the stream is lossless.

## Patterns

| Index | Pattern |
| ---: | --- |
| 0 | Clean |
| 1 | Quantize Blocks |
| 2 | DC Predictor Smear |
| 3 | Block DC Bleed |
| 4 | Coefficient Low-Pass |
| 5 | Coefficient Sign Ring |
| 6 | Coefficient Scramble |
| 7 | Block Slip |
| 8 | Block Echo |
| 9 | Temporal Flow |
| 10 | False Colour |
| 11 | Transform Collapse |
| 12 | Entropy Desync |
| 13 | Scan Shred |
| 14 | Entropy Truncate |

Patterns 0–11 are coefficient-domain failures. Patterns 12–14 are **entropy/bitstream**
failures (see below). These are transform-domain codec failures, not post-process image filters. DC corruption
is differentially coded across blocks, so it bleeds block-by-block in scan order; DC drift
walks each channel independently so the colour spans the whole space (not just the
green/magenta diagonal).

## Controls

- `Intensity`: master amount.
- `Structure`: block displacement and coefficient-order corruption.
- `Persist`: temporal feedback / persistence amount.
- `DC`: DC drift and per-block DC offset.
- `Quant`: quantization coarseness.
- `Use Overrides`: enables direct DCT0 parameter overrides.
- `Coefficient` page: quant scale, AC zero-above, sign-flip, coeff shift, zig-zag reverse,
  block transpose.
- `DC` page: DC drift and DC block offset (amount + period).
- `Block` page: block shift X/Y, block shift period, block repeat, chroma swap.
- `Temporal` page: persistence.
- `Entropy` page: byte flip / byte drop periods, scan slip (period/bytes/window), truncate
  tail — corrupts the `DTE0` byte stream (decoder-desync glitches; CPU only).
- `Advanced` page: direct `Param ID` / `Param Value` override.
- `Reset Glitch`: clears the temporal-feedback history.

The five macro controls use `0..1` as the authored range and `1..2` as overdrive. The
`Pattern` menu can be driven by its displayed name or by a zero-based CHOP/Python index.

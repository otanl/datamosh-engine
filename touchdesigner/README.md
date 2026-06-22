# TouchDesigner Operators

The project ships seven TOP operators. DLL names and TouchDesigner `opType` values remain stable.

| Operator label | DLL | Backend | Processing |
| --- | --- | --- | --- |
| Datamosh Motion TOP | `DatamoshTOP.dll` | `raw_mosh_v1` | CPU |
| Datamosh Motion CUDA TOP | `DatamoshCudaTOP.dll` | `cuda_motion_v1` | CUDA |
| Datamosh Scanline TOP | `DatamoshScanlineTOP.dll` | `scanline_signal_v1` | CPU |
| Datamosh DCT TOP | `DatamoshDctTOP.dll` | `dct_transform_v1` | CPU |
| Datamosh DCT CUDA TOP | `DatamoshDctCudaTOP.dll` | `cuda_dct_v1` | CUDA |
| Datamosh Wavelet TOP | `DatamoshWaveletTOP.dll` | `wavelet_pyramid_v1` | CPU |
| Datamosh Wavelet CUDA TOP | `DatamoshWaveletCudaTOP.dll` | `cuda_wavelet_v1` | CUDA |

## Common Conventions

- `Pattern` index `0` is `Clean` in every operator.
- Pattern names remain available through the TouchDesigner menu; numeric CHOP/Python control uses the documented zero-based index.
- Macro controls use `0..1` as the authored range and `1..2` as overdrive. `Intensity` is the master amount.
- `Reset Glitch` clears persistent decoder/reference history without changing the selected pattern.
- Info CHOP/DAT reports `pattern`, `pattern_index`, `pattern_count`, `operator_version`, and `pattern_schema_version`. CUDA also reports its internal `implementation_version`.
- CPU operators default to clean codec reconstruction. CUDA also defaults to `Clean`.

## Parameters

Datamosh Motion TOP:

- Macros: `Intensity`, `Motion`, `Residual`, `Temporal`, `Bitstream`
- Override pages: `Motion`, `Reference`, `Residual`, `Bitstream`, `Advanced`
- Audio page: CHOP mapping for macros and reset trigger

Turning `Use Overrides` off restores the selected preset before the next frame. Changing or clearing the `Advanced` `Param ID` also removes the previous direct override instead of leaving stale engine state.

Datamosh Motion CUDA TOP:

- Macros: `Intensity`, `Motion`, `Residual`, `Temporal`, `Bitstream`
- Codec page: block size, search radius/step, history, seed, vector decode
- Override pages: `Motion`, `Reference`, `Residual`, `Bitstream`, `Advanced`
- Audio page: CHOP mapping for all five macros and reset trigger
- The CPU Motion TOP's 18 visible detailed parameter IDs are available as GPU-native decoder mutations with the same names and ranges
- CUDA-only codec controls remain available; serialized MSH0 mutation remains a parallel GPU implementation

Datamosh Scanline TOP:

- Macros: `Intensity`, `Timebase`, `Carrier`, `Prediction`, `Packet`
- Override pages: `Timebase`, `Carrier`, `Prediction`, `Packet`, `Advanced`
- Audio page: CHOP mapping for macros and reset trigger

Datamosh DCT TOP:

- Macros: `Intensity`, `Structure`, `Persist`, `DC`, `Quant`
- Override pages: `Coefficient`, `DC`, `Block`, `Temporal`, `Entropy`, `Advanced`
- Entropy patterns use the CPU-only DTE0 variable-length bitstream path
- Audio page: CHOP mapping for macros and reset trigger

Datamosh DCT CUDA TOP:

- Macros: `Intensity`, `Structure`, `Persist`, `DC`, `Quant`
- Codec page: JPEG-style `Quality`
- Override pages: `Coefficient`, `DC`, `Block`, `Temporal`, `Advanced`
- Audio page: CHOP mapping for all five macros and reset trigger
- Coefficient-domain parameter names, ranges, presets, and control order match the CPU DCT operator
- DTE0 entropy patterns and parameters are CPU-only; Advanced entropy parameter IDs produce an explicit warning

Datamosh Wavelet TOP:

- Macros: `Intensity`, `Structure`, `Coefficient`, `History`, `Routing`
- Codec page: wavelet `Quality`, decomposition `Levels`, packet `History Length`
- Override pages: `Structure`, `Coefficient`, `History`, `Routing`, `Advanced`
- Patterns operate on WVT0 subband packets, coefficient bitplanes, temporal concealment, and inverse lifting state

Datamosh Wavelet CUDA TOP:

- Macros: `Intensity`, `Structure`, `Coefficient`, `History`, `Routing`
- Codec page: wavelet `Quality`, decomposition `Levels`, packet `History Length`
- Override pages: `Structure`, `Coefficient`, `History`, `Routing`, `Advanced`
- Audio page: CHOP mapping for all five macros and reset trigger
- Uses the same parameter names, ranges, preset indices, integer transform, packet routing, and control order as the CPU WVT0 operator
- Keeps transform planes, quantized packets, and packet history on the GPU; it does not link `datamosh.dll`

## Index Migration

The 2026-06 operator cleanup moved `Clean` to index `0`.

- Datamosh Motion TOP: every former index moved by `+1`.
- Datamosh Motion CUDA TOP: former `Clean` index `8` moved to `0`; former indices `0` through `7` moved by `+1`; GPU-only indices `9` through `12` are unchanged.
- Datamosh Scanline TOP: indices are unchanged; only several display labels were clarified.
- Datamosh DCT TOP and Datamosh DCT CUDA TOP started at schema version `1`; their menus are operator-specific.
- Datamosh Wavelet TOP and Datamosh Wavelet CUDA TOP share the WVT0 schema version `1` menu and indices.

TouchDesigner may cache custom parameter schemas. After replacing a plugin DLL, recreate existing nodes once before validating saved index automation. Name-based selection is preferred when patch portability matters.

Wavelet CUDA operator version `2` aligns its internal parameter names with the CPU WVT0 operator (`Motion`, `Residual`, `Temporal`, and `Bitstream`, displayed as Structure, Coefficient, History, and Routing). Recreate older Wavelet CUDA nodes after installing this DLL.

## Operator Rename

The Scanline TOP was renamed for naming consistency with the other operators: its DLL `ScanlineSignalTOP.dll` → `DatamoshScanlineTOP.dll` and its `opType` `Scanlinesignal` → `Datamoshscanline`. Because the `opType` changed, `.toe` files built before the rename show the node as missing and must recreate it. The codec backend (`scanline_signal_v1`, C ABI backend `2`) is unchanged.

DCT CUDA operator version `2` similarly aligns its internal macro names with the CPU DCT operator and adds detailed overrides, Advanced control, Audio CHOP mapping, and reset/recreate controls. Recreate older DCT CUDA nodes after installing this DLL.

Motion CUDA operator version `3` adds the CPU Motion TOP's detailed override pages, Advanced parameter control, Audio CHOP mapping, and reset/recreate controls. Recreate older Motion CUDA nodes after installing this DLL.

# TouchDesigner Operators

The project ships five TOP operators. DLL names and TouchDesigner `opType` values remain stable.

| Operator label | DLL | Backend | Processing |
| --- | --- | --- | --- |
| Datamosh Motion TOP | `DatamoshTOP.dll` | `raw_mosh_v1` | CPU |
| Datamosh Motion CUDA TOP | `DatamoshCudaTOP.dll` | `cuda_motion_v1` | CUDA |
| Datamosh Scanline TOP | `ScanlineSignalTOP.dll` | `scanline_signal_v1` | CPU |
| Datamosh DCT TOP | `DatamoshDctTOP.dll` | `dct_transform_v1` | CPU |
| Datamosh DCT CUDA TOP | `DatamoshDctCudaTOP.dll` | `cuda_dct_v1` | CUDA |

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
- Parameters may be driven directly by CHOP export or Python

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
- Coefficient-domain patterns match the CPU DCT operator; DTE0 entropy patterns are CPU-only

## Index Migration

The 2026-06 operator cleanup moved `Clean` to index `0`.

- Datamosh Motion TOP: every former index moved by `+1`.
- Datamosh Motion CUDA TOP: former `Clean` index `8` moved to `0`; former indices `0` through `7` moved by `+1`; GPU-only indices `9` through `12` are unchanged.
- Datamosh Scanline TOP: indices are unchanged; only several display labels were clarified.
- Datamosh DCT TOP and Datamosh DCT CUDA TOP started at schema version `1`; their menus are operator-specific.

TouchDesigner may cache custom parameter schemas. After replacing a plugin DLL, recreate existing nodes once before validating saved index automation. Name-based selection is preferred when patch portability matters.

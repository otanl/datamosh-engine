# Datamosh DCT CUDA TOP

GPU-native reimplementation of the DCT0 transform codec (`src/dct_codec.rs`) as a
TouchDesigner CUDA TOP. All codec state stays on the GPU — like `DatamoshCudaTOP` (the
motion codec) this does **not** call into the Rust core and does **not** link
`datamosh.dll`. It is a parallel implementation of the same glitch *ideas*; parity with the
CPU codec is maintained **by hand** (a change in `dct_codec.rs` does not propagate here).

## Build

```powershell
.\scripts\build-td-dct-cuda-top.cmd        # → target/release/DatamoshDctCudaTOP.dll
```

Requirements: CUDA Toolkit (`nvcc` on `PATH`), MSVC (`cl.exe`, auto-located via `vswhere`),
and the TouchDesigner CUDA TOP SDK headers under
`…\TouchDesigner\Samples\CPlusPlus\CudaTOP`. opType `Datamoshdctcuda` (distinct from the
motion CUDA TOP's `Datamoshcuda` and the CPU DCT TOP's `Datamoshdct`).

## Pipeline (per frame, all on GPU)

1. RGB → YCbCr, luma kept full-res, chroma 4:2:0-subsampled (2×2 average) — one CUDA thread
   per 8×8 block reads the input surface directly (blended with the previous output when
   `Persist` > 0 for temporal feedback).
2. Forward 8×8 DCT + JPEG-style quantization per block (luma + chroma grids).
3. Transform-domain glitch: block remap (ping-pong), per-block coefficient corruption
   (quant, low-pass, sign-flip, coeff-shift, zig-zag reverse, block transpose, DC offset),
   propagating DC-predictor drift (host-precomputed per-channel sign prefix), Cb/Cr swap.
4. Inverse DCT + dequant (with a DC-only fast path), nearest-neighbour chroma upsample,
   YCbCr → RGB, write to the output surface + store for next frame's feedback.

## Parameters

- `Pattern` (12): clean, blocks, dc-smear, bleed, blur, ring, scramble, block-slip, echo,
  flow, false-color, composite — the same presets as the CPU DCT TOP.
- `Intensity` master, plus `Structure` / `Persist` / `DC` / `Quant` macros (0..2), resolved
  on the host exactly as `apply_dct_transform_controls` does.
- `Quality` (1..100) — JPEG-style quantization quality.
- `Reset Glitch` — clears the temporal-feedback history.

The host (`DatamoshDctCudaTOP.cpp`) resolves the pattern preset + macros into the detailed
glitch parameters and passes them to the kernels, so the preset table mirrors
`load_dct_transform_preset`.

## Verifying

This DLL can only run inside TouchDesigner, but the kernels can be exercised headless. The
**parity guard** does exactly that: `tools/dct_parity_check.cu` runs the same input + preset
through the CPU codec (via the C ABI) and these CUDA kernels and reports per-preset MAE,
failing on drift. Build and run:

```powershell
cargo build --release            # produces datamosh.dll.lib
.\scripts\build-dct-parity-check.cmd
.\target\release\dct_parity_check.exe   # run from target/release (needs datamosh.dll)
```

All presets currently match within ~0.2 MAE. Re-run it after editing either the kernels or
`src/dct_codec.rs` to confirm the hand-maintained parity still holds. The preset table is
shared with the TOP via `DatamoshDctCudaPresets.h`, so the check guards the TOP's values too.

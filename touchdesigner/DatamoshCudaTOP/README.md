# Datamosh Motion CUDA TOP

CUDA-native TouchDesigner TOP for the custom predictive glitch codec.

The node keeps motion vectors, residuals, clean references, dirty references, and residual history on the GPU. It reads the input TOP through TouchDesigner's CUDA array API and writes directly to a CUDA output array, avoiding GPU-to-CPU-to-GPU transfers.

The input may be `BGRA8 Fixed`, `RGBA8 Fixed`, `RGBA16 Fixed`, `RGBA16 Float`, or `RGBA32 Float`. Input pixels are converted to the codec's internal BGRA8 representation on the GPU, and the node outputs `BGRA8 Fixed`. This allows direct use with Movie File In TOP outputs that use floating-point textures.

The TOP uses a regular CUDA stream, matching TouchDesigner's CUDA sample, and forces continuous cooking so a transient output failure cannot stop an upstream Movie File In TOP. Its Info CHOP/DAT exposes:

- `input_cooks`: upstream TOP cook counter
- `cook_stage`: last completed plugin stage (`8` means a complete cook)
- `input_format`: numeric `OP_PixelFormat`

These values are intended for diagnosing host-specific CUDA interop issues.

## Build

```powershell
.\scripts\build-td-cuda-top.cmd
```

Requirements:

- NVIDIA GPU
- CUDA Toolkit with `nvcc`
- TouchDesigner CUDA TOP SDK sample headers
- Visual Studio C++ build tools

Output:

```text
target\release\DatamoshCudaTOP.dll
```

The GPU TOP does not depend on `datamosh.dll`. The existing CPU `DatamoshTOP.dll` remains available as a separate fallback operator.

## Patterns

- Clean: reconstructs without intentional corruption for A/B comparison.
- Motion Melt: follows the encoded motion field into older dirty references.
- Temporal Slice Drift: reads history by vertical column and rotates motion vectors between bands.
- Channel Plane Desync: decodes each channel with a different temporal slot and vector orientation.
- Residual Stream Desync: reads residuals with alternating vertical and diagonal stride errors.
- Motion Vector Bank Desync: reads neighboring 2D vector banks and rotates their vector orientation.
- Entropy Byte Slip: decodes residual bytes using column-major, mirrored, and reversed scan orders.
- Residual Codebook Leak: reads rotated or transposed residual tiles from older dictionary slots.
- Codec State Collapse: combines cell-local 2D reference, vector, and residual state damage.
- Row Pitch Fracture: decodes the residual plane with a changing incorrect row pitch.
- Residual Scale Mismatch: decodes residual blocks using the wrong block dimensions and orientation.
- Packet Tile Loss: drops or substitutes deterministic residual packets and freezes their reference tiles.
- History Weave: interleaves dirty-history slots across rows, columns, and time.

These patterns corrupt the GPU codec's motion, residual, byte addressing, and history state. They are not post-process image filters.

The five macro controls use `0..1` as the authored range and `1..2` as overdrive.

Temporal propagation is also pattern-specific. Only Motion Melt follows the encoded motion field directly. Temporal Slice Drift converts horizontal motion into vertical history movement; Channel Plane Desync rotates motion independently per channel; Residual, Entropy, and Codebook patterns disable recursive dirty-history transport and damage their base prediction vectors differently; Vector and Unstable use two-dimensional bank-derived vectors.

`Vector Decode` overrides the decoder's interpretation of motion-vector direction:

- `Pattern`: preset-specific routing
- `Original`: encoded direction
- `Reverse`: inverted encoded direction
- `Vertical`: discards horizontal transport
- `Static`: zeroes transport while retaining residual/history corruption
- `Radial`: interprets vector magnitude as movement toward or away from frame regions

The residual predictor uses the same vector interpretation. Motion Melt uses motion-compensated residuals only in `Pattern` or `Original` mode; other patterns default to co-located residual prediction. This prevents the source motion direction from remaining hidden inside the residual stream after vectors are replaced.

Changing `Pattern` or `Vector Decode` automatically resets dirty history. This matches the CPU TOP's preset behavior and prevents a previous pattern's directional smear from contaminating the newly selected decoder path.

`Pattern` is a TouchDesigner numeric menu: the UI displays names while CHOP export and Python may set the same parameter by zero-based index. For example:

```python
op('datamoshcuda1').par.Pattern = 10
```

Current indices:

| Index | Pattern |
| ---: | --- |
| 0 | Clean |
| 1 | Motion Melt |
| 2 | Temporal Slice Drift |
| 3 | Channel Plane Desync |
| 4 | Residual Stream Desync |
| 5 | Motion Vector Bank Desync |
| 6 | Entropy Byte Slip |
| 7 | Residual Codebook Leak |
| 8 | Codec State Collapse |
| 9 | Row Pitch Fracture |
| 10 | Residual Scale Mismatch |
| 11 | Packet Tile Loss |
| 12 | History Weave |

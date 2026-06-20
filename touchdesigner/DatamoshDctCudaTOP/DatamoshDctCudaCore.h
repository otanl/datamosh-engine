#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// GPU-native reimplementation of the DCT0 transform codec (src/dct_codec.rs). Parity with
// the CPU codec is maintained by hand — a change in dct_codec.rs does NOT propagate here.
//
// The host (DatamoshDctCudaTOP.cpp) resolves a pattern preset + the five macro controls into
// the detailed glitch fields below (mirroring load_dct_transform_preset /
// apply_dct_transform_controls) so the kernels only apply already-resolved parameters.
struct DatamoshDctCudaParams
{
    int pattern = 0;
    int quality = 50;
    int inputFormat = 0;

    // Resolved transform-domain glitch parameters (see DctGlitchParams in dct_codec.rs).
    float quantScale = 1.0f;
    int dcDrift = 0;
    int dcDriftEvery = 0;
    int dcBlockOffset = 0;
    int dcBlockOffsetEvery = 0;
    int acZeroAbove = 0;
    int signFlipEvery = 0;
    int coeffShift = 0;
    int coeffShiftEvery = 0;
    int blockShiftX = 0;
    int blockShiftY = 0;
    int blockShiftEvery = 0;
    int blockRepeatEvery = 0;
    int zigzagReverseEvery = 0;
    int blockTransposeEvery = 0;
    int chromaSwapEvery = 0;
    float persistence = 0.0f;
};

struct DatamoshDctCudaState;

cudaError_t datamoshDctCudaCreate(DatamoshDctCudaState** state, int width, int height);
void datamoshDctCudaDestroy(DatamoshDctCudaState* state);
void datamoshDctCudaReset(DatamoshDctCudaState* state);
cudaError_t datamoshDctCudaProcess(
    DatamoshDctCudaState* state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    const DatamoshDctCudaParams& params,
    cudaStream_t stream);

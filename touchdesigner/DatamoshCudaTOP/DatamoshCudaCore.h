#pragma once

#include <cuda_runtime.h>

#include <cstdint>

enum DatamoshCudaOverride : uint64_t
{
    DATAMOSH_CUDA_OVERRIDE_MV_SCALE = 1ULL << 0,
    DATAMOSH_CUDA_OVERRIDE_MV_JITTER = 1ULL << 1,
    DATAMOSH_CUDA_OVERRIDE_VECTOR_INTERP = 1ULL << 2,
    DATAMOSH_CUDA_OVERRIDE_SAMPLE_DESYNC = 1ULL << 3,
    DATAMOSH_CUDA_OVERRIDE_REFERENCE_LAG = 1ULL << 4,
    DATAMOSH_CUDA_OVERRIDE_REFERENCE_BLEED = 1ULL << 5,
    DATAMOSH_CUDA_OVERRIDE_REFERENCE_LATCH = 1ULL << 6,
    DATAMOSH_CUDA_OVERRIDE_TEMPORAL_DRIFT = 1ULL << 7,
    DATAMOSH_CUDA_OVERRIDE_RESIDUAL_KEEP = 1ULL << 8,
    DATAMOSH_CUDA_OVERRIDE_RESIDUAL_JITTER = 1ULL << 9,
    DATAMOSH_CUDA_OVERRIDE_RESIDUAL_CHANNEL = 1ULL << 10,
    DATAMOSH_CUDA_OVERRIDE_ENTROPY_EVERY = 1ULL << 11,
    DATAMOSH_CUDA_OVERRIDE_ENTROPY_WINDOWS = 1ULL << 12,
    DATAMOSH_CUDA_OVERRIDE_COEFF_SHIFT = 1ULL << 13,
    DATAMOSH_CUDA_OVERRIDE_COEFF_QUANT = 1ULL << 14,
    DATAMOSH_CUDA_OVERRIDE_CODEBOOK_EVERY = 1ULL << 15,
    DATAMOSH_CUDA_OVERRIDE_CODEBOOK_STRIDE = 1ULL << 16,
    DATAMOSH_CUDA_OVERRIDE_CODEBOOK_SHUFFLE = 1ULL << 17,
};

constexpr uint64_t DATAMOSH_CUDA_OVERRIDE_ALL =
    (1ULL << 18) - 1ULL;

struct DatamoshCudaParams
{
    int pattern = 0;
    float intensity = 1.0f;
    float motion = 1.0f;
    float residual = 1.0f;
    float temporal = 1.0f;
    float bitstream = 1.0f;
    int blockSize = 16;
    int searchRadius = 8;
    int searchStep = 4;
    int historySlots = 8;
    int inputFormat = 0;
    int vectorDecode = 0;
    uint32_t seed = 1;

    uint64_t overrideMask = 0;
    float mvScale = 1.0f;
    int mvJitter = 0;
    float vectorInterpolation = 0.0f;
    float sampleAddressDesync = 0.0f;
    int referenceLag = 1;
    float referenceBleed = 0.0f;
    int referenceLatchFrames = 1;
    int temporalSliceDrift = 0;
    float residualKeep = 1.0f;
    int residualAddressJitter = 0;
    int residualChannelShift = 0;
    int entropySlipEvery = 0;
    int entropySlipWindows = 1;
    int coeffShift = 0;
    int coeffQuant = 1;
    int codebookReplaceEvery = 0;
    int codebookStride = 1;
    int codebookShuffleEvery = 0;
};

struct DatamoshCudaState;

cudaError_t datamoshCudaCreate(
    DatamoshCudaState** state,
    int width,
    int height,
    int blockSize,
    int historySlots);
void datamoshCudaDestroy(DatamoshCudaState* state);
void datamoshCudaReset(DatamoshCudaState* state);
cudaError_t datamoshCudaProcess(
    DatamoshCudaState* state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    const DatamoshCudaParams& params,
    cudaStream_t stream);

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// GPU-native reimplementation of WVT0 (src/wavelet_codec.rs). The host resolves
// presets and macro controls before launching the kernels.
struct DatamoshWaveletCudaParams
{
    int pattern = 0;
    int quality = 82;
    int levels = 3;
    int historyLength = 12;
    int inputFormat = 0;

    int packetShift = 0;
    int packetShiftEvery = 0;
    int orientationRotate = 0;
    int orientationRotateEvery = 0;
    int levelFold = 0;
    int levelFoldEvery = 0;
    int channelRoute = 0;
    int channelRouteEvery = 0;
    int packetLossEvery = 0;
    int packetLossConceal = 1;
    int bitplaneClear = 0;
    int bitplaneClearEvery = 0;
    int bitplaneXor = 0;
    int bitplaneXorEvery = 0;
    int signFlipEvery = 0;
    int historyLag = 1;
    int historyBandEvery = 0;
    int lowpassHistoryLag = 0;
    int liftingBias = 0;
    int liftingBiasEvery = 0;
};

struct DatamoshWaveletCudaState;

cudaError_t datamoshWaveletCudaCreate(
    DatamoshWaveletCudaState** state,
    int width,
    int height,
    int levels,
    int historyLength);
void datamoshWaveletCudaDestroy(DatamoshWaveletCudaState* state);
void datamoshWaveletCudaReset(DatamoshWaveletCudaState* state);
cudaError_t datamoshWaveletCudaProcess(
    DatamoshWaveletCudaState* state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    const DatamoshWaveletCudaParams& params,
    cudaStream_t stream);

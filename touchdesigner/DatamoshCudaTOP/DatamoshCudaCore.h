#pragma once

#include <cuda_runtime.h>

#include <cstdint>

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

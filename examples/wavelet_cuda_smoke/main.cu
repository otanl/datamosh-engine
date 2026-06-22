#include "../../touchdesigner/DatamoshWaveletCudaTOP/DatamoshWaveletCudaCore.h"
#include "../../touchdesigner/DatamoshWaveletCudaTOP/DatamoshWaveletCudaPresets.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

cudaSurfaceObject_t createSurface(cudaArray_t array)
{
    cudaResourceDesc resource = {};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = array;
    cudaSurfaceObject_t surface = 0;
    check(cudaCreateSurfaceObject(&surface, &resource), "cudaCreateSurfaceObject");
    return surface;
}

void fillFrame(std::vector<uchar4>& pixels, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const bool box =
                x > width / 5 && x < width * 3 / 5 &&
                y > height / 4 && y < height * 3 / 4;
            pixels[static_cast<size_t>(y) * width + x] = make_uchar4(
                static_cast<unsigned char>((x * 5 + y) & 255),
                static_cast<unsigned char>((x + y * 3) & 255),
                static_cast<unsigned char>(box ? 235 : ((x * 2 + y * 7) & 255)),
                255);
        }
    }
}

uint64_t checksum(const std::vector<uchar4>& pixels)
{
    uint64_t value = 1469598103934665603ULL;
    for (size_t index = 0; index < pixels.size(); index += 17)
    {
        value ^= pixels[index].x;
        value *= 1099511628211ULL;
        value ^= pixels[index].y;
        value *= 1099511628211ULL;
        value ^= pixels[index].z;
        value *= 1099511628211ULL;
    }
    return value;
}

} // namespace

int main(int argc, char** argv)
{
    const int width = argc > 1 ? std::max(16, std::atoi(argv[1])) : 1920;
    const int height = argc > 2 ? std::max(16, std::atoi(argv[2])) : 1080;
    const int frames = argc > 3 ? std::max(1, std::atoi(argv[3])) : 120;
    const size_t rowBytes = static_cast<size_t>(width) * sizeof(uchar4);
    std::vector<uchar4> input(static_cast<size_t>(width) * height);
    std::vector<uchar4> output(input.size());
    fillFrame(input, width, height);

    cudaDeviceProp device = {};
    check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");

    cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
    cudaArray_t inputArray = nullptr;
    cudaArray_t outputArray = nullptr;
    check(
        cudaMallocArray(
            &inputArray, &format, width, height, cudaArraySurfaceLoadStore),
        "cudaMallocArray input");
    check(
        cudaMallocArray(
            &outputArray, &format, width, height, cudaArraySurfaceLoadStore),
        "cudaMallocArray output");
    const cudaSurfaceObject_t inputSurface = createSurface(inputArray);
    const cudaSurfaceObject_t outputSurface = createSurface(outputArray);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");
    check(
        cudaMemcpy2DToArrayAsync(
            inputArray,
            0,
            0,
            input.data(),
            rowBytes,
            rowBytes,
            height,
            cudaMemcpyHostToDevice,
            stream),
        "cudaMemcpy2DToArrayAsync");

    DatamoshWaveletCudaState* state = nullptr;
    check(
        datamoshWaveletCudaCreate(&state, width, height, 3, 12),
        "datamoshWaveletCudaCreate");

    DatamoshWaveletCudaParams clean = waveletcuda::presetParams(0);
    clean.quality = 100;
    clean.inputFormat = 0;
    check(
        datamoshWaveletCudaProcess(
            state, inputSurface, outputSurface, clean, stream),
        "datamoshWaveletCudaProcess clean");
    check(
        cudaMemcpy2DFromArrayAsync(
            output.data(),
            rowBytes,
            outputArray,
            0,
            0,
            rowBytes,
            height,
            cudaMemcpyDeviceToHost,
            stream),
        "cudaMemcpy2DFromArrayAsync clean");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize clean");
    if (std::memcmp(input.data(), output.data(), input.size() * sizeof(uchar4)) != 0)
    {
        std::fprintf(stderr, "quality-100 clean reconstruction is not bit exact\n");
        return 1;
    }

    datamoshWaveletCudaReset(state);
    DatamoshWaveletCudaParams params = waveletcuda::presetParams(10);
    waveletcuda::applyControls(params, 1, 1, 1, 1, 1);
    params.inputFormat = 0;
    for (int warmup = 0; warmup < 8; ++warmup)
        check(
            datamoshWaveletCudaProcess(
                state, inputSurface, outputSurface, params, stream),
            "datamoshWaveletCudaProcess warmup");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
    check(cudaEventCreate(&started), "cudaEventCreate started");
    check(cudaEventCreate(&finished), "cudaEventCreate finished");
    check(cudaEventRecord(started, stream), "cudaEventRecord started");
    for (int frame = 0; frame < frames; ++frame)
        check(
            datamoshWaveletCudaProcess(
                state, inputSurface, outputSurface, params, stream),
            "datamoshWaveletCudaProcess benchmark");
    check(cudaEventRecord(finished, stream), "cudaEventRecord finished");
    check(cudaEventSynchronize(finished), "cudaEventSynchronize finished");

    float milliseconds = 0.0f;
    check(
        cudaEventElapsedTime(&milliseconds, started, finished),
        "cudaEventElapsedTime");
    check(
        cudaMemcpy2DFromArray(
            output.data(),
            rowBytes,
            outputArray,
            0,
            0,
            rowBytes,
            height,
            cudaMemcpyDeviceToHost),
        "cudaMemcpy2DFromArray benchmark");

    const float frameMilliseconds = milliseconds / frames;
    std::printf(
        "wavelet cuda smoke ok: gpu=%s %dx%d frames=%d avg=%.3fms fps=%.1f checksum=%llu\n",
        device.name,
        width,
        height,
        frames,
        frameMilliseconds,
        1000.0f / frameMilliseconds,
        static_cast<unsigned long long>(checksum(output)));

    datamoshWaveletCudaDestroy(state);
    cudaEventDestroy(started);
    cudaEventDestroy(finished);
    cudaStreamDestroy(stream);
    cudaDestroySurfaceObject(inputSurface);
    cudaDestroySurfaceObject(outputSurface);
    cudaFreeArray(inputArray);
    cudaFreeArray(outputArray);
    return 0;
}

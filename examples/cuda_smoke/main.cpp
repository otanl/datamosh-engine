#include "../../touchdesigner/DatamoshCudaTOP/DatamoshCudaCore.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
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

uint64_t checksum(const std::vector<uchar4>& pixels)
{
    uint64_t value = 0;
    for (size_t index = 0; index < pixels.size(); index += 7)
    {
        value = value * 131 + pixels[index].x;
        value = value * 131 + pixels[index].y;
        value = value * 131 + pixels[index].z;
    }
    return value;
}

struct DirectionMetrics
{
    uint64_t horizontalVariation = 0;
    uint64_t verticalVariation = 0;
};

int pixelError(uchar4 input, uchar4 output)
{
    return std::abs(static_cast<int>(input.x) - static_cast<int>(output.x)) +
           std::abs(static_cast<int>(input.y) - static_cast<int>(output.y)) +
           std::abs(static_cast<int>(input.z) - static_cast<int>(output.z));
}

DirectionMetrics directionMetrics(
    const std::vector<uchar4>& input,
    const std::vector<uchar4>& output,
    int width,
    int height)
{
    DirectionMetrics metrics;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int pixel = y * width + x;
            int error = pixelError(input[pixel], output[pixel]);
            if (x > 0)
            {
                int leftError = pixelError(input[pixel - 1], output[pixel - 1]);
                metrics.horizontalVariation += std::abs(error - leftError);
            }
            if (y > 0)
            {
                int aboveError = pixelError(input[pixel - width], output[pixel - width]);
                metrics.verticalVariation += std::abs(error - aboveError);
            }
        }
    }
    return metrics;
}

void fillFrame(std::vector<uchar4>& pixels, int width, int height, int frame)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int movingX = (x + frame * 5) % width;
            int movingY = (y + frame * 3) % height;
            bool square = movingX > width / 4 && movingX < width / 2 &&
                          movingY > height / 4 && movingY < height * 3 / 4;
            pixels[y * width + x] = make_uchar4(
                static_cast<unsigned char>((x * 3 + frame * 7) & 255),
                static_cast<unsigned char>((y * 5) & 255),
                static_cast<unsigned char>(square ? 240 : 24),
                255);
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
    const int width = argc > 1 ? std::max(std::atoi(argv[1]), 16) : 96;
    const int height = argc > 2 ? std::max(std::atoi(argv[2]), 16) : 64;
    const int frameCount = argc > 3 ? std::max(std::atoi(argv[3]), 2) : 12;
    const size_t rowBytes = width * sizeof(uchar4);
    std::vector<uchar4> input(width * height);
    std::vector<uchar4> output(width * height);

    cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
    cudaArray_t inputArray = nullptr;
    cudaArray_t outputArray = nullptr;
    check(cudaMallocArray(&inputArray, &format, width, height, cudaArraySurfaceLoadStore),
          "cudaMallocArray input");
    check(cudaMallocArray(&outputArray, &format, width, height, cudaArraySurfaceLoadStore),
          "cudaMallocArray output");
    cudaSurfaceObject_t inputSurface = createSurface(inputArray);
    cudaSurfaceObject_t outputSurface = createSurface(outputArray);
    cudaStream_t stream = nullptr;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");

    DatamoshCudaState* state = nullptr;
    check(datamoshCudaCreate(&state, width, height, 16, 8), "datamoshCudaCreate");
    DatamoshCudaParams params;
    params.pattern = 7;
    params.intensity = 1.4f;
    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;
    check(cudaEventCreate(&started), "cudaEventCreate started");
    check(cudaEventCreate(&finished), "cudaEventCreate finished");
    float totalMilliseconds = 0.0f;

    uint64_t firstChecksum = 0;
    uint64_t glitchedChecksum = 0;
    for (int frame = 0; frame < frameCount; ++frame)
    {
        fillFrame(input, width, height, frame);
        check(cudaMemcpy2DToArrayAsync(
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
        check(cudaEventRecord(started, stream), "cudaEventRecord started");
        check(datamoshCudaProcess(
                  state, inputSurface, outputSurface, params, stream),
              "datamoshCudaProcess");
        check(cudaEventRecord(finished, stream), "cudaEventRecord finished");
        check(cudaMemcpy2DFromArrayAsync(
                  output.data(),
                  rowBytes,
                  outputArray,
                  0,
                  0,
                  rowBytes,
                  height,
                  cudaMemcpyDeviceToHost,
                  stream),
              "cudaMemcpy2DFromArrayAsync");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, started, finished),
              "cudaEventElapsedTime");
        totalMilliseconds += milliseconds;
        if (frame == 0)
            firstChecksum = checksum(output);
        if (frame == frameCount - 1)
            glitchedChecksum = checksum(output);
    }

    datamoshCudaReset(state);
    fillFrame(input, width, height, 20);
    check(cudaMemcpy2DToArray(
              inputArray, 0, 0, input.data(), rowBytes, rowBytes, height, cudaMemcpyHostToDevice),
          "cudaMemcpy2DToArray reset");
    check(datamoshCudaProcess(state, inputSurface, outputSurface, params, stream),
          "datamoshCudaProcess reset");
    check(cudaMemcpy2DFromArrayAsync(
              output.data(),
              rowBytes,
              outputArray,
              0,
              0,
              rowBytes,
              height,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpy2DFromArrayAsync reset");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize reset");

    const uint64_t resetInputChecksum = checksum(input);
    const uint64_t resetOutputChecksum = checksum(output);
    if (firstChecksum == glitchedChecksum || resetInputChecksum != resetOutputChecksum)
    {
        std::fprintf(
            stderr,
            "cuda smoke mismatch first=%llu glitched=%llu resetIn=%llu resetOut=%llu\n",
            static_cast<unsigned long long>(firstChecksum),
            static_cast<unsigned long long>(glitchedChecksum),
            static_cast<unsigned long long>(resetInputChecksum),
            static_cast<unsigned long long>(resetOutputChecksum));
        return 1;
    }

    params.inputFormat = 1;
    datamoshCudaReset(state);
    for (size_t index = 0; index < input.size(); ++index)
        input[index] = make_uchar4(17, 83, 211, 255);
    check(cudaMemcpy2DToArray(
              inputArray, 0, 0, input.data(), rowBytes, rowBytes, height, cudaMemcpyHostToDevice),
          "cudaMemcpy2DToArray rgba");
    check(datamoshCudaProcess(state, inputSurface, outputSurface, params, stream),
          "datamoshCudaProcess rgba");
    check(cudaMemcpy2DFromArrayAsync(
              output.data(),
              rowBytes,
              outputArray,
              0,
              0,
              rowBytes,
              height,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpy2DFromArrayAsync rgba");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize rgba");
    if (output[0].x != 211 || output[0].y != 83 || output[0].z != 17 || output[0].w != 255)
    {
        std::fprintf(
            stderr,
            "RGBA conversion mismatch: %u %u %u %u\n",
            output[0].x,
            output[0].y,
            output[0].z,
            output[0].w);
        return 1;
    }

    std::vector<ushort4> halfInput(width * height);
    const ushort4 halfPixel = make_ushort4(
        __half_as_ushort(__float2half(0.25f)),
        __half_as_ushort(__float2half(0.5f)),
        __half_as_ushort(__float2half(0.75f)),
        __half_as_ushort(__float2half(1.0f)));
    std::fill(halfInput.begin(), halfInput.end(), halfPixel);
    cudaChannelFormatDesc halfFormat =
        cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    cudaArray_t halfArray = nullptr;
    check(cudaMallocArray(
              &halfArray, &halfFormat, width, height, cudaArraySurfaceLoadStore),
          "cudaMallocArray half");
    cudaSurfaceObject_t halfSurface = createSurface(halfArray);
    const size_t halfRowBytes = width * sizeof(ushort4);
    check(cudaMemcpy2DToArray(
              halfArray,
              0,
              0,
              halfInput.data(),
              halfRowBytes,
              halfRowBytes,
              height,
              cudaMemcpyHostToDevice),
          "cudaMemcpy2DToArray half");
    params.inputFormat = 202;
    datamoshCudaReset(state);
    check(datamoshCudaProcess(state, halfSurface, outputSurface, params, stream),
          "datamoshCudaProcess half");
    check(cudaMemcpy2DFromArrayAsync(
              output.data(),
              rowBytes,
              outputArray,
              0,
              0,
              rowBytes,
              height,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpy2DFromArrayAsync half");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize half");
    if (output[0].x != 191 || output[0].y != 127 || output[0].z != 63 ||
        output[0].w != 255)
    {
        std::fprintf(
            stderr,
            "RGBA16F conversion mismatch: %u %u %u %u\n",
            output[0].x,
            output[0].y,
            output[0].z,
            output[0].w);
        return 1;
    }
    cudaDestroySurfaceObject(halfSurface);
    cudaFreeArray(halfArray);

    params.inputFormat = 0;
    params.intensity = 1.4f;
    constexpr int patternCount = 13;
    std::array<uint64_t, patternCount> patternChecksums = {};
    std::array<DirectionMetrics, patternCount> patternDirections = {};
    for (int pattern = 0; pattern < patternCount; ++pattern)
    {
        params.pattern = pattern;
        datamoshCudaReset(state);
        for (int frame = 0; frame < 16; ++frame)
        {
            fillFrame(input, width, height, frame + 30);
            check(cudaMemcpy2DToArrayAsync(
                      inputArray,
                      0,
                      0,
                      input.data(),
                      rowBytes,
                      rowBytes,
                      height,
                      cudaMemcpyHostToDevice,
                      stream),
                  "cudaMemcpy2DToArrayAsync pattern");
            check(datamoshCudaProcess(
                      state, inputSurface, outputSurface, params, stream),
                  "datamoshCudaProcess pattern");
        }
        check(cudaMemcpy2DFromArrayAsync(
                  output.data(),
                  rowBytes,
                  outputArray,
                  0,
                  0,
                  rowBytes,
                  height,
                  cudaMemcpyDeviceToHost,
                  stream),
              "cudaMemcpy2DFromArrayAsync pattern");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize pattern");
        patternChecksums[pattern] = checksum(output);
        patternDirections[pattern] = directionMetrics(input, output, width, height);
    }

    for (int left = 0; left < patternCount; ++left)
    {
        for (int right = left + 1; right < patternCount; ++right)
        {
            if (patternChecksums[left] == patternChecksums[right])
            {
                std::fprintf(
                    stderr,
                    "patterns %d and %d produced the same checksum\n",
                    left,
                    right);
                return 1;
            }
        }
    }

    double minimumRatio = 1.0e9;
    double maximumRatio = 0.0;
    int isotropicPatterns = 0;
    int directionalPatterns = 0;
    for (int pattern = 0; pattern < patternCount; ++pattern)
    {
        const DirectionMetrics& direction = patternDirections[pattern];
        double ratio = static_cast<double>(direction.horizontalVariation) /
                       std::max<uint64_t>(direction.verticalVariation, 1);
        minimumRatio = std::min(minimumRatio, ratio);
        maximumRatio = std::max(maximumRatio, ratio);
        if (ratio >= 0.8 && ratio <= 1.25)
            ++isotropicPatterns;
        if (ratio >= 1.5 || ratio <= 0.67)
            ++directionalPatterns;
        std::printf(
            "pattern %d checksum=%llu horizontal=%llu vertical=%llu ratio=%.3f\n",
            pattern,
            static_cast<unsigned long long>(patternChecksums[pattern]),
            static_cast<unsigned long long>(direction.horizontalVariation),
            static_cast<unsigned long long>(direction.verticalVariation),
            ratio);
    }
    if (maximumRatio < minimumRatio * 1.5 || isotropicPatterns == 0 ||
        directionalPatterns == 0)
    {
        std::fprintf(
            stderr,
            "directional diversity missing: min=%.3f max=%.3f isotropic=%d directional=%d\n",
            minimumRatio,
            maximumRatio,
            isotropicPatterns,
            directionalPatterns);
        return 1;
    }

    DatamoshCudaParams overrideParams;
    overrideParams.pattern = 8;
    overrideParams.intensity = 1.0f;
    overrideParams.motion = 1.0f;
    overrideParams.residual = 1.0f;
    overrideParams.temporal = 1.0f;
    overrideParams.bitstream = 1.0f;
    overrideParams.overrideMask = DATAMOSH_CUDA_OVERRIDE_ALL;
    overrideParams.mvScale = 1.4f;
    overrideParams.mvJitter = 3;
    overrideParams.vectorInterpolation = 0.6f;
    overrideParams.sampleAddressDesync = 2.0f;
    overrideParams.referenceLag = 3;
    overrideParams.referenceBleed = 0.45f;
    overrideParams.referenceLatchFrames = 4;
    overrideParams.temporalSliceDrift = 1;
    overrideParams.residualKeep = 0.35f;
    overrideParams.residualAddressJitter = 5;
    overrideParams.residualChannelShift = 1;
    overrideParams.entropySlipEvery = 3;
    overrideParams.entropySlipWindows = 4;
    overrideParams.coeffShift = 1;
    overrideParams.coeffQuant = 4;
    overrideParams.codebookReplaceEvery = 5;
    overrideParams.codebookStride = -3;
    overrideParams.codebookShuffleEvery = 7;
    datamoshCudaReset(state);
    for (int frame = 0; frame < 16; ++frame)
    {
        fillFrame(input, width, height, frame + 90);
        check(cudaMemcpy2DToArrayAsync(
                  inputArray,
                  0,
                  0,
                  input.data(),
                  rowBytes,
                  rowBytes,
                  height,
                  cudaMemcpyHostToDevice,
                  stream),
              "cudaMemcpy2DToArrayAsync overrides");
        check(datamoshCudaProcess(
                  state, inputSurface, outputSurface, overrideParams, stream),
              "datamoshCudaProcess overrides");
    }
    check(cudaMemcpy2DFromArrayAsync(
              output.data(),
              rowBytes,
              outputArray,
              0,
              0,
              rowBytes,
              height,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpy2DFromArrayAsync overrides");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize overrides");
    const uint64_t overrideChecksum = checksum(output);
    if (overrideChecksum == patternChecksums[8] ||
        overrideChecksum == checksum(input))
    {
        std::fprintf(
            stderr,
            "manual overrides did not alter clean decode: %llu\n",
            static_cast<unsigned long long>(overrideChecksum));
        return 1;
    }

    params.pattern = 0;
    params.vectorDecode = 4;
    params.intensity = 1.4f;
    params.residual = 1.0f;
    datamoshCudaReset(state);
    std::vector<uchar4> previous(width * height);
    fillFrame(previous, width, height, 70);
    check(cudaMemcpy2DToArrayAsync(
              inputArray,
              0,
              0,
              previous.data(),
              rowBytes,
              rowBytes,
              height,
              cudaMemcpyHostToDevice,
              stream),
          "cudaMemcpy2DToArrayAsync static previous");
    check(datamoshCudaProcess(
              state, inputSurface, outputSurface, params, stream),
          "datamoshCudaProcess static previous");
    fillFrame(input, width, height, 71);
    check(cudaMemcpy2DToArrayAsync(
              inputArray,
              0,
              0,
              input.data(),
              rowBytes,
              rowBytes,
              height,
              cudaMemcpyHostToDevice,
              stream),
          "cudaMemcpy2DToArrayAsync static current");
    check(datamoshCudaProcess(
              state, inputSurface, outputSurface, params, stream),
          "datamoshCudaProcess static current");
    check(cudaMemcpy2DFromArrayAsync(
              output.data(),
              rowBytes,
              outputArray,
              0,
              0,
              rowBytes,
              height,
              cudaMemcpyDeviceToHost,
              stream),
          "cudaMemcpy2DFromArrayAsync static");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize static");
    for (size_t pixel = 0; pixel < output.size(); ++pixel)
    {
        const uchar4 value = output[pixel];
        const uchar4 oldValue = previous[pixel];
        const uchar4 currentValue = input[pixel];
        const bool matchesOld =
            value.x == oldValue.x && value.y == oldValue.y && value.z == oldValue.z;
        const bool matchesCurrent =
            value.x == currentValue.x && value.y == currentValue.y &&
            value.z == currentValue.z;
        if (!matchesOld && !matchesCurrent)
        {
            std::fprintf(
                stderr,
                "static vector decode mixed another coordinate at pixel %llu\n",
                static_cast<unsigned long long>(pixel));
            return 1;
        }
    }

    std::printf(
        "cuda smoke ok: %dx%d frames=%d avg=%.3fms first=%llu glitched=%llu reset=%llu\n",
        width,
        height,
        frameCount,
        totalMilliseconds / frameCount,
        static_cast<unsigned long long>(firstChecksum),
        static_cast<unsigned long long>(glitchedChecksum),
        static_cast<unsigned long long>(resetOutputChecksum));

    datamoshCudaDestroy(state);
    cudaEventDestroy(started);
    cudaEventDestroy(finished);
    cudaStreamDestroy(stream);
    cudaDestroySurfaceObject(inputSurface);
    cudaDestroySurfaceObject(outputSurface);
    cudaFreeArray(inputArray);
    cudaFreeArray(outputArray);
    return 0;
}

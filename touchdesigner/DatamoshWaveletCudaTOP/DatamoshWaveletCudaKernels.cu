#include "DatamoshWaveletCudaCore.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <new>

struct DatamoshWaveletCudaState
{
    int width = 0;
    int height = 0;
    int levels = 0;
    int historyLength = 0;
    int pixels = 0;
    int bandsPerChannel = 0;
    uint64_t frameIndex = 0;

    int32_t* plane = nullptr;
    int32_t* work = nullptr;
    int16_t* coefficients = nullptr;
    int16_t* history = nullptr;
};

namespace {

constexpr int CHANNELS = 3;

struct BandDesc
{
    int channel;
    int level;
    int orientation;
    int originX;
    int originY;
    int width;
    int height;
    int index;
    bool valid;
};

__device__ __forceinline__ int positiveMod(int value, int modulus)
{
    int result = value % modulus;
    return result < 0 ? result + modulus : result;
}

__device__ __forceinline__ bool eventHit(uint64_t ordinal, int every)
{
    return every != 0 && ordinal % static_cast<uint64_t>(every) == 0;
}

__device__ __forceinline__ int roundDiv(int value, int divisor)
{
    if (divisor <= 1)
        return value;
    return value >= 0 ? (value + divisor / 2) / divisor : (value - divisor / 2) / divisor;
}

__device__ __forceinline__ int clampByte(int value)
{
    return max(0, min(255, value));
}

__device__ __forceinline__ int clampCoord(int value, int maximum)
{
    return max(0, min(value, maximum - 1));
}

// Returns BGRA (.x=B, .y=G, .z=R), matching TouchDesigner's CUDA TOP convention.
__device__ __forceinline__ uchar4 readInput(
    cudaSurfaceObject_t input, int x, int y, int width, int height, int format)
{
    x = clampCoord(x, width);
    y = clampCoord(y, height);
    if (format == 0)
    {
        uchar4 pixel;
        surf2Dread(&pixel, input, x * static_cast<int>(sizeof(uchar4)), y, cudaBoundaryModeClamp);
        return pixel;
    }
    if (format == 1)
    {
        uchar4 pixel;
        surf2Dread(&pixel, input, x * static_cast<int>(sizeof(uchar4)), y, cudaBoundaryModeClamp);
        return make_uchar4(pixel.z, pixel.y, pixel.x, pixel.w);
    }
    if (format == 102)
    {
        ushort4 pixel;
        surf2Dread(&pixel, input, x * static_cast<int>(sizeof(ushort4)), y, cudaBoundaryModeClamp);
        return make_uchar4(
            static_cast<unsigned char>(pixel.z >> 8),
            static_cast<unsigned char>(pixel.y >> 8),
            static_cast<unsigned char>(pixel.x >> 8),
            static_cast<unsigned char>(pixel.w >> 8));
    }
    if (format == 202)
    {
        ushort4 pixel;
        surf2Dread(&pixel, input, x * static_cast<int>(sizeof(ushort4)), y, cudaBoundaryModeClamp);
        float4 value = make_float4(
            __half2float(__ushort_as_half(pixel.x)),
            __half2float(__ushort_as_half(pixel.y)),
            __half2float(__ushort_as_half(pixel.z)),
            __half2float(__ushort_as_half(pixel.w)));
        return make_uchar4(
            static_cast<unsigned char>(max(0.0f, min(value.z, 1.0f)) * 255.0f),
            static_cast<unsigned char>(max(0.0f, min(value.y, 1.0f)) * 255.0f),
            static_cast<unsigned char>(max(0.0f, min(value.x, 1.0f)) * 255.0f),
            static_cast<unsigned char>(max(0.0f, min(value.w, 1.0f)) * 255.0f));
    }
    float4 pixel;
    surf2Dread(&pixel, input, x * static_cast<int>(sizeof(float4)), y, cudaBoundaryModeClamp);
    return make_uchar4(
        static_cast<unsigned char>(max(0.0f, min(pixel.z, 1.0f)) * 255.0f),
        static_cast<unsigned char>(max(0.0f, min(pixel.y, 1.0f)) * 255.0f),
        static_cast<unsigned char>(max(0.0f, min(pixel.x, 1.0f)) * 255.0f),
        static_cast<unsigned char>(max(0.0f, min(pixel.w, 1.0f)) * 255.0f));
}

__device__ __forceinline__ int3 rgbToYCoCg(uchar4 pixel)
{
    const int red = pixel.z;
    const int green = pixel.y;
    const int blue = pixel.x;
    const int co = red - blue;
    const int temporary = blue + (co >> 1);
    const int cg = green - temporary;
    const int yy = temporary + (cg >> 1);
    return make_int3(yy, co, cg);
}

__host__ __device__ __forceinline__ int lowLength(int value)
{
    return (value + 1) / 2;
}

__host__ __device__ __forceinline__ int highLength(int value)
{
    return value / 2;
}

__device__ __forceinline__ int quantStep(
    int quality, int levels, int channel, int level, int orientation)
{
    if (quality >= 100)
        return 1;
    const int qualityLoss = 101 - max(1, quality);
    const int base = 1 + qualityLoss * qualityLoss / 180;
    const int chroma = channel == 0 ? 1 : 2;
    if (orientation == 0)
        return max(1, base / 3) * chroma;
    const int fineScale = levels - level + 1;
    return max(1, base * fineScale * chroma);
}

__device__ BandDesc bandByIndex(
    int index, int width, int height, int levels, int bandsPerChannel)
{
    BandDesc band = {};
    const int totalBands = CHANNELS * bandsPerChannel;
    if (index < 0 || index >= totalBands)
    {
        band.valid = false;
        return band;
    }
    band.channel = index / bandsPerChannel;
    const int local = index % bandsPerChannel;
    band.index = index;
    band.valid = true;
    if (local == 0)
    {
        int activeWidth = width;
        int activeHeight = height;
        for (int level = 0; level < levels; ++level)
        {
            activeWidth = lowLength(activeWidth);
            activeHeight = lowLength(activeHeight);
        }
        band.level = levels;
        band.orientation = 0;
        band.originX = 0;
        band.originY = 0;
        band.width = activeWidth;
        band.height = activeHeight;
        return band;
    }

    band.level = (local - 1) / 3 + 1;
    band.orientation = (local - 1) % 3 + 1;
    int activeWidth = width;
    int activeHeight = height;
    for (int level = 1; level <= band.level; ++level)
    {
        const int lowWidth = lowLength(activeWidth);
        const int lowHeight = lowLength(activeHeight);
        const int highWidth = highLength(activeWidth);
        const int highHeight = highLength(activeHeight);
        if (level == band.level)
        {
            band.originX = band.orientation == 1 || band.orientation == 3 ? lowWidth : 0;
            band.originY = band.orientation == 2 || band.orientation == 3 ? lowHeight : 0;
            band.width = band.orientation == 1 || band.orientation == 3 ? highWidth : lowWidth;
            band.height = band.orientation == 2 || band.orientation == 3 ? highHeight : lowHeight;
            return band;
        }
        activeWidth = lowWidth;
        activeHeight = lowHeight;
    }
    band.valid = false;
    return band;
}

__device__ BandDesc bandAt(
    int x, int y, int channel, int width, int height, int levels, int bandsPerChannel)
{
    int activeWidth = width;
    int activeHeight = height;
    for (int level = 1; level <= levels; ++level)
    {
        const int lowWidth = lowLength(activeWidth);
        const int lowHeight = lowLength(activeHeight);
        if (x >= lowWidth || y >= lowHeight)
        {
            int orientation = 0;
            if (x >= lowWidth && y < lowHeight)
                orientation = 1;
            else if (x < lowWidth && y >= lowHeight)
                orientation = 2;
            else
                orientation = 3;
            const int local = 1 + (level - 1) * 3 + (orientation - 1);
            return bandByIndex(
                channel * bandsPerChannel + local,
                width,
                height,
                levels,
                bandsPerChannel);
        }
        activeWidth = lowWidth;
        activeHeight = lowHeight;
    }
    return bandByIndex(channel * bandsPerChannel, width, height, levels, bandsPerChannel);
}

__device__ __forceinline__ int matchingBandIndex(
    int channel, int level, int orientation, int levels, int bandsPerChannel)
{
    if (channel < 0 || channel >= CHANNELS)
        return -1;
    if (orientation == 0)
        return level == levels ? channel * bandsPerChannel : -1;
    if (level < 1 || level > levels || orientation < 1 || orientation > 3)
        return -1;
    return channel * bandsPerChannel + 1 + (level - 1) * 3 + orientation - 1;
}

__device__ __forceinline__ int rotatedOrientation(int orientation, int amount)
{
    if (orientation == 0)
        return 0;
    return positiveMod(orientation - 1 + amount, 3) + 1;
}

__device__ __forceinline__ int wrappedLevel(int level, int amount, int levels)
{
    return positiveMod(level - 1 + amount, levels) + 1;
}

__device__ __forceinline__ int clearLowBits(int value, int bits)
{
    bits = min(30, max(0, bits));
    if (bits == 0)
        return value;
    const int sign = value < 0 ? -1 : (value > 0 ? 1 : 0);
    unsigned int magnitude = static_cast<unsigned int>(abs(value));
    magnitude = (magnitude >> bits) << bits;
    return static_cast<int>(magnitude) * sign;
}

__device__ __forceinline__ int xorBitplane(int value, int bitplane)
{
    const int bit = min(30, max(0, bitplane - 1));
    const int sign = value < 0 ? -1 : 1;
    unsigned int magnitude = static_cast<unsigned int>(abs(value));
    magnitude ^= 1u << bit;
    return static_cast<int>(magnitude) * sign;
}

__device__ __forceinline__ bool historyAvailable(
    const DatamoshWaveletCudaState& state, int lag)
{
    return lag > 0 && lag <= state.historyLength &&
           state.frameIndex >= static_cast<uint64_t>(lag);
}

__device__ __forceinline__ const int16_t* historyPacket(
    const DatamoshWaveletCudaState& state, int lag)
{
    const uint64_t frame = state.frameIndex - static_cast<uint64_t>(lag);
    const int slot = static_cast<int>(frame % static_cast<uint64_t>(state.historyLength));
    return state.history + static_cast<size_t>(slot) * CHANNELS * state.pixels;
}

__global__ void kInputHorizontal(
    DatamoshWaveletCudaState state,
    cudaSurfaceObject_t input,
    int inputFormat)
{
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int pairs = state.width / 2;
    const int lowWidth = lowLength(state.width);
    if (pair >= lowWidth || y >= state.height)
        return;

    const int evenX = pair * 2;
    int3 even = rgbToYCoCg(readInput(
        input, evenX, y, state.width, state.height, inputFormat));
    int3 low = even;
    int3 high = make_int3(0, 0, 0);
    if (pair < pairs)
    {
        int3 odd = rgbToYCoCg(readInput(
            input, evenX + 1, y, state.width, state.height, inputFormat));
        high = make_int3(odd.x - even.x, odd.y - even.y, odd.z - even.z);
        low = make_int3(
            even.x + (high.x >> 1),
            even.y + (high.y >> 1),
            even.z + (high.z >> 1));
    }

    const int row = y * state.width;
    state.work[row + pair] = low.x;
    state.work[state.pixels + row + pair] = low.y;
    state.work[2 * state.pixels + row + pair] = low.z;
    if (pair < pairs)
    {
        state.work[row + lowWidth + pair] = high.x;
        state.work[state.pixels + row + lowWidth + pair] = high.y;
        state.work[2 * state.pixels + row + lowWidth + pair] = high.z;
    }
}

__global__ void kForwardHorizontal(
    DatamoshWaveletCudaState state, int activeWidth, int activeHeight)
{
    const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    const int lowWidth = lowLength(activeWidth);
    const int pairs = activeWidth / 2;
    if (outputX >= activeWidth || y >= activeHeight || channel >= CHANNELS)
        return;

    const int base = channel * state.pixels + y * state.width;
    if (outputX < pairs)
    {
        const int even = state.plane[base + outputX * 2];
        const int odd = state.plane[base + outputX * 2 + 1];
        const int high = odd - even;
        state.work[base + outputX] = even + (high >> 1);
    }
    else if (outputX < lowWidth)
    {
        state.work[base + outputX] = state.plane[base + activeWidth - 1];
    }
    else
    {
        const int pair = outputX - lowWidth;
        state.work[base + outputX] =
            state.plane[base + pair * 2 + 1] - state.plane[base + pair * 2];
    }
}

__global__ void kForwardVertical(
    DatamoshWaveletCudaState state, int activeWidth, int activeHeight)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    const int lowHeight = lowLength(activeHeight);
    const int pairs = activeHeight / 2;
    if (x >= activeWidth || outputY >= activeHeight || channel >= CHANNELS)
        return;

    const int base = channel * state.pixels;
    if (outputY < pairs)
    {
        const int even = state.work[base + outputY * 2 * state.width + x];
        const int odd = state.work[base + (outputY * 2 + 1) * state.width + x];
        const int high = odd - even;
        state.plane[base + outputY * state.width + x] = even + (high >> 1);
    }
    else if (outputY < lowHeight)
    {
        state.plane[base + outputY * state.width + x] =
            state.work[base + (activeHeight - 1) * state.width + x];
    }
    else
    {
        const int pair = outputY - lowHeight;
        state.plane[base + outputY * state.width + x] =
            state.work[base + (pair * 2 + 1) * state.width + x] -
            state.work[base + pair * 2 * state.width + x];
    }
}

__global__ void kQuantize(DatamoshWaveletCudaState state, int quality)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    if (x >= state.width || y >= state.height || channel >= CHANNELS)
        return;
    const BandDesc band = bandAt(
        x, y, channel, state.width, state.height, state.levels, state.bandsPerChannel);
    const int step = quantStep(quality, state.levels, channel, band.level, band.orientation);
    const int index = channel * state.pixels + y * state.width + x;
    const int coefficient = max(-32768, min(32767, roundDiv(state.plane[index], step)));
    state.coefficients[index] = static_cast<int16_t>(coefficient);
}

__global__ void kDecodePackets(
    DatamoshWaveletCudaState state, DatamoshWaveletCudaParams params)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    if (x >= state.width || y >= state.height || channel >= CHANNELS)
        return;

    const BandDesc destination = bandAt(
        x, y, channel, state.width, state.height, state.levels, state.bandsPerChannel);
    BandDesc source = destination;
    bool sourceFromHistory = false;
    int sourceHistoryLag = 0;
    const uint64_t ordinal =
        static_cast<uint64_t>(destination.index) + state.frameIndex;

    if (destination.orientation == 0 && params.lowpassHistoryLag != 0 &&
        historyAvailable(state, params.lowpassHistoryLag))
    {
        sourceFromHistory = true;
        sourceHistoryLag = params.lowpassHistoryLag;
    }
    else if (eventHit(ordinal, params.historyBandEvery) &&
             historyAvailable(state, params.historyLag))
    {
        sourceFromHistory = true;
        sourceHistoryLag = params.historyLag;
    }

    if (params.orientationRotate != 0 &&
        eventHit(ordinal, params.orientationRotateEvery))
    {
        const int index = matchingBandIndex(
            destination.channel,
            destination.level,
            rotatedOrientation(destination.orientation, params.orientationRotate),
            state.levels,
            state.bandsPerChannel);
        if (index >= 0)
        {
            source = bandByIndex(
                index,
                state.width,
                state.height,
                state.levels,
                state.bandsPerChannel);
            sourceFromHistory = false;
        }
    }

    if (params.levelFold != 0 && eventHit(ordinal, params.levelFoldEvery))
    {
        const int index = matchingBandIndex(
            destination.channel,
            wrappedLevel(destination.level, params.levelFold, state.levels),
            destination.orientation,
            state.levels,
            state.bandsPerChannel);
        if (index >= 0)
        {
            source = bandByIndex(
                index,
                state.width,
                state.height,
                state.levels,
                state.bandsPerChannel);
            sourceFromHistory = false;
        }
    }

    if (params.channelRoute != 0 && eventHit(ordinal, params.channelRouteEvery))
    {
        const int index = matchingBandIndex(
            positiveMod(destination.channel + params.channelRoute, CHANNELS),
            destination.level,
            destination.orientation,
            state.levels,
            state.bandsPerChannel);
        if (index >= 0)
        {
            source = bandByIndex(
                index,
                state.width,
                state.height,
                state.levels,
                state.bandsPerChannel);
            sourceFromHistory = false;
        }
    }

    if (params.packetShift != 0 && eventHit(ordinal, params.packetShiftEvery))
    {
        const int totalBands = CHANNELS * state.bandsPerChannel;
        const int index = positiveMod(destination.index + params.packetShift, totalBands);
        source = bandByIndex(
            index, state.width, state.height, state.levels, state.bandsPerChannel);
        sourceFromHistory = false;
    }

    if (eventHit(ordinal, params.packetLossEvery))
    {
        if (params.packetLossConceal && historyAvailable(state, params.historyLag))
        {
            source = destination;
            sourceFromHistory = true;
            sourceHistoryLag = params.historyLag;
        }
        else
        {
            state.plane[channel * state.pixels + y * state.width + x] = 0;
            return;
        }
    }

    const int localX = x - destination.originX;
    const int localY = y - destination.originY;
    const int sourceX = source.originX +
                        (source.width == destination.width
                             ? localX
                             : localX * source.width / max(1, destination.width));
    const int sourceY = source.originY +
                        (source.height == destination.height
                             ? localY
                             : localY * source.height / max(1, destination.height));
    const int sourceIndex =
        source.channel * state.pixels + sourceY * state.width + sourceX;
    const int16_t* sourcePacket = sourceFromHistory
                                      ? historyPacket(state, sourceHistoryLag)
                                      : state.coefficients;
    int value = static_cast<int>(sourcePacket[sourceIndex]);
    const uint64_t coefficientOrdinal =
        ordinal * 1000003ULL +
        static_cast<uint64_t>(localY * destination.width + localX);

    if (params.bitplaneClear != 0 &&
        eventHit(coefficientOrdinal, params.bitplaneClearEvery))
        value = clearLowBits(value, params.bitplaneClear);
    if (params.bitplaneXor != 0 &&
        eventHit(coefficientOrdinal, params.bitplaneXorEvery))
        value = xorBitplane(value, params.bitplaneXor);
    if (eventHit(coefficientOrdinal, params.signFlipEvery))
        value = -value;

    const int step = quantStep(
        params.quality,
        state.levels,
        destination.channel,
        destination.level,
        destination.orientation);
    int reconstructed = value * step;
    if (destination.orientation != 0 && params.liftingBias != 0 &&
        eventHit(coefficientOrdinal, params.liftingBiasEvery))
        reconstructed += params.liftingBias;
    state.plane[channel * state.pixels + y * state.width + x] = reconstructed;
}

__global__ void kInverseVertical(
    DatamoshWaveletCudaState state, int activeWidth, int activeHeight)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int outputY = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    if (x >= activeWidth || outputY >= activeHeight || channel >= CHANNELS)
        return;
    const int lowHeight = lowLength(activeHeight);
    const int pairs = activeHeight / 2;
    const int base = channel * state.pixels;
    if (outputY >= pairs * 2)
    {
        state.work[base + outputY * state.width + x] =
            state.plane[base + (lowHeight - 1) * state.width + x];
        return;
    }
    const int pair = outputY / 2;
    const int low = state.plane[base + pair * state.width + x];
    const int high = state.plane[base + (lowHeight + pair) * state.width + x];
    const int even = low - (high >> 1);
    state.work[base + outputY * state.width + x] =
        (outputY & 1) == 0 ? even : high + even;
}

__global__ void kInverseHorizontal(
    DatamoshWaveletCudaState state, int activeWidth, int activeHeight)
{
    const int outputX = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int channel = blockIdx.z;
    if (outputX >= activeWidth || y >= activeHeight || channel >= CHANNELS)
        return;
    const int lowWidth = lowLength(activeWidth);
    const int pairs = activeWidth / 2;
    const int base = channel * state.pixels + y * state.width;
    if (outputX >= pairs * 2)
    {
        state.plane[base + outputX] = state.work[base + lowWidth - 1];
        return;
    }
    const int pair = outputX / 2;
    const int low = state.work[base + pair];
    const int high = state.work[base + lowWidth + pair];
    const int even = low - (high >> 1);
    state.plane[base + outputX] = (outputX & 1) == 0 ? even : high + even;
}

__global__ void kOutputHorizontal(
    DatamoshWaveletCudaState state, cudaSurfaceObject_t output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= state.width || y >= state.height)
        return;
    const int lowWidth = lowLength(state.width);
    const int pairs = state.width / 2;
    int values[CHANNELS];
    for (int channel = 0; channel < CHANNELS; ++channel)
    {
        const int base = channel * state.pixels + y * state.width;
        if (x >= pairs * 2)
        {
            values[channel] = state.work[base + lowWidth - 1];
        }
        else
        {
            const int pair = x / 2;
            const int low = state.work[base + pair];
            const int high = state.work[base + lowWidth + pair];
            const int even = low - (high >> 1);
            values[channel] = (x & 1) == 0 ? even : high + even;
        }
    }
    const int temporary = values[0] - (values[2] >> 1);
    const int green = values[2] + temporary;
    const int blue = temporary - (values[1] >> 1);
    const int red = values[1] + blue;
    const uchar4 pixel = make_uchar4(
        static_cast<unsigned char>(clampByte(blue)),
        static_cast<unsigned char>(clampByte(green)),
        static_cast<unsigned char>(clampByte(red)),
        255);
    surf2Dwrite(pixel, output, x * static_cast<int>(sizeof(uchar4)), y);
}

cudaError_t allocateDevice(void** pointer, size_t bytes)
{
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess)
        *pointer = nullptr;
    return status;
}

int maxLevels(int width, int height)
{
    int levels = 0;
    while (width > 1 && height > 1)
    {
        ++levels;
        width = lowLength(width);
        height = lowLength(height);
    }
    return levels;
}

} // namespace

cudaError_t datamoshWaveletCudaCreate(
    DatamoshWaveletCudaState** state,
    int width,
    int height,
    int levels,
    int historyLength)
{
    if (!state || width <= 0 || height <= 0 || levels <= 0 ||
        levels > maxLevels(width, height) || levels >= 32 || historyLength <= 0)
        return cudaErrorInvalidValue;

    DatamoshWaveletCudaState* created = new (std::nothrow) DatamoshWaveletCudaState;
    if (!created)
        return cudaErrorMemoryAllocation;
    created->width = width;
    created->height = height;
    created->levels = levels;
    created->historyLength = historyLength;
    created->pixels = width * height;
    created->bandsPerChannel = 1 + levels * 3;

    const size_t planeValues = static_cast<size_t>(CHANNELS) * created->pixels;
    cudaError_t status = cudaSuccess;
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->plane), planeValues * sizeof(int32_t));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->work), planeValues * sizeof(int32_t));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->coefficients), planeValues * sizeof(int16_t));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->history),
            planeValues * static_cast<size_t>(historyLength) * sizeof(int16_t));

    if (status != cudaSuccess)
    {
        datamoshWaveletCudaDestroy(created);
        return status;
    }
    *state = created;
    return cudaSuccess;
}

void datamoshWaveletCudaDestroy(DatamoshWaveletCudaState* state)
{
    if (!state)
        return;
    cudaFree(state->plane);
    cudaFree(state->work);
    cudaFree(state->coefficients);
    cudaFree(state->history);
    delete state;
}

void datamoshWaveletCudaReset(DatamoshWaveletCudaState* state)
{
    if (state)
        state->frameIndex = 0;
}

cudaError_t datamoshWaveletCudaProcess(
    DatamoshWaveletCudaState* state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    const DatamoshWaveletCudaParams& params,
    cudaStream_t stream)
{
    if (!state || !input || !output || params.levels != state->levels ||
        params.historyLength != state->historyLength)
        return cudaErrorInvalidValue;

    dim3 block(16, 16);
    dim3 inputGrid(
        (lowLength(state->width) + block.x - 1) / block.x,
        (state->height + block.y - 1) / block.y);
    kInputHorizontal<<<inputGrid, block, 0, stream>>>(*state, input, params.inputFormat);

    int activeWidth = state->width;
    int activeHeight = state->height;
    for (int level = 1; level <= state->levels; ++level)
    {
        if (level > 1)
        {
            dim3 horizontalGrid(
                (activeWidth + block.x - 1) / block.x,
                (activeHeight + block.y - 1) / block.y,
                CHANNELS);
            kForwardHorizontal<<<horizontalGrid, block, 0, stream>>>(
                *state, activeWidth, activeHeight);
        }
        dim3 verticalGrid(
            (activeWidth + block.x - 1) / block.x,
            (activeHeight + block.y - 1) / block.y,
            CHANNELS);
        kForwardVertical<<<verticalGrid, block, 0, stream>>>(
            *state, activeWidth, activeHeight);
        activeWidth = lowLength(activeWidth);
        activeHeight = lowLength(activeHeight);
    }

    dim3 fullGrid(
        (state->width + block.x - 1) / block.x,
        (state->height + block.y - 1) / block.y,
        CHANNELS);
    kQuantize<<<fullGrid, block, 0, stream>>>(*state, params.quality);
    kDecodePackets<<<fullGrid, block, 0, stream>>>(*state, params);

    int dimensionsWidth[32] = {};
    int dimensionsHeight[32] = {};
    dimensionsWidth[0] = state->width;
    dimensionsHeight[0] = state->height;
    for (int level = 1; level <= state->levels; ++level)
    {
        dimensionsWidth[level] = lowLength(dimensionsWidth[level - 1]);
        dimensionsHeight[level] = lowLength(dimensionsHeight[level - 1]);
    }
    for (int level = state->levels; level >= 1; --level)
    {
        activeWidth = dimensionsWidth[level - 1];
        activeHeight = dimensionsHeight[level - 1];
        dim3 verticalGrid(
            (activeWidth + block.x - 1) / block.x,
            (activeHeight + block.y - 1) / block.y,
            CHANNELS);
        kInverseVertical<<<verticalGrid, block, 0, stream>>>(
            *state, activeWidth, activeHeight);
        if (level > 1)
        {
            dim3 horizontalGrid(
                (activeWidth + block.x - 1) / block.x,
                (activeHeight + block.y - 1) / block.y,
                CHANNELS);
            kInverseHorizontal<<<horizontalGrid, block, 0, stream>>>(
                *state, activeWidth, activeHeight);
        }
    }

    dim3 outputGrid(
        (state->width + block.x - 1) / block.x,
        (state->height + block.y - 1) / block.y);
    kOutputHorizontal<<<outputGrid, block, 0, stream>>>(*state, output);

    const size_t packetBytes =
        static_cast<size_t>(CHANNELS) * state->pixels * sizeof(int16_t);
    const int historySlot =
        static_cast<int>(state->frameIndex % static_cast<uint64_t>(state->historyLength));
    cudaMemcpyAsync(
        state->history + static_cast<size_t>(historySlot) * CHANNELS * state->pixels,
        state->coefficients,
        packetBytes,
        cudaMemcpyDeviceToDevice,
        stream);

    ++state->frameIndex;
    return cudaPeekAtLastError();
}

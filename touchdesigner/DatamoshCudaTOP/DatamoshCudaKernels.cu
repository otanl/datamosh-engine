#include "DatamoshCudaCore.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <new>

struct DatamoshCudaState
{
    int width = 0;
    int height = 0;
    int blockSize = 0;
    int blocksX = 0;
    int blocksY = 0;
    int historySlots = 0;
    uint64_t frameIndex = 0;
    uchar4* cleanHistory = nullptr;
    uchar4* dirtyHistory = nullptr;
    short4* residual = nullptr;
    short4* residualHistory = nullptr;
    short2* motionVectors = nullptr;
};

namespace {

__device__ __forceinline__ int clampCoord(int value, int maximum)
{
    return max(0, min(value, maximum - 1));
}

__device__ __forceinline__ int wrapIndex(int value, int count)
{
    int wrapped = value % count;
    return wrapped < 0 ? wrapped + count : wrapped;
}

__device__ __forceinline__ int wrapCoord(int value, int count)
{
    return wrapIndex(value, count);
}

__device__ __forceinline__ int pixelIndex2D(int x, int y, int width, int height)
{
    return wrapCoord(y, height) * width + wrapCoord(x, width);
}

__device__ __forceinline__ unsigned hashValue(unsigned value)
{
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

__device__ __forceinline__ int luma(uchar4 pixel)
{
    return (29 * static_cast<int>(pixel.x) + 150 * static_cast<int>(pixel.y) +
            77 * static_cast<int>(pixel.z)) >>
           8;
}

__device__ __forceinline__ uchar4 readInput(
    cudaSurfaceObject_t input,
    int x,
    int y,
    int width,
    int height,
    int format)
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
        surf2Dread(
            &pixel,
            input,
            x * static_cast<int>(sizeof(ushort4)),
            y,
            cudaBoundaryModeClamp);
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

__device__ __forceinline__ uchar4 readHistory(
    const uchar4* history,
    int slot,
    int x,
    int y,
    int width,
    int height)
{
    x = clampCoord(x, width);
    y = clampCoord(y, height);
    return history[(slot * height + y) * width + x];
}

__device__ __forceinline__ int historySlot(
    uint64_t frameIndex,
    int available,
    int requestedLag,
    int slots)
{
    int lag = max(1, min(requestedLag, available));
    return static_cast<int>((frameIndex + slots - lag) % slots);
}

__device__ __forceinline__ short2 residualPredictorMotion(
    DatamoshCudaState state,
    DatamoshCudaParams params,
    short2 cleanMotion,
    int blockX,
    int blockY)
{
    int motionMagnitude =
        max(1,
            min(params.searchRadius,
                abs(static_cast<int>(cleanMotion.x)) +
                    abs(static_cast<int>(cleanMotion.y))));
    int motionStrength =
        max(1, static_cast<int>(motionMagnitude * params.motion * params.intensity));

    if (params.vectorDecode == 1)
        return cleanMotion;
    if (params.vectorDecode == 2)
    {
        return make_short2(
            static_cast<short>(-cleanMotion.x),
            static_cast<short>(-cleanMotion.y));
    }
    if (params.vectorDecode == 3)
    {
        int direction = ((blockX + blockY) & 1) ? 1 : -1;
        return make_short2(0, static_cast<short>(direction * motionStrength));
    }
    if (params.vectorDecode == 4)
        return make_short2(0, 0);
    if (params.vectorDecode == 5)
    {
        int centerX = state.blocksX / 2;
        int centerY = state.blocksY / 2;
        int distanceX = blockX - centerX;
        int distanceY = blockY - centerY;
        if (abs(distanceX) >= abs(distanceY))
        {
            return make_short2(
                static_cast<short>((distanceX >= 0 ? 1 : -1) * motionStrength),
                0);
        }
        return make_short2(
            0,
            static_cast<short>((distanceY >= 0 ? 1 : -1) * motionStrength));
    }

    return params.pattern == 0 ? cleanMotion : make_short2(0, 0);
}

__global__ void initializeState(
    DatamoshCudaState state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    int inputFormat)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= state.width || y >= state.height)
        return;

    uchar4 pixel = readInput(input, x, y, state.width, state.height, inputFormat);
    surf2Dwrite(pixel, output, x * static_cast<int>(sizeof(uchar4)), y);
    int index = y * state.width + x;
    for (int slot = 0; slot < state.historySlots; ++slot)
    {
        state.cleanHistory[slot * state.width * state.height + index] = pixel;
        state.dirtyHistory[slot * state.width * state.height + index] = pixel;
        state.residualHistory[slot * state.width * state.height + index] =
            make_short4(0, 0, 0, 0);
    }
    state.residual[index] = make_short4(0, 0, 0, 0);
}

__global__ void encodeMotion(
    DatamoshCudaState state,
    cudaSurfaceObject_t input,
    DatamoshCudaParams params,
    int referenceSlot)
{
    int blockX = blockIdx.x * blockDim.x + threadIdx.x;
    int blockY = blockIdx.y * blockDim.y + threadIdx.y;
    if (blockX >= state.blocksX || blockY >= state.blocksY)
        return;

    int originX = blockX * state.blockSize;
    int originY = blockY * state.blockSize;
    int blockWidth = min(state.blockSize, state.width - originX);
    int blockHeight = min(state.blockSize, state.height - originY);
    int sampleStep = max(state.blockSize / 4, 2);
    int searchStep = max(params.searchStep, 1);
    int bestError = 0x7fffffff;
    int bestCost = 0x7fffffff;
    short2 best = make_short2(0, 0);

    for (int dy = -params.searchRadius; dy <= params.searchRadius; dy += searchStep)
    {
        for (int dx = -params.searchRadius; dx <= params.searchRadius; dx += searchStep)
        {
            int error = 0;
            for (int by = 0; by < blockHeight; by += sampleStep)
            {
                for (int bx = 0; bx < blockWidth; bx += sampleStep)
                {
                    uchar4 current =
                        readInput(
                            input,
                            originX + bx,
                            originY + by,
                            state.width,
                            state.height,
                            params.inputFormat);
                    uchar4 reference = readHistory(
                        state.cleanHistory,
                        referenceSlot,
                        originX + bx + dx,
                        originY + by + dy,
                        state.width,
                        state.height);
                    error += abs(luma(current) - luma(reference));
                }
            }
            int cost = abs(dx) + abs(dy);
            if (error < bestError || (error == bestError && cost < bestCost))
            {
                bestError = error;
                bestCost = cost;
                best = make_short2(static_cast<short>(dx), static_cast<short>(dy));
            }
        }
    }

    state.motionVectors[blockY * state.blocksX + blockX] = best;
}

__global__ void encodeResidual(
    DatamoshCudaState state,
    cudaSurfaceObject_t input,
    DatamoshCudaParams params,
    int referenceSlot)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= state.width || y >= state.height)
        return;

    int blockX = min(x / state.blockSize, state.blocksX - 1);
    int blockY = min(y / state.blockSize, state.blocksY - 1);
    short2 cleanMotion = state.motionVectors[blockY * state.blocksX + blockX];
    short2 residualMotion =
        residualPredictorMotion(state, params, cleanMotion, blockX, blockY);
    uchar4 current =
        readInput(input, x, y, state.width, state.height, params.inputFormat);
    uchar4 prediction = readHistory(
        state.cleanHistory,
        referenceSlot,
        x + residualMotion.x,
        y + residualMotion.y,
        state.width,
        state.height);
    state.residual[y * state.width + x] = make_short4(
        static_cast<short>(static_cast<int>(current.x) - static_cast<int>(prediction.x)),
        static_cast<short>(static_cast<int>(current.y) - static_cast<int>(prediction.y)),
        static_cast<short>(static_cast<int>(current.z) - static_cast<int>(prediction.z)),
        0);
}

__device__ __forceinline__ short readByteSlippedResidual(
    const short4* residual,
    int pixelCount,
    int pixel,
    int channel,
    int slip)
{
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(residual);
    int byteCount = pixelCount * static_cast<int>(sizeof(short4));
    int address = wrapIndex(
        pixel * static_cast<int>(sizeof(short4)) + channel * 2 + slip,
        byteCount);
    int low = bytes[address];
    int high = bytes[wrapIndex(address + 1, byteCount)];
    return static_cast<short>(low | (high << 8));
}

__global__ void decodeFrame(
    DatamoshCudaState state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    DatamoshCudaParams params,
    int currentSlot,
    int availableHistory)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= state.width || y >= state.height)
        return;

    int pixel = y * state.width + x;
    int blockX = min(x / state.blockSize, state.blocksX - 1);
    int blockY = min(y / state.blockSize, state.blocksY - 1);
    int blockIndex = blockY * state.blocksX + blockX;
    unsigned noise = hashValue(
        static_cast<unsigned>(pixel) ^ static_cast<unsigned>(state.frameIndex * 977) ^
        params.seed);

    short2 cleanMotion = state.motionVectors[blockIndex];
    int motionMagnitude =
        max(1,
            min(params.searchRadius,
                abs(static_cast<int>(cleanMotion.x)) +
                    abs(static_cast<int>(cleanMotion.y))));
    int motionStrength =
        max(1, static_cast<int>(motionMagnitude * params.motion * params.intensity));
    short2 motion = cleanMotion;
    if (params.pattern == 1)
    {
        int verticalDirection =
            ((blockX + static_cast<int>(state.frameIndex / 4)) & 1) ? 1 : -1;
        motion =
            make_short2(0, static_cast<short>(verticalDirection * motionStrength));
    }
    else if (params.pattern == 2)
    {
        motion = make_short2(0, 0);
    }
    else if (params.pattern == 3)
    {
        int verticalDirection =
            ((blockX + blockY + static_cast<int>(state.frameIndex / 5)) & 1) ? 1 : -1;
        motion =
            make_short2(0, static_cast<short>(verticalDirection * motionStrength));
    }
    else if (params.pattern == 4)
    {
        int stride = max(1, static_cast<int>(params.motion * params.intensity * 3.0f));
        int direction = static_cast<int>((noise >> 2) & 3U);
        int sourceBlockX = blockX;
        int sourceBlockY = blockY;
        if (direction == 0)
            sourceBlockY += stride;
        else if (direction == 1)
            sourceBlockX -= stride;
        else if (direction == 2)
            sourceBlockY -= stride;
        else
            sourceBlockX += stride;
        int vectorIndex =
            wrapCoord(sourceBlockY, state.blocksY) * state.blocksX +
            wrapCoord(sourceBlockX, state.blocksX);
        short2 bankVector = state.motionVectors[vectorIndex];
        int bankMagnitude =
            max(1,
                min(params.searchRadius,
                    abs(static_cast<int>(bankVector.x)) +
                        abs(static_cast<int>(bankVector.y))));
        int bankStrength =
            max(1, static_cast<int>(bankMagnitude * params.motion * params.intensity));
        if (direction == 0)
            motion = make_short2(0, static_cast<short>(bankStrength));
        else if (direction == 1)
            motion = make_short2(static_cast<short>(-bankStrength), 0);
        else if (direction == 2)
            motion = make_short2(0, static_cast<short>(-bankStrength));
        else
            motion = make_short2(static_cast<short>(bankStrength), 0);
    }
    else if (params.pattern == 5)
    {
        motion = make_short2(0, 0);
    }
    else if (params.pattern == 6)
    {
        int diagonalDirection =
            ((blockX ^ blockY ^ static_cast<int>(state.frameIndex / 6)) & 1) ? 1 : -1;
        motion = make_short2(
            static_cast<short>(diagonalDirection * motionStrength),
            static_cast<short>(-diagonalDirection * motionStrength));
    }
    else if (params.pattern == 7)
    {
        int radius = max(1, static_cast<int>(params.motion * params.intensity * 4.0f));
        int sourceBlockX =
            blockX + static_cast<int>((noise >> 3) % (radius * 2 + 1)) - radius;
        int sourceBlockY =
            blockY + static_cast<int>((noise >> 11) % (radius * 2 + 1)) - radius;
        int vectorIndex =
            wrapCoord(sourceBlockY, state.blocksY) * state.blocksX +
            wrapCoord(sourceBlockX, state.blocksX);
        short2 bankVector = state.motionVectors[vectorIndex];
        int bankMagnitude =
            max(1,
                min(params.searchRadius,
                    abs(static_cast<int>(bankVector.x)) +
                        abs(static_cast<int>(bankVector.y))));
        int bankStrength =
            max(1, static_cast<int>(bankMagnitude * params.motion * params.intensity));
        int orientation = static_cast<int>((noise >> 19) & 3U);
        if (orientation == 0)
            motion = make_short2(0, static_cast<short>(bankStrength));
        else if (orientation == 1)
            motion = make_short2(static_cast<short>(-bankStrength), 0);
        else if (orientation == 2)
            motion = make_short2(0, static_cast<short>(-bankStrength));
        else
            motion = make_short2(static_cast<short>(bankStrength), 0);
    }
    else if (params.pattern == 9 || params.pattern == 10 || params.pattern == 11)
    {
        motion = make_short2(0, 0);
    }
    else if (params.pattern == 12)
    {
        int direction =
            ((y / max(1, state.blockSize / 2) +
              static_cast<int>(state.frameIndex / 3)) &
             1)
                ? 1
                : -1;
        motion = make_short2(0, static_cast<short>(direction * motionStrength));
    }

    short2 residualMotion =
        residualPredictorMotion(state, params, cleanMotion, blockX, blockY);
    if (params.vectorDecode != 0)
        motion = residualMotion;

    int temporalSpan =
        max(1, min(availableHistory, 1 + static_cast<int>(params.temporal * params.intensity * 4)));
    int referenceLag = 1;
    if (params.pattern == 0)
        referenceLag = temporalSpan;
    else if (params.pattern == 1)
        referenceLag = 1 + wrapIndex(x / max(1, state.blockSize / 2) +
                                         static_cast<int>(state.frameIndex),
                                     temporalSpan);
    else if (params.pattern == 7)
    {
        unsigned cellNoise = hashValue(
            static_cast<unsigned>(blockX * 73856093) ^
            static_cast<unsigned>(blockY * 19349663) ^
            static_cast<unsigned>(state.frameIndex / 2) ^ params.seed);
        referenceLag = 1 + static_cast<int>(cellNoise % temporalSpan);
    }
    else if (params.pattern == 11)
    {
        unsigned packetNoise = hashValue(
            static_cast<unsigned>(blockIndex * 2654435761U) ^
            static_cast<unsigned>(state.frameIndex / 2) ^ params.seed);
        referenceLag = 1 + static_cast<int>(packetNoise % temporalSpan);
    }
    else if (params.pattern == 12)
    {
        int weaveCell =
            (x / max(1, state.blockSize) +
             y / max(1, state.blockSize / 2) +
             static_cast<int>(state.frameIndex)) %
            temporalSpan;
        referenceLag = 1 + weaveCell;
    }

    int dirtySlot =
        historySlot(state.frameIndex, availableHistory, referenceLag, state.historySlots);
    short4 residual = state.residual[pixel];
    int activity =
        (abs(static_cast<int>(residual.x)) + abs(static_cast<int>(residual.y)) +
         abs(static_cast<int>(residual.z))) /
            3 +
        (abs(static_cast<int>(cleanMotion.x)) + abs(static_cast<int>(cleanMotion.y))) * 4;
    int threshold = max(2, 18 - static_cast<int>(params.intensity * 6.0f));
    // Pattern 8 = "clean": disable all glitch so the decoder does plain motion-compensated
    // reconstruction. Output should equal the input (1-frame pipeline delay). This is the
    // controlled A/B test against the CPU codec's clean reconstruction: if this still
    // scrolls, the bug is in the core pipeline (surface I/O / history indexing / residual),
    // not the glitch model.
    bool corrupt = (activity >= threshold) && params.pattern != 8;
    bool packetLost = false;

    int residualPixel = pixel;
    if (params.pattern == 3 && corrupt)
    {
        int shift = max(1, static_cast<int>(params.residual * params.intensity * 13.0f));
        int direction = (blockX + blockY + static_cast<int>(state.frameIndex / 4)) & 3;
        int residualX = x;
        int residualY = y;
        if (direction == 0)
            residualY += shift;
        else if (direction == 1)
        {
            residualX -= shift / 3;
            residualY -= shift;
        }
        else if (direction == 2)
        {
            residualX += shift / 2;
            residualY += shift;
        }
        else
            residualY -= shift;
        residualPixel = pixelIndex2D(
            residualX,
            residualY,
            state.width,
            state.height);
        residual = state.residual[residualPixel];
    }
    else if (params.pattern == 5 && corrupt)
    {
        int slip = max(1, static_cast<int>(params.bitstream * params.intensity * 7.0f));
        int scanMode =
            (blockX + blockY + static_cast<int>(state.frameIndex / 3)) & 3;
        int streamPixel = pixel;
        if (scanMode == 0)
            streamPixel = wrapIndex(x * state.height + y, state.width * state.height);
        else if (scanMode == 1)
            streamPixel = pixelIndex2D(
                state.width - 1 - x,
                y,
                state.width,
                state.height);
        else if (scanMode == 2)
            streamPixel = wrapIndex(
                (state.width - 1 - x) * state.height + y,
                state.width * state.height);
        else
            streamPixel = pixelIndex2D(
                x,
                state.height - 1 - y,
                state.width,
                state.height);
        int signedSlip = (noise & 1U) ? slip : -slip;
        residual.x = readByteSlippedResidual(
            state.residual,
            state.width * state.height,
            streamPixel,
            0,
            signedSlip);
        residual.y = readByteSlippedResidual(
            state.residual,
            state.width * state.height,
            streamPixel,
            1,
            -signedSlip);
        residual.z = readByteSlippedResidual(
            state.residual,
            state.width * state.height,
            streamPixel,
            2,
            signedSlip + ((scanMode & 1) ? 2 : -2));
    }
    else if (params.pattern == 6 && corrupt)
    {
        int lag = 1 + static_cast<int>(noise % max(1, availableHistory));
        int slot = historySlot(state.frameIndex, availableHistory, lag, state.historySlots);
        int tileSize = max(4, min(state.blockSize, 16));
        int tileX = x / tileSize;
        int tileY = y / tileSize;
        int localX = x % tileSize;
        int localY = y % tileSize;
        int orientation = static_cast<int>((noise >> 8) & 3U);
        int sourceLocalX = localX;
        int sourceLocalY = localY;
        if (orientation == 1)
        {
            sourceLocalX = localY;
            sourceLocalY = tileSize - 1 - localX;
        }
        else if (orientation == 2)
        {
            sourceLocalX = tileSize - 1 - localX;
            sourceLocalY = tileSize - 1 - localY;
        }
        else if (orientation == 3)
        {
            sourceLocalX = tileSize - 1 - localY;
            sourceLocalY = localX;
        }
        int tileStride = max(1, static_cast<int>(params.bitstream * params.intensity * 3.0f));
        int sourceTileX = tileX + ((noise & 1U) ? tileStride : -tileStride);
        int sourceTileY = tileY + ((noise & 2U) ? -tileStride : tileStride);
        int bankX = sourceTileX * tileSize + sourceLocalX;
        int bankY = sourceTileY * tileSize + sourceLocalY;
        int bankPixel = pixelIndex2D(bankX, bankY, state.width, state.height);
        residual = state.residualHistory[slot * state.width * state.height + bankPixel];
    }
    else if (params.pattern == 7 && corrupt)
    {
        int radius = max(1, static_cast<int>(params.residual * params.intensity * 9.0f));
        int residualX =
            x + static_cast<int>((noise >> 5) % (radius * 2 + 1)) - radius;
        int residualY =
            y + static_cast<int>((noise >> 13) % (radius * 2 + 1)) - radius;
        if (noise & 0x20000U)
        {
            int localX = residualX - blockX * state.blockSize;
            int localY = residualY - blockY * state.blockSize;
            residualX = blockX * state.blockSize + localY;
            residualY = blockY * state.blockSize + state.blockSize - 1 - localX;
        }
        residualPixel =
            pixelIndex2D(residualX, residualY, state.width, state.height);
        residual = state.residual[residualPixel];
    }
    else if (params.pattern == 9 && corrupt)
    {
        int pitchError =
            max(1,
                static_cast<int>(
                    state.blockSize * params.bitstream * params.intensity));
        int pitchDirection =
            ((blockY + static_cast<int>(state.frameIndex / 3)) & 1) ? 1 : -1;
        int decodedPitch = max(1, state.width + pitchDirection * pitchError);
        residualPixel = wrapIndex(
            y * decodedPitch + x,
            state.width * state.height);
        residual = state.residual[residualPixel];
    }
    else if (params.pattern == 10 && corrupt)
    {
        int assumedBlock =
            ((blockX + blockY + static_cast<int>(state.frameIndex / 4)) & 1)
                ? max(4, state.blockSize / 2)
                : min(64, state.blockSize * 2);
        int assumedBlockX = x / assumedBlock;
        int assumedBlockY = y / assumedBlock;
        int assumedLocalX = x % assumedBlock;
        int assumedLocalY = y % assumedBlock;
        int sourceX = assumedBlockX * state.blockSize +
                      assumedLocalX * state.blockSize / assumedBlock;
        int sourceY = assumedBlockY * state.blockSize +
                      assumedLocalY * state.blockSize / assumedBlock;
        if ((noise >> 7) & 1U)
        {
            int localX = sourceX % state.blockSize;
            int localY = sourceY % state.blockSize;
            sourceX = sourceX - localX + localY;
            sourceY = sourceY - localY + state.blockSize - 1 - localX;
        }
        residualPixel =
            pixelIndex2D(sourceX, sourceY, state.width, state.height);
        residual = state.residual[residualPixel];
    }
    else if (params.pattern == 11 && corrupt)
    {
        unsigned packetNoise = hashValue(
            static_cast<unsigned>(blockIndex * 2246822519U) ^
            static_cast<unsigned>(state.frameIndex / 2) ^ params.seed);
        int lossPeriod =
            max(2, 8 - static_cast<int>(params.bitstream * params.intensity * 2.0f));
        packetLost = static_cast<int>(packetNoise % lossPeriod) == 0;
        if (packetLost)
        {
            if (packetNoise & 1U)
            {
                residual = make_short4(0, 0, 0, 0);
            }
            else
            {
                int lag =
                    1 + static_cast<int>((packetNoise >> 8) % max(1, availableHistory));
                int slot =
                    historySlot(state.frameIndex, availableHistory, lag, state.historySlots);
                int packetOffsetX =
                    ((packetNoise >> 16) & 1U) ? state.blockSize : -state.blockSize;
                int sourcePixel = pixelIndex2D(
                    x + packetOffsetX,
                    y,
                    state.width,
                    state.height);
                residual =
                    state.residualHistory[slot * state.width * state.height + sourcePixel];
            }
        }
    }

    float residualKeep = 1.0f;
    if (corrupt)
    {
        if (params.pattern == 0 || params.pattern == 1)
            residualKeep = max(0.0f, 0.35f - params.residual * params.intensity * 0.3f);
        else if (params.pattern == 7)
            residualKeep = 0.15f;
        else if (params.pattern == 10)
            residualKeep = 0.7f;
        else if (params.pattern == 11 && packetLost)
            residualKeep = 0.0f;
        else if (params.pattern == 12)
            residualKeep = 0.12f;
    }

    uchar4 current =
        readInput(input, x, y, state.width, state.height, params.inputFormat);
    uchar4 dirtyPrediction = readHistory(
        state.dirtyHistory,
        dirtySlot,
        x + motion.x,
        y + motion.y,
        state.width,
        state.height);
    short2 baseMotion = residualMotion;
    if (corrupt && params.vectorDecode == 0 && params.pattern == 3)
    {
        baseMotion = motion;
    }
    else if (corrupt && params.vectorDecode == 0 && params.pattern == 5)
    {
        baseMotion = make_short2(0, 0);
    }
    else if (corrupt && params.vectorDecode == 0 && params.pattern == 6)
    {
        baseMotion = motion;
    }
    uchar4 cleanPrediction = readHistory(
        state.cleanHistory,
        historySlot(state.frameIndex, availableHistory, 1, state.historySlots),
        x + baseMotion.x,
        y + baseMotion.y,
        state.width,
        state.height);
    bool recursivePrediction =
        params.pattern == 0 || params.pattern == 1 || params.pattern == 2 ||
        params.pattern == 4 || params.pattern == 7 || params.pattern == 12 ||
        (params.pattern == 11 && packetLost);
    uchar4 prediction =
        corrupt && recursivePrediction ? dirtyPrediction : cleanPrediction;

    int values[3] = {
        static_cast<int>(prediction.x) + static_cast<int>(residual.x * residualKeep),
        static_cast<int>(prediction.y) + static_cast<int>(residual.y * residualKeep),
        static_cast<int>(prediction.z) + static_cast<int>(residual.z * residualKeep),
    };

    if (params.pattern == 2 && corrupt)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            int lag = 1 + wrapIndex(channel + static_cast<int>(state.frameIndex), temporalSpan);
            int slot = historySlot(state.frameIndex, availableHistory, lag, state.historySlots);
            short2 channelMotion;
            if (params.vectorDecode != 0)
                channelMotion = motion;
            else if (channel == 0)
                channelMotion = make_short2(0, static_cast<short>(motionStrength));
            else if (channel == 1)
                channelMotion = make_short2(static_cast<short>(-motionStrength), 0);
            else
                channelMotion =
                    make_short2(0, static_cast<short>(-motionStrength));
            uchar4 channelPrediction = readHistory(
                state.dirtyHistory,
                slot,
                x + channelMotion.x,
                y + channelMotion.y,
                state.width,
                state.height);
            int component = channel == 0 ? channelPrediction.x
                                        : (channel == 1 ? channelPrediction.y : channelPrediction.z);
            int residualComponent =
                channel == 0 ? residual.x : (channel == 1 ? residual.y : residual.z);
            values[channel] = component + static_cast<int>(residualComponent * 0.25f);
        }
    }

    uchar4 outputPixel = make_uchar4(
        static_cast<unsigned char>(max(0, min(values[0], 255))),
        static_cast<unsigned char>(max(0, min(values[1], 255))),
        static_cast<unsigned char>(max(0, min(values[2], 255))),
        current.w);
    surf2Dwrite(outputPixel, output, x * static_cast<int>(sizeof(uchar4)), y);
    state.dirtyHistory[currentSlot * state.width * state.height + pixel] = outputPixel;
}

__global__ void commitState(
    DatamoshCudaState state,
    cudaSurfaceObject_t input,
    DatamoshCudaParams params,
    int currentSlot)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= state.width || y >= state.height)
        return;
    int pixel = y * state.width + x;
    state.cleanHistory[currentSlot * state.width * state.height + pixel] =
        readInput(input, x, y, state.width, state.height, params.inputFormat);
    state.residualHistory[currentSlot * state.width * state.height + pixel] =
        state.residual[pixel];
}

cudaError_t allocateDevice(void** pointer, size_t bytes)
{
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess)
        *pointer = nullptr;
    return status;
}

} // namespace

cudaError_t datamoshCudaCreate(
    DatamoshCudaState** state,
    int width,
    int height,
    int blockSize,
    int historySlots)
{
    if (!state || width <= 0 || height <= 0 || blockSize <= 0 || historySlots < 2)
        return cudaErrorInvalidValue;

    DatamoshCudaState* created = new (std::nothrow) DatamoshCudaState;
    if (!created)
        return cudaErrorMemoryAllocation;
    created->width = width;
    created->height = height;
    created->blockSize = blockSize;
    created->blocksX = (width + blockSize - 1) / blockSize;
    created->blocksY = (height + blockSize - 1) / blockSize;
    created->historySlots = historySlots;

    size_t pixels = static_cast<size_t>(width) * height;
    size_t historyPixels = pixels * historySlots;
    cudaError_t status = allocateDevice(
        reinterpret_cast<void**>(&created->cleanHistory),
        historyPixels * sizeof(uchar4));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->dirtyHistory),
            historyPixels * sizeof(uchar4));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->residual),
            pixels * sizeof(short4));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->residualHistory),
            historyPixels * sizeof(short4));
    if (status == cudaSuccess)
        status = allocateDevice(
            reinterpret_cast<void**>(&created->motionVectors),
            static_cast<size_t>(created->blocksX) * created->blocksY * sizeof(short2));

    if (status != cudaSuccess)
    {
        datamoshCudaDestroy(created);
        return status;
    }

    *state = created;
    return cudaSuccess;
}

void datamoshCudaDestroy(DatamoshCudaState* state)
{
    if (!state)
        return;
    cudaFree(state->cleanHistory);
    cudaFree(state->dirtyHistory);
    cudaFree(state->residual);
    cudaFree(state->residualHistory);
    cudaFree(state->motionVectors);
    delete state;
}

void datamoshCudaReset(DatamoshCudaState* state)
{
    if (state)
        state->frameIndex = 0;
}

cudaError_t datamoshCudaProcess(
    DatamoshCudaState* state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    const DatamoshCudaParams& params,
    cudaStream_t stream)
{
    if (!state || !input || !output)
        return cudaErrorInvalidValue;

    dim3 pixelBlock(16, 16);
    dim3 pixelGrid(
        (state->width + pixelBlock.x - 1) / pixelBlock.x,
        (state->height + pixelBlock.y - 1) / pixelBlock.y);

    if (state->frameIndex == 0)
    {
        initializeState<<<pixelGrid, pixelBlock, 0, stream>>>(
            *state, input, output, params.inputFormat);
        state->frameIndex = 1;
        return cudaPeekAtLastError();
    }

    int availableHistory =
        min(static_cast<int>(state->frameIndex), state->historySlots);
    int referenceSlot =
        static_cast<int>((state->frameIndex + state->historySlots - 1) % state->historySlots);
    int currentSlot = static_cast<int>(state->frameIndex % state->historySlots);

    dim3 motionBlock(8, 8);
    dim3 motionGrid(
        (state->blocksX + motionBlock.x - 1) / motionBlock.x,
        (state->blocksY + motionBlock.y - 1) / motionBlock.y);
    encodeMotion<<<motionGrid, motionBlock, 0, stream>>>(
        *state, input, params, referenceSlot);
    encodeResidual<<<pixelGrid, pixelBlock, 0, stream>>>(
        *state, input, params, referenceSlot);
    decodeFrame<<<pixelGrid, pixelBlock, 0, stream>>>(
        *state,
        input,
        output,
        params,
        currentSlot,
        availableHistory);
    commitState<<<pixelGrid, pixelBlock, 0, stream>>>(*state, input, params, currentSlot);
    ++state->frameIndex;
    return cudaPeekAtLastError();
}

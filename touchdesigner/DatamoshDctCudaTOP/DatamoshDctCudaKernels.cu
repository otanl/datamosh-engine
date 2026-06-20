#include "DatamoshDctCudaCore.h"

#include <cuda_fp16.h>

#include <cmath>
#include <new>
#include <vector>

// GPU-native DCT0 codec. One CUDA thread handles one 8x8 block for the transform stages;
// per-pixel stages use a 2D grid. Behavioural parity with src/dct_codec.rs is by hand.

struct DatamoshDctCudaState
{
    int width = 0;
    int height = 0;
    int cw = 0;   // chroma plane width  (ceil(width/2))
    int ch = 0;   // chroma plane height (ceil(height/2))
    int blocksX = 0;
    int blocksY = 0;
    int cblocksX = 0;
    int cblocksY = 0;
    int lumaBlocks = 0;
    int chromaBlocks = 0;
    int lastQuality = -1;
    uint64_t frameIndex = 0;

    float* y = nullptr;          // width*height reconstructed luma (level-shifted)
    float* cb = nullptr;         // cw*ch
    float* cr = nullptr;         // cw*ch
    float* coeffLuma = nullptr;  // lumaBlocks*64
    float* coeffCb = nullptr;    // chromaBlocks*64
    float* coeffCr = nullptr;    // chromaBlocks*64
    float* snapLuma = nullptr;   // remap ping-pong
    float* snapCb = nullptr;
    float* snapCr = nullptr;
    float* quantLuma = nullptr;  // 64
    float* quantChroma = nullptr;// 64
    int* signLuma = nullptr;     // DC-drift prefix, capacity lumaBlocks+1
    int* signCb = nullptr;       // capacity chromaBlocks+1
    int* signCr = nullptr;
    uchar4* prevOutput = nullptr;// width*height (persistence feedback)
};

namespace {

constexpr int BLOCK = 8;
constexpr int BLOCK_AREA = 64;

// Standard JPEG quantization tables (natural row-major order), mirroring dct_codec.rs.
__constant__ float c_dctMatrix[BLOCK][BLOCK];
__constant__ int c_zigzag[BLOCK_AREA];

const unsigned short kQuantLuma[BLOCK_AREA] = {
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
    56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104,
    113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
};
const unsigned short kQuantChroma[BLOCK_AREA] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
};
const int kZigzag[BLOCK_AREA] = {
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
};

__device__ __forceinline__ int clampCoord(int value, int maximum)
{
    return max(0, min(value, maximum - 1));
}

// Returns BGRA-ordered (.x=B, .y=G, .z=R), matching the motion TOP convention.
__device__ __forceinline__ uchar4 readInput(
    cudaSurfaceObject_t input, int x, int y, int width, int height, int format)
{
    x = clampCoord(x, width);
    y = clampCoord(y, height);
    if (format == 0)
    {
        uchar4 pixel;
        surf2Dread(&pixel, input, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
        return pixel;
    }
    if (format == 1)
    {
        uchar4 pixel;
        surf2Dread(&pixel, input, x * (int)sizeof(uchar4), y, cudaBoundaryModeClamp);
        return make_uchar4(pixel.z, pixel.y, pixel.x, pixel.w);
    }
    if (format == 102)
    {
        ushort4 pixel;
        surf2Dread(&pixel, input, x * (int)sizeof(ushort4), y, cudaBoundaryModeClamp);
        return make_uchar4((unsigned char)(pixel.z >> 8), (unsigned char)(pixel.y >> 8),
                           (unsigned char)(pixel.x >> 8), (unsigned char)(pixel.w >> 8));
    }
    if (format == 202)
    {
        ushort4 pixel;
        surf2Dread(&pixel, input, x * (int)sizeof(ushort4), y, cudaBoundaryModeClamp);
        float4 v = make_float4(__half2float(__ushort_as_half(pixel.x)),
                               __half2float(__ushort_as_half(pixel.y)),
                               __half2float(__ushort_as_half(pixel.z)),
                               __half2float(__ushort_as_half(pixel.w)));
        return make_uchar4((unsigned char)(max(0.0f, min(v.z, 1.0f)) * 255.0f),
                           (unsigned char)(max(0.0f, min(v.y, 1.0f)) * 255.0f),
                           (unsigned char)(max(0.0f, min(v.x, 1.0f)) * 255.0f),
                           (unsigned char)(max(0.0f, min(v.w, 1.0f)) * 255.0f));
    }
    float4 pixel;
    surf2Dread(&pixel, input, x * (int)sizeof(float4), y, cudaBoundaryModeClamp);
    return make_uchar4((unsigned char)(max(0.0f, min(pixel.z, 1.0f)) * 255.0f),
                       (unsigned char)(max(0.0f, min(pixel.y, 1.0f)) * 255.0f),
                       (unsigned char)(max(0.0f, min(pixel.x, 1.0f)) * 255.0f),
                       (unsigned char)(max(0.0f, min(pixel.w, 1.0f)) * 255.0f));
}

// Read input pixel, optionally blended with the previous output (persistence feedback).
__device__ __forceinline__ uchar4 sourcePixel(
    cudaSurfaceObject_t input, const uchar4* prev, bool prevValid, float persistence,
    int x, int y, int width, int height, int format)
{
    uchar4 px = readInput(input, x, y, width, height, format);
    if (prevValid && persistence > 0.0f)
    {
        uchar4 p = prev[clampCoord(y, height) * width + clampCoord(x, width)];
        float inv = 1.0f - persistence;
        px.x = (unsigned char)min(255.0f, max(0.0f, px.x * inv + p.x * persistence + 0.5f));
        px.y = (unsigned char)min(255.0f, max(0.0f, px.y * inv + p.y * persistence + 0.5f));
        px.z = (unsigned char)min(255.0f, max(0.0f, px.z * inv + p.z * persistence + 0.5f));
    }
    return px;
}

__device__ __forceinline__ void rgbToYcc(uchar4 px, int& yy, int& cb, int& cr)
{
    int r = px.z, g = px.y, b = px.x;
    yy = (77 * r + 150 * g + 29 * b + 128) >> 8;
    cb = (((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128);
    cr = (((128 * r - 107 * g - 21 * b + 128) >> 8) + 128);
}

__device__ __forceinline__ void forwardDct(const float* blk, float* out)
{
    float tmp[BLOCK_AREA];
    for (int i = 0; i < BLOCK; ++i)
        for (int j = 0; j < BLOCK; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < BLOCK; ++k)
                s += c_dctMatrix[i][k] * blk[k * BLOCK + j];
            tmp[i * BLOCK + j] = s;
        }
    for (int i = 0; i < BLOCK; ++i)
        for (int j = 0; j < BLOCK; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < BLOCK; ++k)
                s += tmp[i * BLOCK + k] * c_dctMatrix[j][k];
            out[i * BLOCK + j] = s;
        }
}

__device__ __forceinline__ void inverseDct(const float* freq, float* out)
{
    float tmp[BLOCK_AREA];
    for (int i = 0; i < BLOCK; ++i)
        for (int j = 0; j < BLOCK; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < BLOCK; ++k)
                s += c_dctMatrix[k][i] * freq[k * BLOCK + j];
            tmp[i * BLOCK + j] = s;
        }
    for (int i = 0; i < BLOCK; ++i)
        for (int j = 0; j < BLOCK; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < BLOCK; ++k)
                s += tmp[i * BLOCK + k] * c_dctMatrix[k][j];
            out[i * BLOCK + j] = s;
        }
}

__device__ __forceinline__ void rotateZigzagAc(float* block, int amount)
{
    const int ac = BLOCK_AREA - 1;
    int shift = ((amount % ac) + ac) % ac;
    if (shift == 0)
        return;
    float ordered[BLOCK_AREA - 1];
    for (int slot = 0; slot < ac; ++slot)
        ordered[slot] = block[c_zigzag[slot + 1]];
    for (int slot = 0; slot < ac; ++slot)
        block[c_zigzag[slot + 1]] = ordered[(slot + shift) % ac];
}

__device__ __forceinline__ void reverseZigzagAc(float* block)
{
    for (int i = 1; i <= (BLOCK_AREA - 1) / 2; ++i)
    {
        int a = c_zigzag[i];
        int b = c_zigzag[BLOCK_AREA - i];
        float t = block[a];
        block[a] = block[b];
        block[b] = t;
    }
}

__device__ __forceinline__ void transposeBlock(float* block)
{
    for (int i = 0; i < BLOCK; ++i)
        for (int j = i + 1; j < BLOCK; ++j)
        {
            float t = block[i * BLOCK + j];
            block[i * BLOCK + j] = block[j * BLOCK + i];
            block[j * BLOCK + i] = t;
        }
}

// ---- encode ----

__global__ void kEncodeLuma(
    DatamoshDctCudaState state, cudaSurfaceObject_t input, DatamoshDctCudaParams params,
    bool prevValid)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= state.lumaBlocks)
        return;
    int bx = b % state.blocksX;
    int by = b / state.blocksX;
    float blk[BLOCK_AREA];
    for (int ry = 0; ry < BLOCK; ++ry)
        for (int rx = 0; rx < BLOCK; ++rx)
        {
            int sx = min(bx * BLOCK + rx, state.width - 1);
            int sy = min(by * BLOCK + ry, state.height - 1);
            uchar4 px = sourcePixel(input, state.prevOutput, prevValid, params.persistence, sx, sy,
                                    state.width, state.height, params.inputFormat);
            int yy, cb, cr;
            rgbToYcc(px, yy, cb, cr);
            blk[ry * BLOCK + rx] = (float)yy - 128.0f;
        }
    float freq[BLOCK_AREA];
    forwardDct(blk, freq);
    int base = b * BLOCK_AREA;
    for (int k = 0; k < BLOCK_AREA; ++k)
        state.coeffLuma[base + k] = roundf(freq[k] / state.quantLuma[k]);
}

__global__ void kEncodeChroma(
    DatamoshDctCudaState state, cudaSurfaceObject_t input, DatamoshDctCudaParams params,
    bool prevValid)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= state.chromaBlocks)
        return;
    int bx = b % state.cblocksX;
    int by = b / state.cblocksX;
    float cbBlk[BLOCK_AREA];
    float crBlk[BLOCK_AREA];
    for (int ry = 0; ry < BLOCK; ++ry)
        for (int rx = 0; rx < BLOCK; ++rx)
        {
            int cx = min(bx * BLOCK + rx, state.cw - 1);
            int cy = min(by * BLOCK + ry, state.ch - 1);
            int cbSum = 0, crSum = 0, cnt = 0;
            for (int dy = 0; dy < 2; ++dy)
                for (int dx = 0; dx < 2; ++dx)
                {
                    int sx = min(2 * cx + dx, state.width - 1);
                    int sy = min(2 * cy + dy, state.height - 1);
                    uchar4 px = sourcePixel(input, state.prevOutput, prevValid, params.persistence,
                                            sx, sy, state.width, state.height, params.inputFormat);
                    int yy, cb, cr;
                    rgbToYcc(px, yy, cb, cr);
                    cbSum += cb;
                    crSum += cr;
                    ++cnt;
                }
            cbBlk[ry * BLOCK + rx] = (float)cbSum / cnt - 128.0f;
            crBlk[ry * BLOCK + rx] = (float)crSum / cnt - 128.0f;
        }
    float fcb[BLOCK_AREA], fcr[BLOCK_AREA];
    forwardDct(cbBlk, fcb);
    forwardDct(crBlk, fcr);
    int base = b * BLOCK_AREA;
    for (int k = 0; k < BLOCK_AREA; ++k)
    {
        state.coeffCb[base + k] = roundf(fcb[k] / state.quantChroma[k]);
        state.coeffCr[base + k] = roundf(fcr[k] / state.quantChroma[k]);
    }
}

// ---- glitch ----

__global__ void kRemap(
    float* coeff, const float* snap, int blocksX, int blocksY, int shiftX, int shiftY,
    int shiftEvery, int repeatEvery)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blocksX * blocksY;
    if (b >= total)
        return;
    int bx = b % blocksX;
    int by = b / blocksX;
    long ordinal = (long)b + 1;
    int srcBlock = b;
    if (shiftEvery != 0 && (shiftX != 0 || shiftY != 0) && ordinal % shiftEvery == 0)
    {
        int sx = ((bx + shiftX) % blocksX + blocksX) % blocksX;
        int sy = ((by + shiftY) % blocksY + blocksY) % blocksY;
        srcBlock = sy * blocksX + sx;
    }
    else if (repeatEvery != 0 && ordinal % repeatEvery == 0)
    {
        int prev = b > 0 ? b - 1 : 0;
        srcBlock = (prev % blocksX) + (prev / blocksX % blocksY) * blocksX;
    }
    int dst = b * BLOCK_AREA;
    int src = srcBlock * BLOCK_AREA;
    for (int k = 0; k < BLOCK_AREA; ++k)
        coeff[dst + k] = snap[src + k];
}

__global__ void kCorrupt(
    float* coeff, int blocksX, int blocksY, DatamoshDctCudaParams params, int channel,
    const int* signSum)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int total = blocksX * blocksY;
    if (b >= total)
        return;
    long ordinal = (long)b + 1;
    int base = b * BLOCK_AREA;
    float blk[BLOCK_AREA];
    for (int k = 0; k < BLOCK_AREA; ++k)
        blk[k] = coeff[base + k];

    bool coeffShift =
        params.coeffShift != 0 && params.coeffShiftEvery != 0 && ordinal % params.coeffShiftEvery == 0;
    bool zigzagReverse = params.zigzagReverseEvery != 0 && ordinal % params.zigzagReverseEvery == 0;
    bool blockTranspose =
        params.blockTransposeEvery != 0 && ordinal % params.blockTransposeEvery == 0;
    bool dcOffset = params.dcBlockOffset != 0 && params.dcBlockOffsetEvery != 0 &&
                    ordinal % params.dcBlockOffsetEvery == 0;
    if (coeffShift)
        rotateZigzagAc(blk, params.coeffShift);
    if (zigzagReverse)
        reverseZigzagAc(blk);
    if (blockTranspose)
        transposeBlock(blk);

    bool requant = params.quantScale > 1.0f;
    bool lowPass = params.acZeroAbove != 0;
    bool signFlip = params.signFlipEvery != 0 && ordinal % params.signFlipEvery == 0;
    for (int zz = 0; zz < BLOCK_AREA; ++zz)
    {
        int idx = c_zigzag[zz];
        if (zz != 0 && lowPass && zz > params.acZeroAbove)
        {
            blk[idx] = 0.0f;
            continue;
        }
        if (requant)
            blk[idx] = roundf(blk[idx] / params.quantScale) * params.quantScale;
        if (signFlip && zz != 0)
            blk[idx] = -blk[idx];
    }
    if (dcOffset)
        blk[0] += (float)params.dcBlockOffset;
    if (params.dcDrift != 0 && params.dcDriftEvery != 0)
    {
        int T = (int)(ordinal / params.dcDriftEvery);
        blk[0] += (float)params.dcDrift * (float)signSum[T];
    }
    blk[0] = fminf(32767.0f, fmaxf(-32768.0f, blk[0]));

    for (int k = 0; k < BLOCK_AREA; ++k)
        coeff[base + k] = blk[k];
    (void)channel;
}

__global__ void kChromaSwap(float* coeffCb, float* coeffCr, int chromaBlocks, int every)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= chromaBlocks)
        return;
    long ordinal = (long)b + 1;
    if (ordinal % every != 0)
        return;
    int base = b * BLOCK_AREA;
    for (int k = 0; k < BLOCK_AREA; ++k)
    {
        float t = coeffCb[base + k];
        coeffCb[base + k] = coeffCr[base + k];
        coeffCr[base + k] = t;
    }
}

// ---- decode ----

__device__ __forceinline__ void decodeBlock(const float* coeff, const float* quant, float* out)
{
    bool acZero = true;
    for (int k = 1; k < BLOCK_AREA; ++k)
        if (coeff[k] != 0.0f)
        {
            acZero = false;
            break;
        }
    if (acZero)
    {
        float flat = coeff[0] * quant[0] * 0.125f;
        for (int k = 0; k < BLOCK_AREA; ++k)
            out[k] = flat;
        return;
    }
    float freq[BLOCK_AREA];
    for (int k = 0; k < BLOCK_AREA; ++k)
        freq[k] = coeff[k] * quant[k];
    inverseDct(freq, out);
}

__global__ void kDecodeLuma(DatamoshDctCudaState state)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= state.lumaBlocks)
        return;
    int bx = b % state.blocksX;
    int by = b / state.blocksX;
    float out[BLOCK_AREA];
    decodeBlock(&state.coeffLuma[b * BLOCK_AREA], state.quantLuma, out);
    for (int ry = 0; ry < BLOCK; ++ry)
    {
        int py = by * BLOCK + ry;
        if (py >= state.height)
            break;
        for (int rx = 0; rx < BLOCK; ++rx)
        {
            int px = bx * BLOCK + rx;
            if (px >= state.width)
                break;
            state.y[py * state.width + px] = out[ry * BLOCK + rx];
        }
    }
}

__global__ void kDecodeChroma(DatamoshDctCudaState state)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= state.chromaBlocks)
        return;
    int bx = b % state.cblocksX;
    int by = b / state.cblocksX;
    float ocb[BLOCK_AREA], ocr[BLOCK_AREA];
    decodeBlock(&state.coeffCb[b * BLOCK_AREA], state.quantChroma, ocb);
    decodeBlock(&state.coeffCr[b * BLOCK_AREA], state.quantChroma, ocr);
    for (int ry = 0; ry < BLOCK; ++ry)
    {
        int py = by * BLOCK + ry;
        if (py >= state.ch)
            break;
        for (int rx = 0; rx < BLOCK; ++rx)
        {
            int px = bx * BLOCK + rx;
            if (px >= state.cw)
                break;
            state.cb[py * state.cw + px] = ocb[ry * BLOCK + rx];
            state.cr[py * state.cw + px] = ocr[ry * BLOCK + rx];
        }
    }
}

__global__ void kCombine(DatamoshDctCudaState state, cudaSurfaceObject_t output)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= state.width || y >= state.height)
        return;
    int ci = (y / 2) * state.cw + (x / 2);
    int yi = (int)(state.y[y * state.width + x] + 128.0f);
    int cbi = (int)(state.cb[ci] + 128.0f) - 128;
    int cri = (int)(state.cr[ci] + 128.0f) - 128;
    int r = min(255, max(0, yi + ((359 * cri + 128) >> 8)));
    int g = min(255, max(0, yi - ((88 * cbi + 183 * cri + 128) >> 8)));
    int b = min(255, max(0, yi + ((454 * cbi + 128) >> 8)));
    uchar4 out = make_uchar4((unsigned char)b, (unsigned char)g, (unsigned char)r, 255);
    surf2Dwrite(out, output, x * (int)sizeof(uchar4), y);
    state.prevOutput[y * state.width + x] = out;
}

cudaError_t allocateDevice(void** pointer, size_t bytes)
{
    cudaError_t status = cudaMalloc(pointer, bytes);
    if (status != cudaSuccess)
        *pointer = nullptr;
    return status;
}

uint64_t hashU64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

float qualityScale(int quality)
{
    float q = (float)min(100, max(1, quality));
    return q < 50.0f ? 5000.0f / q : 200.0f - 2.0f * q;
}

void scaledTable(const unsigned short* table, float scale, float* out)
{
    for (int i = 0; i < BLOCK_AREA; ++i)
    {
        float v = floorf(((float)table[i] * scale + 50.0f) / 100.0f);
        out[i] = fminf(255.0f, fmaxf(1.0f, v));
    }
}

// Host: build the per-channel DC-drift sign prefix used by kCorrupt.
void buildSignSum(int channel, int blockCount, int every, std::vector<int>& out)
{
    int maxT = every > 0 ? blockCount / every + 1 : 0;
    out.assign(maxT + 1, 0);
    int running = 0;
    for (int t = 1; t <= maxT; ++t)
    {
        uint64_t h = hashU64((uint64_t)t);
        int sign = ((h >> channel) & 1ULL) == 0 ? 1 : -1;
        running += sign;
        out[t] = running;
    }
}

} // namespace

cudaError_t datamoshDctCudaCreate(DatamoshDctCudaState** state, int width, int height)
{
    if (!state || width <= 0 || height <= 0)
        return cudaErrorInvalidValue;

    DatamoshDctCudaState* s = new (std::nothrow) DatamoshDctCudaState;
    if (!s)
        return cudaErrorMemoryAllocation;
    s->width = width;
    s->height = height;
    s->cw = (width + 1) / 2;
    s->ch = (height + 1) / 2;
    s->blocksX = (width + BLOCK - 1) / BLOCK;
    s->blocksY = (height + BLOCK - 1) / BLOCK;
    s->cblocksX = (s->cw + BLOCK - 1) / BLOCK;
    s->cblocksY = (s->ch + BLOCK - 1) / BLOCK;
    s->lumaBlocks = s->blocksX * s->blocksY;
    s->chromaBlocks = s->cblocksX * s->cblocksY;

    size_t lumaCoeff = (size_t)s->lumaBlocks * BLOCK_AREA;
    size_t chromaCoeff = (size_t)s->chromaBlocks * BLOCK_AREA;
    cudaError_t st = cudaSuccess;
    if (st == cudaSuccess) st = allocateDevice((void**)&s->y, (size_t)width * height * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->cb, (size_t)s->cw * s->ch * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->cr, (size_t)s->cw * s->ch * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->coeffLuma, lumaCoeff * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->coeffCb, chromaCoeff * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->coeffCr, chromaCoeff * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->snapLuma, lumaCoeff * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->snapCb, chromaCoeff * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->snapCr, chromaCoeff * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->quantLuma, BLOCK_AREA * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->quantChroma, BLOCK_AREA * sizeof(float));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->signLuma, ((size_t)s->lumaBlocks + 2) * sizeof(int));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->signCb, ((size_t)s->chromaBlocks + 2) * sizeof(int));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->signCr, ((size_t)s->chromaBlocks + 2) * sizeof(int));
    if (st == cudaSuccess) st = allocateDevice((void**)&s->prevOutput, (size_t)width * height * sizeof(uchar4));

    if (st == cudaSuccess)
    {
        float matrix[BLOCK][BLOCK];
        for (int u = 0; u < BLOCK; ++u)
        {
            double cu = u == 0 ? sqrt(1.0 / BLOCK) : sqrt(2.0 / BLOCK);
            for (int x = 0; x < BLOCK; ++x)
            {
                double angle = (2.0 * x + 1.0) * u * 3.14159265358979323846 / (2.0 * BLOCK);
                matrix[u][x] = (float)(cu * cos(angle));
            }
        }
        st = cudaMemcpyToSymbol(c_dctMatrix, matrix, sizeof(matrix));
        if (st == cudaSuccess)
            st = cudaMemcpyToSymbol(c_zigzag, kZigzag, sizeof(kZigzag));
    }

    if (st != cudaSuccess)
    {
        datamoshDctCudaDestroy(s);
        return st;
    }
    *state = s;
    return cudaSuccess;
}

void datamoshDctCudaDestroy(DatamoshDctCudaState* state)
{
    if (!state)
        return;
    cudaFree(state->y);
    cudaFree(state->cb);
    cudaFree(state->cr);
    cudaFree(state->coeffLuma);
    cudaFree(state->coeffCb);
    cudaFree(state->coeffCr);
    cudaFree(state->snapLuma);
    cudaFree(state->snapCb);
    cudaFree(state->snapCr);
    cudaFree(state->quantLuma);
    cudaFree(state->quantChroma);
    cudaFree(state->signLuma);
    cudaFree(state->signCb);
    cudaFree(state->signCr);
    cudaFree(state->prevOutput);
    delete state;
}

void datamoshDctCudaReset(DatamoshDctCudaState* state)
{
    if (state)
        state->frameIndex = 0;
}

cudaError_t datamoshDctCudaProcess(
    DatamoshDctCudaState* state,
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    const DatamoshDctCudaParams& params,
    cudaStream_t stream)
{
    if (!state || !input || !output)
        return cudaErrorInvalidValue;

    // Update the quantization tables when quality changes.
    if (params.quality != state->lastQuality)
    {
        float scale = qualityScale(params.quality);
        float ql[BLOCK_AREA], qc[BLOCK_AREA];
        scaledTable(kQuantLuma, scale, ql);
        scaledTable(kQuantChroma, scale, qc);
        cudaMemcpyAsync(state->quantLuma, ql, sizeof(ql), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(state->quantChroma, qc, sizeof(qc), cudaMemcpyHostToDevice, stream);
        state->lastQuality = params.quality;
    }

    // Build DC-drift sign prefixes (one per plane channel) when drift is active.
    if (params.dcDrift != 0 && params.dcDriftEvery != 0)
    {
        std::vector<int> sl, sb, sr;
        buildSignSum(0, state->lumaBlocks, params.dcDriftEvery, sl);
        buildSignSum(1, state->chromaBlocks, params.dcDriftEvery, sb);
        buildSignSum(2, state->chromaBlocks, params.dcDriftEvery, sr);
        cudaMemcpyAsync(state->signLuma, sl.data(), sl.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(state->signCb, sb.data(), sb.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(state->signCr, sr.data(), sr.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    bool prevValid = state->frameIndex > 0;
    int threads = 256;
    int lumaGrid = (state->lumaBlocks + threads - 1) / threads;
    int chromaGrid = (state->chromaBlocks + threads - 1) / threads;

    kEncodeLuma<<<lumaGrid, threads, 0, stream>>>(*state, input, params, prevValid);
    kEncodeChroma<<<chromaGrid, threads, 0, stream>>>(*state, input, params, prevValid);

    bool remap = (params.blockShiftEvery != 0 && (params.blockShiftX != 0 || params.blockShiftY != 0)) ||
                 params.blockRepeatEvery != 0;
    if (remap)
    {
        size_t lumaBytes = (size_t)state->lumaBlocks * BLOCK_AREA * sizeof(float);
        size_t chromaBytes = (size_t)state->chromaBlocks * BLOCK_AREA * sizeof(float);
        cudaMemcpyAsync(state->snapLuma, state->coeffLuma, lumaBytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(state->snapCb, state->coeffCb, chromaBytes, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(state->snapCr, state->coeffCr, chromaBytes, cudaMemcpyDeviceToDevice, stream);
        kRemap<<<lumaGrid, threads, 0, stream>>>(state->coeffLuma, state->snapLuma, state->blocksX,
            state->blocksY, params.blockShiftX, params.blockShiftY, params.blockShiftEvery,
            params.blockRepeatEvery);
        kRemap<<<chromaGrid, threads, 0, stream>>>(state->coeffCb, state->snapCb, state->cblocksX,
            state->cblocksY, params.blockShiftX, params.blockShiftY, params.blockShiftEvery,
            params.blockRepeatEvery);
        kRemap<<<chromaGrid, threads, 0, stream>>>(state->coeffCr, state->snapCr, state->cblocksX,
            state->cblocksY, params.blockShiftX, params.blockShiftY, params.blockShiftEvery,
            params.blockRepeatEvery);
    }

    kCorrupt<<<lumaGrid, threads, 0, stream>>>(state->coeffLuma, state->blocksX, state->blocksY,
        params, 0, state->signLuma);
    kCorrupt<<<chromaGrid, threads, 0, stream>>>(state->coeffCb, state->cblocksX, state->cblocksY,
        params, 1, state->signCb);
    kCorrupt<<<chromaGrid, threads, 0, stream>>>(state->coeffCr, state->cblocksX, state->cblocksY,
        params, 2, state->signCr);

    if (params.chromaSwapEvery != 0)
        kChromaSwap<<<chromaGrid, threads, 0, stream>>>(state->coeffCb, state->coeffCr,
            state->chromaBlocks, params.chromaSwapEvery);

    kDecodeLuma<<<lumaGrid, threads, 0, stream>>>(*state);
    kDecodeChroma<<<chromaGrid, threads, 0, stream>>>(*state);

    dim3 pixelBlock(16, 16);
    dim3 pixelGrid((state->width + 15) / 16, (state->height + 15) / 16);
    kCombine<<<pixelGrid, pixelBlock, 0, stream>>>(*state, output);

    ++state->frameIndex;
    return cudaPeekAtLastError();
}

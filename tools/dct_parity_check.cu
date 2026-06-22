// CPU<->CUDA DCT0 parity guard.
//
// Runs the SAME input frame + SAME preset through (a) the CPU DCT0 codec via the C ABI
// (datamosh.dll) and (b) the real CUDA kernels (datamoshDctCudaProcess), and reports the
// mean-abs-error per preset. The two paths are not bit-identical (fast even/odd vs naive DCT,
// i16 vs float coefficients), so a small MAE is expected; a large MAE means the hand-maintained
// CUDA kernels / preset table have DRIFTED from src/dct_codec.rs. Exit code 1 on drift.
//
// Build: scripts/build-dct-parity-check.cmd   Run from target/release (needs datamosh.dll).
#include "datamosh_ffi.h"

#include "DatamoshDctCudaCore.h"
#include "DatamoshDctCudaPresets.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const int W = 256, H = 192;
// Observed max across presets is ~0.2 (float DCT precision + i16/float coeff rounding). A real
// drift — a missed preset value or an algorithm change — moves the glitch pattern and spikes
// MAE into the tens, so this catches it with a wide margin.
static const double THRESHOLD = 4.0;

static void makeImage(std::vector<unsigned char>& rgb, std::vector<uchar4>& bgra)
{
    const unsigned char bars[8][3] = {
        {220, 220, 220}, {220, 220, 0}, {0, 220, 220}, {0, 220, 0},
        {220, 0, 220}, {220, 0, 0}, {0, 0, 220}, {20, 20, 20}};
    rgb.resize((size_t)W * H * 3);
    bgra.resize((size_t)W * H);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
        {
            const unsigned char* c = bars[(x * 8) / W];
            unsigned char r = c[0], g = c[1], b = c[2];
            if (y >= 80 && y < 120)
            {
                int k = (x / 3) % 3; // fine vertical detail band
                if (k == 0) r = 235;
                if (k == 1) g = 215;
                if (k == 2) b = 235;
            }
            int i = y * W + x;
            rgb[i * 3 + 0] = r;
            rgb[i * 3 + 1] = g;
            rgb[i * 3 + 2] = b;
            bgra[i] = make_uchar4(b, g, r, 255);
        }
}

static void runCudaParams(
    DatamoshDctCudaParams p,
    int frames,
    const std::vector<uchar4>& in,
    std::vector<uchar4>& out)
{
    p.inputFormat = 0; // BGRA8, matches the test buffer

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t inArr, outArr;
    cudaMallocArray(&inArr, &desc, W, H, cudaArraySurfaceLoadStore);
    cudaMallocArray(&outArr, &desc, W, H, cudaArraySurfaceLoadStore);
    cudaMemcpy2DToArray(inArr, 0, 0, in.data(), W * sizeof(uchar4), W * sizeof(uchar4), H,
                        cudaMemcpyHostToDevice);
    cudaResourceDesc rd = {};
    rd.resType = cudaResourceTypeArray;
    cudaSurfaceObject_t inS = 0, outS = 0;
    rd.res.array.array = inArr;
    cudaCreateSurfaceObject(&inS, &rd);
    rd.res.array.array = outArr;
    cudaCreateSurfaceObject(&outS, &rd);

    DatamoshDctCudaState* st = nullptr;
    datamoshDctCudaCreate(&st, W, H);
    for (int frame = 0; frame < frames; ++frame)
        datamoshDctCudaProcess(st, inS, outS, p, 0);
    cudaDeviceSynchronize();

    out.resize((size_t)W * H);
    cudaMemcpy2DFromArray(out.data(), W * sizeof(uchar4), outArr, 0, 0, W * sizeof(uchar4), H,
                          cudaMemcpyDeviceToHost);
    datamoshDctCudaDestroy(st);
    cudaDestroySurfaceObject(inS);
    cudaDestroySurfaceObject(outS);
    cudaFreeArray(inArr);
    cudaFreeArray(outArr);
}

static void runCuda(
    int idx,
    float intensity,
    int frames,
    const std::vector<uchar4>& in,
    std::vector<uchar4>& out)
{
    DatamoshDctCudaParams p = dctcuda::presetParams(idx);
    dctcuda::applyControls(p, intensity, 1, 1, 1, 1);
    p.quality = 50;
    runCudaParams(p, frames, in, out);
}

int main()
{
    std::vector<unsigned char> inRgb;
    std::vector<uchar4> inBgra;
    makeImage(inRgb, inBgra);
    size_t len = (size_t)W * H * 3;

    int fails = 0;
    double worst = 0.0;
    const char* worstName = "-";
    constexpr int frames = 2;
    printf(
        "CPU<->CUDA DCT0 parity (%d frames, MAE per preset, threshold %.1f)\n",
        frames,
        THRESHOLD);

    std::vector<uchar4> cudaClean;
    runCuda(0, 0.0f, frames, inBgra, cudaClean);

    for (int idx = 0; idx < dctcuda::patternCount(); ++idx)
    {
        const char* name = dctcuda::kPatternNames[idx];
        DatamoshMoshEngine* eng =
            datamosh_mosh_engine_new_with_backend(DATAMOSH_BACKEND_DCT_TRANSFORM_V1, W, H);
        if (!eng)
        {
            printf("  %-12s : engine creation failed\n", name);
            ++fails;
            continue;
        }
        datamosh_mosh_engine_set_preset(eng, name);
        std::vector<unsigned char> cpuOut(len, 0);
        datamosh_mosh_engine_set_controls(eng, 1, 1, 1, 1, 1);
        int status = 0;
        for (int frame = 0; frame < frames && status == 0; ++frame)
            status =
                datamosh_mosh_engine_process_rgb24(eng, inRgb.data(), len, cpuOut.data(), len);
        datamosh_mosh_engine_free(eng);
        if (status != 0)
        {
            printf("  %-12s : CPU process status %d\n", name, status);
            ++fails;
            continue;
        }

        std::vector<uchar4> cudaOut;
        runCuda(idx, 1.0f, frames, inBgra, cudaOut);

        double err = 0.0;
        for (int i = 0; i < W * H; ++i)
        {
            err += std::abs((int)cpuOut[i * 3 + 0] - (int)cudaOut[i].z);
            err += std::abs((int)cpuOut[i * 3 + 1] - (int)cudaOut[i].y);
            err += std::abs((int)cpuOut[i * 3 + 2] - (int)cudaOut[i].x);
        }
        double mae = err / ((double)W * H * 3);
        bool ok = mae <= THRESHOLD;
        printf("  %-12s : MAE=%6.2f  %s\n", name, mae, ok ? "ok" : "DRIFT");
        if (!ok)
            ++fails;
        if (mae > worst)
        {
            worst = mae;
            worstName = name;
        }

        std::vector<uchar4> cudaBypass;
        runCuda(idx, 0.0f, frames, inBgra, cudaBypass);
        if (std::memcmp(
                cudaBypass.data(),
                cudaClean.data(),
                cudaClean.size() * sizeof(uchar4)) != 0)
        {
            printf("  %-12s : intensity-zero bypass DRIFT\n", name);
            ++fails;
        }
    }

    {
        struct Override
        {
            const char* id;
            float value;
        };
        constexpr Override overrides[] = {
            {"quality", 72},
            {"quant_scale", 3.5f},
            {"dc_drift", -9},
            {"dc_drift_every", 13},
            {"dc_block_offset", 17},
            {"dc_block_offset_every", 5},
            {"ac_zero_above", 9},
            {"coeff_sign_flip_every", 7},
            {"coeff_shift", -5},
            {"coeff_shift_every", 3},
            {"block_shift_x", -2},
            {"block_shift_y", 1},
            {"block_shift_every", 11},
            {"block_repeat_every", 17},
            {"zigzag_reverse_every", 19},
            {"block_transpose_every", 23},
            {"chroma_swap_every", 5},
            {"persistence", 0.35f},
        };
        DatamoshMoshEngine* engine =
            datamosh_mosh_engine_new_with_backend(
                DATAMOSH_BACKEND_DCT_TRANSFORM_V1, W, H);
        int status = engine ? datamosh_mosh_engine_set_preset(engine, "composite") : -1;
        DatamoshDctCudaParams params =
            dctcuda::presetParams(dctcuda::patternIndex("composite"));
        for (const Override& overrideValue : overrides)
        {
            if (status == 0)
                status = datamosh_mosh_engine_set_parameter(
                    engine, overrideValue.id, overrideValue.value);
            dctcuda::setParameter(params, overrideValue.id, overrideValue.value);
        }
        if (status == 0)
            status = datamosh_mosh_engine_set_controls(
                engine, 0.75f, 3.0f, 1.5f, 2.5f, 0.6f);
        dctcuda::applyControls(params, 0.75f, 3.0f, 1.5f, 2.5f, 0.6f);

        std::vector<unsigned char> cpuOut(len, 0);
        for (int frame = 0; frame < frames && status == 0; ++frame)
            status = datamosh_mosh_engine_process_rgb24(
                engine, inRgb.data(), len, cpuOut.data(), len);
        std::vector<uchar4> cudaOut;
        if (status == 0)
            runCudaParams(params, frames, inBgra, cudaOut);

        double err = 0.0;
        if (status == 0)
        {
            for (int i = 0; i < W * H; ++i)
            {
                err += std::abs((int)cpuOut[i * 3 + 0] - (int)cudaOut[i].z);
                err += std::abs((int)cpuOut[i * 3 + 1] - (int)cudaOut[i].y);
                err += std::abs((int)cpuOut[i * 3 + 2] - (int)cudaOut[i].x);
            }
        }
        const double mae =
            status == 0 ? err / ((double)W * H * 3) : THRESHOLD + 1.0;
        const bool ok = status == 0 && mae <= THRESHOLD;
        printf("  %-12s : MAE=%6.2f  %s\n", "manual", mae, ok ? "ok" : "DRIFT");
        if (!ok)
            ++fails;
        if (mae > worst)
        {
            worst = mae;
            worstName = "manual";
        }
        if (engine)
            datamosh_mosh_engine_free(engine);
    }
    printf("worst: %s MAE=%.2f  => %s\n", worstName, worst, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

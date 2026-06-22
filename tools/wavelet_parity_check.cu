// CPU<->CUDA WVT0 parity guard.
//
// Runs the same moving RGB frames and presets through the Rust CPU codec and
// the CUDA implementation. WVT0 uses integer transforms on both paths, so the
// expected result is bit exact.
#include "datamosh_ffi.h"

#include "DatamoshWaveletCudaCore.h"
#include "DatamoshWaveletCudaPresets.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int W = 127;
constexpr int H = 95;
constexpr int FRAMES = 9;
constexpr double THRESHOLD = 0.0;

bool cudaOk(cudaError_t status, const char* operation)
{
    if (status == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

void makeFrame(
    int frame,
    std::vector<unsigned char>& rgb,
    std::vector<uchar4>& bgra)
{
    rgb.resize(static_cast<size_t>(W) * H * 3);
    bgra.resize(static_cast<size_t>(W) * H);
    const int boxX = (frame * 7) % (W - 28);
    const int boxY = (frame * 5) % (H - 22);
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            int red = (x * 3 + y + frame * 11) & 255;
            int green = (x + y * 2 + frame * 17) & 255;
            int blue = (x * 2 + y * 3 + frame * 5) & 255;
            if (x >= boxX && x < boxX + 28 && y >= boxY && y < boxY + 22)
            {
                red = 245;
                green = 35 + (frame * 13 & 63);
                blue = 210;
            }
            const size_t pixel = static_cast<size_t>(y) * W + x;
            rgb[pixel * 3 + 0] = static_cast<unsigned char>(red);
            rgb[pixel * 3 + 1] = static_cast<unsigned char>(green);
            rgb[pixel * 3 + 2] = static_cast<unsigned char>(blue);
            bgra[pixel] = make_uchar4(
                static_cast<unsigned char>(blue),
                static_cast<unsigned char>(green),
                static_cast<unsigned char>(red),
                255);
        }
    }
}

bool createSurface(
    cudaArray_t* array,
    cudaSurfaceObject_t* surface)
{
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    if (!cudaOk(
            cudaMallocArray(array, &desc, W, H, cudaArraySurfaceLoadStore),
            "cudaMallocArray"))
        return false;
    cudaResourceDesc resource = {};
    resource.resType = cudaResourceTypeArray;
    resource.res.array.array = *array;
    if (!cudaOk(
            cudaCreateSurfaceObject(surface, &resource),
            "cudaCreateSurfaceObject"))
    {
        cudaFreeArray(*array);
        *array = nullptr;
        return false;
    }
    return true;
}

} // namespace

int main()
{
    cudaArray_t inputArray = nullptr;
    cudaArray_t outputArray = nullptr;
    cudaSurfaceObject_t inputSurface = 0;
    cudaSurfaceObject_t outputSurface = 0;
    if (!createSurface(&inputArray, &inputSurface) ||
        !createSurface(&outputArray, &outputSurface))
        return 1;

    std::vector<unsigned char> inputRgb;
    std::vector<uchar4> inputBgra;
    std::vector<unsigned char> cpuOutput(static_cast<size_t>(W) * H * 3);
    std::vector<uchar4> cudaOutput(static_cast<size_t>(W) * H);

    int failures = 0;
    double worst = 0.0;
    const char* worstName = "-";
    std::printf(
        "CPU<->CUDA WVT0 parity (%d moving frames, threshold %.1f)\n",
        FRAMES,
        THRESHOLD);

    for (int preset = 0; preset < waveletcuda::patternCount(); ++preset)
    {
        const char* name = waveletcuda::kPatternNames[preset];
        DatamoshMoshEngine* cpu = datamosh_mosh_engine_new_with_backend(
            DATAMOSH_BACKEND_WAVELET_PYRAMID_V1, W, H);
        DatamoshWaveletCudaState* gpu = nullptr;
        if (!cpu ||
            !cudaOk(
                datamoshWaveletCudaCreate(&gpu, W, H, 3, 12),
                "datamoshWaveletCudaCreate"))
        {
            std::printf("  %-20s : state creation failed\n", name);
            if (cpu)
                datamosh_mosh_engine_free(cpu);
            ++failures;
            continue;
        }

        int status = datamosh_mosh_engine_set_preset(cpu, name);
        status = status == 0
                     ? datamosh_mosh_engine_set_controls(cpu, 1, 1, 1, 1, 1)
                     : status;
        DatamoshWaveletCudaParams params = waveletcuda::presetParams(preset);
        waveletcuda::applyControls(params, 1, 1, 1, 1, 1);
        params.inputFormat = 0;

        for (int frame = 0; frame < FRAMES && status == 0; ++frame)
        {
            makeFrame(frame, inputRgb, inputBgra);
            status = datamosh_mosh_engine_process_rgb24(
                cpu,
                inputRgb.data(),
                inputRgb.size(),
                cpuOutput.data(),
                cpuOutput.size());
            if (!cudaOk(
                    cudaMemcpy2DToArray(
                        inputArray,
                        0,
                        0,
                        inputBgra.data(),
                        W * sizeof(uchar4),
                        W * sizeof(uchar4),
                        H,
                        cudaMemcpyHostToDevice),
                    "cudaMemcpy2DToArray") ||
                !cudaOk(
                    datamoshWaveletCudaProcess(
                        gpu, inputSurface, outputSurface, params, 0),
                    "datamoshWaveletCudaProcess"))
            {
                status = -1;
            }
        }

        if (status == 0 &&
            cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize") &&
            cudaOk(
                cudaMemcpy2DFromArray(
                    cudaOutput.data(),
                    W * sizeof(uchar4),
                    outputArray,
                    0,
                    0,
                    W * sizeof(uchar4),
                    H,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy2DFromArray"))
        {
            double error = 0.0;
            for (int pixel = 0; pixel < W * H; ++pixel)
            {
                error += std::abs(
                    static_cast<int>(cpuOutput[pixel * 3 + 0]) -
                    static_cast<int>(cudaOutput[pixel].z));
                error += std::abs(
                    static_cast<int>(cpuOutput[pixel * 3 + 1]) -
                    static_cast<int>(cudaOutput[pixel].y));
                error += std::abs(
                    static_cast<int>(cpuOutput[pixel * 3 + 2]) -
                    static_cast<int>(cudaOutput[pixel].x));
            }
            const double mae = error / (static_cast<double>(W) * H * 3);
            const bool ok = mae <= THRESHOLD;
            std::printf("  %-20s : MAE=%7.3f  %s\n", name, mae, ok ? "ok" : "DRIFT");
            if (!ok)
                ++failures;
            if (mae > worst)
            {
                worst = mae;
                worstName = name;
            }
        }
        else
        {
            std::printf("  %-20s : processing failed (%d)\n", name, status);
            ++failures;
        }

        datamoshWaveletCudaDestroy(gpu);
        datamosh_mosh_engine_free(cpu);
    }

    {
        struct Override
        {
            const char* id;
            float value;
        };
        constexpr Override overrides[] = {
            {"quality", 91},
            {"levels", 2},
            {"history_len", 7},
            {"packet_shift", -3},
            {"packet_shift_every", 4},
            {"orientation_rotate", 2},
            {"orientation_rotate_every", 5},
            {"level_fold", -1},
            {"level_fold_every", 6},
            {"channel_route", 1},
            {"channel_route_every", 7},
            {"packet_loss_every", 11},
            {"packet_loss_conceal", 1},
            {"bitplane_clear", 3},
            {"bitplane_clear_every", 4},
            {"bitplane_xor", 5},
            {"bitplane_xor_every", 13},
            {"sign_flip_every", 17},
            {"history_lag", 3},
            {"history_band_every", 5},
            {"lowpass_history_lag", 2},
            {"lifting_bias", -9},
            {"lifting_bias_every", 7},
        };
        DatamoshMoshEngine* cpu = datamosh_mosh_engine_new_with_backend(
            DATAMOSH_BACKEND_WAVELET_PYRAMID_V1, W, H);
        DatamoshWaveletCudaState* gpu = nullptr;
        int status = cpu ? datamosh_mosh_engine_set_preset(
                               cpu, "hierarchy-collapse")
                         : -1;
        DatamoshWaveletCudaParams params =
            waveletcuda::presetParams(waveletcuda::patternIndex("hierarchy-collapse"));
        for (const Override& overrideValue : overrides)
        {
            if (status == 0)
                status = datamosh_mosh_engine_set_parameter(
                    cpu, overrideValue.id, overrideValue.value);
            waveletcuda::setParameter(
                params, overrideValue.id, overrideValue.value, 12);
        }
        if (status == 0)
            status = datamosh_mosh_engine_set_controls(
                cpu, 0.5f, 3.0f, 1.25f, 2.5f, 0.75f);
        waveletcuda::applyControls(
            params, 0.5f, 3.0f, 1.25f, 2.5f, 0.75f);
        params.inputFormat = 0;
        if (status == 0)
            status = cudaOk(
                         datamoshWaveletCudaCreate(
                             &gpu,
                             W,
                             H,
                             params.levels,
                             params.historyLength),
                         "datamoshWaveletCudaCreate overrides")
                         ? 0
                         : -1;

        for (int frame = 0; frame < FRAMES && status == 0; ++frame)
        {
            makeFrame(frame + 20, inputRgb, inputBgra);
            status = datamosh_mosh_engine_process_rgb24(
                cpu,
                inputRgb.data(),
                inputRgb.size(),
                cpuOutput.data(),
                cpuOutput.size());
            if (!cudaOk(
                    cudaMemcpy2DToArray(
                        inputArray,
                        0,
                        0,
                        inputBgra.data(),
                        W * sizeof(uchar4),
                        W * sizeof(uchar4),
                        H,
                        cudaMemcpyHostToDevice),
                    "cudaMemcpy2DToArray overrides") ||
                !cudaOk(
                    datamoshWaveletCudaProcess(
                        gpu, inputSurface, outputSurface, params, 0),
                    "datamoshWaveletCudaProcess overrides"))
                status = -1;
        }

        double mae = THRESHOLD + 1.0;
        if (status == 0 &&
            cudaOk(cudaDeviceSynchronize(), "cudaDeviceSynchronize overrides") &&
            cudaOk(
                cudaMemcpy2DFromArray(
                    cudaOutput.data(),
                    W * sizeof(uchar4),
                    outputArray,
                    0,
                    0,
                    W * sizeof(uchar4),
                    H,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy2DFromArray overrides"))
        {
            double error = 0.0;
            for (int pixel = 0; pixel < W * H; ++pixel)
            {
                error += std::abs(
                    static_cast<int>(cpuOutput[pixel * 3 + 0]) -
                    static_cast<int>(cudaOutput[pixel].z));
                error += std::abs(
                    static_cast<int>(cpuOutput[pixel * 3 + 1]) -
                    static_cast<int>(cudaOutput[pixel].y));
                error += std::abs(
                    static_cast<int>(cpuOutput[pixel * 3 + 2]) -
                    static_cast<int>(cudaOutput[pixel].x));
            }
            mae = error / (static_cast<double>(W) * H * 3);
        }
        const bool ok = status == 0 && mae <= THRESHOLD;
        std::printf(
            "  %-20s : MAE=%7.3f  %s\n",
            "manual-overrides",
            mae,
            ok ? "ok" : "DRIFT");
        if (!ok)
            ++failures;
        if (mae > worst)
        {
            worst = mae;
            worstName = "manual-overrides";
        }
        datamoshWaveletCudaDestroy(gpu);
        if (cpu)
            datamosh_mosh_engine_free(cpu);
    }

    cudaDestroySurfaceObject(inputSurface);
    cudaDestroySurfaceObject(outputSurface);
    cudaFreeArray(inputArray);
    cudaFreeArray(outputArray);
    std::printf(
        "worst: %s MAE=%.3f => %s\n",
        worstName,
        worst,
        failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

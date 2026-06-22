#define DATAMOSH_STATIC
#include "../../include/datamosh_ffi.h"

#include <windows.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

template <typename Fn>
Fn load_symbol(HMODULE dll, const char* name) {
    FARPROC proc = GetProcAddress(dll, name);
    if (!proc) {
        std::cerr << "missing symbol: " << name << "\n";
        std::exit(2);
    }
    return reinterpret_cast<Fn>(proc);
}

int main(int argc, char** argv) {
    const char* dll_path = argc > 1 ? argv[1] : "target\\release\\datamosh.dll";
    HMODULE dll = LoadLibraryA(dll_path);
    if (!dll) {
        std::cerr << "failed to load " << dll_path << "\n";
        return 2;
    }

    using status_message_fn = const char* (*)(int32_t);
    using backend_count_fn = size_t (*)();
    using default_backend_fn = uint32_t (*)();
    using backend_name_fn = const char* (*)(uint32_t);
    using new_with_backend_fn = DatamoshMoshEngine* (*)(uint32_t, size_t, size_t);
    using engine_backend_fn = uint32_t (*)(const DatamoshMoshEngine*);
    using free_fn = void (*)(DatamoshMoshEngine*);
    using set_preset_fn = int32_t (*)(DatamoshMoshEngine*, const char*);
    using set_controls_fn = int32_t (*)(DatamoshMoshEngine*, float, float, float, float, float);
    using set_parameter_fn = int32_t (*)(DatamoshMoshEngine*, const char*, float);
    using reset_glitch_fn = int32_t (*)(DatamoshMoshEngine*);
    using process_rgba8_fn =
        int32_t (*)(DatamoshMoshEngine*, const uint8_t*, size_t, uint8_t*, size_t);

    auto status_message = load_symbol<status_message_fn>(dll, "datamosh_status_message");
    auto backend_count =
        load_symbol<backend_count_fn>(dll, "datamosh_mosh_engine_backend_count");
    auto default_backend =
        load_symbol<default_backend_fn>(dll, "datamosh_mosh_engine_default_backend");
    auto backend_name =
        load_symbol<backend_name_fn>(dll, "datamosh_mosh_engine_backend_name");
    auto engine_new =
        load_symbol<new_with_backend_fn>(dll, "datamosh_mosh_engine_new_with_backend");
    auto engine_backend =
        load_symbol<engine_backend_fn>(dll, "datamosh_mosh_engine_backend");
    auto engine_free = load_symbol<free_fn>(dll, "datamosh_mosh_engine_free");
    auto set_preset = load_symbol<set_preset_fn>(dll, "datamosh_mosh_engine_set_preset");
    auto set_controls =
        load_symbol<set_controls_fn>(dll, "datamosh_mosh_engine_set_controls");
    auto set_parameter =
        load_symbol<set_parameter_fn>(dll, "datamosh_mosh_engine_set_parameter");
    auto reset_glitch =
        load_symbol<reset_glitch_fn>(dll, "datamosh_mosh_engine_reset_glitch");
    auto process_rgba8 =
        load_symbol<process_rgba8_fn>(dll, "datamosh_mosh_engine_process_rgba8");

    const uint32_t backend = default_backend();
    if (backend_count() < 3 || backend != DATAMOSH_BACKEND_RAW_MOSH_V1) {
        std::cerr << "unexpected backend table\n";
        return 3;
    }

    const size_t width = 8;
    const size_t height = 8;
    DatamoshMoshEngine* engine = engine_new(backend, width, height);
    if (!engine) {
        std::cerr << "failed to create engine\n";
        return 3;
    }

    auto check = [&](int32_t status, const char* operation) {
        if (status != DATAMOSH_STATUS_OK) {
            std::cerr << operation << " failed: " << status_message(status) << "\n";
            engine_free(engine);
            std::exit(4);
        }
    };

    check(set_preset(engine, "codebook"), "set_preset");
    check(set_parameter(engine, "codebook_replace_every", 4.0f), "set_parameter");
    check(set_controls(engine, 1.0f, 0.8f, 1.0f, 0.7f, 1.0f), "set_controls");

    std::vector<uint8_t> input(width * height * 4);
    std::vector<uint8_t> output(input.size(), 0);
    for (size_t i = 0; i < width * height; ++i) {
        input[i * 4 + 0] = static_cast<uint8_t>((i * 3) & 0xff);
        input[i * 4 + 1] = static_cast<uint8_t>((i * 5) & 0xff);
        input[i * 4 + 2] = static_cast<uint8_t>((i * 7) & 0xff);
        input[i * 4 + 3] = static_cast<uint8_t>(128 + (i & 0x7f));
    }

    check(process_rgba8(engine, input.data(), input.size(), output.data(), output.size()),
          "process_rgba8 first frame");
    if (output != input) {
        std::cerr << "first keyframe output did not match input\n";
        engine_free(engine);
        return 5;
    }

    check(reset_glitch(engine), "reset_glitch");
    for (size_t i = 0; i < width * height; ++i) {
        input[i * 4 + 0] ^= 0x33;
        input[i * 4 + 1] ^= 0x55;
        input[i * 4 + 2] ^= 0x77;
    }
    check(process_rgba8(engine, input.data(), input.size(), output.data(), output.size()),
          "process_rgba8 after reset");
    for (size_t i = 0; i < width * height; ++i) {
        if (output[i * 4 + 3] != input[i * 4 + 3]) {
            std::cerr << "alpha channel was not preserved\n";
            engine_free(engine);
            return 5;
        }
    }

    const std::string raw_backend_name = backend_name(engine_backend(engine));
    engine_free(engine);

    engine = engine_new(DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1, width, height);
    if (!engine) {
        std::cerr << "failed to create scanline engine\n";
        return 3;
    }
    check(set_preset(engine, "plane"), "scanline set_preset");
    check(set_parameter(engine, "phase_offset", 2.0f), "scanline set_parameter");
    check(set_controls(engine, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f),
          "scanline set_controls");
    check(process_rgba8(engine, input.data(), input.size(), output.data(), output.size()),
          "scanline process_rgba8");
    for (size_t i = 0; i < width * height; ++i) {
        if (output[i * 4 + 3] != input[i * 4 + 3]) {
            std::cerr << "scanline alpha channel was not preserved\n";
            engine_free(engine);
            return 5;
        }
    }

    const std::string scanline_backend_name = backend_name(engine_backend(engine));
    engine_free(engine);

    engine = engine_new(DATAMOSH_BACKEND_DCT_TRANSFORM_V1, width, height);
    if (!engine) {
        std::cerr << "failed to create DCT engine\n";
        return 3;
    }
    check(set_preset(engine, "desync"), "DCT set_preset");
    check(set_controls(engine, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f),
          "DCT set_controls");
    check(process_rgba8(engine, input.data(), input.size(), output.data(), output.size()),
          "DCT process_rgba8");
    for (size_t i = 0; i < width * height; ++i) {
        if (output[i * 4 + 3] != input[i * 4 + 3]) {
            std::cerr << "DCT alpha channel was not preserved\n";
            engine_free(engine);
            return 5;
        }
    }

    std::cout << "cpp smoke ok: backends " << raw_backend_name << ", "
              << scanline_backend_name << ", " << backend_name(engine_backend(engine)) << "\n";
    engine_free(engine);
    FreeLibrary(dll);
    return 0;
}

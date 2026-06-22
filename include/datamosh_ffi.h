#ifndef DATAMOSH_FFI_H
#define DATAMOSH_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && !defined(DATAMOSH_STATIC)
#define DATAMOSH_API __declspec(dllimport)
#else
#define DATAMOSH_API
#endif

#define DATAMOSH_STATUS_OK 0
#define DATAMOSH_STATUS_NULL_POINTER -1
#define DATAMOSH_STATUS_INVALID_UTF8 -2
#define DATAMOSH_STATUS_INVALID_ARGUMENT -3
#define DATAMOSH_STATUS_PROCESS_ERROR -4
#define DATAMOSH_STATUS_PANIC -255

#define DATAMOSH_BACKEND_RAW_MOSH_V1 1u
#define DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1 2u
#define DATAMOSH_BACKEND_DCT_TRANSFORM_V1 3u
#define DATAMOSH_BACKEND_WAVELET_PYRAMID_V1 4u

typedef struct DatamoshMoshEngine DatamoshMoshEngine;

DATAMOSH_API const char* datamosh_status_message(int32_t status);

DATAMOSH_API size_t datamosh_mosh_engine_backend_count(void);
DATAMOSH_API uint32_t datamosh_mosh_engine_default_backend(void);
DATAMOSH_API const char* datamosh_mosh_engine_backend_name(uint32_t backend);

DATAMOSH_API DatamoshMoshEngine* datamosh_mosh_engine_new(size_t width, size_t height);
DATAMOSH_API DatamoshMoshEngine* datamosh_mosh_engine_new_with_backend(
    uint32_t backend,
    size_t width,
    size_t height
);
DATAMOSH_API uint32_t datamosh_mosh_engine_backend(const DatamoshMoshEngine* engine);
DATAMOSH_API void datamosh_mosh_engine_free(DatamoshMoshEngine* engine);

DATAMOSH_API int32_t datamosh_mosh_engine_set_preset(
    DatamoshMoshEngine* engine,
    const char* preset
);
DATAMOSH_API int32_t datamosh_mosh_engine_set_controls(
    DatamoshMoshEngine* engine,
    /* Controls use 0..1 for the authored range and 1..2 for overdrive. */
    float intensity,
    float motion,
    float residual,
    float temporal,
    float bitstream
);
DATAMOSH_API int32_t datamosh_mosh_engine_reset_controls(DatamoshMoshEngine* engine);
DATAMOSH_API int32_t datamosh_mosh_engine_set_parameter(
    DatamoshMoshEngine* engine,
    const char* id,
    float value
);
DATAMOSH_API int32_t datamosh_mosh_engine_reset_glitch(DatamoshMoshEngine* engine);

DATAMOSH_API int32_t datamosh_mosh_engine_process_rgb24(
    DatamoshMoshEngine* engine,
    const uint8_t* input,
    size_t input_len,
    uint8_t* output,
    size_t output_len
);
DATAMOSH_API int32_t datamosh_mosh_engine_process_rgba8(
    DatamoshMoshEngine* engine,
    const uint8_t* input,
    size_t input_len,
    uint8_t* output,
    size_t output_len
);

#undef DATAMOSH_API

#ifdef __cplusplus
}
#endif

#endif

#pragma once

// Shared pattern/preset resolution for the DCT CUDA TOP, hand-mirrored from
// load_dct_transform_preset / apply_dct_transform_controls (src/dct_codec.rs).
//
// This lives in a header so the TOP (DatamoshDctCudaTOP.cpp) and the CPU<->CUDA parity check
// (tools/dct_parity_check.cu) use the IDENTICAL preset table — that is what lets the parity
// check actually guard the TOP's hand-maintained values against the Rust codec.

#include "DatamoshDctCudaCore.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace dctcuda {

// Coefficient-domain presets only (the CUDA TOP has no entropy stage). Order matches the CPU
// DCT TOP and load_dct_transform_preset.
// Non-const elements so it decays to `const char**` for TD's appendMenu.
inline const char* kPatternNames[] = {
    "clean", "blocks", "dc-smear", "bleed", "blur", "ring",
    "scramble", "block-slip", "echo", "flow", "false-color", "composite",
};

inline int patternCount()
{
    return static_cast<int>(sizeof(kPatternNames) / sizeof(kPatternNames[0]));
}

inline int patternIndex(const char* name)
{
    if (!name)
        return 0;
    for (int i = 0; i < patternCount(); ++i)
        if (!std::strcmp(name, kPatternNames[i]))
            return i;
    return 0;
}

inline DatamoshDctCudaParams presetParams(int index)
{
    DatamoshDctCudaParams p;
    switch (index)
    {
        case 1: // blocks
            p.quantScale = 8.0f;
            break;
        case 2: // dc-smear
            p.dcDrift = 12;
            p.dcDriftEvery = 80;
            p.persistence = 0.5f;
            break;
        case 3: // bleed
            p.dcBlockOffset = 24;
            p.dcBlockOffsetEvery = 5;
            break;
        case 4: // blur
            p.acZeroAbove = 3;
            p.quantScale = 2.0f;
            break;
        case 5: // ring
            p.signFlipEvery = 3;
            p.blockTransposeEvery = 5;
            break;
        case 6: // scramble
            p.coeffShift = 7;
            p.coeffShiftEvery = 2;
            p.zigzagReverseEvery = 11;
            break;
        case 7: // block-slip
            p.blockShiftX = 3;
            p.blockShiftY = 1;
            p.blockShiftEvery = 4;
            break;
        case 8: // echo
            p.blockRepeatEvery = 5;
            break;
        case 9: // flow
            p.dcBlockOffset = 18;
            p.dcBlockOffsetEvery = 6;
            p.persistence = 0.7f;
            break;
        case 10: // false-color
            p.chromaSwapEvery = 2;
            p.dcDrift = 8;
            p.dcDriftEvery = 50;
            break;
        case 11: // composite
            p.quantScale = 5.0f;
            p.dcDrift = 7;
            p.dcDriftEvery = 96;
            p.dcBlockOffset = 12;
            p.dcBlockOffsetEvery = 11;
            p.acZeroAbove = 6;
            p.signFlipEvery = 17;
            p.coeffShift = 5;
            p.coeffShiftEvery = 7;
            p.blockShiftX = 2;
            p.blockShiftY = 1;
            p.blockShiftEvery = 19;
            p.blockRepeatEvery = 23;
            p.zigzagReverseEvery = 29;
            p.blockTransposeEvery = 31;
            p.chromaSwapEvery = 37;
            p.persistence = 0.4f;
            break;
        default: // clean
            break;
    }
    return p;
}

inline int scaleInt(int value, float amount)
{
    return static_cast<int>(std::lround(static_cast<float>(value) * amount));
}

inline int scalePeriod(int value, float amount)
{
    if (value == 0 || amount <= 0.0f)
        return 0;
    return std::max(1, static_cast<int>(std::lround(static_cast<float>(value) / amount)));
}

inline int scaleAcCutoff(int cutoff, float amount)
{
    if (cutoff == 0 || amount <= 0.0f)
        return 0;
    constexpr float clean = 63.0f;
    return std::clamp(
        static_cast<int>(std::lround(clean + (static_cast<float>(cutoff) - clean) * amount)),
        1,
        63);
}

// Hand-mirrored from apply_dct_transform_controls.
inline void applyControls(
    DatamoshDctCudaParams& p, float intensity, float structure, float persist, float dc,
    float quant)
{
    float master = std::max(0.0f, intensity);
    float quantAmt = master * std::max(0.0f, quant);
    float dcAmt = master * std::max(0.0f, dc);
    float structAmt = master * std::max(0.0f, structure);
    p.quantScale = std::max(1.0f, 1.0f + (p.quantScale - 1.0f) * quantAmt);
    p.dcDrift = scaleInt(p.dcDrift, dcAmt);
    p.dcDriftEvery = scalePeriod(p.dcDriftEvery, dcAmt);
    p.dcBlockOffset = scaleInt(p.dcBlockOffset, dcAmt);
    p.dcBlockOffsetEvery = scalePeriod(p.dcBlockOffsetEvery, dcAmt);
    p.acZeroAbove = scaleAcCutoff(p.acZeroAbove, quantAmt);
    p.signFlipEvery = scalePeriod(p.signFlipEvery, structAmt);
    p.coeffShift = scaleInt(p.coeffShift, structAmt);
    p.coeffShiftEvery = scalePeriod(p.coeffShiftEvery, structAmt);
    p.blockShiftX = scaleInt(p.blockShiftX, structAmt);
    p.blockShiftY = scaleInt(p.blockShiftY, structAmt);
    p.blockShiftEvery = scalePeriod(p.blockShiftEvery, structAmt);
    p.blockRepeatEvery = scalePeriod(p.blockRepeatEvery, structAmt);
    p.zigzagReverseEvery = scalePeriod(p.zigzagReverseEvery, structAmt);
    p.blockTransposeEvery = scalePeriod(p.blockTransposeEvery, structAmt);
    p.chromaSwapEvery = scalePeriod(p.chromaSwapEvery, structAmt);
    p.persistence =
        std::clamp(p.persistence * master * std::max(0.0f, persist), 0.0f, 0.98f);
}

} // namespace dctcuda

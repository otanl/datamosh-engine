#pragma once

#include "DatamoshWaveletCudaCore.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace waveletcuda {

inline const char* kPatternNames[] = {
    "clean",
    "subband-slip",
    "orientation-cross",
    "scale-fold",
    "bitplane-rain",
    "lowpass-ghost",
    "temporal-weave",
    "packet-loss",
    "lifting-drift",
    "chroma-pyramid",
    "hierarchy-collapse",
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

inline DatamoshWaveletCudaParams presetParams(int index)
{
    DatamoshWaveletCudaParams p;
    switch (index)
    {
        case 1:
            p.packetShift = 2;
            p.packetShiftEvery = 5;
            break;
        case 2:
            p.orientationRotate = 1;
            p.orientationRotateEvery = 3;
            break;
        case 3:
            p.levelFold = 1;
            p.levelFoldEvery = 4;
            break;
        case 4:
            p.bitplaneClear = 2;
            p.bitplaneClearEvery = 3;
            p.bitplaneXor = 4;
            p.bitplaneXorEvery = 11;
            break;
        case 5:
            p.lowpassHistoryLag = 5;
            break;
        case 6:
            p.historyLag = 6;
            p.historyBandEvery = 3;
            break;
        case 7:
            p.historyLag = 3;
            p.packetLossEvery = 5;
            p.packetLossConceal = 1;
            break;
        case 8:
            p.liftingBias = 12;
            p.liftingBiasEvery = 7;
            break;
        case 9:
            p.channelRoute = 1;
            p.channelRouteEvery = 4;
            p.levelFold = -1;
            p.levelFoldEvery = 7;
            break;
        case 10:
            p.packetShift = 5;
            p.packetShiftEvery = 4;
            p.orientationRotate = 1;
            p.orientationRotateEvery = 5;
            p.levelFold = 1;
            p.levelFoldEvery = 6;
            p.bitplaneClear = 2;
            p.bitplaneClearEvery = 3;
            p.historyLag = 4;
            p.historyBandEvery = 5;
            p.packetLossEvery = 9;
            p.packetLossConceal = 1;
            p.liftingBias = 8;
            p.liftingBiasEvery = 11;
            break;
        default:
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

inline int scaleUnsigned(int value, float amount)
{
    return std::max(0, static_cast<int>(std::lround(static_cast<float>(value) * amount)));
}

inline void applyControls(
    DatamoshWaveletCudaParams& p,
    float intensity,
    float structure,
    float coefficient,
    float history,
    float routing)
{
    const float master = std::clamp(intensity, 0.0f, 2.0f);
    const float structureAmount =
        std::clamp(master * std::clamp(structure, 0.0f, 2.0f), 0.0f, 4.0f);
    const float coefficientAmount =
        std::clamp(master * std::clamp(coefficient, 0.0f, 2.0f), 0.0f, 4.0f);
    const float historyAmount =
        std::clamp(master * std::clamp(history, 0.0f, 2.0f), 0.0f, 4.0f);
    const float routingAmount =
        std::clamp(master * std::clamp(routing, 0.0f, 2.0f), 0.0f, 4.0f);

    p.packetShift = scaleInt(p.packetShift, structureAmount);
    p.packetShiftEvery = scalePeriod(p.packetShiftEvery, structureAmount);
    p.orientationRotate = scaleInt(p.orientationRotate, structureAmount);
    p.orientationRotateEvery = scalePeriod(p.orientationRotateEvery, structureAmount);
    p.levelFold = scaleInt(p.levelFold, structureAmount);
    p.levelFoldEvery = scalePeriod(p.levelFoldEvery, structureAmount);

    p.bitplaneClear = scaleUnsigned(p.bitplaneClear, coefficientAmount);
    p.bitplaneClearEvery = scalePeriod(p.bitplaneClearEvery, coefficientAmount);
    p.bitplaneXor = scaleUnsigned(p.bitplaneXor, coefficientAmount);
    p.bitplaneXorEvery = scalePeriod(p.bitplaneXorEvery, coefficientAmount);
    p.signFlipEvery = scalePeriod(p.signFlipEvery, coefficientAmount);
    p.liftingBias = scaleInt(p.liftingBias, coefficientAmount);
    p.liftingBiasEvery = scalePeriod(p.liftingBiasEvery, coefficientAmount);

    p.historyLag = 1 + scaleUnsigned(std::max(0, p.historyLag - 1), historyAmount);
    p.historyBandEvery = scalePeriod(p.historyBandEvery, historyAmount);
    p.lowpassHistoryLag = scaleUnsigned(p.lowpassHistoryLag, historyAmount);

    p.channelRoute = scaleInt(p.channelRoute, routingAmount);
    p.channelRouteEvery = scalePeriod(p.channelRouteEvery, routingAmount);
    p.packetLossEvery = scalePeriod(p.packetLossEvery, routingAmount);
}

inline int roundedInt(float value, int minimum, int maximum)
{
    if (!std::isfinite(value))
        value = 0.0f;
    return std::clamp(static_cast<int>(std::lround(value)), minimum, maximum);
}

inline bool setParameter(
    DatamoshWaveletCudaParams& p,
    const char* id,
    float value,
    int maximumLevels)
{
    if (!id || !*id)
        return false;
    if (!std::isfinite(value))
        value = 0.0f;

    if (!std::strcmp(id, "quality"))
        p.quality = roundedInt(value, 1, 100);
    else if (!std::strcmp(id, "levels"))
        p.levels = roundedInt(value, 1, std::max(1, maximumLevels));
    else if (!std::strcmp(id, "history_len"))
        p.historyLength = roundedInt(value, 1, 128);
    else if (!std::strcmp(id, "packet_shift"))
        p.packetShift = roundedInt(value, INT16_MIN, INT16_MAX);
    else if (!std::strcmp(id, "packet_shift_every"))
        p.packetShiftEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "orientation_rotate"))
        p.orientationRotate = roundedInt(value, INT8_MIN, INT8_MAX);
    else if (!std::strcmp(id, "orientation_rotate_every"))
        p.orientationRotateEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "level_fold"))
        p.levelFold = roundedInt(value, INT8_MIN, INT8_MAX);
    else if (!std::strcmp(id, "level_fold_every"))
        p.levelFoldEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "channel_route"))
        p.channelRoute = roundedInt(value, INT8_MIN, INT8_MAX);
    else if (!std::strcmp(id, "channel_route_every"))
        p.channelRouteEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "packet_loss_every"))
        p.packetLossEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "packet_loss_conceal"))
        p.packetLossConceal = value >= 0.5f ? 1 : 0;
    else if (!std::strcmp(id, "bitplane_clear"))
        p.bitplaneClear = roundedInt(value, 0, 30);
    else if (!std::strcmp(id, "bitplane_clear_every"))
        p.bitplaneClearEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "bitplane_xor"))
        p.bitplaneXor = roundedInt(value, 0, 30);
    else if (!std::strcmp(id, "bitplane_xor_every"))
        p.bitplaneXorEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "sign_flip_every"))
        p.signFlipEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "history_lag"))
        p.historyLag = std::max(1, roundedInt(value, 1, INT32_MAX));
    else if (!std::strcmp(id, "history_band_every"))
        p.historyBandEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "lowpass_history_lag"))
        p.lowpassHistoryLag = std::max(0, roundedInt(value, 0, INT32_MAX));
    else if (!std::strcmp(id, "lifting_bias"))
        p.liftingBias = roundedInt(value, INT16_MIN, INT16_MAX);
    else if (!std::strcmp(id, "lifting_bias_every"))
        p.liftingBiasEvery = std::max(0, roundedInt(value, 0, INT32_MAX));
    else
        return false;
    return true;
}

} // namespace waveletcuda

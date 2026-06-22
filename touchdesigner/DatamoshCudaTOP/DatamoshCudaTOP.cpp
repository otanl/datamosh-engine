#include "DatamoshCudaTOP.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kImplementationVersion = 17;
constexpr int kOperatorVersion = 3;
constexpr int kPatternSchemaVersion = 2;

const char* kPatternNames[] = {
    "clean",
    "melt",
    "drift",
    "plane",
    "residue",
    "vector",
    "entropy",
    "codebook",
    "unstable",
    "pitch",
    "scale",
    "packetloss",
    "weave",
};

const char* kPatternLabels[] = {
    "Clean",
    "Motion Melt",
    "Temporal Slice Drift",
    "Channel Plane Desync",
    "Residual Stream Desync",
    "Motion Vector Bank Desync",
    "Entropy Byte Slip",
    "Residual Codebook Leak",
    "Codec State Collapse",
    "Row Pitch Fracture",
    "Residual Scale Mismatch",
    "Packet Tile Loss",
    "History Weave",
};

// UI order is stable across operators; CUDA kernel IDs remain ABI-compatible.
constexpr int kKernelPatternIds[] = {
    8,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    9,
    10,
    11,
    12,
};
static_assert(std::size(kPatternNames) == std::size(kKernelPatternIds));

const char* kVectorDecodeNames[] = {
    "pattern",
    "original",
    "reverse",
    "vertical",
    "static",
    "radial",
};

const char* kVectorDecodeLabels[] = {
    "Pattern",
    "Original",
    "Reverse",
    "Vertical",
    "Static",
    "Radial",
};

struct ParameterBinding
{
    const char* parName;
    const char* id;
};

constexpr ParameterBinding kParameterBindings[] = {
    {"Mvscale", "mv_scale"},
    {"Mvjitter", "mv_jitter"},
    {"Vectorinterp", "mv_field_interpolation"},
    {"Sampledesync", "sample_address_desync"},
    {"Reflag", "reference_lag"},
    {"Refbleed", "reference_bleed"},
    {"Reflatch", "reference_latch_frames"},
    {"Temporaldrift", "temporal_slice_drift"},
    {"Residkeep", "residual_keep"},
    {"Residjitter", "residual_address_jitter"},
    {"Residchannel", "residual_channel_shift"},
    {"Entropyevery", "entropy_slip_every"},
    {"Entropywindows", "entropy_slip_windows"},
    {"Coeffshift", "coeff_shift"},
    {"Coeffquant", "coeff_quant"},
    {"Codebookevery", "codebook_replace_every"},
    {"Codebookstride", "codebook_stride"},
    {"Codebookshuffle", "codebook_shuffle_every"},
};

int patternCount()
{
    return static_cast<int>(std::size(kPatternNames));
}

int kernelPatternId(int uiPatternIndex)
{
    return kKernelPatternIds[std::clamp(uiPatternIndex, 0, patternCount() - 1)];
}

int patternIndex(const char* name)
{
    if (!name)
        return 0;
    for (int index = 0; index < patternCount(); ++index)
    {
        if (!std::strcmp(name, kPatternNames[index]))
            return index;
    }
    return 0;
}

void configureFloat(
    OP_NumericParameter& parameter,
    const char* name,
    const char* label,
    double defaultValue,
    double sliderMin,
    double sliderMax,
    const char* page = "Datamosh")
{
    parameter.name = name;
    parameter.label = label;
    parameter.page = page;
    parameter.defaultValues[0] = defaultValue;
    parameter.minSliders[0] = sliderMin;
    parameter.maxSliders[0] = sliderMax;
    parameter.minValues[0] = sliderMin;
    parameter.maxValues[0] = sliderMax;
    parameter.clampMins[0] = true;
    parameter.clampMaxes[0] = true;
}

void appendInt(
    OP_ParameterManager* manager,
    const char* name,
    const char* label,
    int defaultValue,
    int minimum,
    int maximum)
{
    OP_NumericParameter parameter;
    parameter.name = name;
    parameter.label = label;
    parameter.page = "Codec";
    parameter.defaultValues[0] = defaultValue;
    parameter.minSliders[0] = minimum;
    parameter.maxSliders[0] = maximum;
    parameter.minValues[0] = minimum;
    parameter.maxValues[0] = maximum;
    parameter.clampMins[0] = true;
    parameter.clampMaxes[0] = true;
    manager->appendInt(parameter);
}

void appendFloat(
    OP_ParameterManager* manager,
    const char* page,
    const char* name,
    const char* label,
    double value,
    double sliderMin,
    double sliderMax,
    double minValue,
    double maxValue,
    bool clampMin = true,
    bool clampMax = true)
{
    OP_NumericParameter parameter;
    parameter.name = name;
    parameter.label = label;
    parameter.page = page;
    parameter.defaultValues[0] = value;
    parameter.minSliders[0] = sliderMin;
    parameter.maxSliders[0] = sliderMax;
    parameter.minValues[0] = minValue;
    parameter.maxValues[0] = maxValue;
    parameter.clampMins[0] = clampMin;
    parameter.clampMaxes[0] = clampMax;
    manager->appendFloat(parameter);
}

void appendToggle(
    OP_ParameterManager* manager,
    const char* page,
    const char* name,
    const char* label,
    double value = 0.0)
{
    OP_NumericParameter parameter;
    parameter.name = name;
    parameter.label = label;
    parameter.page = page;
    parameter.defaultValues[0] = value;
    parameter.minValues[0] = 0.0;
    parameter.maxValues[0] = 1.0;
    parameter.clampMins[0] = true;
    parameter.clampMaxes[0] = true;
    manager->appendToggle(parameter);
}

void appendString(
    OP_ParameterManager* manager,
    const char* page,
    const char* name,
    const char* label,
    const char* value = "")
{
    OP_StringParameter parameter;
    parameter.name = name;
    parameter.label = label;
    parameter.page = page;
    parameter.defaultValue = value;
    manager->appendString(parameter);
}

void appendCHOP(
    OP_ParameterManager* manager,
    const char* page,
    const char* name,
    const char* label)
{
    OP_StringParameter parameter;
    parameter.name = name;
    parameter.label = label;
    parameter.page = page;
    parameter.defaultValue = "";
    manager->appendCHOP(parameter);
}

int roundedInt(float value, int minimum, int maximum)
{
    if (!std::isfinite(value))
        value = 0.0f;
    return std::clamp(
        static_cast<int>(std::lround(value)), minimum, maximum);
}

int scaleInt(int value, float amount)
{
    return roundedInt(
        static_cast<float>(value) * amount, INT32_MIN, INT32_MAX);
}

int scalePeriod(int value, float amount)
{
    if (value <= 0 || amount <= 0.0f)
        return 0;
    return std::max(
        1,
        static_cast<int>(
            std::lround(static_cast<float>(value) / amount)));
}

bool setOverrideParameter(
    DatamoshCudaParams& params,
    const char* id,
    float value)
{
    if (!id || !*id)
        return false;
    if (!std::isfinite(value))
        value = 0.0f;

    if (!std::strcmp(id, "mv_scale"))
    {
        params.mvScale = std::clamp(value, 0.0f, 2.0f);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_MV_SCALE;
    }
    else if (!std::strcmp(id, "mv_jitter"))
    {
        params.mvJitter = roundedInt(value, 0, 16);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_MV_JITTER;
    }
    else if (!std::strcmp(id, "mv_field_interpolation"))
    {
        params.vectorInterpolation = std::clamp(value, 0.0f, 1.0f);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_VECTOR_INTERP;
    }
    else if (!std::strcmp(id, "sample_address_desync"))
    {
        params.sampleAddressDesync = std::clamp(value, 0.0f, 4.0f);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_SAMPLE_DESYNC;
    }
    else if (!std::strcmp(id, "reference_lag"))
    {
        params.referenceLag = roundedInt(value, 1, 32);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_REFERENCE_LAG;
    }
    else if (!std::strcmp(id, "reference_bleed"))
    {
        params.referenceBleed = std::clamp(value, 0.0f, 1.0f);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_REFERENCE_BLEED;
    }
    else if (!std::strcmp(id, "reference_latch_frames"))
    {
        params.referenceLatchFrames = roundedInt(value, 1, 64);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_REFERENCE_LATCH;
    }
    else if (!std::strcmp(id, "temporal_slice_drift"))
    {
        params.temporalSliceDrift = roundedInt(value, -16, 16);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_TEMPORAL_DRIFT;
    }
    else if (!std::strcmp(id, "residual_keep"))
    {
        params.residualKeep = std::clamp(value, -2.0f, 2.0f);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_RESIDUAL_KEEP;
    }
    else if (!std::strcmp(id, "residual_address_jitter"))
    {
        params.residualAddressJitter = roundedInt(value, 0, 32);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_RESIDUAL_JITTER;
    }
    else if (!std::strcmp(id, "residual_channel_shift"))
    {
        params.residualChannelShift = roundedInt(value, -4, 4);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_RESIDUAL_CHANNEL;
    }
    else if (!std::strcmp(id, "entropy_slip_every"))
    {
        params.entropySlipEvery = roundedInt(value, 0, 64);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_ENTROPY_EVERY;
    }
    else if (!std::strcmp(id, "entropy_slip_windows"))
    {
        params.entropySlipWindows = roundedInt(value, 0, 64);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_ENTROPY_WINDOWS;
    }
    else if (!std::strcmp(id, "coeff_shift"))
    {
        params.coeffShift = roundedInt(value, -32, 32);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_COEFF_SHIFT;
    }
    else if (!std::strcmp(id, "coeff_quant"))
    {
        params.coeffQuant = roundedInt(value, 1, 64);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_COEFF_QUANT;
    }
    else if (!std::strcmp(id, "codebook_replace_every"))
    {
        params.codebookReplaceEvery = roundedInt(value, 0, 64);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_CODEBOOK_EVERY;
    }
    else if (!std::strcmp(id, "codebook_stride"))
    {
        params.codebookStride = roundedInt(value, -128, 128);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_CODEBOOK_STRIDE;
    }
    else if (!std::strcmp(id, "codebook_shuffle_every"))
    {
        params.codebookShuffleEvery = roundedInt(value, 0, 64);
        params.overrideMask |= DATAMOSH_CUDA_OVERRIDE_CODEBOOK_SHUFFLE;
    }
    else if (!std::strcmp(id, "block_size"))
        params.blockSize = roundedInt(value, 4, 64);
    else if (!std::strcmp(id, "search_radius"))
        params.searchRadius = roundedInt(value, 0, 64);
    else if (!std::strcmp(id, "search_step"))
        params.searchStep = roundedInt(value, 1, 16);
    else if (!std::strcmp(id, "history_slots"))
        params.historySlots = roundedInt(value, 2, 16);
    else if (!std::strcmp(id, "seed"))
        params.seed = static_cast<uint32_t>(roundedInt(value, 0, 65535));
    else
        return false;
    return true;
}

void applyOverrideControls(DatamoshCudaParams& params)
{
    const float master = std::clamp(params.intensity, 0.0f, 2.0f);
    const float motion =
        std::clamp(master * std::clamp(params.motion, 0.0f, 2.0f), 0.0f, 4.0f);
    const float residual =
        std::clamp(master * std::clamp(params.residual, 0.0f, 2.0f), 0.0f, 4.0f);
    const float temporal =
        std::clamp(master * std::clamp(params.temporal, 0.0f, 2.0f), 0.0f, 4.0f);
    const float bitstream =
        std::clamp(master * std::clamp(params.bitstream, 0.0f, 2.0f), 0.0f, 4.0f);

    params.mvScale = 1.0f + (params.mvScale - 1.0f) * motion;
    params.mvJitter = scaleInt(params.mvJitter, motion);
    params.vectorInterpolation *= motion;
    params.sampleAddressDesync *= motion;

    params.referenceLag =
        1 + scaleInt(std::max(0, params.referenceLag - 1), temporal);
    params.referenceBleed *= temporal;
    params.referenceLatchFrames =
        1 + scaleInt(std::max(0, params.referenceLatchFrames - 1), temporal);
    params.temporalSliceDrift =
        scaleInt(params.temporalSliceDrift, temporal);

    params.residualKeep =
        1.0f + (params.residualKeep - 1.0f) * residual;
    params.residualAddressJitter =
        scaleInt(params.residualAddressJitter, residual);
    params.residualChannelShift =
        scaleInt(params.residualChannelShift, residual);

    params.entropySlipEvery =
        scalePeriod(params.entropySlipEvery, bitstream);
    params.entropySlipWindows =
        std::max(0, scaleInt(params.entropySlipWindows, bitstream));
    params.coeffShift = scaleInt(params.coeffShift, bitstream);
    params.coeffQuant =
        1 + scaleInt(std::max(0, params.coeffQuant - 1), bitstream);
    params.codebookReplaceEvery =
        scalePeriod(params.codebookReplaceEvery, bitstream);
    params.codebookStride = scaleInt(params.codebookStride, bitstream);
    params.codebookShuffleEvery =
        scalePeriod(params.codebookShuffleEvery, bitstream);
}

float clampFloat(float value, float minimum, float maximum)
{
    return std::clamp(value, minimum, maximum);
}

bool parseChannelIndex(const std::string& text, int32_t& index)
{
    if (text.empty())
        return false;
    char* end = nullptr;
    const long value = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0' || value < 0 || value > INT32_MAX)
        return false;
    index = static_cast<int32_t>(value);
    return true;
}

float latestChopValue(
    const OP_CHOPInput* chop,
    const std::string& channel,
    bool& found)
{
    found = false;
    if (!chop || chop->numChannels <= 0 || chop->numSamples <= 0 ||
        !chop->channelData)
        return 0.0f;
    int32_t index = -1;
    if (parseChannelIndex(channel, index))
    {
        if (index >= 0 && index < chop->numChannels)
        {
            const float* data = chop->getChannelData(index);
            if (data)
            {
                found = true;
                return data[chop->numSamples - 1];
            }
        }
        return 0.0f;
    }
    if (channel.empty() || !chop->nameData)
        return 0.0f;
    for (int32_t i = 0; i < chop->numChannels; ++i)
    {
        const char* name = chop->getChannelName(i);
        if (name && !std::strcmp(name, channel.c_str()))
        {
            const float* data = chop->getChannelData(i);
            if (data)
            {
                found = true;
                return data[chop->numSamples - 1];
            }
        }
    }
    return 0.0f;
}

float blendAudioValue(
    float base,
    float sample,
    float amount,
    float gain,
    float bias)
{
    const float target = sample * gain + bias;
    return clampFloat(
        base * (1.0f - amount) + target * amount, 0.0f, 8.0f);
}

void setupSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
    if (*surface)
    {
        cudaResourceDesc existing = {};
        cudaGetSurfaceObjectResourceDesc(&existing, *surface);
        if (existing.resType != cudaResourceTypeArray || existing.res.array.array != array)
        {
            cudaDestroySurfaceObject(*surface);
            *surface = 0;
        }
    }

    if (!*surface)
    {
        cudaResourceDesc resource = {};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = array;
        cudaCreateSurfaceObject(surface, &resource);
    }
}

} // namespace

DatamoshCudaTOP::DatamoshCudaTOP(const OP_NodeInfo* info, TOP_Context* context)
    : myNodeInfo(info), myContext(context)
{
    cudaError_t status = cudaStreamCreate(&myStream);
    if (status != cudaSuccess)
        setCudaError("cudaStreamCreate", status);
}

DatamoshCudaTOP::~DatamoshCudaTOP()
{
    releaseState();
    if (myInputSurface)
        cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface)
        cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)
        cudaStreamDestroy(myStream);
}

void DatamoshCudaTOP::getGeneralInfo(TOP_GeneralInfo* info, const OP_Inputs*, void*)
{
    info->cookEveryFrame = true;
    info->cookEveryFrameIfAsked = true;
}

void DatamoshCudaTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    ++myExecuteCount;
    myCookStage = 1;
    myError.clear();
    myWarning.clear();

    const OP_TOPInput* input = inputs->getInputTOP(0);
    if (!input)
    {
        myWarning = "Connect a TOP input";
        return;
    }
    myInputCooks = input->totalCooks;
    myCookStage = 2;
    const OP_PixelFormat inputFormat = input->textureDesc.pixelFormat;
    myInputFormat = static_cast<int32_t>(inputFormat);
    const bool supportedFormat =
        inputFormat == OP_PixelFormat::BGRA8Fixed ||
        inputFormat == OP_PixelFormat::RGBA8Fixed ||
        inputFormat == OP_PixelFormat::RGBA16Fixed ||
        inputFormat == OP_PixelFormat::RGBA16Float ||
        inputFormat == OP_PixelFormat::RGBA32Float;
    if (input->textureDesc.texDim != OP_TexDim::e2D || !supportedFormat)
    {
        myError = "GPU codec requires a 2D BGRA/RGBA 8/16/32-bit input";
        return;
    }

    DatamoshCudaParams params;
    const char* pattern = inputs->getParString("Pattern");
    myPatternIndex = std::clamp(inputs->getParInt("Pattern"), 0, patternCount() - 1);
    params.pattern = kernelPatternId(myPatternIndex);
    myPatternName = kPatternNames[myPatternIndex];
    if (pattern && patternIndex(pattern) == myPatternIndex)
        myPatternName = pattern;
    params.blockSize = std::clamp(inputs->getParInt("Blocksize"), 4, 64);
    params.searchRadius = std::clamp(inputs->getParInt("Searchradius"), 0, 32);
    params.searchStep = std::clamp(inputs->getParInt("Searchstep"), 1, 16);
    params.historySlots = std::clamp(inputs->getParInt("History"), 2, 16);
    params.inputFormat = static_cast<int>(inputFormat);
    const char* vectorDecode = inputs->getParString("Vectordecode");
    params.vectorDecode = 0;
    if (vectorDecode)
    {
        for (int index = 0; index < static_cast<int>(std::size(kVectorDecodeNames)); ++index)
        {
            if (!std::strcmp(vectorDecode, kVectorDecodeNames[index]))
            {
                params.vectorDecode = index;
                break;
            }
        }
    }
    if (myPatternIndex != myLastPatternIndex ||
        params.vectorDecode != myLastVectorDecode)
    {
        myResetPending = true;
        myLastPatternIndex = myPatternIndex;
        myLastVectorDecode = params.vectorDecode;
    }
    params.seed = static_cast<uint32_t>(std::max(inputs->getParInt("Seed"), 0));

    myUseParams = inputs->getParInt("Useparams") != 0;
    if (myLastUseParams && !myUseParams)
        myResetPending = true;
    myLastUseParams = myUseParams;
    if (myUseParams)
    {
        params.overrideMask = DATAMOSH_CUDA_OVERRIDE_ALL;
        for (const ParameterBinding& binding : kParameterBindings)
        {
            setOverrideParameter(
                params,
                binding.id,
                static_cast<float>(inputs->getParDouble(binding.parName)));
        }
    }

    const char* parameterId = inputs->getParString("Paramid");
    myParameterId = parameterId ? parameterId : "";
    if (!myLastParameterId.empty() && myParameterId != myLastParameterId)
        myResetPending = true;
    myLastParameterId = myParameterId;
    if (!myParameterId.empty() &&
        !setOverrideParameter(
            params,
            myParameterId.c_str(),
            static_cast<float>(inputs->getParDouble("Paramvalue"))))
    {
        myWarning =
            "Unsupported CUDA motion parameter: " + myParameterId;
    }

    myIntensity = static_cast<float>(inputs->getParDouble("Intensity"));
    myMotion = static_cast<float>(inputs->getParDouble("Motion"));
    myResidual = static_cast<float>(inputs->getParDouble("Residual"));
    myTemporal = static_cast<float>(inputs->getParDouble("Temporal"));
    myBitstream = static_cast<float>(inputs->getParDouble("Bitstream"));
    applyAudioControlInputs(inputs);
    params.intensity = std::clamp(myIntensity, 0.0f, 2.0f);
    params.motion = std::clamp(myMotion, 0.0f, 2.0f);
    params.residual = std::clamp(myResidual, 0.0f, 2.0f);
    params.temporal = std::clamp(myTemporal, 0.0f, 2.0f);
    params.bitstream = std::clamp(myBitstream, 0.0f, 2.0f);
    applyOverrideControls(params);

    const int width = static_cast<int>(input->textureDesc.width);
    const int height = static_cast<int>(input->textureDesc.height);

    OP_CUDAAcquireInfo acquireInfo;
    acquireInfo.stream = myStream;
    const OP_CUDAArrayInfo* inputInfo = input->getCUDAArray(acquireInfo, nullptr);
    myCookStage = 3;

    TOP_CUDAOutputInfo outputRequest;
    outputRequest.textureDesc = input->textureDesc;
    outputRequest.textureDesc.pixelFormat = OP_PixelFormat::BGRA8Fixed;
    outputRequest.stream = myStream;
    const OP_CUDAArrayInfo* outputInfo = output->createCUDAArray(outputRequest, nullptr);
    myCookStage = 4;
    if (!inputInfo || !outputInfo)
    {
        myError = "TouchDesigner could not acquire CUDA arrays";
        return;
    }

    if (!myContext->beginCUDAOperations(nullptr))
    {
        myError = "TouchDesigner could not begin CUDA operations";
        return;
    }
    myCookStage = 5;

    if (!ensureState(width, height, params))
    {
        myContext->endCUDAOperations(nullptr);
        return;
    }
    myCookStage = 6;

    setupSurface(&myInputSurface, inputInfo->cudaArray);
    setupSurface(&myOutputSurface, outputInfo->cudaArray);
    if (myResetPending)
    {
        datamoshCudaReset(myState);
        myResetPending = false;
    }

    cudaError_t status =
        datamoshCudaProcess(myState, myInputSurface, myOutputSurface, params, myStream);
    if (status == cudaSuccess)
    {
        ++myProcessedFrames;
        myCookStage = 7;
    }
    else
        setCudaError("datamoshCudaProcess", status);

    myContext->endCUDAOperations(nullptr);
    if (status == cudaSuccess)
        myCookStage = 8;
}

int32_t DatamoshCudaTOP::getNumInfoCHOPChans(void*)
{
    return 19;
}

void DatamoshCudaTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
    switch (index)
    {
        case 0:
            chan->name->setString("execute_count");
            chan->value = static_cast<float>(myExecuteCount);
            break;
        case 1:
            chan->name->setString("processed_frames");
            chan->value = static_cast<float>(myProcessedFrames);
            break;
        case 2:
            chan->name->setString("width");
            chan->value = static_cast<float>(myWidth);
            break;
        case 3:
            chan->name->setString("height");
            chan->value = static_cast<float>(myHeight);
            break;
        case 4:
            chan->name->setString("input_cooks");
            chan->value = static_cast<float>(myInputCooks);
            break;
        case 5:
            chan->name->setString("cook_stage");
            chan->value = static_cast<float>(myCookStage);
            break;
        case 6:
            chan->name->setString("input_format");
            chan->value = static_cast<float>(myInputFormat);
            break;
        case 7:
            chan->name->setString("pattern_index");
            chan->value = static_cast<float>(myPatternIndex);
            break;
        case 8:
            chan->name->setString("implementation_version");
            chan->value = static_cast<float>(kImplementationVersion);
            break;
        case 9:
            chan->name->setString("pattern_count");
            chan->value = static_cast<float>(patternCount());
            break;
        case 10:
            chan->name->setString("operator_version");
            chan->value = static_cast<float>(kOperatorVersion);
            break;
        case 11:
            chan->name->setString("pattern_schema_version");
            chan->value = static_cast<float>(kPatternSchemaVersion);
            break;
        case 12:
            chan->name->setString("audio_active");
            chan->value = myAudioActive ? 1.0f : 0.0f;
            break;
        case 13:
            chan->name->setString("audio_reset");
            chan->value = myAudioResetValue;
            break;
        case 14:
            chan->name->setString("intensity");
            chan->value = myIntensity;
            break;
        case 15:
            chan->name->setString("motion");
            chan->value = myMotion;
            break;
        case 16:
            chan->name->setString("residual");
            chan->value = myResidual;
            break;
        case 17:
            chan->name->setString("temporal");
            chan->value = myTemporal;
            break;
        default:
            chan->name->setString("bitstream");
            chan->value = myBitstream;
            break;
    }
}

bool DatamoshCudaTOP::getInfoDATSize(OP_InfoDATSize* info, void*)
{
    info->rows = 14;
    info->cols = 2;
    info->byColumn = false;
    return true;
}

void DatamoshCudaTOP::getInfoDATEntries(
    int32_t index,
    int32_t,
    OP_InfoDATEntries* entries,
    void*)
{
    const char* key = "";
    std::string value;
    switch (index)
    {
        case 0:
            key = "backend";
            value = "cuda_motion_v1";
            break;
        case 1:
            key = "pattern";
            value = myPatternName;
            break;
        case 2:
            key = "resolution";
            value = std::to_string(myWidth) + "x" + std::to_string(myHeight);
            break;
        case 3:
            key = "history_slots";
            value = std::to_string(myHistorySlots);
            break;
        case 4:
            key = "audio";
            value = myAudioActive ? "active" : "inactive";
            break;
        case 5:
            key = "param_overrides";
            if (myUseParams && !myParameterId.empty())
                value = "dedicated+advanced";
            else if (myUseParams)
                value = "dedicated";
            else if (!myParameterId.empty())
                value = "advanced";
            else
                value = "off";
            break;
        case 6:
            key = "input_cooks";
            value = std::to_string(myInputCooks);
            break;
        case 7:
            key = "cook_stage";
            value = std::to_string(myCookStage);
            break;
        case 8:
            key = "input_format";
            value = std::to_string(myInputFormat);
            break;
        case 9:
            key = "pattern_index";
            value = std::to_string(myPatternIndex);
            break;
        case 10:
            key = "implementation_version";
            value = std::to_string(kImplementationVersion);
            break;
        case 11:
            key = "pattern_count";
            value = std::to_string(patternCount());
            break;
        case 12:
            key = "operator_version";
            value = std::to_string(kOperatorVersion);
            break;
        default:
            key = "pattern_schema_version";
            value = std::to_string(kPatternSchemaVersion);
            break;
    }
    entries->values[0]->setString(key);
    entries->values[1]->setString(value.c_str());
}

void DatamoshCudaTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError.c_str());
}

void DatamoshCudaTOP::getWarningString(OP_String* warning, void*)
{
    warning->setString(myWarning.c_str());
}

void DatamoshCudaTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    OP_StringParameter pattern;
    pattern.name = "Pattern";
    pattern.label = "Pattern";
    pattern.page = "Datamosh";
    pattern.defaultValue = "clean";
    manager->appendMenu(pattern, patternCount(), kPatternNames, kPatternLabels);

    for (const auto& definition : std::array<std::array<const char*, 2>, 5>{{
             {"Intensity", "Intensity"},
             {"Motion", "Motion"},
             {"Residual", "Residual"},
             {"Temporal", "Temporal"},
             {"Bitstream", "Bitstream"},
         }})
    {
        OP_NumericParameter parameter;
        configureFloat(parameter, definition[0], definition[1], 1.0, 0.0, 2.0);
        manager->appendFloat(parameter);
    }
    appendToggle(manager, "Datamosh", "Useparams", "Use Overrides");

    appendInt(manager, "Blocksize", "Block Size", 16, 4, 64);
    appendInt(manager, "Searchradius", "Search Radius", 8, 0, 32);
    appendInt(manager, "Searchstep", "Search Step", 4, 1, 16);
    appendInt(manager, "History", "History", 8, 2, 16);
    appendInt(manager, "Seed", "Seed", 1, 0, 65535);

    OP_StringParameter vectorDecode;
    vectorDecode.name = "Vectordecode";
    vectorDecode.label = "Vector Decode";
    vectorDecode.page = "Codec";
    vectorDecode.defaultValue = "pattern";
    manager->appendStringMenu(
        vectorDecode,
        static_cast<int32_t>(std::size(kVectorDecodeNames)),
        kVectorDecodeNames,
        kVectorDecodeLabels);

    appendFloat(manager, "Motion", "Mvscale", "MV Scale", 1.0, 0.0, 2.0, 0.0, 2.0);
    appendFloat(manager, "Motion", "Mvjitter", "MV Jitter", 0.0, 0.0, 16.0, 0.0, 16.0);
    appendFloat(
        manager, "Motion", "Vectorinterp", "Vector Interp", 0.0, 0.0, 1.0, 0.0, 1.0);
    appendFloat(
        manager, "Motion", "Sampledesync", "Sample Desync", 0.0, 0.0, 4.0, 0.0, 4.0);

    appendFloat(
        manager, "Reference", "Reflag", "Reference Lag", 1.0, 1.0, 16.0, 1.0, 32.0);
    appendFloat(
        manager, "Reference", "Refbleed", "Reference Bleed", 0.0, 0.0, 1.0, 0.0, 1.0);
    appendFloat(
        manager, "Reference", "Reflatch", "Reference Latch", 1.0, 1.0, 32.0, 1.0, 64.0);
    appendFloat(
        manager,
        "Reference",
        "Temporaldrift",
        "Temporal Drift",
        0.0,
        -16.0,
        16.0,
        -16.0,
        16.0,
        false,
        false);

    appendFloat(
        manager,
        "Residual",
        "Residkeep",
        "Residual Keep",
        1.0,
        -2.0,
        2.0,
        -2.0,
        2.0,
        false,
        false);
    appendFloat(
        manager, "Residual", "Residjitter", "Residual Jitter", 0.0, 0.0, 32.0, 0.0, 32.0);
    appendFloat(
        manager,
        "Residual",
        "Residchannel",
        "Residual Channel",
        0.0,
        -4.0,
        4.0,
        -4.0,
        4.0,
        false,
        false);

    appendFloat(
        manager,
        "Bitstream",
        "Entropyevery",
        "Entropy Period",
        0.0,
        0.0,
        64.0,
        0.0,
        64.0);
    appendFloat(
        manager,
        "Bitstream",
        "Entropywindows",
        "Entropy Windows",
        1.0,
        0.0,
        16.0,
        0.0,
        64.0);
    appendFloat(
        manager,
        "Bitstream",
        "Coeffshift",
        "Coeff Shift",
        0.0,
        -32.0,
        32.0,
        -32.0,
        32.0,
        false,
        false);
    appendFloat(
        manager, "Bitstream", "Coeffquant", "Coeff Quant", 1.0, 1.0, 32.0, 1.0, 64.0);
    appendFloat(
        manager,
        "Bitstream",
        "Codebookevery",
        "Codebook Period",
        0.0,
        0.0,
        32.0,
        0.0,
        64.0);
    appendFloat(
        manager,
        "Bitstream",
        "Codebookstride",
        "Codebook Stride",
        1.0,
        -64.0,
        64.0,
        -128.0,
        128.0,
        false,
        false);
    appendFloat(
        manager,
        "Bitstream",
        "Codebookshuffle",
        "Codebook Shuffle",
        0.0,
        0.0,
        32.0,
        0.0,
        64.0);

    appendString(manager, "Advanced", "Paramid", "Param ID");
    appendFloat(
        manager,
        "Advanced",
        "Paramvalue",
        "Param Value",
        0.0,
        -64.0,
        64.0,
        -4096.0,
        4096.0,
        false,
        false);

    appendToggle(manager, "Audio", "Audioenable", "Audio Enable");
    appendCHOP(manager, "Audio", "Controlchop", "Control CHOP");
    appendFloat(
        manager, "Audio", "Audioamount", "Audio Amount", 1.0, 0.0, 1.0, 0.0, 1.0);
    appendFloat(
        manager,
        "Audio",
        "Audiogain",
        "Audio Gain",
        1.0,
        0.0,
        4.0,
        -64.0,
        64.0,
        false,
        false);
    appendFloat(
        manager,
        "Audio",
        "Audiobias",
        "Audio Bias",
        0.0,
        -2.0,
        2.0,
        -64.0,
        64.0,
        false,
        false);
    appendString(manager, "Audio", "Intensitychan", "Intensity Chan", "0");
    appendString(manager, "Audio", "Motionchan", "Motion Chan");
    appendString(manager, "Audio", "Residualchan", "Residual Chan");
    appendString(manager, "Audio", "Temporalchan", "Temporal Chan");
    appendString(manager, "Audio", "Bitstreamchan", "Bitstream Chan");
    appendString(manager, "Audio", "Resetchan", "Reset Chan");
    appendFloat(
        manager,
        "Audio",
        "Resetthreshold",
        "Reset Threshold",
        0.75,
        0.0,
        1.0,
        0.0,
        64.0);
    appendFloat(
        manager, "Audio", "Resetrearm", "Reset Rearm", 0.25, 0.0, 1.0, 0.0, 64.0);

    OP_NumericParameter reset;
    reset.name = "Resetglitch";
    reset.label = "Reset Glitch";
    reset.page = "Datamosh";
    manager->appendPulse(reset);

    OP_NumericParameter recreate;
    recreate.name = "Recreate";
    recreate.label = "Recreate Engine";
    recreate.page = "Datamosh";
    manager->appendPulse(recreate);
}

void DatamoshCudaTOP::pulsePressed(const char* name, void*)
{
    if (!std::strcmp(name, "Resetglitch"))
        myResetPending = true;
    else if (!std::strcmp(name, "Recreate"))
    {
        releaseState();
        myResetPending = true;
    }
}

void DatamoshCudaTOP::applyAudioControlInputs(
    const OP_Inputs* inputs)
{
    myAudioActive = false;
    myAudioResetValue = 0.0f;
    if (inputs->getParInt("Audioenable") == 0)
        return;

    const OP_CHOPInput* chop = inputs->getParCHOP("Controlchop");
    if (!chop)
        return;

    const float amount = clampFloat(
        static_cast<float>(inputs->getParDouble("Audioamount")), 0.0f, 1.0f);
    const float gain = static_cast<float>(inputs->getParDouble("Audiogain"));
    const float bias = static_cast<float>(inputs->getParDouble("Audiobias"));
    auto applyChannel = [&](const char* parName, float& destination) {
        const char* channelName = inputs->getParString(parName);
        bool found = false;
        const float value = latestChopValue(
            chop, channelName ? channelName : "", found);
        if (found)
        {
            destination =
                blendAudioValue(destination, value, amount, gain, bias);
            myAudioActive = true;
        }
    };

    applyChannel("Intensitychan", myIntensity);
    applyChannel("Motionchan", myMotion);
    applyChannel("Residualchan", myResidual);
    applyChannel("Temporalchan", myTemporal);
    applyChannel("Bitstreamchan", myBitstream);

    const char* resetChannel = inputs->getParString("Resetchan");
    bool resetFound = false;
    myAudioResetValue = latestChopValue(
        chop, resetChannel ? resetChannel : "", resetFound);
    if (!resetFound)
        return;

    myAudioActive = true;
    const float threshold =
        static_cast<float>(inputs->getParDouble("Resetthreshold"));
    const float rearm =
        static_cast<float>(inputs->getParDouble("Resetrearm"));
    if (myAudioResetArmed && myAudioResetValue >= threshold)
    {
        myResetPending = true;
        myAudioResetArmed = false;
    }
    else if (myAudioResetValue <= rearm)
    {
        myAudioResetArmed = true;
    }
}

bool DatamoshCudaTOP::ensureState(
    int width,
    int height,
    const DatamoshCudaParams& params)
{
    if (myState && width == myWidth && height == myHeight &&
        params.blockSize == myBlockSize && params.historySlots == myHistorySlots)
        return true;

    releaseState();
    cudaError_t status =
        datamoshCudaCreate(&myState, width, height, params.blockSize, params.historySlots);
    if (status != cudaSuccess)
    {
        setCudaError("datamoshCudaCreate", status);
        return false;
    }

    myWidth = width;
    myHeight = height;
    myBlockSize = params.blockSize;
    myHistorySlots = params.historySlots;
    myResetPending = true;
    return true;
}

void DatamoshCudaTOP::releaseState()
{
    if (myState)
    {
        datamoshCudaDestroy(myState);
        myState = nullptr;
    }
    myWidth = 0;
    myHeight = 0;
    myBlockSize = 0;
    myHistorySlots = 0;
}

void DatamoshCudaTOP::setCudaError(const char* operation, cudaError_t error)
{
    myError = operation;
    myError += ": ";
    myError += cudaGetErrorString(error);
}

extern "C" {

DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;
    info->executeMode = TOP_ExecuteMode::CUDA;
    info->customOPInfo.opType->setString("Datamoshcuda");
    info->customOPInfo.opLabel->setString("Datamosh Motion CUDA TOP");
    info->customOPInfo.opIcon->setString("DMC");
    info->customOPInfo.authorName->setString("datamosh");
    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(
    const OP_NodeInfo* info,
    TOP_Context* context)
{
    return new DatamoshCudaTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<DatamoshCudaTOP*>(instance);
}

}

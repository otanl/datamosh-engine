#include "DatamoshWaveletCudaTOP.h"

#include "DatamoshWaveletCudaPresets.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kImplementationVersion = 2;
constexpr int kOperatorVersion = 2;
constexpr int kPatternSchemaVersion = 1;

using namespace waveletcuda;

const char* kPatternLabels[] = {
    "Clean",
    "Subband Packet Slip",
    "Orientation Cross",
    "Scale Fold",
    "Bitplane Rain",
    "Lowpass Ghost",
    "Temporal Pyramid Weave",
    "Packet Loss Concealment",
    "Lifting State Drift",
    "Chroma Pyramid Route",
    "Hierarchy Collapse",
};

struct ParameterBinding
{
    const char* parName;
    const char* id;
};

constexpr ParameterBinding kParameterBindings[] = {
    {"Packetshift", "packet_shift"},
    {"Packetshiftperiod", "packet_shift_every"},
    {"Orientation", "orientation_rotate"},
    {"Orientationperiod", "orientation_rotate_every"},
    {"Levelfold", "level_fold"},
    {"Levelfoldperiod", "level_fold_every"},
    {"Channelroute", "channel_route"},
    {"Channelperiod", "channel_route_every"},
    {"Packetloss", "packet_loss_every"},
    {"Packetconceal", "packet_loss_conceal"},
    {"Bitclear", "bitplane_clear"},
    {"Bitclearperiod", "bitplane_clear_every"},
    {"Bitxor", "bitplane_xor"},
    {"Bitxorperiod", "bitplane_xor_every"},
    {"Signflip", "sign_flip_every"},
    {"Historylag", "history_lag"},
    {"Historyperiod", "history_band_every"},
    {"Lowpasslag", "lowpass_history_lag"},
    {"Liftingbias", "lifting_bias"},
    {"Liftingperiod", "lifting_bias_every"},
};

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
    return clampFloat(base * (1.0f - amount) + target * amount, 0.0f, 8.0f);
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

int maxLevels(int width, int height)
{
    int levels = 0;
    while (width > 1 && height > 1)
    {
        ++levels;
        width = (width + 1) / 2;
        height = (height + 1) / 2;
    }
    return std::max(1, levels);
}

} // namespace

DatamoshWaveletCudaTOP::DatamoshWaveletCudaTOP(
    const OP_NodeInfo* info, TOP_Context* context)
    : myNodeInfo(info), myContext(context)
{
    cudaError_t status = cudaStreamCreate(&myStream);
    if (status != cudaSuccess)
        setCudaError("cudaStreamCreate", status);
}

DatamoshWaveletCudaTOP::~DatamoshWaveletCudaTOP()
{
    releaseState();
    if (myInputSurface)
        cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface)
        cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)
        cudaStreamDestroy(myStream);
}

void DatamoshWaveletCudaTOP::getGeneralInfo(
    TOP_GeneralInfo* info, const OP_Inputs*, void*)
{
    info->cookEveryFrame = true;
    info->cookEveryFrameIfAsked = true;
}

void DatamoshWaveletCudaTOP::execute(
    TOP_Output* output, const OP_Inputs* inputs, void*)
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
    myCookStage = 2;

    myPatternIndex = std::clamp(inputs->getParInt("Pattern"), 0, patternCount() - 1);
    const char* pattern = inputs->getParString("Pattern");
    myPatternName = kPatternNames[myPatternIndex];
    if (pattern && patternIndex(pattern) == myPatternIndex)
        myPatternName = pattern;
    if (myPatternIndex != myLastPatternIndex)
    {
        myResetPending = true;
        myLastPatternIndex = myPatternIndex;
    }

    const int width = static_cast<int>(input->textureDesc.width);
    const int height = static_cast<int>(input->textureDesc.height);
    DatamoshWaveletCudaParams params = presetParams(myPatternIndex);
    params.pattern = myPatternIndex;

    params.quality = std::clamp(
        static_cast<int>(std::lround(inputs->getParDouble("Quality"))), 1, 100);
    params.levels = std::clamp(
        static_cast<int>(std::lround(inputs->getParDouble("Levels"))),
        1,
        maxLevels(width, height));
    params.historyLength = std::clamp(
        static_cast<int>(std::lround(inputs->getParDouble("Historylen"))), 1, 128);

    myUseParams = inputs->getParInt("Useparams") != 0;
    if (myLastUseParams && !myUseParams)
        myResetPending = true;
    myLastUseParams = myUseParams;
    if (myUseParams)
    {
        for (const ParameterBinding& binding : kParameterBindings)
        {
            setParameter(
                params,
                binding.id,
                static_cast<float>(inputs->getParDouble(binding.parName)),
                maxLevels(width, height));
        }
    }

    const char* parameterId = inputs->getParString("Paramid");
    myParameterId = parameterId ? parameterId : "";
    if (!myLastParameterId.empty() && myParameterId != myLastParameterId)
        myResetPending = true;
    myLastParameterId = myParameterId;
    if (!myParameterId.empty() &&
        !setParameter(
            params,
            myParameterId.c_str(),
            static_cast<float>(inputs->getParDouble("Paramvalue")),
            maxLevels(width, height)))
    {
        myWarning = "Unknown WVT0 parameter: " + myParameterId;
    }

    myIntensity = static_cast<float>(inputs->getParDouble("Intensity"));
    myStructure = static_cast<float>(inputs->getParDouble("Motion"));
    myCoefficient = static_cast<float>(inputs->getParDouble("Residual"));
    myHistory = static_cast<float>(inputs->getParDouble("Temporal"));
    myRouting = static_cast<float>(inputs->getParDouble("Bitstream"));
    applyAudioControlInputs(inputs);
    applyControls(
        params,
        myIntensity,
        myStructure,
        myCoefficient,
        myHistory,
        myRouting);
    params.inputFormat = static_cast<int>(inputFormat);
    if (params.quality != myQuality)
    {
        myQuality = params.quality;
        myResetPending = true;
    }
    const int levels = params.levels;
    const int historyLength = params.historyLength;

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

    if (!ensureState(width, height, levels, historyLength))
    {
        myContext->endCUDAOperations(nullptr);
        return;
    }
    setupSurface(&myInputSurface, inputInfo->cudaArray);
    setupSurface(&myOutputSurface, outputInfo->cudaArray);
    if (myResetPending)
    {
        datamoshWaveletCudaReset(myState);
        myResetPending = false;
    }
    myCookStage = 6;

    cudaError_t status =
        datamoshWaveletCudaProcess(myState, myInputSurface, myOutputSurface, params, myStream);
    if (status == cudaSuccess)
    {
        ++myProcessedFrames;
        myCookStage = 7;
    }
    else
    {
        setCudaError("datamoshWaveletCudaProcess", status);
    }

    myContext->endCUDAOperations(nullptr);
    if (status == cudaSuccess)
        myCookStage = 8;
}

int32_t DatamoshWaveletCudaTOP::getNumInfoCHOPChans(void*)
{
    return 21;
}

void DatamoshWaveletCudaTOP::getInfoCHOPChan(
    int32_t index, OP_InfoCHOPChan* chan, void*)
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
            chan->name->setString("levels");
            chan->value = static_cast<float>(myLevels);
            break;
        case 5:
            chan->name->setString("history_length");
            chan->value = static_cast<float>(myHistoryLength);
            break;
        case 6:
            chan->name->setString("input_cooks");
            chan->value = static_cast<float>(myInputCooks);
            break;
        case 7:
            chan->name->setString("cook_stage");
            chan->value = static_cast<float>(myCookStage);
            break;
        case 8:
            chan->name->setString("input_format");
            chan->value = static_cast<float>(myInputFormat);
            break;
        case 9:
            chan->name->setString("pattern_index");
            chan->value = static_cast<float>(myPatternIndex);
            break;
        case 10:
            chan->name->setString("implementation_version");
            chan->value = static_cast<float>(kImplementationVersion);
            break;
        case 11:
            chan->name->setString("pattern_count");
            chan->value = static_cast<float>(patternCount());
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
            chan->name->setString("structure");
            chan->value = myStructure;
            break;
        case 16:
            chan->name->setString("coefficient");
            chan->value = myCoefficient;
            break;
        case 17:
            chan->name->setString("history");
            chan->value = myHistory;
            break;
        case 18:
            chan->name->setString("routing");
            chan->value = myRouting;
            break;
        case 19:
            chan->name->setString("operator_version");
            chan->value = static_cast<float>(kOperatorVersion);
            break;
        default:
            chan->name->setString("pattern_schema_version");
            chan->value = static_cast<float>(kPatternSchemaVersion);
            break;
    }
}

bool DatamoshWaveletCudaTOP::getInfoDATSize(OP_InfoDATSize* info, void*)
{
    info->rows = 14;
    info->cols = 2;
    info->byColumn = false;
    return true;
}

void DatamoshWaveletCudaTOP::getInfoDATEntries(
    int32_t index, int32_t, OP_InfoDATEntries* entries, void*)
{
    const char* key = "";
    std::string value;
    switch (index)
    {
        case 0:
            key = "backend";
            value = "cuda_wavelet_v1";
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
            key = "levels";
            value = std::to_string(myLevels);
            break;
        case 4:
            key = "history_length";
            value = std::to_string(myHistoryLength);
            break;
        case 5:
            key = "audio";
            value = myAudioActive ? "active" : "inactive";
            break;
        case 6:
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
        case 7:
            key = "input_cooks";
            value = std::to_string(myInputCooks);
            break;
        case 8:
            key = "cook_stage";
            value = std::to_string(myCookStage);
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
            key = "operator_version";
            value = std::to_string(kOperatorVersion);
            break;
        case 12:
            key = "pattern_count";
            value = std::to_string(patternCount());
            break;
        default:
            key = "pattern_schema_version";
            value = std::to_string(kPatternSchemaVersion);
            break;
    }
    entries->values[0]->setString(key);
    entries->values[1]->setString(value.c_str());
}

void DatamoshWaveletCudaTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError.c_str());
}

void DatamoshWaveletCudaTOP::getWarningString(OP_String* warning, void*)
{
    warning->setString(myWarning.c_str());
}

void DatamoshWaveletCudaTOP::setupParameters(
    OP_ParameterManager* manager, void*)
{
    OP_StringParameter pattern;
    pattern.name = "Pattern";
    pattern.label = "Pattern";
    pattern.page = "Datamosh";
    pattern.defaultValue = "clean";
    manager->appendMenu(pattern, patternCount(), kPatternNames, kPatternLabels);

    appendFloat(manager, "Datamosh", "Intensity", "Intensity", 1.0, 0.0, 2.0, 0.0, 2.0);
    appendFloat(manager, "Datamosh", "Motion", "Structure", 1.0, 0.0, 2.0, 0.0, 2.0);
    appendFloat(
        manager, "Datamosh", "Residual", "Coefficient", 1.0, 0.0, 2.0, 0.0, 2.0);
    appendFloat(manager, "Datamosh", "Temporal", "History", 1.0, 0.0, 2.0, 0.0, 2.0);
    appendFloat(manager, "Datamosh", "Bitstream", "Routing", 1.0, 0.0, 2.0, 0.0, 2.0);
    appendToggle(manager, "Datamosh", "Useparams", "Use Overrides");

    appendFloat(manager, "Codec", "Quality", "Quality", 82.0, 1.0, 100.0, 1.0, 100.0);
    appendFloat(manager, "Codec", "Levels", "Levels", 3.0, 1.0, 6.0, 1.0, 12.0);
    appendFloat(
        manager, "Codec", "Historylen", "History Length", 12.0, 1.0, 32.0, 1.0, 128.0);

    appendFloat(
        manager,
        "Structure",
        "Packetshift",
        "Packet Shift",
        0.0,
        -16.0,
        16.0,
        -64.0,
        64.0,
        false,
        false);
    appendFloat(
        manager,
        "Structure",
        "Packetshiftperiod",
        "Packet Shift Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager,
        "Structure",
        "Orientation",
        "Orientation Rotate",
        0.0,
        -3.0,
        3.0,
        -12.0,
        12.0,
        false,
        false);
    appendFloat(
        manager,
        "Structure",
        "Orientationperiod",
        "Orientation Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager,
        "Structure",
        "Levelfold",
        "Level Fold",
        0.0,
        -4.0,
        4.0,
        -12.0,
        12.0,
        false,
        false);
    appendFloat(
        manager,
        "Structure",
        "Levelfoldperiod",
        "Level Fold Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);

    appendFloat(
        manager, "Coefficient", "Bitclear", "Bitplanes Clear", 0.0, 0.0, 8.0, 0.0, 30.0);
    appendFloat(
        manager,
        "Coefficient",
        "Bitclearperiod",
        "Bit Clear Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager, "Coefficient", "Bitxor", "Bitplane XOR", 0.0, 0.0, 12.0, 0.0, 30.0);
    appendFloat(
        manager,
        "Coefficient",
        "Bitxorperiod",
        "Bit XOR Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager,
        "Coefficient",
        "Signflip",
        "Sign Flip Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager,
        "Coefficient",
        "Liftingbias",
        "Lifting Bias",
        0.0,
        -64.0,
        64.0,
        -512.0,
        512.0,
        false,
        false);
    appendFloat(
        manager,
        "Coefficient",
        "Liftingperiod",
        "Lifting Bias Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);

    appendFloat(
        manager, "History", "Historylag", "History Lag", 1.0, 1.0, 16.0, 1.0, 128.0);
    appendFloat(
        manager,
        "History",
        "Historyperiod",
        "History Band Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager,
        "History",
        "Lowpasslag",
        "Lowpass History Lag",
        0.0,
        0.0,
        16.0,
        0.0,
        128.0);

    appendFloat(
        manager,
        "Routing",
        "Channelroute",
        "Channel Route",
        0.0,
        -2.0,
        2.0,
        -8.0,
        8.0,
        false,
        false);
    appendFloat(
        manager,
        "Routing",
        "Channelperiod",
        "Channel Route Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendFloat(
        manager,
        "Routing",
        "Packetloss",
        "Packet Loss Period",
        0.0,
        0.0,
        64.0,
        0.0,
        512.0);
    appendToggle(
        manager, "Routing", "Packetconceal", "Conceal From History", 1.0);

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
    appendString(manager, "Audio", "Motionchan", "Structure Chan");
    appendString(manager, "Audio", "Residualchan", "Coefficient Chan");
    appendString(manager, "Audio", "Temporalchan", "History Chan");
    appendString(manager, "Audio", "Bitstreamchan", "Routing Chan");
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

void DatamoshWaveletCudaTOP::pulsePressed(const char* name, void*)
{
    if (!std::strcmp(name, "Resetglitch"))
        myResetPending = true;
    else if (!std::strcmp(name, "Recreate"))
    {
        releaseState();
        myResetPending = true;
    }
}

void DatamoshWaveletCudaTOP::applyAudioControlInputs(
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
    applyChannel("Motionchan", myStructure);
    applyChannel("Residualchan", myCoefficient);
    applyChannel("Temporalchan", myHistory);
    applyChannel("Bitstreamchan", myRouting);

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

bool DatamoshWaveletCudaTOP::ensureState(
    int width, int height, int levels, int historyLength)
{
    if (myState && width == myWidth && height == myHeight &&
        levels == myLevels && historyLength == myHistoryLength)
        return true;

    releaseState();
    cudaError_t status = datamoshWaveletCudaCreate(
        &myState, width, height, levels, historyLength);
    if (status != cudaSuccess)
    {
        setCudaError("datamoshWaveletCudaCreate", status);
        return false;
    }
    myWidth = width;
    myHeight = height;
    myLevels = levels;
    myHistoryLength = historyLength;
    myResetPending = true;
    return true;
}

void DatamoshWaveletCudaTOP::releaseState()
{
    if (myState)
    {
        datamoshWaveletCudaDestroy(myState);
        myState = nullptr;
    }
    myWidth = 0;
    myHeight = 0;
    myLevels = 0;
    myHistoryLength = 0;
}

void DatamoshWaveletCudaTOP::setCudaError(
    const char* operation, cudaError_t error)
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
    info->customOPInfo.opType->setString("Datamoshwaveletcuda");
    info->customOPInfo.opLabel->setString("Datamosh Wavelet CUDA TOP");
    info->customOPInfo.opIcon->setString("WVC");
    info->customOPInfo.authorName->setString("datamosh");
    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(
    const OP_NodeInfo* info, TOP_Context* context)
{
    return new DatamoshWaveletCudaTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(
    TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<DatamoshWaveletCudaTOP*>(instance);
}

}

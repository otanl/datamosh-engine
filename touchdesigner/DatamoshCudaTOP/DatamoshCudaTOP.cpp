#include "DatamoshCudaTOP.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>

namespace {

constexpr int kImplementationVersion = 16;
constexpr int kOperatorVersion = 2;
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
    params.intensity = std::clamp(static_cast<float>(inputs->getParDouble("Intensity")), 0.0f, 4.0f);
    params.motion = std::clamp(static_cast<float>(inputs->getParDouble("Motion")), 0.0f, 4.0f);
    params.residual = std::clamp(static_cast<float>(inputs->getParDouble("Residual")), 0.0f, 4.0f);
    params.temporal = std::clamp(static_cast<float>(inputs->getParDouble("Temporal")), 0.0f, 4.0f);
    params.bitstream = std::clamp(static_cast<float>(inputs->getParDouble("Bitstream")), 0.0f, 4.0f);
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
    return 12;
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
        default:
            chan->name->setString("pattern_schema_version");
            chan->value = static_cast<float>(kPatternSchemaVersion);
            break;
    }
}

bool DatamoshCudaTOP::getInfoDATSize(OP_InfoDATSize* info, void*)
{
    info->rows = 12;
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
            key = "input_cooks";
            value = std::to_string(myInputCooks);
            break;
        case 5:
            key = "cook_stage";
            value = std::to_string(myCookStage);
            break;
        case 6:
            key = "input_format";
            value = std::to_string(myInputFormat);
            break;
        case 7:
            key = "pattern_index";
            value = std::to_string(myPatternIndex);
            break;
        case 8:
            key = "implementation_version";
            value = std::to_string(kImplementationVersion);
            break;
        case 9:
            key = "pattern_count";
            value = std::to_string(patternCount());
            break;
        case 10:
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

    OP_NumericParameter reset;
    reset.name = "Resetglitch";
    reset.label = "Reset Glitch";
    reset.page = "Datamosh";
    manager->appendPulse(reset);
}

void DatamoshCudaTOP::pulsePressed(const char* name, void*)
{
    if (!std::strcmp(name, "Resetglitch"))
        myResetPending = true;
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

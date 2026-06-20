#include "DatamoshDctCudaTOP.h"

#include "DatamoshDctCudaPresets.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace {

constexpr int kImplementationVersion = 1;
constexpr int kOperatorVersion = 1;
constexpr int kPatternSchemaVersion = 1;

// Pattern table + preset resolution live in DatamoshDctCudaPresets.h so the parity check
// (tools/dct_parity_check.cu) uses the identical values.
using namespace dctcuda;

const char* kPatternLabels[] = {
    "Clean", "Quantize Blocks", "DC Predictor Smear", "Block DC Bleed", "Coefficient Low-Pass",
    "Coefficient Sign Ring", "Coefficient Scramble", "Block Slip", "Block Echo", "Temporal Flow",
    "False Colour", "Transform Collapse",
};

void configureFloat(
    OP_NumericParameter& parameter, const char* name, const char* label, double defaultValue,
    double sliderMin, double sliderMax, const char* page = "Datamosh")
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

DatamoshDctCudaTOP::DatamoshDctCudaTOP(const OP_NodeInfo* info, TOP_Context* context)
    : myNodeInfo(info), myContext(context)
{
    cudaError_t status = cudaStreamCreate(&myStream);
    if (status != cudaSuccess)
        setCudaError("cudaStreamCreate", status);
}

DatamoshDctCudaTOP::~DatamoshDctCudaTOP()
{
    releaseState();
    if (myInputSurface)
        cudaDestroySurfaceObject(myInputSurface);
    if (myOutputSurface)
        cudaDestroySurfaceObject(myOutputSurface);
    if (myStream)
        cudaStreamDestroy(myStream);
}

void DatamoshDctCudaTOP::getGeneralInfo(TOP_GeneralInfo* info, const OP_Inputs*, void*)
{
    info->cookEveryFrame = true;
    info->cookEveryFrameIfAsked = true;
}

void DatamoshDctCudaTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
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
        inputFormat == OP_PixelFormat::BGRA8Fixed || inputFormat == OP_PixelFormat::RGBA8Fixed ||
        inputFormat == OP_PixelFormat::RGBA16Fixed || inputFormat == OP_PixelFormat::RGBA16Float ||
        inputFormat == OP_PixelFormat::RGBA32Float;
    if (input->textureDesc.texDim != OP_TexDim::e2D || !supportedFormat)
    {
        myError = "GPU codec requires a 2D BGRA/RGBA 8/16/32-bit input";
        return;
    }

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

    DatamoshDctCudaParams params = presetParams(myPatternIndex);
    params.pattern = myPatternIndex;
    applyControls(
        params,
        static_cast<float>(inputs->getParDouble("Intensity")),
        static_cast<float>(inputs->getParDouble("Structure")),
        static_cast<float>(inputs->getParDouble("Persist")),
        static_cast<float>(inputs->getParDouble("Dc")),
        static_cast<float>(inputs->getParDouble("Quant")));
    params.quality = std::clamp(inputs->getParInt("Quality"), 1, 100);
    params.inputFormat = static_cast<int>(inputFormat);

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

    if (!ensureState(width, height))
    {
        myContext->endCUDAOperations(nullptr);
        return;
    }
    myCookStage = 6;

    setupSurface(&myInputSurface, inputInfo->cudaArray);
    setupSurface(&myOutputSurface, outputInfo->cudaArray);
    if (myResetPending)
    {
        datamoshDctCudaReset(myState);
        myResetPending = false;
    }

    cudaError_t status =
        datamoshDctCudaProcess(myState, myInputSurface, myOutputSurface, params, myStream);
    if (status == cudaSuccess)
    {
        ++myProcessedFrames;
        myCookStage = 7;
    }
    else
        setCudaError("datamoshDctCudaProcess", status);

    myContext->endCUDAOperations(nullptr);
    if (status == cudaSuccess)
        myCookStage = 8;
}

int32_t DatamoshDctCudaTOP::getNumInfoCHOPChans(void*)
{
    return 10;
}

void DatamoshDctCudaTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
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
        default:
            chan->name->setString("pattern_count");
            chan->value = static_cast<float>(patternCount());
            break;
    }
}

bool DatamoshDctCudaTOP::getInfoDATSize(OP_InfoDATSize* info, void*)
{
    info->rows = 8;
    info->cols = 2;
    info->byColumn = false;
    return true;
}

void DatamoshDctCudaTOP::getInfoDATEntries(int32_t index, int32_t, OP_InfoDATEntries* entries, void*)
{
    const char* key = "";
    std::string value;
    switch (index)
    {
        case 0:
            key = "backend";
            value = "cuda_dct_v1";
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
            key = "input_cooks";
            value = std::to_string(myInputCooks);
            break;
        case 4:
            key = "cook_stage";
            value = std::to_string(myCookStage);
            break;
        case 5:
            key = "pattern_index";
            value = std::to_string(myPatternIndex);
            break;
        case 6:
            key = "implementation_version";
            value = std::to_string(kImplementationVersion);
            break;
        default:
            key = "pattern_schema_version";
            value = std::to_string(kPatternSchemaVersion);
            break;
    }
    entries->values[0]->setString(key);
    entries->values[1]->setString(value.c_str());
}

void DatamoshDctCudaTOP::getErrorString(OP_String* error, void*)
{
    error->setString(myError.c_str());
}

void DatamoshDctCudaTOP::getWarningString(OP_String* warning, void*)
{
    warning->setString(myWarning.c_str());
}

void DatamoshDctCudaTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    OP_StringParameter pattern;
    pattern.name = "Pattern";
    pattern.label = "Pattern";
    pattern.page = "Datamosh";
    pattern.defaultValue = "clean";
    manager->appendMenu(pattern, patternCount(), kPatternNames, kPatternLabels);

    for (const auto& def : std::array<std::array<const char*, 2>, 5>{{
             {"Intensity", "Intensity"},
             {"Structure", "Structure"},
             {"Persist", "Persist"},
             {"Dc", "DC"},
             {"Quant", "Quant"},
         }})
    {
        OP_NumericParameter parameter;
        configureFloat(parameter, def[0], def[1], 1.0, 0.0, 2.0);
        manager->appendFloat(parameter);
    }

    OP_NumericParameter quality;
    quality.name = "Quality";
    quality.label = "Quality";
    quality.page = "Codec";
    quality.defaultValues[0] = 50;
    quality.minSliders[0] = 1;
    quality.maxSliders[0] = 100;
    quality.minValues[0] = 1;
    quality.maxValues[0] = 100;
    quality.clampMins[0] = true;
    quality.clampMaxes[0] = true;
    manager->appendInt(quality);

    OP_NumericParameter reset;
    reset.name = "Resetglitch";
    reset.label = "Reset Glitch";
    reset.page = "Datamosh";
    manager->appendPulse(reset);
}

void DatamoshDctCudaTOP::pulsePressed(const char* name, void*)
{
    if (!std::strcmp(name, "Resetglitch"))
        myResetPending = true;
}

bool DatamoshDctCudaTOP::ensureState(int width, int height)
{
    if (myState && width == myWidth && height == myHeight)
        return true;

    releaseState();
    cudaError_t status = datamoshDctCudaCreate(&myState, width, height);
    if (status != cudaSuccess)
    {
        setCudaError("datamoshDctCudaCreate", status);
        return false;
    }
    myWidth = width;
    myHeight = height;
    myResetPending = true;
    return true;
}

void DatamoshDctCudaTOP::releaseState()
{
    if (myState)
    {
        datamoshDctCudaDestroy(myState);
        myState = nullptr;
    }
    myWidth = 0;
    myHeight = 0;
}

void DatamoshDctCudaTOP::setCudaError(const char* operation, cudaError_t error)
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
    info->customOPInfo.opType->setString("Datamoshdctcuda");
    info->customOPInfo.opLabel->setString("Datamosh DCT CUDA TOP");
    info->customOPInfo.opIcon->setString("DXC");
    info->customOPInfo.authorName->setString("datamosh");
    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new DatamoshDctCudaTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<DatamoshDctCudaTOP*>(instance);
}

}

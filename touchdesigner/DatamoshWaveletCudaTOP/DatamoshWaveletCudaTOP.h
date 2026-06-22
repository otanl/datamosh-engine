#pragma once

#include "DatamoshWaveletCudaCore.h"
#include "TOP_CPlusPlusBase.h"

#include <cstdint>
#include <string>

using namespace TD;

class DatamoshWaveletCudaTOP final : public TOP_CPlusPlusBase
{
public:
    DatamoshWaveletCudaTOP(const OP_NodeInfo* info, TOP_Context* context);
    ~DatamoshWaveletCudaTOP() override;

    void getGeneralInfo(TOP_GeneralInfo* info, const OP_Inputs* inputs, void*) override;
    void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override;
    int32_t getNumInfoCHOPChans(void*) override;
    void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override;
    bool getInfoDATSize(OP_InfoDATSize* info, void*) override;
    void getInfoDATEntries(int32_t index, int32_t nEntries, OP_InfoDATEntries* entries, void*)
        override;
    void getErrorString(OP_String* error, void*) override;
    void getWarningString(OP_String* warning, void*) override;
    void setupParameters(OP_ParameterManager* manager, void*) override;
    void pulsePressed(const char* name, void*) override;

private:
    bool ensureState(int width, int height, int levels, int historyLength);
    void applyAudioControlInputs(const OP_Inputs* inputs);
    void releaseState();
    void setCudaError(const char* operation, cudaError_t error);

    const OP_NodeInfo* myNodeInfo = nullptr;
    TOP_Context* myContext = nullptr;
    DatamoshWaveletCudaState* myState = nullptr;
    cudaStream_t myStream = nullptr;
    cudaSurfaceObject_t myInputSurface = 0;
    cudaSurfaceObject_t myOutputSurface = 0;
    int myWidth = 0;
    int myHeight = 0;
    int myLevels = 0;
    int myHistoryLength = 0;
    int myQuality = 82;
    uint64_t myExecuteCount = 0;
    uint64_t myProcessedFrames = 0;
    int64_t myInputCooks = 0;
    int32_t myCookStage = 0;
    int32_t myInputFormat = -1;
    bool myResetPending = true;
    bool myUseParams = false;
    bool myLastUseParams = false;
    bool myAudioActive = false;
    bool myAudioResetArmed = true;
    float myAudioResetValue = 0.0f;
    float myIntensity = 1.0f;
    float myStructure = 1.0f;
    float myCoefficient = 1.0f;
    float myHistory = 1.0f;
    float myRouting = 1.0f;
    std::string myParameterId;
    std::string myLastParameterId;
    std::string myPatternName = "clean";
    int32_t myPatternIndex = 0;
    int32_t myLastPatternIndex = -1;
    std::string myError;
    std::string myWarning;
};

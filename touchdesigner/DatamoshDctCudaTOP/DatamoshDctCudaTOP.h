#pragma once

#include "DatamoshDctCudaCore.h"
#include "TOP_CPlusPlusBase.h"

#include <cstdint>
#include <string>

using namespace TD;

class DatamoshDctCudaTOP final : public TOP_CPlusPlusBase
{
public:
    DatamoshDctCudaTOP(const OP_NodeInfo* info, TOP_Context* context);
    ~DatamoshDctCudaTOP() override;

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
    bool ensureState(int width, int height);
    void releaseState();
    void setCudaError(const char* operation, cudaError_t error);

    const OP_NodeInfo* myNodeInfo = nullptr;
    TOP_Context* myContext = nullptr;
    DatamoshDctCudaState* myState = nullptr;
    cudaStream_t myStream = nullptr;
    cudaSurfaceObject_t myInputSurface = 0;
    cudaSurfaceObject_t myOutputSurface = 0;
    int myWidth = 0;
    int myHeight = 0;
    uint64_t myExecuteCount = 0;
    uint64_t myProcessedFrames = 0;
    int64_t myInputCooks = 0;
    int32_t myCookStage = 0;
    int32_t myInputFormat = -1;
    bool myResetPending = true;
    std::string myPatternName = "clean";
    int32_t myPatternIndex = 0;
    int32_t myLastPatternIndex = -1;
    std::string myError;
    std::string myWarning;
};

#include "TOP_CPlusPlusBase.h"

#define DATAMOSH_STATIC
#include "../../include/datamosh_ffi.h"

#include <windows.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace TD;

namespace {

constexpr int32_t kOperatorVersion = 3;
constexpr const char* kDefaultPattern = "clean";

#if defined(SCANLINE_SIGNAL_TOP)
constexpr int32_t kPatternSchemaVersion = 1;
constexpr const char* kOperatorLabel = "Datamosh Scanline TOP";
const char* kPatternNames[] = {
    "clean",
    "timebase-tear",
    "clock-skew",
    "sync-dropout",
    "chroma-sequence",
    "burst-seed-loss",
    "carrier-xor",
    "predictor-ghost",
    "rle-runaway",
    "plane-crosswire",
    "composite-collapse",
};

const char* kPatternLabels[] = {
    "Clean",
    "Timebase Tear",
    "Sample Clock Skew",
    "Line Sync Dropout",
    "Chroma Sequence Desync",
    "Burst Seed Loss",
    "Carrier Codeword XOR",
    "Temporal Predictor Ghost",
    "Luma RLE Runaway",
    "Luma/Chroma Crosswire",
    "Signal State Collapse",
};
#elif defined(DCT_TRANSFORM_TOP)
constexpr int32_t kPatternSchemaVersion = 1;
constexpr const char* kOperatorLabel = "Datamosh DCT TOP";
const char* kPatternNames[] = {
    "clean",
    "blocks",
    "dc-smear",
    "bleed",
    "blur",
    "ring",
    "scramble",
    "block-slip",
    "echo",
    "flow",
    "false-color",
    "composite",
    "desync",
    "shred",
    "truncate",
};

const char* kPatternLabels[] = {
    "Clean",
    "Quantize Blocks",
    "DC Predictor Smear",
    "Block DC Bleed",
    "Coefficient Low-Pass",
    "Coefficient Sign Ring",
    "Coefficient Scramble",
    "Block Slip",
    "Block Echo",
    "Temporal Flow",
    "False Colour",
    "Transform Collapse",
    "Entropy Desync",
    "Scan Shred",
    "Entropy Truncate",
};
#else
constexpr int32_t kPatternSchemaVersion = 2;
constexpr const char* kOperatorLabel = "Datamosh Motion TOP";
const char* kPatternNames[] = {
    "clean",
    "melt",
    "drift",
    "plane",
    "residue",
    "vector",
    "entropy",
    "coeff",
    "codebook",
    "unstable",
};

const char* kPatternLabels[] = {
    "Clean",
    "Motion Melt",
    "Temporal Slice Drift",
    "Channel Plane Desync",
    "Residual Stream Desync",
    "Motion Vector Bank Desync",
    "Entropy Byte Slip",
    "Transform Coefficient Drift",
    "Residual Codebook Leak",
    "Codec State Collapse",
};
#endif
static_assert(
    sizeof(kPatternNames) / sizeof(kPatternNames[0]) ==
    sizeof(kPatternLabels) / sizeof(kPatternLabels[0]));

struct ParameterBinding
{
    const char* parName;
    const char* id;
};

constexpr ParameterBinding kParameterBindings[] = {
#if defined(SCANLINE_SIGNAL_TOP)
    {"Lineshift", "line_shift"},
    {"Shiftperiod", "line_shift_every"},
    {"Shiftdrift", "line_shift_drift"},
    {"Lineoffset", "line_index_offset"},
    {"Lineperiod", "line_index_every"},
    {"Linestride", "line_index_stride"},
    {"Syncloss", "sync_loss_every"},
    {"Fieldsyncloss", "field_sync_loss_every"},
    {"Fieldparity", "field_parity_flip_every"},
    {"Phaseoffset", "phase_offset"},
    {"Phasedrift", "phase_drift"},
    {"Burstloss", "burst_loss_every"},
    {"Chromagroup", "chroma_group_delta"},
    {"Chromasequence", "chroma_sequence_offset"},
    {"Chromasequenceperiod", "chroma_sequence_every"},
    {"Chromaseedloss", "chroma_seed_loss_every"},
    {"Chromaxor", "chroma_xor_mask"},
    {"Chromaxorperiod", "chroma_xor_every"},
    {"Carriersign", "carrier_sign_flip_every"},
    {"Predictorflip", "predictor_flip_every"},
    {"Predictorlag", "predictor_lag"},
    {"Predictorline", "predictor_line_offset"},
    {"Predictorlineperiod", "predictor_line_offset_every"},
    {"Quantoffset", "quant_offset"},
    {"Quantperiod", "quant_offset_every"},
    {"Lumaslip", "luma_payload_slip"},
    {"Lumaslipperiod", "luma_payload_slip_every"},
    {"Chromaslip", "chroma_payload_slip"},
    {"Chromaslipperiod", "chroma_payload_slip_every"},
    {"Runlength", "luma_run_delta"},
    {"Runperiod", "luma_run_delta_every"},
    {"Packetlength", "packet_length_delta"},
    {"Packetperiod", "packet_length_delta_every"},
    {"Planeswap", "payload_swap_every"},
    {"Historylag", "history_line_weave"},
    {"Historyperiod", "history_line_weave_every"},
#elif defined(DCT_TRANSFORM_TOP)
    {"Quantscale", "quant_scale"},
    {"Dcdrift", "dc_drift"},
    {"Dcdriftperiod", "dc_drift_every"},
    {"Dcoffset", "dc_block_offset"},
    {"Dcoffsetperiod", "dc_block_offset_every"},
    {"Aczero", "ac_zero_above"},
    {"Signflip", "coeff_sign_flip_every"},
    {"Coeffshift", "coeff_shift"},
    {"Coeffshiftperiod", "coeff_shift_every"},
    {"Blockshiftx", "block_shift_x"},
    {"Blockshifty", "block_shift_y"},
    {"Blockshiftperiod", "block_shift_every"},
    {"Blockrepeat", "block_repeat_every"},
    {"Zigzagreverse", "zigzag_reverse_every"},
    {"Blocktranspose", "block_transpose_every"},
    {"Chromaswap", "chroma_swap_every"},
    {"Byteflip", "byte_flip_every"},
    {"Bytedrop", "drop_every"},
    {"Slipevery", "slip_every"},
    {"Slipbytes", "slip_bytes"},
    {"Slipwindow", "slip_window"},
    {"Truncate", "truncate_tail"},
    {"Persistence", "persistence"},
#else
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
#endif
};

int32_t patternCount()
{
    return static_cast<int32_t>(sizeof(kPatternNames) / sizeof(kPatternNames[0]));
}

std::string presetForPattern(const std::string& pattern)
{
#if !defined(SCANLINE_SIGNAL_TOP) && !defined(DCT_TRANSFORM_TOP)
    if (pattern == "entropyhard")
        return "entropy-hard";
    if (pattern == "coeffhard")
        return "coeff-hard";
    if (pattern == "codebookhard")
        return "codebook-hard";
#endif
    if (!pattern.empty())
        return pattern;
    return "clean";
}

using status_message_fn = const char* (*)(int32_t);
using default_backend_fn = uint32_t (*)();
using backend_name_fn = const char* (*)(uint32_t);
using engine_new_fn = DatamoshMoshEngine* (*)(uint32_t, size_t, size_t);
using engine_free_fn = void (*)(DatamoshMoshEngine*);
using set_preset_fn = int32_t (*)(DatamoshMoshEngine*, const char*);
using set_controls_fn = int32_t (*)(DatamoshMoshEngine*, float, float, float, float, float);
using set_parameter_fn = int32_t (*)(DatamoshMoshEngine*, const char*, float);
using reset_glitch_fn = int32_t (*)(DatamoshMoshEngine*);
using process_rgba8_fn =
    int32_t (*)(DatamoshMoshEngine*, const uint8_t*, size_t, uint8_t*, size_t);

template <typename Fn>
Fn loadSymbol(HMODULE dll, const char* name)
{
    FARPROC proc = GetProcAddress(dll, name);
    return reinterpret_cast<Fn>(proc);
}

std::string parentDir(const char* path)
{
    if (!path || !path[0])
        return {};

    std::string text(path);
    size_t pos = text.find_last_of("\\/");
    if (pos == std::string::npos)
        return {};
    return text.substr(0, pos);
}

class DatamoshRuntime
{
public:
    ~DatamoshRuntime()
    {
        unload();
    }

    bool loadNearPlugin(const char* pluginPath, std::string& error)
    {
        if (dll)
            return true;

        std::vector<std::string> candidates;
        std::string dir = parentDir(pluginPath);
        if (!dir.empty())
            candidates.push_back(dir + "\\datamosh.dll");
        candidates.push_back("datamosh.dll");
        candidates.push_back("target\\release\\datamosh.dll");

        for (const std::string& candidate : candidates)
        {
            HMODULE loaded = LoadLibraryA(candidate.c_str());
            if (!loaded)
                continue;

            dll = loaded;
            dllPath = candidate;
            if (loadSymbols(error))
                return true;

            unload();
        }

        error = "Could not load datamosh.dll next to the TOP plugin";
        return false;
    }

    void unload()
    {
        if (dll)
        {
            FreeLibrary(dll);
            dll = nullptr;
        }
        dllPath.clear();
        statusMessage = nullptr;
        defaultBackend = nullptr;
        backendName = nullptr;
        engineNew = nullptr;
        engineFree = nullptr;
        setPreset = nullptr;
        setControls = nullptr;
        setParameter = nullptr;
        resetGlitch = nullptr;
        processRgba8 = nullptr;
    }

    const char* statusText(int32_t status) const
    {
        if (statusMessage)
            return statusMessage(status);
        return "datamosh runtime unavailable";
    }

    const char* backendText(uint32_t backend) const
    {
        if (backendName)
            return backendName(backend);
        return "";
    }

    bool loaded() const
    {
        return dll != nullptr;
    }

    std::string dllPath;
    status_message_fn statusMessage = nullptr;
    default_backend_fn defaultBackend = nullptr;
    backend_name_fn backendName = nullptr;
    engine_new_fn engineNew = nullptr;
    engine_free_fn engineFree = nullptr;
    set_preset_fn setPreset = nullptr;
    set_controls_fn setControls = nullptr;
    set_parameter_fn setParameter = nullptr;
    reset_glitch_fn resetGlitch = nullptr;
    process_rgba8_fn processRgba8 = nullptr;

private:
    bool loadSymbols(std::string& error)
    {
        statusMessage = loadSymbol<status_message_fn>(dll, "datamosh_status_message");
        defaultBackend =
            loadSymbol<default_backend_fn>(dll, "datamosh_mosh_engine_default_backend");
        backendName = loadSymbol<backend_name_fn>(dll, "datamosh_mosh_engine_backend_name");
        engineNew = loadSymbol<engine_new_fn>(dll, "datamosh_mosh_engine_new_with_backend");
        engineFree = loadSymbol<engine_free_fn>(dll, "datamosh_mosh_engine_free");
        setPreset = loadSymbol<set_preset_fn>(dll, "datamosh_mosh_engine_set_preset");
        setControls = loadSymbol<set_controls_fn>(dll, "datamosh_mosh_engine_set_controls");
        setParameter = loadSymbol<set_parameter_fn>(dll, "datamosh_mosh_engine_set_parameter");
        resetGlitch = loadSymbol<reset_glitch_fn>(dll, "datamosh_mosh_engine_reset_glitch");
        processRgba8 = loadSymbol<process_rgba8_fn>(dll, "datamosh_mosh_engine_process_rgba8");

        if (statusMessage && defaultBackend && backendName && engineNew && engineFree &&
            setPreset && setControls && setParameter && resetGlitch && processRgba8)
        {
            return true;
        }

        error = "datamosh.dll is missing one or more required C ABI symbols";
        return false;
    }

    HMODULE dll = nullptr;
};

void configureFloat(OP_NumericParameter& np,
                    const char* name,
                    const char* label,
                    double value,
                    double sliderMin,
                    double sliderMax,
                    double minValue,
                    double maxValue,
                    bool clampMin = true,
                    bool clampMax = true,
                    const char* page = "Datamosh")
{
    np.name = name;
    np.label = label;
    np.page = page;
    np.defaultValues[0] = value;
    np.minSliders[0] = sliderMin;
    np.maxSliders[0] = sliderMax;
    np.minValues[0] = minValue;
    np.maxValues[0] = maxValue;
    np.clampMins[0] = clampMin;
    np.clampMaxes[0] = clampMax;
}

void appendFloat(OP_ParameterManager* manager,
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
    OP_NumericParameter np;
    configureFloat(np, name, label, value, sliderMin, sliderMax, minValue, maxValue, clampMin,
                   clampMax, page);
    manager->appendFloat(np);
}

void appendToggle(OP_ParameterManager* manager,
                  const char* page,
                  const char* name,
                  const char* label,
                  double value = 0.0)
{
    OP_NumericParameter np;
    np.name = name;
    np.label = label;
    np.page = page;
    np.defaultValues[0] = value;
    np.minValues[0] = 0.0;
    np.maxValues[0] = 1.0;
    np.clampMins[0] = true;
    np.clampMaxes[0] = true;
    manager->appendToggle(np);
}

void appendString(OP_ParameterManager* manager,
                  const char* page,
                  const char* name,
                  const char* label,
                  const char* value = "")
{
    OP_StringParameter sp;
    sp.name = name;
    sp.label = label;
    sp.page = page;
    sp.defaultValue = value;
    manager->appendString(sp);
}

void appendCHOP(OP_ParameterManager* manager, const char* page, const char* name, const char* label)
{
    OP_StringParameter sp;
    sp.name = name;
    sp.label = label;
    sp.page = page;
    sp.defaultValue = "";
    manager->appendCHOP(sp);
}

float clampFloat(float value, float minValue, float maxValue)
{
    if (value < minValue)
        return minValue;
    if (value > maxValue)
        return maxValue;
    return value;
}

bool parseChannelIndex(const std::string& text, int32_t& index)
{
    if (text.empty())
        return false;

    char* end = nullptr;
    long value = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != '\0' || value < 0 || value > INT32_MAX)
        return false;

    index = static_cast<int32_t>(value);
    return true;
}

float latestChopValue(const OP_CHOPInput* chop, const std::string& channel, bool& found)
{
    found = false;
    if (!chop || chop->numChannels <= 0 || chop->numSamples <= 0 || !chop->channelData)
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
        if (name && std::strcmp(name, channel.c_str()) == 0)
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

float blendAudioValue(float base, float sample, float amount, float gain, float bias)
{
    float target = sample * gain + bias;
    float mixed = base * (1.0f - amount) + target * amount;
    return clampFloat(mixed, 0.0f, 8.0f);
}

} // namespace

class DatamoshTOP : public TOP_CPlusPlusBase
{
public:
    DatamoshTOP(const OP_NodeInfo* info, TOP_Context* context)
        : myNodeInfo(info), myContext(context)
    {
    }

    ~DatamoshTOP() override
    {
        freeEngine();
    }

    void getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*) override
    {
        ginfo->cookEveryFrameIfAsked = true;
    }

    void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override
    {
        ++myExecuteCount;

        const OP_TOPInput* top = inputs->getInputTOP(0);
        if (top)
        {
            OP_TOPInputDownloadOptions opts;
            opts.pixelFormat = OP_PixelFormat::RGBA8Fixed;
            opts.colorSpace = OP_ColorSpace::Passthrough;
            opts.verticalFlip = false;

            OP_SmartRef<OP_TOPDownloadResult> downRes = top->downloadTexture(opts, nullptr);
            if (myPrevDownRes)
            {
                processDownload(output, inputs, myPrevDownRes.operator->());
            }
            else
            {
                uploadBlack(output, top->textureDesc.width, top->textureDesc.height);
            }
            myPrevDownRes = std::move(downRes);
            return;
        }

        myPrevDownRes.release();
        uploadBlack(output, 320, 180);
    }

    int32_t getNumInfoCHOPChans(void*) override
    {
        return 16;
    }

    void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override
    {
        switch (index)
        {
            case 0:
                chan->name->setString("execute_count");
                chan->value = static_cast<float>(myExecuteCount);
                break;
            case 1:
                chan->name->setString("runtime_loaded");
                chan->value = myRuntime.loaded() ? 1.0f : 0.0f;
                break;
            case 2:
                chan->name->setString("engine_ready");
                chan->value = myEngine ? 1.0f : 0.0f;
                break;
            case 3:
                chan->name->setString("width");
                chan->value = static_cast<float>(myWidth);
                break;
            case 4:
                chan->name->setString("height");
                chan->value = static_cast<float>(myHeight);
                break;
            case 5:
                chan->name->setString("last_status");
                chan->value = static_cast<float>(myLastStatus);
                break;
            case 6:
                chan->name->setString("audio_active");
                chan->value = myAudioActive ? 1.0f : 0.0f;
                break;
            case 7:
                chan->name->setString("audio_reset");
                chan->value = myAudioResetValue;
                break;
            case 8:
                chan->name->setString("intensity");
                chan->value = myIntensity;
                break;
            case 9:
#if defined(SCANLINE_SIGNAL_TOP)
                chan->name->setString("timebase");
#else
                chan->name->setString("motion");
#endif
                chan->value = myMotion;
                break;
            case 10:
#if defined(SCANLINE_SIGNAL_TOP)
                chan->name->setString("carrier");
#else
                chan->name->setString("residual");
#endif
                chan->value = myResidual;
                break;
            case 11:
#if defined(SCANLINE_SIGNAL_TOP)
                chan->name->setString("prediction");
#else
                chan->name->setString("temporal");
#endif
                chan->value = myTemporal;
                break;
            case 12:
#if defined(SCANLINE_SIGNAL_TOP)
                chan->name->setString("packet");
#else
                chan->name->setString("bitstream");
#endif
                chan->value = myBitstream;
                break;
            case 13:
                chan->name->setString("operator_version");
                chan->value = static_cast<float>(kOperatorVersion);
                break;
            case 14:
                chan->name->setString("pattern_count");
                chan->value = static_cast<float>(patternCount());
                break;
            default:
                chan->name->setString("pattern_schema_version");
                chan->value = static_cast<float>(kPatternSchemaVersion);
                break;
        }
    }

    bool getInfoDATSize(OP_InfoDATSize* infoSize, void*) override
    {
        infoSize->rows = 12;
        infoSize->cols = 2;
        infoSize->byColumn = false;
        return true;
    }

    void getInfoDATEntries(int32_t index, int32_t, OP_InfoDATEntries* entries, void*) override
    {
        switch (index)
        {
            case 0:
                entries->values[0]->setString("backend");
                entries->values[1]->setString(myRuntime.backendText(myBackend));
                break;
            case 1:
                entries->values[0]->setString("pattern");
                entries->values[1]->setString(myPattern.c_str());
                break;
            case 2:
                entries->values[0]->setString("pattern_index");
                entries->values[1]->setString(myPatternIndexText.c_str());
                break;
            case 3:
                entries->values[0]->setString("preset");
                entries->values[1]->setString(myPreset.c_str());
                break;
            case 4:
                entries->values[0]->setString("dll");
                entries->values[1]->setString(myRuntime.dllPath.c_str());
                break;
            case 5:
                entries->values[0]->setString("status");
                entries->values[1]->setString(myLastStatusText.c_str());
                break;
            case 6:
                entries->values[0]->setString("audio");
                entries->values[1]->setString(myAudioActive ? "active" : "inactive");
                break;
            case 7:
                entries->values[0]->setString("param_overrides");
                if (myUseParams && !myParameterId.empty())
                    entries->values[1]->setString("dedicated+advanced");
                else if (myUseParams)
                    entries->values[1]->setString("dedicated");
                else if (!myParameterId.empty())
                    entries->values[1]->setString("advanced");
                else
                    entries->values[1]->setString("off");
                break;
            case 8:
                entries->values[0]->setString("resolution");
                entries->values[1]->setString(myResolutionText.c_str());
                break;
            case 9:
                entries->values[0]->setString("operator_version");
                entries->values[1]->setString(std::to_string(kOperatorVersion).c_str());
                break;
            case 10:
                entries->values[0]->setString("pattern_count");
                entries->values[1]->setString(std::to_string(patternCount()).c_str());
                break;
            default:
                entries->values[0]->setString("pattern_schema_version");
                entries->values[1]->setString(std::to_string(kPatternSchemaVersion).c_str());
                break;
        }
    }

    void getWarningString(OP_String* warning, void*) override
    {
        if (!myWarning.empty())
            warning->setString(myWarning.c_str());
    }

    void getErrorString(OP_String* error, void*) override
    {
        if (!myError.empty())
            error->setString(myError.c_str());
    }

    void getInfoPopupString(OP_String* info, void*) override
    {
        std::string text = kOperatorLabel;
        text += "\nOperator version: ";
        text += std::to_string(kOperatorVersion);
        text += "\nBackend: ";
        text += myRuntime.backendText(myBackend);
        text += "\nPattern: ";
        text += myPattern;
        text += "\nPreset: ";
        text += myPreset;
        text += "\nStatus: ";
        text += myLastStatusText;
        info->setString(text.c_str());
    }

    void setupParameters(OP_ParameterManager* manager, void*) override
    {
        {
            OP_StringParameter sp;
            sp.name = "Pattern";
            sp.label = "Pattern";
            sp.page = "Datamosh";
            sp.defaultValue = kDefaultPattern;
            manager->appendMenu(
                sp,
                static_cast<int32_t>(sizeof(kPatternNames) / sizeof(kPatternNames[0])),
                kPatternNames,
                kPatternLabels);
        }

        {
            OP_NumericParameter np;
            configureFloat(np, "Intensity", "Intensity", 1.0, 0.0, 2.0, 0.0, 2.0);
            manager->appendFloat(np);
        }
        {
            OP_NumericParameter np;
#if defined(SCANLINE_SIGNAL_TOP)
            configureFloat(np, "Motion", "Timebase", 1.0, 0.0, 2.0, 0.0, 2.0);
#elif defined(DCT_TRANSFORM_TOP)
            configureFloat(np, "Motion", "Structure", 1.0, 0.0, 2.0, 0.0, 2.0);
#else
            configureFloat(np, "Motion", "Motion", 1.0, 0.0, 2.0, 0.0, 2.0);
#endif
            manager->appendFloat(np);
        }
        {
            OP_NumericParameter np;
#if defined(SCANLINE_SIGNAL_TOP)
            configureFloat(np, "Residual", "Carrier", 1.0, 0.0, 2.0, 0.0, 2.0);
#elif defined(DCT_TRANSFORM_TOP)
            configureFloat(np, "Residual", "Persist", 1.0, 0.0, 2.0, 0.0, 2.0);
#else
            configureFloat(np, "Residual", "Residual", 1.0, 0.0, 2.0, 0.0, 2.0);
#endif
            manager->appendFloat(np);
        }
        {
            OP_NumericParameter np;
#if defined(SCANLINE_SIGNAL_TOP)
            configureFloat(np, "Temporal", "Prediction", 1.0, 0.0, 2.0, 0.0, 2.0);
#elif defined(DCT_TRANSFORM_TOP)
            configureFloat(np, "Temporal", "DC", 1.0, 0.0, 2.0, 0.0, 2.0);
#else
            configureFloat(np, "Temporal", "Temporal", 1.0, 0.0, 2.0, 0.0, 2.0);
#endif
            manager->appendFloat(np);
        }
        {
            OP_NumericParameter np;
#if defined(SCANLINE_SIGNAL_TOP)
            configureFloat(np, "Bitstream", "Packet", 1.0, 0.0, 2.0, 0.0, 2.0);
#elif defined(DCT_TRANSFORM_TOP)
            configureFloat(np, "Bitstream", "Quant", 1.0, 0.0, 2.0, 0.0, 2.0);
#else
            configureFloat(np, "Bitstream", "Bitstream", 1.0, 0.0, 2.0, 0.0, 2.0);
#endif
            manager->appendFloat(np);
        }

        appendToggle(manager, "Datamosh", "Useparams", "Use Overrides", 0.0);
#if defined(SCANLINE_SIGNAL_TOP)
        appendFloat(manager, "Timebase", "Lineshift", "Line Shift", 0.0, -64.0, 64.0, -256.0,
                    256.0, false, false);
        appendFloat(manager, "Timebase", "Shiftperiod", "Shift Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Timebase", "Shiftdrift", "Shift Drift", 0.0, -8.0, 8.0, -256.0,
                    256.0, false, false);
        appendFloat(manager, "Timebase", "Lineoffset", "Line Address Offset", 0.0, -64.0, 64.0,
                    -256.0, 256.0, false, false);
        appendFloat(manager, "Timebase", "Lineperiod", "Line Address Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Timebase", "Linestride", "Line Address Stride", 0.0, -16.0, 16.0,
                    -64.0, 64.0, false, false);
        appendFloat(manager, "Timebase", "Syncloss", "Sync Loss Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Timebase", "Fieldsyncloss", "Field Sync Loss", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Timebase", "Fieldparity", "Field Parity Flip", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Carrier", "Phaseoffset", "Burst Phase Offset", 0.0, -4.0, 4.0,
                    -16.0, 16.0, false, false);
        appendFloat(manager, "Carrier", "Phasedrift", "Burst Phase Drift", 0.0, -4.0, 4.0,
                    -16.0, 16.0, false, false);
        appendFloat(manager, "Carrier", "Burstloss", "Burst Loss Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Carrier", "Chromagroup", "Chroma Group Delta", 0.0, -3.0, 3.0,
                    -6.0, 6.0, false, false);
        appendFloat(manager, "Carrier", "Chromasequence", "Chroma Sequence Offset", 0.0, -4.0,
                    4.0, -4.0, 4.0, false, false);
        appendFloat(manager, "Carrier", "Chromasequenceperiod", "Chroma Sequence Period", 0.0, 0.0,
                    64.0, 0.0, 512.0);
        appendFloat(manager, "Carrier", "Chromaseedloss", "Chroma Seed Loss", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Carrier", "Chromaxor", "Chroma XOR Mask", 0.0, 0.0, 255.0, 0.0,
                    255.0);
        appendFloat(manager, "Carrier", "Chromaxorperiod", "Chroma XOR Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Carrier", "Carriersign", "Carrier Sign Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Prediction", "Predictorflip", "Predictor Flip Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Prediction", "Predictorlag", "Predictor Lag", 1.0, 1.0, 16.0, 1.0,
                    64.0);
        appendFloat(manager, "Prediction", "Predictorline", "Predictor Line Offset", 0.0, -64.0,
                    64.0, -256.0, 256.0, false, false);
        appendFloat(manager, "Prediction", "Predictorlineperiod", "Predictor Line Period", 0.0, 0.0,
                    64.0, 0.0, 512.0);
        appendFloat(manager, "Carrier", "Quantoffset", "Quantizer Offset", 0.0, -16.0, 16.0,
                    -64.0, 64.0, false, false);
        appendFloat(manager, "Carrier", "Quantperiod", "Quantizer Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Packet", "Lumaslip", "Luma Payload Slip", 0.0, -32.0, 32.0,
                    -256.0, 256.0, false, false);
        appendFloat(manager, "Packet", "Lumaslipperiod", "Luma Slip Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Packet", "Chromaslip", "Chroma Payload Slip", 0.0, -32.0, 32.0,
                    -256.0, 256.0, false, false);
        appendFloat(manager, "Packet", "Chromaslipperiod", "Chroma Slip Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Packet", "Runlength", "RLE Run Delta", 0.0, -32.0, 32.0, -127.0,
                    127.0, false, false);
        appendFloat(manager, "Packet", "Runperiod", "RLE Run Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Packet", "Packetlength", "Packet Length Delta", 0.0, -64.0, 64.0,
                    -1024.0, 1024.0, false, false);
        appendFloat(manager, "Packet", "Packetperiod", "Packet Length Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Packet", "Planeswap", "Plane Swap Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Prediction", "Historylag", "History Weave Lag", 0.0, 0.0, 16.0, 0.0,
                    64.0);
        appendFloat(manager, "Prediction", "Historyperiod", "History Weave Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
#elif defined(DCT_TRANSFORM_TOP)
        appendFloat(manager, "Codec", "Quality", "Quality", 50.0, 1.0, 100.0, 1.0, 100.0);
        appendFloat(manager, "Coefficient", "Quantscale", "Quant Scale", 1.0, 1.0, 16.0, 1.0,
                    64.0);
        appendFloat(manager, "Coefficient", "Aczero", "AC Zero Above", 0.0, 0.0, 63.0, 0.0, 63.0);
        appendFloat(manager, "Coefficient", "Signflip", "Sign Flip Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Coefficient", "Coeffshift", "Coeff Shift", 0.0, -32.0, 32.0, -63.0,
                    63.0, false, false);
        appendFloat(manager, "Coefficient", "Coeffshiftperiod", "Coeff Shift Period", 0.0, 0.0,
                    64.0, 0.0, 512.0);
        appendFloat(manager, "Coefficient", "Zigzagreverse", "Zigzag Reverse Period", 0.0, 0.0,
                    64.0, 0.0, 512.0);
        appendFloat(manager, "Coefficient", "Blocktranspose", "Block Transpose Period", 0.0, 0.0,
                    64.0, 0.0, 512.0);
        appendFloat(manager, "DC", "Dcdrift", "DC Drift", 0.0, -64.0, 64.0, -256.0, 256.0, false,
                    false);
        appendFloat(manager, "DC", "Dcdriftperiod", "DC Drift Period", 0.0, 0.0, 256.0, 0.0,
                    1024.0);
        appendFloat(manager, "DC", "Dcoffset", "DC Block Offset", 0.0, -64.0, 64.0, -256.0, 256.0,
                    false, false);
        appendFloat(manager, "DC", "Dcoffsetperiod", "DC Offset Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Block", "Blockshiftx", "Block Shift X", 0.0, -16.0, 16.0, -64.0,
                    64.0, false, false);
        appendFloat(manager, "Block", "Blockshifty", "Block Shift Y", 0.0, -16.0, 16.0, -64.0,
                    64.0, false, false);
        appendFloat(manager, "Block", "Blockshiftperiod", "Block Shift Period", 0.0, 0.0, 64.0,
                    0.0, 512.0);
        appendFloat(manager, "Block", "Blockrepeat", "Block Repeat Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Block", "Chromaswap", "Chroma Swap Period", 0.0, 0.0, 64.0, 0.0,
                    512.0);
        appendFloat(manager, "Temporal", "Persistence", "Persistence", 0.0, 0.0, 0.98, 0.0, 0.98);
        appendFloat(manager, "Entropy", "Byteflip", "Byte Flip Period", 0.0, 0.0, 4000.0, 0.0,
                    65536.0);
        appendFloat(manager, "Entropy", "Bytedrop", "Byte Drop Period", 0.0, 0.0, 4000.0, 0.0,
                    65536.0);
        appendFloat(manager, "Entropy", "Slipevery", "Slip Period", 0.0, 0.0, 16.0, 0.0, 256.0);
        appendFloat(manager, "Entropy", "Slipbytes", "Slip Bytes", 0.0, -32.0, 32.0, -256.0, 256.0,
                    false, false);
        appendFloat(manager, "Entropy", "Slipwindow", "Slip Window", 64.0, 2.0, 256.0, 1.0,
                    4096.0);
        appendFloat(manager, "Entropy", "Truncate", "Truncate Tail", 0.0, 0.0, 1.0, 0.0, 1.0);
#else
        appendFloat(manager, "Motion", "Mvscale", "MV Scale", 1.0, 0.0, 2.0, 0.0, 2.0);
        appendFloat(manager, "Motion", "Mvjitter", "MV Jitter", 0.0, 0.0, 16.0, 0.0, 16.0);
        appendFloat(manager, "Motion", "Vectorinterp", "Vector Interp", 0.0, 0.0, 1.0, 0.0,
                    1.0);
        appendFloat(manager, "Motion", "Sampledesync", "Sample Desync", 0.0, 0.0, 4.0, 0.0,
                    4.0);
        appendFloat(manager, "Reference", "Reflag", "Reference Lag", 1.0, 1.0, 16.0, 1.0, 32.0);
        appendFloat(manager, "Reference", "Refbleed", "Reference Bleed", 0.0, 0.0, 1.0, 0.0,
                    1.0);
        appendFloat(manager, "Reference", "Reflatch", "Reference Latch", 1.0, 1.0, 32.0, 1.0,
                    64.0);
        appendFloat(manager, "Reference", "Temporaldrift", "Temporal Drift", 0.0, -16.0, 16.0,
                    -16.0, 16.0, false, false);
        appendFloat(manager, "Residual", "Residkeep", "Residual Keep", 1.0, -2.0, 2.0, -2.0,
                    2.0, false, false);
        appendFloat(manager, "Residual", "Residjitter", "Residual Jitter", 0.0, 0.0, 32.0, 0.0,
                    32.0);
        appendFloat(manager, "Residual", "Residchannel", "Residual Channel", 0.0, -4.0, 4.0,
                    -4.0, 4.0, false, false);
        appendFloat(manager, "Bitstream", "Entropyevery", "Entropy Period", 0.0, 0.0, 64.0, 0.0,
                    64.0);
        appendFloat(manager, "Bitstream", "Entropywindows", "Entropy Windows", 1.0, 0.0, 16.0,
                    0.0, 64.0);
        appendFloat(manager, "Bitstream", "Coeffshift", "Coeff Shift", 0.0, -32.0, 32.0, -32.0,
                    32.0, false, false);
        appendFloat(manager, "Bitstream", "Coeffquant", "Coeff Quant", 1.0, 1.0, 32.0, 1.0, 64.0);
        appendFloat(manager, "Bitstream", "Codebookevery", "Codebook Period", 0.0, 0.0, 32.0,
                    0.0, 64.0);
        appendFloat(manager, "Bitstream", "Codebookstride", "Codebook Stride", 1.0, -64.0, 64.0,
                    -128.0, 128.0, false, false);
        appendFloat(manager, "Bitstream", "Codebookshuffle", "Codebook Shuffle", 0.0, 0.0, 32.0,
                    0.0, 64.0);
#endif

        {
            OP_StringParameter sp;
            sp.name = "Paramid";
            sp.label = "Param ID";
            sp.page = "Advanced";
            sp.defaultValue = "";
            manager->appendString(sp);
        }
        {
            OP_NumericParameter np;
            configureFloat(np, "Paramvalue", "Param Value", 0.0, -64.0, 64.0, -4096.0, 4096.0,
                           false, false, "Advanced");
            manager->appendFloat(np);
        }

        appendToggle(manager, "Audio", "Audioenable", "Audio Enable", 0.0);
        appendCHOP(manager, "Audio", "Controlchop", "Control CHOP");
        appendFloat(manager, "Audio", "Audioamount", "Audio Amount", 1.0, 0.0, 1.0, 0.0, 1.0);
        appendFloat(manager, "Audio", "Audiogain", "Audio Gain", 1.0, 0.0, 4.0, -64.0, 64.0,
                    false, false);
        appendFloat(manager, "Audio", "Audiobias", "Audio Bias", 0.0, -2.0, 2.0, -64.0, 64.0,
                    false, false);
        appendString(manager, "Audio", "Intensitychan", "Intensity Chan", "0");
#if defined(SCANLINE_SIGNAL_TOP)
        appendString(manager, "Audio", "Motionchan", "Timebase Chan", "");
        appendString(manager, "Audio", "Residualchan", "Carrier Chan", "");
        appendString(manager, "Audio", "Temporalchan", "Prediction Chan", "");
        appendString(manager, "Audio", "Bitstreamchan", "Packet Chan", "");
#else
        appendString(manager, "Audio", "Motionchan", "Motion Chan", "");
        appendString(manager, "Audio", "Residualchan", "Residual Chan", "");
        appendString(manager, "Audio", "Temporalchan", "Temporal Chan", "");
        appendString(manager, "Audio", "Bitstreamchan", "Bitstream Chan", "");
#endif
        appendString(manager, "Audio", "Resetchan", "Reset Chan", "");
        appendFloat(manager, "Audio", "Resetthreshold", "Reset Threshold", 0.75, 0.0, 1.0, 0.0,
                    64.0);
        appendFloat(manager, "Audio", "Resetrearm", "Reset Rearm", 0.25, 0.0, 1.0, 0.0, 64.0);

        {
            OP_NumericParameter np;
            np.name = "Resetglitch";
            np.label = "Reset Glitch";
            np.page = "Datamosh";
            manager->appendPulse(np);
        }
        {
            OP_NumericParameter np;
            np.name = "Recreate";
            np.label = "Recreate Engine";
            np.page = "Datamosh";
            manager->appendPulse(np);
        }
    }

    void pulsePressed(const char* name, void*) override
    {
        if (!std::strcmp(name, "Resetglitch"))
        {
            myResetPending = true;
            if (myEngine && myRuntime.resetGlitch)
            {
                setStatus(myRuntime.resetGlitch(myEngine));
                myResetPending = false;
            }
        }
        else if (!std::strcmp(name, "Recreate"))
        {
            freeEngine();
            myPrevDownRes.release();
            clearMessages();
        }
    }

private:
    void processDownload(TOP_Output* output, const OP_Inputs* inputs, OP_TOPDownloadResult* downRes)
    {
        if (!downRes)
        {
            uploadBlack(output, 320, 180);
            return;
        }

        const uint32_t width = downRes->textureDesc.width;
        const uint32_t height = downRes->textureDesc.height;
        const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;

        if (width == 0 || height == 0 || downRes->size < bytes ||
            downRes->textureDesc.pixelFormat != OP_PixelFormat::RGBA8Fixed)
        {
            myWarning = "Input download did not produce RGBA8 2D pixels";
            uploadBlack(output, std::max<uint32_t>(width, 1), std::max<uint32_t>(height, 1));
            return;
        }

        const uint8_t* inputPixels = static_cast<const uint8_t*>(downRes->getData());
        if (!inputPixels)
        {
            myWarning = "Input download returned null data";
            uploadBlack(output, width, height);
            return;
        }

        updateEngineParameters(inputs);

        myOutput.resize(bytes);
        if (!ensureEngine(width, height, inputs))
        {
            std::memcpy(myOutput.data(), inputPixels, bytes);
            uploadPixels(output, myOutput.data(), width, height, downRes->textureDesc);
            return;
        }

        int32_t status =
            myRuntime.processRgba8(myEngine, inputPixels, bytes, myOutput.data(), myOutput.size());
        setStatus(status);
        if (status != DATAMOSH_STATUS_OK)
        {
            std::memcpy(myOutput.data(), inputPixels, bytes);
        }

        uploadPixels(output, myOutput.data(), width, height, downRes->textureDesc);
    }

    void updateEngineParameters(const OP_Inputs* inputs)
    {
#if defined(SCANLINE_SIGNAL_TOP)
        myRequestedBackend = DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1;
#elif defined(DCT_TRANSFORM_TOP)
        myRequestedBackend = DATAMOSH_BACKEND_DCT_TRANSFORM_V1;
#else
        myRequestedBackend = DATAMOSH_BACKEND_RAW_MOSH_V1;
#endif
        const char* pattern = inputs->getParString("Pattern");
        myRequestedPatternIndex =
            std::clamp(inputs->getParInt("Pattern"), 0, patternCount() - 1);
        myRequestedPattern = pattern ? pattern : kPatternNames[myRequestedPatternIndex];
        myRequestedPreset = presetForPattern(myRequestedPattern);

        myIntensity = static_cast<float>(inputs->getParDouble("Intensity"));
        myMotion = static_cast<float>(inputs->getParDouble("Motion"));
        myResidual = static_cast<float>(inputs->getParDouble("Residual"));
        myTemporal = static_cast<float>(inputs->getParDouble("Temporal"));
        myBitstream = static_cast<float>(inputs->getParDouble("Bitstream"));
#if defined(DCT_TRANSFORM_TOP)
        myDctQuality =
            std::clamp(static_cast<float>(inputs->getParDouble("Quality")), 1.0f, 100.0f);
#endif
        myUseParams = inputs->getParInt("Useparams") != 0;

        const char* parameter = inputs->getParString("Paramid");
        myParameterId = parameter ? parameter : "";
        myParameterValue = static_cast<float>(inputs->getParDouble("Paramvalue"));

        applyAudioControlInputs(inputs);
    }

    void applyAudioControlInputs(const OP_Inputs* inputs)
    {
        myAudioActive = false;
        myAudioResetValue = 0.0f;

        if (inputs->getParInt("Audioenable") == 0)
            return;

        const OP_CHOPInput* chop = inputs->getParCHOP("Controlchop");
        if (!chop)
            return;

        const float amount =
            clampFloat(static_cast<float>(inputs->getParDouble("Audioamount")), 0.0f, 1.0f);
        const float gain = static_cast<float>(inputs->getParDouble("Audiogain"));
        const float bias = static_cast<float>(inputs->getParDouble("Audiobias"));

        auto applyChannel = [&](const char* parName, float& destination) {
            const char* channelName = inputs->getParString(parName);
            bool found = false;
            float value = latestChopValue(chop, channelName ? channelName : "", found);
            if (found)
            {
                destination = blendAudioValue(destination, value, amount, gain, bias);
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
        myAudioResetValue = latestChopValue(chop, resetChannel ? resetChannel : "", resetFound);
        if (resetFound)
        {
            myAudioActive = true;
            const float threshold =
                static_cast<float>(inputs->getParDouble("Resetthreshold"));
            const float rearm = static_cast<float>(inputs->getParDouble("Resetrearm"));
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
    }

    bool ensureEngine(uint32_t width, uint32_t height, const OP_Inputs* inputs)
    {
        if (!myRuntime.loaded())
        {
            std::string error;
            if (!myRuntime.loadNearPlugin(myNodeInfo ? myNodeInfo->pluginPath : nullptr, error))
            {
                myError = error;
                myLastStatusText = error;
                return false;
            }
        }

        if (!myEngine || myWidth != width || myHeight != height || myBackend != myRequestedBackend)
        {
            freeEngine();
            myEngine = myRuntime.engineNew(myRequestedBackend, width, height);
            if (!myEngine)
            {
                myError = "Could not create datamosh engine for requested backend/resolution";
                myLastStatus = DATAMOSH_STATUS_INVALID_ARGUMENT;
                myLastStatusText = myError;
                return false;
            }
            myBackend = myRequestedBackend;
            myWidth = width;
            myHeight = height;
            updateResolutionText();
            myPreset.clear();
        }

        const bool removingDedicatedOverrides = myAppliedUseParams && !myUseParams;
        const bool replacingAdvancedOverride =
            !myAppliedParameterId.empty() && myParameterId != myAppliedParameterId;
        const bool reloadPreset =
            myRequestedPreset != myPreset || removingDedicatedOverrides ||
            replacingAdvancedOverride;

        if (reloadPreset)
        {
            setStatus(myRuntime.setPreset(myEngine, myRequestedPreset.c_str()));
            if (myLastStatus != DATAMOSH_STATUS_OK)
                return false;
            myPreset = myRequestedPreset;
            resetAppliedOverrideState();
        }
        myPattern = myRequestedPattern;
        myPatternIndex = myRequestedPatternIndex;
        updatePatternIndexText();

#if defined(DCT_TRANSFORM_TOP)
        if (myDctQuality != myAppliedDctQuality)
        {
            setStatus(myRuntime.setParameter(myEngine, "quality", myDctQuality));
            if (myLastStatus != DATAMOSH_STATUS_OK)
                return false;
            myAppliedDctQuality = myDctQuality;
        }
#endif

        if (myUseParams)
        {
            constexpr size_t bindingCount =
                sizeof(kParameterBindings) / sizeof(kParameterBindings[0]);
            const bool applyAll =
                !myAppliedUseParams || myAppliedParameterValues.size() != bindingCount;
            if (myAppliedParameterValues.size() != bindingCount)
                myAppliedParameterValues.resize(bindingCount);

            for (size_t index = 0; index < bindingCount; ++index)
            {
                const ParameterBinding& binding = kParameterBindings[index];
                const float value = static_cast<float>(inputs->getParDouble(binding.parName));
                if (!applyAll && value == myAppliedParameterValues[index])
                    continue;

                setStatus(myRuntime.setParameter(myEngine, binding.id, value));
                if (myLastStatus != DATAMOSH_STATUS_OK)
                    return false;
                myAppliedParameterValues[index] = value;
            }
            myAppliedUseParams = true;
        }
        else
        {
            myAppliedUseParams = false;
            myAppliedParameterValues.clear();
        }

        if (!myParameterId.empty())
        {
            if (myParameterId != myAppliedParameterId ||
                myParameterValue != myAppliedParameterValue)
            {
                setStatus(
                    myRuntime.setParameter(myEngine, myParameterId.c_str(), myParameterValue));
                if (myLastStatus != DATAMOSH_STATUS_OK)
                    return false;
                myAppliedParameterId = myParameterId;
                myAppliedParameterValue = myParameterValue;
            }
        }
        else
        {
            myAppliedParameterId.clear();
            myAppliedParameterValue = 0.0f;
        }

        if (myResetPending)
        {
            setStatus(myRuntime.resetGlitch(myEngine));
            myResetPending = false;
            if (myLastStatus != DATAMOSH_STATUS_OK)
                return false;
        }

        setStatus(myRuntime.setControls(
            myEngine, myIntensity, myMotion, myResidual, myTemporal, myBitstream));

        return myLastStatus == DATAMOSH_STATUS_OK;
    }

    void uploadPixels(TOP_Output* output,
                      const uint8_t* pixels,
                      uint32_t width,
                      uint32_t height,
                      const OP_TextureDesc& sourceDesc)
    {
        const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        OP_SmartRef<TOP_Buffer> buf =
            myContext->createOutputBuffer(bytes, TOP_BufferFlags::None, nullptr);
        if (!buf || !buf->data)
        {
            myError = "Could not allocate TOP output buffer";
            return;
        }

        std::memcpy(buf->data, pixels, bytes);

        TOP_UploadInfo info;
        info.textureDesc = sourceDesc;
        info.textureDesc.width = width;
        info.textureDesc.height = height;
        info.textureDesc.depth = 1;
        info.textureDesc.texDim = OP_TexDim::e2D;
        info.textureDesc.pixelFormat = OP_PixelFormat::RGBA8Fixed;
        info.colorSpace = OP_ColorSpace::Passthrough;

        output->uploadBuffer(&buf, info, nullptr);
    }

    void uploadBlack(TOP_Output* output, uint32_t width, uint32_t height)
    {
        width = std::max<uint32_t>(width, 1);
        height = std::max<uint32_t>(height, 1);
        const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        myOutput.assign(bytes, 0);
        for (size_t i = 3; i < bytes; i += 4)
            myOutput[i] = 255;

        OP_TextureDesc desc;
        desc.width = width;
        desc.height = height;
        desc.depth = 1;
        desc.texDim = OP_TexDim::e2D;
        desc.pixelFormat = OP_PixelFormat::RGBA8Fixed;
        uploadPixels(output, myOutput.data(), width, height, desc);
    }

    void freeEngine()
    {
        if (myEngine && myRuntime.engineFree)
        {
            myRuntime.engineFree(myEngine);
            myEngine = nullptr;
        }
        myWidth = 0;
        myHeight = 0;
        myPreset.clear();
        resetAppliedOverrideState();
#if defined(DCT_TRANSFORM_TOP)
        myAppliedDctQuality = 50.0f;
#endif
        updateResolutionText();
    }

    void resetAppliedOverrideState()
    {
        myAppliedUseParams = false;
        myAppliedParameterValues.clear();
        myAppliedParameterId.clear();
        myAppliedParameterValue = 0.0f;
    }

    void setStatus(int32_t status)
    {
        myLastStatus = status;
        myLastStatusText = myRuntime.statusText(status);
        if (status == DATAMOSH_STATUS_OK)
            clearMessages();
        else
            myWarning = myLastStatusText;
    }

    void clearMessages()
    {
        myWarning.clear();
        myError.clear();
        if (myLastStatusText.empty())
            myLastStatusText = "ok";
    }

    void updateResolutionText()
    {
        char text[64] = {};
        std::snprintf(text, sizeof(text), "%ux%u", myWidth, myHeight);
        myResolutionText = text;
    }

    void updatePatternIndexText()
    {
        char text[64] = {};
        std::snprintf(text, sizeof(text), "%d", std::clamp(myPatternIndex, 0, patternCount() - 1));
        myPatternIndexText = text;
    }

    const OP_NodeInfo* myNodeInfo = nullptr;
    TOP_Context* myContext = nullptr;
    DatamoshRuntime myRuntime;
    DatamoshMoshEngine* myEngine = nullptr;
    OP_SmartRef<OP_TOPDownloadResult> myPrevDownRes;
    std::vector<uint8_t> myOutput;

    uint32_t myWidth = 0;
    uint32_t myHeight = 0;
    uint32_t myBackend = 0;
#if defined(SCANLINE_SIGNAL_TOP)
    uint32_t myRequestedBackend = DATAMOSH_BACKEND_SCANLINE_SIGNAL_V1;
#elif defined(DCT_TRANSFORM_TOP)
    uint32_t myRequestedBackend = DATAMOSH_BACKEND_DCT_TRANSFORM_V1;
#else
    uint32_t myRequestedBackend = DATAMOSH_BACKEND_RAW_MOSH_V1;
#endif
    int32_t myLastStatus = DATAMOSH_STATUS_OK;
    int myExecuteCount = 0;

    float myIntensity = 1.0f;
    float myMotion = 1.0f;
    float myResidual = 1.0f;
    float myTemporal = 1.0f;
    float myBitstream = 1.0f;
#if defined(DCT_TRANSFORM_TOP)
    float myDctQuality = 50.0f;
    float myAppliedDctQuality = 50.0f;
#endif
    float myParameterValue = 0.0f;
    float myAudioResetValue = 0.0f;
    bool myUseParams = false;
    bool myAppliedUseParams = false;
    bool myAudioActive = false;
    bool myAudioResetArmed = true;
    bool myResetPending = false;
    std::string myParameterId;
    std::vector<float> myAppliedParameterValues;
    std::string myAppliedParameterId;
    float myAppliedParameterValue = 0.0f;
#if defined(SCANLINE_SIGNAL_TOP)
    std::string myPattern = "clean";
    std::string myRequestedPattern = "clean";
    int32_t myPatternIndex = 0;
    int32_t myRequestedPatternIndex = 0;
    std::string myPreset = "clean";
    std::string myRequestedPreset = "clean";
#else
    std::string myPattern = "clean";
    std::string myRequestedPattern = "clean";
    int32_t myPatternIndex = 0;
    int32_t myRequestedPatternIndex = 0;
    std::string myPreset = "clean";
    std::string myRequestedPreset = "clean";
#endif
    std::string myLastStatusText = "ok";
    std::string myWarning;
    std::string myError;
#if defined(SCANLINE_SIGNAL_TOP)
    std::string myPatternIndexText = "0";
#else
    std::string myPatternIndexText = "0";
#endif
    std::string myResolutionText = "0x0";
};

extern "C" {

DLLEXPORT
void FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;

    info->executeMode = TOP_ExecuteMode::CPUMem;
#if defined(SCANLINE_SIGNAL_TOP)
    info->customOPInfo.opType->setString("Scanlinesignal");
    info->customOPInfo.opLabel->setString(kOperatorLabel);
    info->customOPInfo.opIcon->setString("SCN");
#elif defined(DCT_TRANSFORM_TOP)
    info->customOPInfo.opType->setString("Datamoshdct");
    info->customOPInfo.opLabel->setString(kOperatorLabel);
    info->customOPInfo.opIcon->setString("DCT");
#else
    info->customOPInfo.opType->setString("Datamosh");
    info->customOPInfo.opLabel->setString(kOperatorLabel);
    info->customOPInfo.opIcon->setString("DMS");
#endif
    info->customOPInfo.authorName->setString("datamosh");
    info->customOPInfo.authorEmail->setString("");
    info->customOPInfo.minInputs = 1;
    info->customOPInfo.maxInputs = 1;
    info->customOPInfo.opHelpURL->setString("");
}

DLLEXPORT
TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new DatamoshTOP(info, context);
}

DLLEXPORT
void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<DatamoshTOP*>(instance);
}

}

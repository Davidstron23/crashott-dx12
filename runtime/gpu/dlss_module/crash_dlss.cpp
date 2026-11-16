// CrashDlss.dll (crash_dlss.h): DLSS through NGX's Vulkan or Direct3D 12 API, for the renderer (upscaler.cpp) and
// for the launcher's and the installer's "can this PC run DLSS?" (CrashDlss_Probe). On Linux it is
// libCrashDlss.so, with NGX's Linux library (libnvsdk_ngx.a) and NVIDIA's libnvidia-ngx-dlss.so.*.
#include "crash_dlss.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#define CRASH_DLSS_EXPORT extern "C" __declspec(dllexport)
#else
#define CRASH_DLSS_EXPORT extern "C" __attribute__((visibility("default")))
// The Vulkan loader for the probe, as on Windows.
using HMODULE = void*;
#define LoadLibraryW(name) dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL)
#define GetProcAddress(library, name) dlsym(library, name)
#define FreeLibrary(library) dlclose(library)
// NGX's data folder (a wide path, as the API takes it), and its parents. Nothing here may throw: this
// library's libstdc++ exceptions can't cross into the game's libc++ (std::filesystem::path throws on
// "Área" under the "C" locale, and the unwinders don't mix), so the UTF-32 -> UTF-8 conversion is by hand.
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
inline std::string Utf8(const wchar_t* path)
{
    std::string utf8;
    for (const wchar_t* p = path; *p; p++)
    {
        uint32_t c = uint32_t(*p);
        if (c < 0x80)
            utf8 += char(c);
        else if (c < 0x800)
            utf8 += char(0xC0 | (c >> 6)), utf8 += char(0x80 | (c & 0x3F));
        else if (c < 0x10000)
            utf8 += char(0xE0 | (c >> 12)), utf8 += char(0x80 | ((c >> 6) & 0x3F)), utf8 += char(0x80 | (c & 0x3F));
        else
            utf8 += char(0xF0 | (c >> 18)), utf8 += char(0x80 | ((c >> 12) & 0x3F)), utf8 += char(0x80 | ((c >> 6) & 0x3F)),
                utf8 += char(0x80 | (c & 0x3F));
    }
    return utf8;
}

inline void CreateDirectoryW(const wchar_t* path, void*)
{
    std::string utf8 = Utf8(path);
    for (size_t at = 1; at <= utf8.size(); at++)
        if (at == utf8.size() || utf8[at] == '/')
            mkdir(utf8.substr(0, at).c_str(), 0755);
}

// NGX finds its feature library (libnvidia-ngx-dlss.so.*) by its real path, and fails on one that
// isn't ASCII ("Área de trabalho": "feature not found", 0xBAD00004; a symbolic link doesn't help, it
// is resolved). Then NVIDIA's files are copied (when changed) to the first ASCII folder of
// $XDG_CACHE_HOME/crashrecomp-ngx, ~/.cache/crashrecomp-ngx, $XDG_RUNTIME_DIR/crashrecomp-ngx and
// /tmp/crashrecomp-ngx-<uid>, and that folder is NGX's. The data folder may have any name.
inline bool Ascii(const std::string& text)
{
    return std::all_of(text.begin(), text.end(), [](char c) { return uint8_t(c) < 0x80; });
}

inline bool CopyFile(const std::string& from, const std::string& to)
{
    struct stat a{}, b{};
    if (stat(from.c_str(), &a) != 0)
        return false;
    if (stat(to.c_str(), &b) == 0 && a.st_size == b.st_size)
        return true;
    FILE* in = fopen(from.c_str(), "rb");
    FILE* out = in ? fopen((to + ".part").c_str(), "wb") : nullptr;
    bool ok = in && out;
    char buffer[1 << 16];
    size_t read;
    while (ok && (read = fread(buffer, 1, sizeof(buffer), in)) > 0)
        ok = fwrite(buffer, 1, read, out) == read;
    if (in)
        fclose(in);
    if (out)
        ok = fclose(out) == 0 && ok;
    return ok && rename((to + ".part").c_str(), to.c_str()) == 0;
}

inline std::wstring AsciiFeatureDir(const wchar_t* featureDir)
{
    std::string from = Utf8(featureDir);
    if (Ascii(from))
        return featureDir;
    std::vector<std::string> bases;
    if (const char* cache = getenv("XDG_CACHE_HOME"); cache && *cache)
        bases.push_back(std::string(cache) + "/crashrecomp-ngx");
    if (const char* home = getenv("HOME"); home && *home)
        bases.push_back(std::string(home) + "/.cache/crashrecomp-ngx");
    if (const char* runtime = getenv("XDG_RUNTIME_DIR"); runtime && *runtime)
        bases.push_back(std::string(runtime) + "/crashrecomp-ngx");
    bases.push_back("/tmp/crashrecomp-ngx-" + std::to_string(getuid()));
    for (const std::string& base : bases)
    {
        if (!Ascii(base))
            continue;
        std::wstring wide(base.begin(), base.end());
        CreateDirectoryW(wide.c_str(), nullptr);
        bool copied = false, ok = true;
        if (DIR* dir = opendir(from.c_str()))
        {
            while (dirent* entry = readdir(dir))
            {
                std::string name = entry->d_name;
                if (name.rfind("libnvidia-ngx-", 0) == 0)
                {
                    ok = ok && CopyFile(from + "/" + name, base + "/" + name);
                    copied = true;
                }
            }
            closedir(dir);
        }
        if (copied && ok)
            return wide;
    }
    return featureDir;
}
#endif

namespace
{
    // Project id required by NVSDK_NGX_VULKAN_Init_with_ProjectID for custom engines (any GUID-like
    // string; this one is CrashRecomp's).
    constexpr const char* NGX_PROJECT_ID = "6c7a3f1e-0b58-4d6e-9a3c-56550e4c7261";

    CrashDlssLogFn g_log = nullptr;
    VkDevice g_device = VK_NULL_HANDLE;
    bool g_initialized = false;
    bool g_d3d12 = false;           // NGX was started on a Direct3D 12 device (g_d3d12Device), else on g_device
    void* g_d3d12Device = nullptr;  // ID3D12Device
    NVSDK_NGX_Parameter* g_parameters = nullptr;
    NVSDK_NGX_Handle* g_feature = nullptr;
    uint32_t g_featureKey[5] = {};  // render w/h, output w/h, quality + flags of the feature

    void Log(const char* format, ...)
    {
        if (!g_log)
            return;
        char text[1024];
        va_list args;
        va_start(args, format);
        vsnprintf(text, sizeof(text), format, args);
        va_end(args);
        g_log(text);
    }

    void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
    {
        std::string text = message ? message : "";
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r'))
            text.pop_back();
        Log("[dlss] %s", text.c_str());
    }

    // NGX on the Vulkan `vulkan` or, when it is null, on the Direct3D 12 device `d3d12`, its capability
    // parameters, and whether DLSS Super Resolution is available. `why` gets the reason when it isn't.
    // Leaves NGX initialized (g_initialized / g_parameters).
    bool StartNgx(const wchar_t* featureDirectory, const wchar_t* dataDirectory, const CrashDlssDevice* vulkan, void* d3d12, std::string& why)
    {
#ifndef _WIN32
        std::wstring featureDir = AsciiFeatureDir(featureDirectory ? featureDirectory : L".");
        const wchar_t* paths[] = { featureDir.c_str() };
#else
        const wchar_t* paths[] = { featureDirectory ? featureDirectory : L"." };
#endif
        NVSDK_NGX_FeatureCommonInfo info{};
        info.PathListInfo.Path = paths;
        info.PathListInfo.Length = 1;
        info.LoggingInfo.LoggingCallback = NgxLog;
        info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
        const wchar_t* dataDir = dataDirectory && *dataDirectory ? dataDirectory : paths[0];
        CreateDirectoryW(dataDir, nullptr);
        NVSDK_NGX_Result result;
#ifdef _WIN32
        if (!vulkan)
            result = NVSDK_NGX_D3D12_Init_with_ProjectID(NGX_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", dataDir,
                                                         static_cast<ID3D12Device*>(d3d12), &info, NVSDK_NGX_Version_API);
        else
#endif
            result = NVSDK_NGX_VULKAN_Init_with_ProjectID(NGX_PROJECT_ID, NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", dataDir, vulkan->instance,
                                                          vulkan->physical, vulkan->device, vulkan->getInstanceProcAddr,
                                                          vulkan->getDeviceProcAddr, &info, NVSDK_NGX_Version_API);
        char text[160];
        if (NVSDK_NGX_FAILED(result))
        {
            snprintf(text, sizeof(text), "NGX initialization failed (0x%08X; not an NVIDIA RTX GPU or an old driver)", uint32_t(result));
            why = text;
            return false;
        }
        g_initialized = true;
        g_d3d12 = !vulkan;
        g_d3d12Device = d3d12;
        g_device = vulkan ? vulkan->device : VK_NULL_HANDLE;
        NVSDK_NGX_Result capability;
#ifdef _WIN32
        if (g_d3d12)
            capability = NVSDK_NGX_D3D12_GetCapabilityParameters(&g_parameters);
        else
#endif
            capability = NVSDK_NGX_VULKAN_GetCapabilityParameters(&g_parameters);
        if (NVSDK_NGX_FAILED(capability) || !g_parameters)
        {
            g_parameters = nullptr;
            why = "no NGX capability parameters";
            return false;
        }
        int available = 0, needsDriver = 0;
        NVSDK_NGX_Parameter_GetI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_Available, &available);
        NVSDK_NGX_Parameter_GetI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver);
        if (!available)
        {
            int initResult = 0;
            NVSDK_NGX_Parameter_GetI(g_parameters, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
            snprintf(text, sizeof(text), "not available on this GPU (needs updated driver: %s, result 0x%08X)", needsDriver ? "yes" : "no",
                     uint32_t(initResult));
            why = text;
            return false;
        }
        return true;
    }

    void ReleaseFeatureNow()
    {
        if (g_feature)
        {
#ifdef _WIN32
            if (g_d3d12)
                NVSDK_NGX_D3D12_ReleaseFeature(g_feature);
            else
#endif
                NVSDK_NGX_VULKAN_ReleaseFeature(g_feature);
        }
        g_feature = nullptr;
    }

    void StopNgx()
    {
        ReleaseFeatureNow();
#ifdef _WIN32
        if (g_d3d12)
        {
            if (g_parameters)
                NVSDK_NGX_D3D12_DestroyParameters(g_parameters);
            if (g_initialized)
                NVSDK_NGX_D3D12_Shutdown1(static_cast<ID3D12Device*>(g_d3d12Device));
        }
        else
#endif
        {
            if (g_parameters)
                NVSDK_NGX_VULKAN_DestroyParameters(g_parameters);
            if (g_initialized)
                NVSDK_NGX_VULKAN_Shutdown1(g_device);
        }
        g_parameters = nullptr;
        g_initialized = false;
        g_d3d12 = false;
        g_d3d12Device = nullptr;
        g_device = VK_NULL_HANDLE;
    }

    NVSDK_NGX_PerfQuality_Value Quality(int32_t quality)
    {
        switch (quality)
        {
        case 1: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case 2: return NVSDK_NGX_PerfQuality_Value_Balanced;
        case 3: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case 4: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
        default: return NVSDK_NGX_PerfQuality_Value_DLAA;
        }
    }

    const char* QualityName(int32_t quality)
    {
        static const char* names[] = { "native", "quality", "balanced", "performance", "ultra_performance" };
        return quality >= 0 && quality < 5 ? names[quality] : "?";
    }

    NVSDK_NGX_Resource_VK Resource(const CrashDlssImage& image, bool readWrite)
    {
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, range, image.format, image.width, image.height, readWrite);
    }

    // The feature for this frame's sizes, mode and flags (an earlier one for other values is released
    // first); `create` makes it on the API's command list. Same messages and warning cap for both APIs.
    template <class Frame, class Create>
    bool EnsureFeature(const Frame& frame, Create create)
    {
        int flags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
                    (frame.depthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0);
        uint32_t key[5] = { frame.renderWidth, frame.renderHeight, frame.outputWidth, frame.outputHeight,
                            uint32_t(frame.quality) | (uint32_t(flags) << 8) };
        if (g_feature && memcmp(key, g_featureKey, sizeof(key)) != 0)
            ReleaseFeatureNow();
        if (g_feature)
            return true;
        NVSDK_NGX_DLSS_Create_Params params{};
        params.Feature.InWidth = frame.renderWidth;
        params.Feature.InHeight = frame.renderHeight;
        params.Feature.InTargetWidth = frame.outputWidth;
        params.Feature.InTargetHeight = frame.outputHeight;
        params.Feature.InPerfQualityValue = Quality(frame.quality);
        params.InFeatureCreateFlags = flags;
        // Presets: the DLL's defaults (transformer K for DLAA/quality/balanced, M/L for the
        // performance modes in DLSS 4.5).
        NVSDK_NGX_Result result = create(params);
        if (NVSDK_NGX_FAILED(result))
        {
            static uint32_t logged;
            if (logged++ < 4)
                Log("[gpu] DLSS: could not create the feature %ux%u -> %ux%u (0x%08X)", frame.renderWidth, frame.renderHeight,
                    frame.outputWidth, frame.outputHeight, uint32_t(result));
            g_feature = nullptr;
            return false;
        }
        memcpy(g_featureKey, key, sizeof(key));
        Log("[gpu] DLSS feature created: %ux%u -> %ux%u, %s", frame.renderWidth, frame.renderHeight, frame.outputWidth, frame.outputHeight,
            QualityName(frame.quality));
        return true;
    }

    bool EvaluateFailed(NVSDK_NGX_Result result)
    {
        if (!NVSDK_NGX_FAILED(result))
            return false;
        static uint32_t logged;
        if (logged++ < 4)
            Log("[gpu] DLSS: evaluation failed (0x%08X)", uint32_t(result));
        return true;
    }

    void Copy(char* out, uint32_t size, const std::string& text)
    {
        if (out && size)
            snprintf(out, size, "%s", text.c_str());
    }
}

CRASH_DLSS_EXPORT uint32_t CrashDlss_ApiVersion()
{
    return CRASH_DLSS_API_VERSION;
}

CRASH_DLSS_EXPORT int32_t CrashDlss_RequiredExtensions(uint32_t* instanceCount, const char* const** instanceNames, uint32_t* deviceCount,
                                                       const char* const** deviceNames)
{
    // NGX's list (the same for every NVIDIA GPU; the device enables the ones the driver has).
    unsigned int ic = 0, dc = 0;
    const char** in = nullptr;
    const char** dn = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ic, &in, &dc, &dn)))
        return 0;
    *instanceCount = ic, *instanceNames = in, *deviceCount = dc, *deviceNames = dn;
    return 1;
}

CRASH_DLSS_EXPORT int32_t CrashDlss_Init(const CrashDlssDevice* device)
{
    g_log = device->log;
    std::string why;
    if (StartNgx(device->featureDir, device->dataDir, device, nullptr, why))
        return 1;
    Log("[gpu] DLSS: %s", why.c_str());
    return 0;
}

CRASH_DLSS_EXPORT int32_t CrashDlss_Evaluate(const CrashDlssFrame* frame)
{
    if (!g_parameters)
        return 0;
    if (!EnsureFeature(*frame, [&](NVSDK_NGX_DLSS_Create_Params& create) {
            return NGX_VULKAN_CREATE_DLSS_EXT1(g_device, frame->commandBuffer, 1, 1, &g_feature, g_parameters, &create);
        }))
        return 0;
    NVSDK_NGX_Resource_VK color = Resource(frame->color, false), depth = Resource(frame->depth, false),
                          motion = Resource(frame->motion, false), output = Resource(frame->output, true);
    NVSDK_NGX_VK_DLSS_Eval_Params eval{};
    eval.Feature.pInColor = &color;
    eval.Feature.pInOutput = &output;
    eval.pInDepth = &depth;
    eval.pInMotionVectors = &motion;
    eval.InJitterOffsetX = frame->jitterX;
    eval.InJitterOffsetY = frame->jitterY;
    eval.InRenderSubrectDimensions = { frame->renderWidth, frame->renderHeight };
    eval.InReset = frame->reset ? 1 : 0;
    // Motion vectors are in UV units: scale to render pixels.
    eval.InMVScaleX = float(frame->renderWidth);
    eval.InMVScaleY = float(frame->renderHeight);
    eval.InFrameTimeDeltaInMsec = frame->frameTimeMs;
    return EvaluateFailed(NGX_VULKAN_EVALUATE_DLSS_EXT(frame->commandBuffer, g_feature, g_parameters, &eval)) ? 0 : 1;
}

#ifdef _WIN32
// Direct3D 12: NGX's own helpers for it (nvsdk_ngx_helpers_d3d.h) are not in the SDK files this project
// takes, so the parameters they would set are set here, the same ones as the Vulkan helpers do.
CRASH_DLSS_EXPORT int32_t CrashDlss_InitD3D12(const CrashDlssD3D12Device* device)
{
    g_log = device->log;
    std::string why;
    if (StartNgx(device->featureDir, device->dataDir, nullptr, device->device, why))
        return 1;
    Log("[gpu] DLSS: %s", why.c_str());
    return 0;
}

CRASH_DLSS_EXPORT int32_t CrashDlss_EvaluateD3D12(const CrashDlssD3D12Frame* frame)
{
    if (!g_parameters || !g_d3d12)
        return 0;
    auto commandList = static_cast<ID3D12GraphicsCommandList*>(frame->commandList);
    NVSDK_NGX_Parameter* p = g_parameters;
    if (!EnsureFeature(*frame, [&](NVSDK_NGX_DLSS_Create_Params& create) {
            NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_CreationNodeMask, 1);
            NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
            NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_Width, create.Feature.InWidth);
            NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_Height, create.Feature.InHeight);
            NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_OutWidth, create.Feature.InTargetWidth);
            NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_OutHeight, create.Feature.InTargetHeight);
            NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_PerfQualityValue, create.Feature.InPerfQualityValue);
            NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags, create.InFeatureCreateFlags);
            NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
            return NVSDK_NGX_D3D12_CreateFeature(commandList, NVSDK_NGX_Feature_SuperSampling, p, &g_feature);
        }))
        return 0;
    NVSDK_NGX_Parameter_SetD3d12Resource(p, NVSDK_NGX_Parameter_Color, static_cast<ID3D12Resource*>(frame->color));
    NVSDK_NGX_Parameter_SetD3d12Resource(p, NVSDK_NGX_Parameter_Output, static_cast<ID3D12Resource*>(frame->output));
    NVSDK_NGX_Parameter_SetD3d12Resource(p, NVSDK_NGX_Parameter_Depth, static_cast<ID3D12Resource*>(frame->depth));
    NVSDK_NGX_Parameter_SetD3d12Resource(p, NVSDK_NGX_Parameter_MotionVectors, static_cast<ID3D12Resource*>(frame->motion));
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_Jitter_Offset_X, frame->jitterX);
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_Jitter_Offset_Y, frame->jitterY);
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_Sharpness, 0.0f);
    NVSDK_NGX_Parameter_SetI(p, NVSDK_NGX_Parameter_Reset, frame->reset ? 1 : 0);
    // Motion vectors are in UV units: scale to render pixels.
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_MV_Scale_X, float(frame->renderWidth));
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_MV_Scale_Y, float(frame->renderHeight));
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_FrameTimeDeltaInMsec, frame->frameTimeMs);
    NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, frame->renderWidth);
    NVSDK_NGX_Parameter_SetUI(p, NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, frame->renderHeight);
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    NVSDK_NGX_Parameter_SetF(p, NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    return EvaluateFailed(NVSDK_NGX_D3D12_EvaluateFeature_C(commandList, g_feature, p, nullptr)) ? 0 : 1;
}
#endif

CRASH_DLSS_EXPORT void CrashDlss_ReleaseFeature()
{
    ReleaseFeatureNow();
}

CRASH_DLSS_EXPORT void CrashDlss_Shutdown()
{
    StopNgx();
}

CRASH_DLSS_EXPORT int32_t CrashDlss_Probe(const wchar_t* featureDir, const wchar_t* dataDir, char* gpuName, uint32_t gpuNameSize, char* reason,
                                          uint32_t reasonSize)
{
    Copy(gpuName, gpuNameSize, "");
    Copy(reason, reasonSize, "");
    if (g_initialized)
    {
        Copy(reason, reasonSize, "NGX is already in use in this process");
        return 0;
    }
    HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
    if (!vulkan)
    {
        Copy(reason, reasonSize, "no Vulkan driver (the Vulkan loader)");
        return 0;
    }
    auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr"));
    auto createInstance = gipa ? reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance")) : nullptr;
    auto enumerateInstanceExtensions =
        gipa ? reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(gipa(nullptr, "vkEnumerateInstanceExtensionProperties")) : nullptr;
    if (!createInstance || !enumerateInstanceExtensions)
    {
        FreeLibrary(vulkan);
        Copy(reason, reasonSize, "the Vulkan loader has no vkCreateInstance");
        return 0;
    }
    unsigned int ngxInstanceCount = 0, ngxDeviceCount = 0;
    const char** ngxInstance = nullptr;
    const char** ngxDevice = nullptr;
    NVSDK_NGX_VULKAN_RequiredExtensions(&ngxInstanceCount, &ngxInstance, &ngxDeviceCount, &ngxDevice);

    // The instance, with the NGX instance extensions the loader has.
    uint32_t count = 0;
    enumerateInstanceExtensions(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> have(count);
    enumerateInstanceExtensions(nullptr, &count, have.data());
    auto has = [](const std::vector<VkExtensionProperties>& list, const char* name) {
        for (const VkExtensionProperties& e : list)
            if (strcmp(e.extensionName, name) == 0)
                return true;
        return false;
    };
    std::vector<const char*> instanceExtensions;
    for (unsigned int i = 0; i < ngxInstanceCount; i++)
        if (has(have, ngxInstance[i]))
            instanceExtensions.push_back(ngxInstance[i]);
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "CrashRecomp DLSS probe";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instanceInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instanceInfo.pApplicationInfo = &app;
    instanceInfo.enabledExtensionCount = uint32_t(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    VkInstance instance = VK_NULL_HANDLE;
    if (createInstance(&instanceInfo, nullptr, &instance) != VK_SUCCESS)
    {
        FreeLibrary(vulkan);
        Copy(reason, reasonSize, "vkCreateInstance failed (no Vulkan 1.3 driver)");
        return 0;
    }
#define INSTANCE_FN(name) auto name = reinterpret_cast<PFN_##name>(gipa(instance, #name))
    INSTANCE_FN(vkDestroyInstance);
    INSTANCE_FN(vkEnumeratePhysicalDevices);
    INSTANCE_FN(vkGetPhysicalDeviceProperties);
    INSTANCE_FN(vkGetPhysicalDeviceQueueFamilyProperties);
    INSTANCE_FN(vkEnumerateDeviceExtensionProperties);
    INSTANCE_FN(vkCreateDevice);
    INSTANCE_FN(vkGetDeviceProcAddr);
#undef INSTANCE_FN

    // The NVIDIA GPU (a discrete one first).
    count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties chosen{};
    for (VkPhysicalDevice candidate : devices)
    {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        if (properties.vendorID != 0x10DE)
            continue;
        if (!physical || (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && chosen.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU))
            physical = candidate, chosen = properties;
    }
    int32_t available = 0;
    std::string why;
    if (!physical)
        why = "no NVIDIA GPU";
    else
    {
        Copy(gpuName, gpuNameSize, chosen.deviceName);
        count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
        uint32_t family = UINT32_MAX;
        for (uint32_t i = 0; i < count && family == UINT32_MAX; i++)
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                family = i;
        count = 0;
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
        std::vector<VkExtensionProperties> deviceHave(count);
        vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, deviceHave.data());
        std::vector<const char*> deviceExtensions;
        for (unsigned int i = 0; i < ngxDeviceCount; i++)
            // VK_EXT_buffer_device_address: the core feature replaces it (both together are invalid).
            if (strcmp(ngxDevice[i], "VK_EXT_buffer_device_address") != 0 && has(deviceHave, ngxDevice[i]))
                deviceExtensions.push_back(ngxDevice[i]);
        float priority = 1.0f;
        VkDeviceQueueCreateInfo queue{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queue.queueFamilyIndex = family == UINT32_MAX ? 0 : family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        features12.bufferDeviceAddress = VK_TRUE;  // NGX's features want device addresses
        VkDeviceCreateInfo deviceInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features12 };
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queue;
        deviceInfo.enabledExtensionCount = uint32_t(deviceExtensions.size());
        deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();
        VkDevice device = VK_NULL_HANDLE;
        if (vkCreateDevice(physical, &deviceInfo, nullptr, &device) != VK_SUCCESS)
            why = "vkCreateDevice failed on the NVIDIA GPU";
        else
        {
            CrashDlssDevice info{ instance, physical, device, gipa, vkGetDeviceProcAddr, dataDir, featureDir, nullptr };
            available = StartNgx(featureDir, dataDir, &info, nullptr, why) ? 1 : 0;
            StopNgx();
            auto vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(device, "vkDestroyDevice"));
            vkDestroyDevice(device, nullptr);
        }
    }
    vkDestroyInstance(instance, nullptr);
    FreeLibrary(vulkan);
    Copy(reason, reasonSize, available ? std::string("DLSS available") : why);
    return available;
}

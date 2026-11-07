#pragma once

// The GPU the game will pick and what it can run, shared by the launcher, the installer (which
// DLSS/FSR files and options to offer) and the runtime's Direct3D 12 path (gpu/upscaler.cpp, whose
// Available() answers by the same rule when no Vulkan device exists).
//
//   - DetectGpu: the GPU the game will pick (Vulkan, as rhi_vulkan.cpp chooses it: discrete, then
//     integrated, CRASH_VULKAN_DEVICE), its vendor and whether its driver has the extensions FSR 3.1
//     calls. CRASH_GPU_VENDOR (settings/gpu_vendor.h) pretends another vendor.

#include "../platform/platform.h"
#include "../settings/gpu_vendor.h"

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES 1
#endif
#include <vulkan/vulkan.h>

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace gpu
{
    struct GpuInfo
    {
        uint32_t vendor = 0;       // PCI vendor id (0 = no Vulkan GPU found)
        std::string name;
        bool vulkan = false;       // a Vulkan 1.3 GPU
        bool fsrExtensions = false;  // the device extensions amd_fidelityfx_vk.dll calls
        bool forced = false;       // vendor from CRASH_GPU_VENDOR
    };

    inline GpuInfo DetectGpu()
    {
        GpuInfo gpu;
#ifdef _WIN32
        HMODULE vulkan = LoadLibraryW(L"vulkan-1.dll");
        auto gipa = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(vulkan, "vkGetInstanceProcAddr")) : nullptr;
#else
        void* vulkan = plat::OpenLibrary("libvulkan.so.1");
        auto gipa = vulkan ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(plat::FindSymbol(vulkan, "vkGetInstanceProcAddr")) : nullptr;
#endif
        auto createInstance = gipa ? reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance")) : nullptr;
        VkInstance instance = VK_NULL_HANDLE;
        VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "CrashRecomp GPU check";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo create{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        create.pApplicationInfo = &app;
        if (createInstance && createInstance(&create, nullptr, &instance) == VK_SUCCESS)
        {
            auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
            auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
            auto extensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(gipa(instance, "vkEnumerateDeviceExtensionProperties"));
            auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
            uint32_t count = 0;
            enumerate(instance, &count, nullptr);
            std::vector<VkPhysicalDevice> devices(count);
            enumerate(instance, &count, devices.data());
            int best = -1;
            const char* wanted = getenv("CRASH_VULKAN_DEVICE");
            for (VkPhysicalDevice device : devices)
            {
                VkPhysicalDeviceProperties p{};
                properties(device, &p);
                if (p.apiVersion < VK_API_VERSION_1_3)
                    continue;
                int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
                if (wanted && strstr(p.deviceName, wanted))
                    score += 10;
                if (score <= best)
                    continue;
                best = score;
                gpu.vendor = p.vendorID;
                gpu.name = p.deviceName;
                gpu.vulkan = true;
                uint32_t n = 0;
                extensions(device, nullptr, &n, nullptr);
                std::vector<VkExtensionProperties> have(n);
                extensions(device, nullptr, &n, have.data());
                auto has = [&](const char* name) {
                    return std::any_of(have.begin(), have.end(), [&](const VkExtensionProperties& e) { return strcmp(e.extensionName, name) == 0; });
                };
                // The list of runtime/gpu/upscaler.cpp (FSR_DEVICE_EXTENSIONS).
                gpu.fsrExtensions = has(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME) && has(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME) &&
                                    has(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) && has(VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME);
            }
            destroy(instance, nullptr);
        }
#ifdef _WIN32
        if (vulkan)
            FreeLibrary(vulkan);
#else
        if (vulkan)
            plat::CloseLibrary(vulkan);
#endif
        if (uint32_t forced = settings::ForcedGpuVendor())
        {
            gpu.vendor = forced;
            gpu.forced = true;
        }
        return gpu;
    }
}

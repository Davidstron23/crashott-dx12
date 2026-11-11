// Direct3D 12 backend of the rendering hardware interface (rhi.h), Windows only.
//
// The device, one direct queue and command list, a flip-model swap chain, the upload ring, clears,
// the read-back of the back buffer (screenshots) and the pass pipelines (the present, the overlay),
// whose HLSL is compiled to DXIL by DXC (dxcompiler.dll, loaded at run time) and signed by dxil.dll,
// which Direct3D 12 requires (docs/decisions/0002-dxil-signing.md).
// Spec 002 adds the game's own draws: textures (with depth and stencil), DXIL shaders, draw pipelines and Draw.
//
// Adapted from the backend this project had until 2026-10-23 (4cac6ecb22f7:runtime/gpu/rhi_d3d12.cpp).

#include "rhi.h"
#include "rhi_ring.h"
#include "../diagnostics.h"
#include "../platform/platform.h"
#include "../platform/text.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <unknwn.h>
#include <objidl.h>  // IStream (WIN32_LEAN_AND_MEAN leaves the COM headers out)
// dxcapi.h's __declspec(uuid(...)) is unknown to mingw's clang (the GUIDs are declared below).
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include <dxcapi.h>
#pragma clang diagnostic pop

#ifdef __MINGW32__
// mingw's __uuidof needs the GUIDs declared (as in rhi_vulkan_shaders.cpp).
__CRT_UUID_DECL(IDxcBlob, 0x8BA5FB08, 0x5195, 0x40E2, 0xAC, 0x58, 0x0D, 0x98, 0x9C, 0x3A, 0x01, 0x02)
__CRT_UUID_DECL(IDxcBlobUtf8, 0x3DA636C9, 0xBA71, 0x4024, 0xA3, 0x01, 0x30, 0xCB, 0xF1, 0x25, 0x30, 0x5B)
__CRT_UUID_DECL(IDxcResult, 0x58346CDA, 0xDDE7, 0x4497, 0x94, 0x61, 0x6F, 0x87, 0xAF, 0x5E, 0x06, 0x59)
__CRT_UUID_DECL(IDxcCompiler3, 0x228B4687, 0x5A6A, 0x4730, 0x90, 0x0C, 0x97, 0x02, 0xB2, 0x20, 0x3F, 0x54)
__CRT_UUID_DECL(IDxcVersionInfo, 0xB04F5B50, 0x2059, 0x4F12, 0xA8, 0xFF, 0xA1, 0xE0, 0xCD, 0xE1, 0xCC, 0x7E)
__CRT_UUID_DECL(IDxcVersionInfo2, 0xFB6904C4, 0x42F0, 0x4B62, 0x9C, 0x46, 0x98, 0x3A, 0xF7, 0xDA, 0x7C, 0x83)
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace gpu::rhi
{
    namespace
    {
        constexpr uint32_t BACK_BUFFER_COUNT = 2;
        constexpr uint32_t REMOVE_AT_PRESENT = 120;  // CRASH_D3D12_FAIL=removed: a few seconds after the start, always the same frame
        constexpr uint32_t TIMESTAMPS = 64;  // the device's timestamp slots (the contract in rhi.h)
        constexpr uint64_t UPLOAD_RING_SIZE = 256ull << 20;
        constexpr uint32_t RTV_DESCRIPTORS = 4096;
        constexpr uint32_t DSV_DESCRIPTORS = 1024;
        constexpr uint32_t CPU_SRV_DESCRIPTORS = 32768;
        constexpr uint32_t SHADER_VISIBLE_DESCRIPTORS = 65536;
        constexpr uint32_t SAMPLER_DESCRIPTORS = 2048;  // blocks of 32, cached by the samplers a draw uses
        // The last shader-visible slot is kept for the upload ring's raw SRV (vertex data of the game's draws, spec 002).
        constexpr uint32_t VERTEX_DATA_DESCRIPTOR = SHADER_VISIBLE_DESCRIPTORS - 1;
        constexpr uint32_t NO_DESCRIPTOR = UINT32_MAX;
        constexpr D3D12_RESOURCE_STATES STATE_SHADER_RESOURCE =
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;

        // Root parameters of the draw root signature (the binding model at the top of rhi.h).
        enum RootParameter : uint32_t
        {
            ROOT_SYSTEM_CONSTANTS = 0,  // b0
            ROOT_VS_CONSTANTS = 1,      // b1
            ROOT_PS_CONSTANTS = 2,      // b2
            ROOT_TEXTURES = 3,          // t0-t31
            ROOT_VERTEX_DATA = 4,       // t0 space1: the upload ring
            ROOT_EFFECTS = 5,           // b3: graphics option constants (graphics_options.h)
            ROOT_SAMPLERS = 6,          // s0-s31
            ROOT_PARAMETER_COUNT,
        };

        // Indices of a CPU descriptor heap, with a free list (textures destroyed give theirs back).
        struct DescriptorPool
        {
            ComPtr<ID3D12DescriptorHeap> heap;
            uint32_t size = 0, count = 0, capacity = 0;
            std::vector<uint32_t> free;
            const char* name = "";

            uint32_t Allocate()
            {
                if (!free.empty())
                {
                    uint32_t index = free.back();
                    free.pop_back();
                    return index;
                }
                if (count >= capacity)
                    Fatal("[gpu] out of %s descriptors", name);
                return count++;
            }
            void Release(uint32_t index)
            {
                if (index != NO_DESCRIPTOR)
                    free.push_back(index);
            }
            D3D12_CPU_DESCRIPTOR_HANDLE Cpu(uint32_t index) const
            {
                D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
                handle.ptr += size_t(index) * size;
                return handle;
            }
        };

        // The swap chain's back buffers and the game's textures.
        class D3D12Texture : public Texture
        {
        public:
            ComPtr<ID3D12Resource> resource;
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            uint32_t srv[3] = { NO_DESCRIPTOR, NO_DESCRIPTOR, NO_DESCRIPTOR };  // CPU SRV per View, made on first use
            uint32_t rtv = NO_DESCRIPTOR, dsv = NO_DESCRIPTOR;
        };

        class D3D12Shader : public Shader
        {
        public:
            std::vector<uint8_t> code;  // DXIL
        };

        // The overlay's image upload slices (an upload heap buffer, mapped for its whole life).
        class D3D12Buffer : public Buffer
        {
        public:
            ComPtr<ID3D12Resource> resource;
        };

        class D3D12Pipeline : public Pipeline
        {
        public:
            ComPtr<ID3D12PipelineState> state;
            ComPtr<ID3D12RootSignature> rootSignature;  // each pass has its own (draw pipelines use the device's)
            uint32_t textureCount = 0;
            bool bufferInput = false;
        };

        // The vertex shader of every pass without one of its own: a full-screen triangle from SV_VertexID.
        const char* FULLSCREEN_VS = R"(
float4 main(uint id : SV_VertexID) : SV_Position
{
    float2 uv = float2((id << 1) & 2, id & 2);
    return float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
}
)";

        D3D12_TEXTURE_ADDRESS_MODE AddressMode(Address address)
        {
            switch (address)
            {
            case Address::Wrap: return D3D12_TEXTURE_ADDRESS_MODE_WRAP;
            case Address::Mirror: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR;
            case Address::MirrorOnce: return D3D12_TEXTURE_ADDRESS_MODE_MIRROR_ONCE;
            case Address::Border: return D3D12_TEXTURE_ADDRESS_MODE_BORDER;
            default: return D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
            }
        }

        D3D12_SAMPLER_DESC SamplerOf(const SamplerDesc& sampler)
        {
            D3D12_SAMPLER_DESC desc{};
            if (sampler.anisotropy > 1)
            {
                desc.Filter = D3D12_FILTER_ANISOTROPIC;
                desc.MaxAnisotropy = sampler.anisotropy;
            }
            else
            {
                auto type = [](Filter filter) { return filter == Filter::Linear ? D3D12_FILTER_TYPE_LINEAR : D3D12_FILTER_TYPE_POINT; };
                desc.Filter = D3D12_ENCODE_BASIC_FILTER(type(sampler.minFilter), type(sampler.magFilter), type(sampler.mipFilter),
                                                        D3D12_FILTER_REDUCTION_TYPE_STANDARD);
                desc.MaxAnisotropy = 1;
            }
            desc.AddressU = AddressMode(sampler.addressU);
            desc.AddressV = AddressMode(sampler.addressV);
            desc.AddressW = AddressMode(sampler.addressW);
            desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
            float border = sampler.border == BorderColor::OpaqueWhite ? 1.0f : 0.0f;
            desc.BorderColor[0] = desc.BorderColor[1] = desc.BorderColor[2] = border;
            desc.BorderColor[3] = sampler.border == BorderColor::TransparentBlack ? 0.0f : 1.0f;
            desc.MinLOD = sampler.minLod;
            desc.MaxLOD = sampler.maxLod;
            return desc;
        }

        D3D_PRIMITIVE_TOPOLOGY PrimitiveTopology(Topology topology)
        {
            switch (topology)
            {
            case Topology::PointList: return D3D_PRIMITIVE_TOPOLOGY_POINTLIST;
            case Topology::LineList: return D3D_PRIMITIVE_TOPOLOGY_LINELIST;
            case Topology::LineStrip: return D3D_PRIMITIVE_TOPOLOGY_LINESTRIP;
            case Topology::TriangleStrip: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP;
            default: return D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
            }
        }

        D3D12_PRIMITIVE_TOPOLOGY_TYPE TopologyType(Topology topology)
        {
            switch (topology)
            {
            case Topology::PointList: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
            case Topology::LineList: case Topology::LineStrip: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE;
            default: return D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            }
        }

        D3D12_BLEND BlendFactor(uint32_t factor, bool alpha)
        {
            switch (factor)
            {
            case 0: return D3D12_BLEND_ZERO;
            case 1: return D3D12_BLEND_ONE;
            case 4: return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;          // SrcColor
            case 5: return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;  // OneMinusSrcColor
            case 6: return D3D12_BLEND_SRC_ALPHA;
            case 7: return D3D12_BLEND_INV_SRC_ALPHA;
            case 8: return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;        // DstColor
            case 9: return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
            case 10: return D3D12_BLEND_DEST_ALPHA;
            case 11: return D3D12_BLEND_INV_DEST_ALPHA;
            case 12: return D3D12_BLEND_BLEND_FACTOR;       // ConstantColor
            case 13: return D3D12_BLEND_INV_BLEND_FACTOR;   // OneMinusConstantColor
            case 14: return D3D12_BLEND_ALPHA_FACTOR;       // ConstantAlpha (the factor's alpha, as Vulkan's CONSTANT_ALPHA)
            case 15: return D3D12_BLEND_INV_ALPHA_FACTOR;   // OneMinusConstantAlpha
            case 16: return alpha ? D3D12_BLEND_ONE : D3D12_BLEND_SRC_ALPHA_SAT;
            default: return D3D12_BLEND_ONE;
            }
        }

        // Factors 2, 3 and 17+ and ops 5+ are not Xenos blend values this backend knows (RF-6).
        bool BlendFactorKnown(uint32_t factor) { return factor <= 1 || (factor >= 4 && factor <= 16); }

        D3D12_BLEND_OP BlendOp(uint32_t op)
        {
            switch (op)
            {
            case 1: return D3D12_BLEND_OP_SUBTRACT;
            case 2: return D3D12_BLEND_OP_MIN;
            case 3: return D3D12_BLEND_OP_MAX;
            case 4: return D3D12_BLEND_OP_REV_SUBTRACT;
            default: return D3D12_BLEND_OP_ADD;
            }
        }

        D3D12_COMPARISON_FUNC CompareFunc(uint32_t func)
        {
            return D3D12_COMPARISON_FUNC(D3D12_COMPARISON_FUNC_NEVER + (func & 7));
        }

        D3D12_STENCIL_OP StencilOp(uint32_t op)
        {
            constexpr D3D12_STENCIL_OP ops[8] = { D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_ZERO, D3D12_STENCIL_OP_REPLACE,
                                                  D3D12_STENCIL_OP_INCR_SAT, D3D12_STENCIL_OP_DECR_SAT, D3D12_STENCIL_OP_INVERT,
                                                  D3D12_STENCIL_OP_INCR, D3D12_STENCIL_OP_DECR };
            return ops[op & 7];
        }

        bool IsDepthFormat(DXGI_FORMAT format)
        {
            return format == DXGI_FORMAT_D24_UNORM_S8_UINT || format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        }

        enum FormatUse : uint32_t { USE_SAMPLE, USE_RENDER_TARGET, USE_DEPTH_STENCIL };

        // The typeless format of a resource whose views differ: depth/stencil, and gamma render targets (RGBA8
        // drawn through its sRGB view, sampled through the plain one).
        DXGI_FORMAT ResourceFormat(const TextureDesc& desc)
        {
            if (desc.format == DXGI_FORMAT_D24_UNORM_S8_UINT)
                return DXGI_FORMAT_R24G8_TYPELESS;
            if (desc.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
                return DXGI_FORMAT_R32G8X24_TYPELESS;
            if ((desc.usage & USAGE_RENDER_TARGET) &&
                (desc.format == DXGI_FORMAT_R8G8B8A8_UNORM || desc.format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB))
                return DXGI_FORMAT_R8G8B8A8_TYPELESS;
            return desc.format;
        }

        // HLSL -> DXIL ------------------------------------------------------------------------------

        // dxcompiler.dll, loaded at run time (as rhi_vulkan_shaders.cpp does for SPIR-V). DXC's objects are not
        // thread-safe: one compiler per thread (as rhi_vulkan_shaders.cpp), so the
        // warm-up thread never makes the recording thread wait. g_dxc is the loading thread's, used for the fingerprint.
        std::once_flag g_dxcOnce;
        DxcCreateInstanceProc g_dxcCreate = nullptr;
        IDxcCompiler3* g_dxc = nullptr;
        thread_local IDxcCompiler3* t_dxc = nullptr;

        void ComputeDxcFingerprint();

        void LoadDxc()
        {
            void* module = plat::OpenLibrary("dxcompiler.dll");
            if (!module)
            {
                Log("[gpu] Direct3D 12: dxcompiler.dll not found next to the exe");
                return;
            }
            auto create = reinterpret_cast<DxcCreateInstanceProc>(plat::FindSymbol(module, "DxcCreateInstance"));
            if (!create || FAILED(create(CLSID_DxcCompiler, __uuidof(IDxcCompiler3), reinterpret_cast<void**>(&g_dxc))))
            {
                Log("[gpu] Direct3D 12: dxcompiler.dll has no DxcCreateInstance / compiler");
                g_dxc = nullptr;
                return;
            }
            g_dxcCreate = create;
            t_dxc = g_dxc;
            ComputeDxcFingerprint();
        }

        // Whether DXC signed the container: "DXBC", then a 16-byte hash. dxil.dll (the validator) fills it in;
        // all zeros means nobody did (no dxil.dll next to the exe), and 0x01 x16 is the "bypass" hash of an
        // unvalidated shader, which only a recent Agility SDK on Windows 11 24H2+ accepts (ADR 0002).
        bool DxilSigned(const std::vector<uint8_t>& code)
        {
            if (code.size() < 20 || memcmp(code.data(), "DXBC", 4) != 0)
                return false;
            bool allZero = true, allBypass = true;
            for (size_t i = 4; i < 20; i++)
            {
                allZero = allZero && code[i] == 0x00;
                allBypass = allBypass && code[i] == 0x01;
            }
            return !allZero && !allBypass;
        }

        // Whether dxil.dll can be loaded from where dxcompiler.dll looks (the exe's folder first).
        bool DxilDllFound()
        {
            HMODULE module = LoadLibraryExW(L"dxil.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (module)
                FreeLibrary(module);
            return module != nullptr;
        }

        // DXC's arguments but the profile: the one list both the compiler call and the cache fingerprint use (change it, change both).
        const wchar_t* const DXC_ARGUMENTS[] = { L"-E", L"main", L"-HV", L"2018", L"-Wno-conversion", L"-Wno-ignored-attributes" };

        // `dump` is userdata\shadercache (empty: nothing is written); the failed HLSL goes to its sibling shaderdump\.
        // The disk cache is CompileDxil's, below; the caller has already loaded DXC.
        bool CompileDxilRaw(Stage stage, const std::string& hlsl, const char* name, const std::filesystem::path& dump, std::vector<uint8_t>& code,
                            uint64_t* compiles, uint64_t* failures)
        {
            const wchar_t* profile = stage == Stage::Pixel ? L"ps_6_0" : stage == Stage::Geometry ? L"gs_6_0" : L"vs_6_0";
            const wchar_t* arguments[2 + std::size(DXC_ARGUMENTS)] = { L"-T", profile };
            std::copy(std::begin(DXC_ARGUMENTS), std::end(DXC_ARGUMENTS), arguments + 2);
            DxcBuffer source{ hlsl.data(), hlsl.size(), DXC_CP_UTF8 };
            if (!t_dxc && FAILED(g_dxcCreate(CLSID_DxcCompiler, __uuidof(IDxcCompiler3), reinterpret_cast<void**>(&t_dxc))))
                t_dxc = nullptr;
            if (!t_dxc)
                return false;
            IDxcResult* result = nullptr;
            if (compiles)
                (*compiles)++;
            HRESULT status = t_dxc->Compile(&source, arguments, uint32_t(std::size(arguments)), nullptr, __uuidof(IDxcResult),
                                            reinterpret_cast<void**>(&result));
            if (SUCCEEDED(status) && result)
                result->GetStatus(&status);
            IDxcBlob* object = nullptr;
            if (SUCCEEDED(status) && result)
                result->GetOutput(DXC_OUT_OBJECT, __uuidof(IDxcBlob), reinterpret_cast<void**>(&object), nullptr);
            if (FAILED(status) || !object || object->GetBufferSize() == 0)
            {
                if (failures)
                    (*failures)++;
                std::string message = "?";
                IDxcBlobUtf8* errors = nullptr;
                if (result && SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, __uuidof(IDxcBlobUtf8), reinterpret_cast<void**>(&errors), nullptr)) && errors)
                {
                    message.assign(errors->GetStringPointer(), errors->GetStringLength());
                    errors->Release();
                }
                if (message.size() > 1500)
                    message.resize(1500);
                Log("[gpu] shader %s failed to compile (DXIL):\n%s", name, message.c_str());
                DumpFailedShader(dump, name, hlsl);
                if (object)
                    object->Release();
                if (result)
                    result->Release();
                return false;
            }
            const uint8_t* bytes = static_cast<const uint8_t*>(object->GetBufferPointer());
            code.assign(bytes, bytes + object->GetBufferSize());
            object->Release();
            result->Release();
            return true;
        }

        // DXIL disk cache (spec 002): userdata\shadercache_dxil\<hash>.dxil, a sibling of the SPIR-V one so that Direct3D 12 never
        // touches shadercache\. File = header + the DXIL container. The name is the hash of the stage and the final HLSL; the header
        // says which compiler and arguments made it, so an entry of another DXC is replaced, not trusted.
        constexpr uint32_t DXIL_CACHE_VERSION = 2;  // bump when the header changes (the arguments are in the fingerprint)
        struct DxilCacheHeader
        {
            char magic[4];
            uint32_t version;
            uint64_t fingerprint;  // DXC (version, flags, commit) + arguments
            uint32_t stage;
            uint32_t blobHash;     // HashBytes of the DXIL that follows, folded to 32 bits: catches altered bytes of the right size
            uint64_t hlslHash;
            uint64_t size;         // bytes of the DXIL that follow
        };
        static_assert(sizeof(DxilCacheHeader) == 40);

        uint32_t BlobHash(const uint8_t* data, size_t size)
        {
            uint64_t hash = HashBytes(data, size);
            return uint32_t(hash ^ (hash >> 32));
        }

        uint64_t g_dxcFingerprint = 0;  // set by LoadDxc (0 = unknown compiler: nothing is cached)

        void ComputeDxcFingerprint()
        {
            IDxcVersionInfo* version = nullptr;
            if (FAILED(g_dxc->QueryInterface(__uuidof(IDxcVersionInfo), reinterpret_cast<void**>(&version))) || !version)
                return;
            uint32_t major = 0, minor = 0, flags = 0, commits = 0;
            std::string commit;
            version->GetVersion(&major, &minor);
            version->GetFlags(&flags);
            IDxcVersionInfo2* version2 = nullptr;
            if (SUCCEEDED(version->QueryInterface(__uuidof(IDxcVersionInfo2), reinterpret_cast<void**>(&version2))) && version2)
            {
                char* hash = nullptr;
                if (SUCCEEDED(version2->GetCommitInfo(&commits, &hash)) && hash)
                {
                    commit = hash;
                    CoTaskMemFree(hash);
                }
                version2->Release();
            }
            version->Release();
            char text[256];
            snprintf(text, sizeof(text), "dxc %u.%u/%u/%u/%s", major, minor, flags, commits, commit.c_str());
            uint64_t fingerprint = HashBytes(text, strlen(text));
            for (const wchar_t* argument : DXC_ARGUMENTS)
                fingerprint = HashBytes(argument, wcslen(argument) * sizeof(wchar_t), fingerprint);
            g_dxcFingerprint = fingerprint;
        }

        bool ValidDxilCacheEntry(const std::vector<uint8_t>& file, Stage stage, uint64_t hlslHash, std::vector<uint8_t>& code)
        {
            DxilCacheHeader header;
            if (file.size() <= sizeof(header))
                return false;
            memcpy(&header, file.data(), sizeof(header));
            if (memcmp(header.magic, "CDXC", 4) != 0 || header.version != DXIL_CACHE_VERSION || header.fingerprint != g_dxcFingerprint ||
                header.stage != uint32_t(stage) || header.hlslHash != hlslHash || header.size != file.size() - sizeof(header))
                return false;
            code.assign(file.begin() + sizeof(header), file.end());
            if (header.blobHash != BlobHash(code.data(), code.size()))
                return false;
            // DXBC container: its size field (offset 24) must match, and dxil.dll must have signed it.
            if (code.size() < 28)
                return false;
            uint32_t containerSize = 0;
            memcpy(&containerSize, code.data() + 24, 4);
            return containerSize == code.size() && DxilSigned(code);
        }

        // The game's and the passes' DXIL, from the disk cache when a valid entry is there (RF-9), compiled and stored otherwise
        // (RF-8). A damaged or old entry is compiled again and overwritten (RF-10). `cache` empty: no disk cache.
        bool CompileDxil(Stage stage, const std::string& hlsl, const char* name, const std::filesystem::path& dump, const std::filesystem::path& cache,
                         std::vector<uint8_t>& code, uint64_t* compiles, uint64_t* failures, uint64_t* cached)
        {
            std::call_once(g_dxcOnce, LoadDxc);
            if (!g_dxc)
                return false;
            uint64_t hlslHash = HashBytes(hlsl.data(), hlsl.size());
            char keyText[32];
            snprintf(keyText, sizeof(keyText), "dxil-%u/%d", DXIL_CACHE_VERSION, int(stage));
            uint64_t key = HashBytes(hlsl.data(), hlsl.size(), HashBytes(keyText, strlen(keyText)));
            bool useCache = !cache.empty() && g_dxcFingerprint != 0;
            std::vector<uint8_t> file;
            if (useCache && ReadCachedShader(cache, key, "dxil", file) && ValidDxilCacheEntry(file, stage, hlslHash, code))
            {
                if (cached)
                    (*cached)++;
                return true;
            }
            if (!CompileDxilRaw(stage, hlsl, name, dump, code, compiles, failures))
                return false;
            // Only a signed container is worth keeping: the reader would throw an unsigned one away.
            if (useCache && DxilSigned(code))
            {
                DxilCacheHeader header{ { 'C', 'D', 'X', 'C' }, DXIL_CACHE_VERSION, g_dxcFingerprint, uint32_t(stage), BlobHash(code.data(), code.size()), hlslHash, code.size() };
                file.resize(sizeof(header) + code.size());
                memcpy(file.data(), &header, sizeof(header));
                memcpy(file.data() + sizeof(header), code.data(), code.size());
                WriteCachedShader(cache, key, "dxil", file.data(), file.size());  // a failed write only means no cache
            }
            return true;
        }

        // userdata\shadercache -> userdata\shadercache_dxil (empty stays empty).
        std::filesystem::path DxilCacheFolder(const std::filesystem::path& shaderCache)
        {
            return shaderCache.empty() ? shaderCache : shaderCache.parent_path() / "shadercache_dxil";
        }

        class D3D12Device final : public Device
        {
        public:
            ~D3D12Device() override
            {
                // A fallback to Vulkan reuses the window: the GPU must be idle and the swap chain released first.
                if (m_queue && m_fence && m_list && m_fenceEvent)
                    WaitIdle();
                if (m_fenceEvent)
                    CloseHandle(m_fenceEvent);
            }

            bool Init(const DeviceInfo& info);
            void FailShaders() { m_failShaders = true; }
            void RemoveOnPurpose() { m_removeOnPurpose = true; }

            const std::string& AdapterName() const override { return m_adapterName; }

            void Submit() override;
            void WaitIdle() override;
            Upload AllocateUpload(uint64_t size, uint64_t alignment) override;
            void HoldUploads(bool hold) override { hold ? m_uploadRing.Hold() : m_uploadRing.Release(); }

            // Textures of the game's draws. Buffers are the overlay's.
            Texture* CreateTexture(const TextureDesc& desc) override;
            void DestroyTexture(Texture* texture) override;
            Texture* NullTexture() override { return nullptr; }
            Buffer* CreateBuffer(uint64_t size) override;
            void DestroyBuffer(Buffer* buffer) override;

            void CopyUploadToTexture(Texture* texture, uint32_t level, uint64_t offset, uint32_t rowPitch) override;
            void CopyTexture(Texture* destination, Texture* source, const Rect* region) override;
            void ClearColor(Texture* texture, const float color[4], const Rect* rect) override;
            void ClearDepthStencil(Texture* texture, float depth, uint8_t stencil, const Rect* rect) override;

            Shader* CreateShader(Stage stage, const std::string& hlsl, const char* name, Layout layout) override;
            Pipeline* GetDrawPipeline(const DrawPipelineKey& key, bool deferrable, bool* deferred) override;
            void PrecompileProgress(uint32_t& done, uint32_t& total) override { done = total = 0; }
            void SetShaderCompilation(ShaderCompilation mode) override { NoteShaderCompilation(mode); }
            void PrewarmShader(Stage, const std::string&) override {}
            Pipeline* CreatePassPipeline(const PassPipelineDesc& desc) override;

            void Draw(const DrawCall& draw) override;
            void DrawPass(const PassDraw& pass) override;

            DXGI_FORMAT BackBufferFormat() const override { return DXGI_FORMAT_R8G8B8A8_UNORM; }
            void BackBufferSize(uint32_t& width, uint32_t& height) const override
            {
                width = m_backBufferWidth;
                height = m_backBufferHeight;
            }
            bool Resize(uint32_t width, uint32_t height) override;
            Texture* BeginPresent() override { return &m_backBuffers[m_swapChain->GetCurrentBackBufferIndex()]; }
            void Present(bool vsync) override;
            // video.window = fullscreen is the borderless window, as on Vulkan (spec 004 may add exclusive mode).
            void UpdateFullscreen(bool, bool) override {}
            void LeaveFullscreen() override {}
            bool ReadBack(Texture* texture, std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height) override;

            // Timestamps are for the lighting remaster; native access is for the upscalers (specs 004, issues #6/#7).
            void WriteTimestamp(uint32_t index) override;
            void ResolveTimestamps(uint32_t first, uint32_t count) override;
            bool ReadTimestamps(uint32_t first, uint32_t count, uint64_t* ticks) override;
            double TimestampFrequency() const override { return double(m_timestampFrequency); }

            NativeHandles Native() override { return {}; }
            bool DeviceExtensionEnabled(const char*) const override { return false; }
            void* NativeTexture(Texture*) override { return nullptr; }
            void* NativeView(Texture*) override { return nullptr; }
            void PrepareNative(Texture*, NativeState) override {}
            void InvalidateState() override { m_heapsBound = false; }

            Timing TakeTiming() override
            {
                Timing timing = m_timing;
                m_timing = {};
                return timing;
            }
            void LogStats() override;

        private:
            void CheckDeviceLost(const char* where);
            void WaitForFence(uint64_t value);
            void RetireCompleted();
            void CollectGarbage();
            uint32_t AllocateDescriptors(uint32_t count);
            D3D12_CPU_DESCRIPTOR_HANDLE GpuHeapCpu(uint32_t index) const;
            D3D12_GPU_DESCRIPTOR_HANDLE GpuHeapGpu(uint32_t index) const;
            void Transition(D3D12Texture* texture, D3D12_RESOURCE_STATES state);
            void BindDescriptorHeaps();
            void CreateBackBuffers();
            bool CreateDrawRootSignature();
            uint32_t ViewSrv(D3D12Texture* texture, View view);
            D3D12_CPU_DESCRIPTOR_HANDLE Rtv(D3D12Texture* texture);
            D3D12_CPU_DESCRIPTOR_HANDLE Dsv(D3D12Texture* texture);
            void ReleaseDescriptors(D3D12Texture* texture);
            void LogSwapChain();

            std::string m_adapterName;
            ComPtr<IDXGIAdapter1> m_adapter;     // the chosen adapter, kept for the diagnostics (spec 006, T3)
            DXGI_ADAPTER_DESC1 m_adapterDesc{};
            bool ChooseAdapter();
            std::filesystem::path m_shaderCache;      // userdata\shadercache: only the failed HLSL goes near it (shaderdump\)
            std::filesystem::path m_dxilCache;        // userdata\shadercache_dxil
            bool m_compilationNoted = false;
            void NoteShaderCompilation(ShaderCompilation mode);
            HWND m_hwnd = nullptr;
            ComPtr<IDXGIFactory4> m_factory;
            ComPtr<ID3D12Device> m_device;
            ComPtr<ID3D12CommandQueue> m_queue;
            ComPtr<IDXGISwapChain3> m_swapChain;
            D3D12Texture m_backBuffers[BACK_BUFFER_COUNT];
            uint32_t m_backBufferWidth = 1280, m_backBufferHeight = 720;
            uint32_t m_failedWidth = 0, m_failedHeight = 0;  // the size ResizeBuffers refused: not tried again until the size changes
            uint32_t m_safe = 0;                              // SafeMode bits (rhi.h)
            int m_presentMode = -1;                           // the last Present's mode (0 V-Sync, 1 tearing, 2 no tearing), to log a change once
            bool m_presentFailed = false;                     // a failed Present is logged once
            UINT m_swapFlags = 0;                             // DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING when Windows admits it (creation and ResizeBuffers must agree)
            ComPtr<ID3D12Fence> m_fence;
            uint64_t m_fenceValue = 0;
            HANDLE m_fenceEvent = nullptr;
            ComPtr<ID3D12GraphicsCommandList> m_list;
            ComPtr<ID3D12CommandAllocator> m_allocator;
            std::deque<std::pair<ComPtr<ID3D12CommandAllocator>, uint64_t>> m_allocatorPool;
            std::deque<uint64_t> m_frameFences;
            bool m_heapsBound = false;

            // Timestamps: resolved into a readback buffer, read by the caller frames later (past the frames in flight).
            ComPtr<ID3D12QueryHeap> m_queryHeap;
            ComPtr<ID3D12Resource> m_queryReadback;
            uint64_t m_timestampFrequency = 0;

            // Descriptors: RTVs, the DSVs, the CPU-only SRVs (the null one and every texture's view), the samplers and the shader-visible heap, where
            // every pass copies the views it samples (a ring over all of it but the last slot).
            DescriptorPool m_rtvs, m_dsvs, m_cpuSrvs;
            ComPtr<ID3D12DescriptorHeap> m_gpuSrvHeap, m_samplerHeap;
            uint32_t m_srvSize = 0, m_samplerSize = 0;
            uint32_t m_samplerCount = 0;
            std::unordered_map<uint64_t, uint32_t> m_samplerBlocks;  // hash of a draw's samplers -> first descriptor of its block of 32
            uint32_t m_nullTextureSrv = 0;
            D3D12_CPU_DESCRIPTOR_HANDLE m_nullRtv{};  // a draw's missing render target
            Ring m_descriptorRing;

            // Upload ring (one upload heap buffer, mapped for its whole life).
            ComPtr<ID3D12Resource> m_uploadBuffer;
            uint8_t* m_uploadCpu = nullptr;
            D3D12_GPU_VIRTUAL_ADDRESS m_uploadGpu = 0;
            Ring m_uploadRing;

            // The game's shaders and draw pipelines (a null state = that draw is skipped), kept for the device's life.
            std::vector<std::unique_ptr<D3D12Shader>> m_shaders;
            ComPtr<ID3D12RootSignature> m_drawRootSignature;
            std::unordered_map<uint64_t, std::unique_ptr<D3D12Pipeline>> m_drawPipelines;

            // RF-6 / RF-7: what is already in the log, so each reason and each shader is noted once.
            std::unordered_set<std::string> m_warned;
            std::unordered_map<uint32_t, bool> m_formatSupport;  // DXGI format * 4 + FormatUse -> usable
            bool WarnOnce(const std::string& reason);
            bool FormatSupported(DXGI_FORMAT format, FormatUse use);

            // The presenter's pipelines, kept for the device's life.
            std::vector<std::unique_ptr<D3D12Pipeline>> m_passPipelines;
            std::vector<uint8_t> m_fullscreenVs;  // DXIL of FULLSCREEN_VS
            bool m_failShaders = false;           // CRASH_D3D12_FAIL=shaders (spec 001): every pass pipeline fails, to see the fallback to Vulkan
            bool m_removeOnPurpose = false;       // CRASH_D3D12_FAIL=removed (spec 006): the device is removed at Present number REMOVE_AT_PRESENT
            uint32_t m_presentCount = 0;

            // Buffers given back by the renderer, released once the GPU is past the last submission that may use them.
            struct Garbage
            {
                uint64_t fence;
                std::unique_ptr<D3D12Buffer> buffer;
                ComPtr<ID3D12Resource> resource;  // a destroyed texture's
            };
            std::deque<Garbage> m_garbage;

            Timing m_timing;
            struct
            {
                uint64_t submits = 0, passPipelines = 0, drawPipelines = 0, dxilCompiles = 0, dxilFailures = 0;
                uint64_t shaderCompiles = 0, shaderCached = 0, shaderFailures = 0;  // the game's shaders (CreateShader), RF-11
            } m_stats;
        };

        // Submission ------------------------------------------------------------------------------

        // A lost device ends the game with the reason, as Vulkan's `device lost` does (spec 006, RF-1). GetDeviceRemovedReason is S_OK
        // while the device lives, so a call from a failure that was not a loss (a size that does not fit) does nothing.
        void D3D12Device::CheckDeviceLost(const char* where)
        {
            HRESULT reason = m_device->GetDeviceRemovedReason();
            if (reason == S_OK)
                return;
            const char* name = reason == DXGI_ERROR_DEVICE_HUNG ? "DXGI_ERROR_DEVICE_HUNG"
                             : reason == DXGI_ERROR_DEVICE_REMOVED ? "DXGI_ERROR_DEVICE_REMOVED"
                             : reason == DXGI_ERROR_DEVICE_RESET ? "DXGI_ERROR_DEVICE_RESET"
                             : reason == DXGI_ERROR_DRIVER_INTERNAL_ERROR ? "DXGI_ERROR_DRIVER_INTERNAL_ERROR"
                             : reason == DXGI_ERROR_INVALID_CALL ? "DXGI_ERROR_INVALID_CALL" : "unknown";
            Fatal("[gpu] Direct3D 12: device lost (the GPU stopped or the driver reset it) %s: %s (0x%08X)", where, name, unsigned(reason));
        }

        void D3D12Device::WaitForFence(uint64_t value)
        {
            if (m_fence->GetCompletedValue() < value)
            {
                m_fence->SetEventOnCompletion(value, m_fenceEvent);
                WaitForSingleObject(m_fenceEvent, INFINITE);
            }
            // A lost device leaves the fence at UINT64_MAX: every wait would look done (the rings' loops would spin on a GPU that is gone).
            if (m_fence->GetCompletedValue() == UINT64_MAX)
                CheckDeviceLost("waiting for the GPU");
        }

        void D3D12Device::RetireCompleted()
        {
            uint64_t completed = m_fence->GetCompletedValue();
            m_uploadRing.Retire(completed);
            m_descriptorRing.Retire(completed);
        }

        void D3D12Device::CollectGarbage()
        {
            uint64_t completed = m_fence->GetCompletedValue();
            while (!m_garbage.empty() && m_garbage.front().fence <= completed)
                m_garbage.pop_front();
        }

        void D3D12Device::Submit()
        {
            m_list->Close();
            ID3D12CommandList* lists[] = { m_list.Get() };
            m_queue->ExecuteCommandLists(1, lists);
            m_fenceValue++;
            m_queue->Signal(m_fence.Get(), m_fenceValue);
            m_uploadRing.OnSubmit(m_fenceValue);
            m_descriptorRing.OnSubmit(m_fenceValue);
            m_allocatorPool.push_back({ m_allocator, m_fenceValue });
            m_stats.submits++;

            // Reuse the oldest allocator the GPU has finished with; a new one if there is none.
            m_allocator.Reset();
            uint64_t completed = m_fence->GetCompletedValue();
            if (!m_allocatorPool.empty() && m_allocatorPool.front().second <= completed)
            {
                m_allocator = m_allocatorPool.front().first;
                m_allocatorPool.pop_front();
                m_allocator->Reset();
            }
            else
            {
                m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator));
            }
            m_list->Reset(m_allocator.Get(), nullptr);
            InvalidateState();
            CollectGarbage();
        }

        void D3D12Device::WaitIdle()
        {
            Submit();
            WaitForFence(m_fenceValue);
            RetireCompleted();
            CollectGarbage();
        }

        Device::Upload D3D12Device::AllocateUpload(uint64_t size, uint64_t alignment)
        {
            // Fence values are only queried when the ring is full.
            uint64_t offset = m_uploadRing.TryAllocate(size, alignment);
            while (offset == UINT64_MAX)
            {
                RetireCompleted();
                offset = m_uploadRing.TryAllocate(size, alignment);
                if (offset != UINT64_MAX)
                    break;
                if (size > UPLOAD_RING_SIZE)
                    return {};
                LARGE_INTEGER waitStart, waitEnd;
                QueryPerformanceCounter(&waitStart);
                if (m_uploadRing.HasPending())
                    WaitForFence(m_uploadRing.OldestPendingFence());
                else
                    WaitIdle();
                QueryPerformanceCounter(&waitEnd);
                m_timing.ringWaitTicks += uint64_t(waitEnd.QuadPart - waitStart.QuadPart);
            }
            return { m_uploadCpu + offset, offset };
        }

        // Shader-visible CBV/SRV/UAV descriptors for one pass.
        uint32_t D3D12Device::AllocateDescriptors(uint32_t count)
        {
            uint64_t offset = m_descriptorRing.TryAllocate(count, 1);
            while (offset == UINT64_MAX)
            {
                RetireCompleted();
                offset = m_descriptorRing.TryAllocate(count, 1);
                if (offset != UINT64_MAX)
                    break;
                LARGE_INTEGER waitStart, waitEnd;
                QueryPerformanceCounter(&waitStart);
                if (m_descriptorRing.HasPending())
                    WaitForFence(m_descriptorRing.OldestPendingFence());
                else
                    WaitIdle();
                QueryPerformanceCounter(&waitEnd);
                m_timing.ringWaitTicks += uint64_t(waitEnd.QuadPart - waitStart.QuadPart);
            }
            return uint32_t(offset);
        }

        D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::GpuHeapCpu(uint32_t index) const
        {
            D3D12_CPU_DESCRIPTOR_HANDLE handle = m_gpuSrvHeap->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += size_t(index) * m_srvSize;
            return handle;
        }

        D3D12_GPU_DESCRIPTOR_HANDLE D3D12Device::GpuHeapGpu(uint32_t index) const
        {
            D3D12_GPU_DESCRIPTOR_HANDLE handle = m_gpuSrvHeap->GetGPUDescriptorHandleForHeapStart();
            handle.ptr += uint64_t(index) * m_srvSize;
            return handle;
        }

        void D3D12Device::BindDescriptorHeaps()
        {
            if (m_heapsBound)
                return;
            ID3D12DescriptorHeap* heaps[] = { m_gpuSrvHeap.Get(), m_samplerHeap.Get() };
            m_list->SetDescriptorHeaps(2, heaps);
            m_heapsBound = true;
        }

        void D3D12Device::Transition(D3D12Texture* texture, D3D12_RESOURCE_STATES state)
        {
            if (texture->state == state)
                return;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = texture->resource.Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = texture->state;
            barrier.Transition.StateAfter = state;
            // Safe mode "barriers": a global UAV barrier (every earlier write) with each transition.
            D3D12_RESOURCE_BARRIER barriers[2] = { barrier, {} };
            barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            m_list->ResourceBarrier((m_safe & SAFE_BARRIERS) ? 2 : 1, barriers);
            texture->state = state;
        }

        // The whole of `source` into the top-left of `destination` (same format; the smaller of the two sizes), or `region` at the
        // same place in both.
        void D3D12Device::CopyTexture(Texture* destination, Texture* source, const Rect* region)
        {
            auto* to = static_cast<D3D12Texture*>(destination);
            auto* from = static_cast<D3D12Texture*>(source);
            Transition(from, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(to, D3D12_RESOURCE_STATE_COPY_DEST);
            if (!region && from->desc.width == to->desc.width && from->desc.height == to->desc.height)
            {
                m_list->CopyResource(to->resource.Get(), from->resource.Get());
                return;
            }
            D3D12_TEXTURE_COPY_LOCATION sourceLocation{}, destinationLocation{};
            sourceLocation.pResource = from->resource.Get();
            sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            destinationLocation.pResource = to->resource.Get();
            destinationLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_BOX box{ 0, 0, 0, std::min(from->desc.width, to->desc.width), std::min(from->desc.height, to->desc.height), 1 };
            if (region)
                box = D3D12_BOX{ UINT(region->left), UINT(region->top), 0, UINT(region->right), UINT(region->bottom), 1 };
            m_list->CopyTextureRegion(&destinationLocation, box.left, box.top, 0, &sourceLocation, &box);
        }

        void D3D12Device::ClearColor(Texture* texture, const float color[4], const Rect* rect)
        {
            auto* d3dTexture = static_cast<D3D12Texture*>(texture);
            Transition(d3dTexture, D3D12_RESOURCE_STATE_RENDER_TARGET);
            D3D12_RECT d3dRect = rect ? D3D12_RECT{ rect->left, rect->top, rect->right, rect->bottom } : D3D12_RECT{};
            m_list->ClearRenderTargetView(Rtv(d3dTexture), color, rect ? 1 : 0, rect ? &d3dRect : nullptr);
        }

        // Buffers and passes ----------------------------------------------------------------------

        Buffer* D3D12Device::CreateBuffer(uint64_t size)
        {
            auto buffer = std::make_unique<D3D12Buffer>();
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = size;
            desc.Height = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels = 1;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_UPLOAD };
            D3D12_RANGE noRead{ 0, 0 };
            if (FAILED(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&buffer->resource))) ||
                FAILED(buffer->resource->Map(0, &noRead, reinterpret_cast<void**>(&buffer->cpu))))
            {
                if (WarnOnce("upload buffer"))
                    Log("[gpu] could not create a %llu-byte upload buffer (device removed reason 0x%08X)", (unsigned long long)size,
                        unsigned(m_device->GetDeviceRemovedReason()));
                return nullptr;
            }
            buffer->size = size;
            return buffer.release();
        }

        void D3D12Device::DestroyBuffer(Buffer* buffer)
        {
            if (!buffer)
                return;
            Garbage garbage{ m_fenceValue + 1 };
            garbage.buffer.reset(static_cast<D3D12Buffer*>(buffer));
            m_garbage.push_back(std::move(garbage));
        }

        // Textures --------------------------------------------------------------------------------

        Texture* D3D12Device::CreateTexture(const TextureDesc& desc)
        {
            auto texture = std::make_unique<D3D12Texture>();
            texture->desc = desc;
            // RF-6: a format the adapter cannot sample or draw to gives no texture, and what uses it goes without.
            FormatUse use = (desc.usage & USAGE_DEPTH_STENCIL) ? USE_DEPTH_STENCIL : (desc.usage & USAGE_RENDER_TARGET) ? USE_RENDER_TARGET : USE_SAMPLE;
            if (!FormatSupported(use == USE_RENDER_TARGET ? desc.RenderFormat() : desc.format, use))
            {
                if (WarnOnce("rf6 texture " + std::to_string(uint32_t(desc.format))))
                    Log("[gpu] Direct3D 12: textures of format DXGI %u are not created (not supported here)", uint32_t(desc.format));
                return nullptr;
            }
            D3D12_RESOURCE_DESC resourceDesc{};
            resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            resourceDesc.Width = desc.width;
            resourceDesc.Height = desc.height;
            resourceDesc.DepthOrArraySize = 1;
            resourceDesc.MipLevels = UINT16(desc.levels);
            resourceDesc.Format = ResourceFormat(desc);
            resourceDesc.SampleDesc.Count = 1;
            D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_DEFAULT };
            D3D12_CLEAR_VALUE clear{};
            bool hasClear = false;
            if (desc.usage & USAGE_DEPTH_STENCIL)
            {
                resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
                clear.Format = desc.format;
                clear.DepthStencil.Depth = 1.0f;
                hasClear = true;
                texture->state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
            }
            else if (desc.usage & USAGE_RENDER_TARGET)
            {
                resourceDesc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
                clear.Format = desc.RenderFormat();
                hasClear = true;
                texture->state = D3D12_RESOURCE_STATE_RENDER_TARGET;
            }
            else
            {
                texture->state = D3D12_RESOURCE_STATE_COPY_DEST;
            }
            if (FAILED(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &resourceDesc, texture->state, hasClear ? &clear : nullptr,
                                                         IID_PPV_ARGS(&texture->resource))))
            {
                if (WarnOnce(std::string("texture ") + desc.name + " " + std::to_string(uint32_t(desc.format))))
                    Log("[gpu] could not create a %ux%u texture (%s, DXGI %u)", desc.width, desc.height, desc.name, uint32_t(desc.format));
                return nullptr;
            }
            if (desc.usage & USAGE_DEPTH_STENCIL)
            {
                m_list->ClearDepthStencilView(Dsv(texture.get()), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
            }
            else if (desc.usage & USAGE_RENDER_TARGET)
            {
                const float zero[4] = {};  // the contract of CreateTexture: render targets start cleared
                m_list->ClearRenderTargetView(Rtv(texture.get()), zero, 0, nullptr);
            }
            return texture.release();
        }

        void D3D12Device::ReleaseDescriptors(D3D12Texture* texture)
        {
            for (uint32_t& srv : texture->srv)
            {
                m_cpuSrvs.Release(srv);
                srv = NO_DESCRIPTOR;
            }
            m_rtvs.Release(texture->rtv);
            m_dsvs.Release(texture->dsv);
            texture->rtv = texture->dsv = NO_DESCRIPTOR;
        }

        void D3D12Device::DestroyTexture(Texture* texture)
        {
            if (!texture)
                return;
            auto* d3dTexture = static_cast<D3D12Texture*>(texture);
            // The CPU descriptors are copied when a draw is recorded, so they can be reused now; the resource waits for the GPU.
            ReleaseDescriptors(d3dTexture);
            Garbage garbage{ m_fenceValue + 1 };
            garbage.resource = std::move(d3dTexture->resource);
            delete d3dTexture;
            m_garbage.push_back(std::move(garbage));
        }

        uint32_t D3D12Device::ViewSrv(D3D12Texture* texture, View view)
        {
            if (!texture)
                return m_nullTextureSrv;
            uint32_t& index = texture->srv[uint32_t(view)];
            if (index != NO_DESCRIPTOR)
                return index;
            index = m_cpuSrvs.Allocate();
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = texture->desc.levels;
            bool isFloat = texture->desc.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
            switch (view)
            {
            case View::Depth:
                srv.Format = isFloat ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
                break;
            case View::Stencil:
                srv.Format = isFloat ? DXGI_FORMAT_X32_TYPELESS_G8X24_UINT : DXGI_FORMAT_X24_TYPELESS_G8_UINT;
                srv.Texture2D.PlaneSlice = 1;
                break;
            default:
                srv.Format = IsDepthFormat(texture->desc.format) ? (isFloat ? DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS : DXGI_FORMAT_R24_UNORM_X8_TYPELESS)
                                                                  : texture->desc.format;
                if (srv.Format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB && ResourceFormat(texture->desc) == DXGI_FORMAT_R8G8B8A8_TYPELESS)
                    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
                break;
            }
            m_device->CreateShaderResourceView(texture->resource.Get(), &srv, m_cpuSrvs.Cpu(index));
            return index;
        }

        D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::Rtv(D3D12Texture* texture)
        {
            if (texture->rtv == NO_DESCRIPTOR)
            {
                texture->rtv = m_rtvs.Allocate();
                D3D12_RENDER_TARGET_VIEW_DESC rtv{};
                rtv.Format = texture->desc.RenderFormat();
                rtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
                m_device->CreateRenderTargetView(texture->resource.Get(), &rtv, m_rtvs.Cpu(texture->rtv));
            }
            return m_rtvs.Cpu(texture->rtv);
        }

        D3D12_CPU_DESCRIPTOR_HANDLE D3D12Device::Dsv(D3D12Texture* texture)
        {
            if (texture->dsv == NO_DESCRIPTOR)
            {
                texture->dsv = m_dsvs.Allocate();
                D3D12_DEPTH_STENCIL_VIEW_DESC dsv{};
                dsv.Format = texture->desc.format;
                dsv.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
                m_device->CreateDepthStencilView(texture->resource.Get(), &dsv, m_dsvs.Cpu(texture->dsv));
            }
            return m_dsvs.Cpu(texture->dsv);
        }

        void D3D12Device::ClearDepthStencil(Texture* texture, float depth, uint8_t stencil, const Rect* rect)
        {
            auto* d3dTexture = static_cast<D3D12Texture*>(texture);
            Transition(d3dTexture, D3D12_RESOURCE_STATE_DEPTH_WRITE);
            D3D12_RECT d3dRect = rect ? D3D12_RECT{ rect->left, rect->top, rect->right, rect->bottom } : D3D12_RECT{};
            m_list->ClearDepthStencilView(Dsv(d3dTexture), D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, depth, stencil, rect ? 1 : 0,
                                          rect ? &d3dRect : nullptr);
        }

        void D3D12Device::CopyUploadToTexture(Texture* texture, uint32_t level, uint64_t offset, uint32_t rowPitch)
        {
            auto* d3dTexture = static_cast<D3D12Texture*>(texture);
            Transition(d3dTexture, D3D12_RESOURCE_STATE_COPY_DEST);
            D3D12_RESOURCE_DESC desc = d3dTexture->resource->GetDesc();
            D3D12_TEXTURE_COPY_LOCATION source{};
            source.pResource = m_uploadBuffer.Get();
            source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            // rhi.h's Footprint follows D3D12's rules: only the offset and the pitch of the renderer's layout are patched in.
            m_device->GetCopyableFootprints(&desc, level, 1, 0, &source.PlacedFootprint, nullptr, nullptr, nullptr);
            source.PlacedFootprint.Offset = offset;
            source.PlacedFootprint.Footprint.RowPitch = rowPitch;
            D3D12_TEXTURE_COPY_LOCATION destination{};
            destination.pResource = d3dTexture->resource.Get();
            destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            destination.SubresourceIndex = level;
            m_list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        }

        // Shaders, draw pipelines and draws ---------------------------------------------------------

        // The recording thread is the only caller (the warm-up only fills the disk cache), so m_shaders needs no lock.
        Shader* D3D12Device::CreateShader(Stage stage, const std::string& hlsl, const char* name, Layout)
        {
            auto shader = std::make_unique<D3D12Shader>();
            uint64_t compiles = m_stats.shaderCompiles;
            LARGE_INTEGER start, end;
            QueryPerformanceCounter(&start);
            bool compiled = CompileDxil(stage, hlsl, name, m_shaderCache, m_dxilCache, shader->code, &m_stats.shaderCompiles, &m_stats.shaderFailures,
                                        &m_stats.shaderCached);
            QueryPerformanceCounter(&end);
            if (m_stats.shaderCompiles != compiles)  // not from the disk cache: RF-8 counts it
            {
                m_timing.shaders++;
                m_timing.shaderTicks += uint64_t(end.QuadPart - start.QuadPart);
            }
            if (!compiled)
            {
                // RF-7 (the renderer asks once per shader: it remembers the failure). DXC's message is in the line before.
                Log("[gpu] Direct3D 12: shader %s does not compile to DXIL: what uses it is not drawn", name);
                return nullptr;
            }
            m_shaders.push_back(std::move(shader));
            return m_shaders.back().get();
        }

        // The binding model of rhi.h for the game's draws: b0-b2 as root CBVs, b3 as root constants, t0-t31 as a table, the
        // upload ring (vertex data) as t0 space1, s0-s31 as a table of the game's samplers.
        bool D3D12Device::CreateDrawRootSignature()
        {
            D3D12_DESCRIPTOR_RANGE textureRange{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 32, 0, 0, 0 };
            D3D12_DESCRIPTOR_RANGE vertexRange{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 1, 0 };
            D3D12_DESCRIPTOR_RANGE samplerRange{ D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 32, 0, 0, 0 };
            D3D12_ROOT_PARAMETER parameters[ROOT_PARAMETER_COUNT]{};
            parameters[ROOT_SYSTEM_CONSTANTS].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[ROOT_SYSTEM_CONSTANTS].Descriptor = { 0, 0 };
            parameters[ROOT_SYSTEM_CONSTANTS].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameters[ROOT_VS_CONSTANTS].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[ROOT_VS_CONSTANTS].Descriptor = { 1, 0 };
            parameters[ROOT_VS_CONSTANTS].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
            parameters[ROOT_PS_CONSTANTS].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[ROOT_PS_CONSTANTS].Descriptor = { 2, 0 };
            parameters[ROOT_PS_CONSTANTS].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
            parameters[ROOT_TEXTURES].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[ROOT_TEXTURES].DescriptorTable = { 1, &textureRange };
            parameters[ROOT_TEXTURES].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameters[ROOT_VERTEX_DATA].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[ROOT_VERTEX_DATA].DescriptorTable = { 1, &vertexRange };
            parameters[ROOT_VERTEX_DATA].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
            parameters[ROOT_EFFECTS].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameters[ROOT_EFFECTS].Constants = { 3, 0, graphics::EFFECT_CONSTANT_DWORDS };
            parameters[ROOT_EFFECTS].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameters[ROOT_SAMPLERS].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[ROOT_SAMPLERS].DescriptorTable = { 1, &samplerRange };
            parameters[ROOT_SAMPLERS].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_ROOT_SIGNATURE_DESC desc{ ROOT_PARAMETER_COUNT, parameters, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE };
            ComPtr<ID3DBlob> blob, errors;
            if (FAILED(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)))
            {
                Log("[gpu] Direct3D 12: root signature: %s", errors ? static_cast<const char*>(errors->GetBufferPointer()) : "?");
                return false;
            }
            return SUCCEEDED(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&m_drawRootSignature)));
        }

        // The part of a DrawPipelineKey that goes into a Direct3D 12 PSO, so keys that differ only in what the PSO ignores
        // (slots past colorCount, depth state without a depth target, the exact topology of one topology type, the shader
        // hashes next to the shader pointers) share one PSO. Bias and stencil masks are PSO state in Direct3D 12.
        uint64_t PsoHash(const DrawPipelineKey& key)
        {
            struct
            {
                Shader* vs;
                Shader* ps;
                Shader* gs;
                uint32_t colorFormats[4];
                uint32_t blend[4];
                uint32_t colorMask;
                uint32_t colorCount;
                uint32_t depthFormat;
                uint32_t depthControl;
                uint32_t stencilMasks;
                int32_t depthBias;
                float slopeBias;
                uint8_t cull, topologyType, depthClip, stripCut;
            } pso;
            memset(&pso, 0, sizeof(pso));  // padding included: the struct is hashed as bytes
            pso.vs = key.vs;
            pso.ps = key.ps;
            pso.gs = key.gs;
            pso.colorCount = std::min(key.colorCount, 4u);
            for (uint32_t i = 0; i < pso.colorCount; i++)
            {
                pso.colorFormats[i] = key.colorFormats[i];
                pso.blend[i] = key.blend[i];
            }
            pso.colorMask = key.colorMask & ((1ull << (4 * pso.colorCount)) - 1);
            pso.depthFormat = key.depthFormat;
            if (key.depthFormat)
            {
                pso.depthControl = key.depthControl;
                pso.stencilMasks = key.stencilMasks & 0xFFFF00;
            }
            pso.depthBias = key.depthBias;
            pso.slopeBias = key.slopeBias;
            pso.cull = uint8_t(key.cull);
            pso.topologyType = uint8_t(TopologyType(key.topology));
            pso.depthClip = key.depthClip;
            pso.stripCut = key.stripCut;
            return HashBytes(&pso, sizeof(pso));
        }

        // One line in the log per reason (RF-6) or per shader (RF-7), whatever the number of draws.
        bool D3D12Device::WarnOnce(const std::string& reason)
        {
            return m_warned.insert(reason).second;
        }

        // Whether a format can be sampled, drawn to or used as a depth target here: one query per format and use.
        bool D3D12Device::FormatSupported(DXGI_FORMAT format, FormatUse use)
        {
            uint32_t id = uint32_t(format) * 4 + use;
            auto found = m_formatSupport.find(id);
            if (found != m_formatSupport.end())
                return found->second;
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{ format };
            D3D12_FORMAT_SUPPORT1 needed = use == USE_DEPTH_STENCIL ? D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL
                                           : use == USE_RENDER_TARGET ? D3D12_FORMAT_SUPPORT1_RENDER_TARGET
                                                                      : D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE;
            bool ok = SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) &&
                      (support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) && (support.Support1 & needed);
            m_formatSupport[id] = ok;
            return ok;
        }

        // Blend, depth and stencil follow RB_BLENDCONTROL and RB_DEPTHCONTROL. What this backend does not translate (RF-6: a blend
        // factor or op that is not Xenos', a render or depth format the adapter cannot draw to) gives a null pipeline, so the draw
        // is skipped, with one log line per reason; a shader or a PSO that fails (RF-7) the same, one line per shader. The result
        // is cached by PsoHash like a PSO.
        Pipeline* D3D12Device::GetDrawPipeline(const DrawPipelineKey& key, bool, bool*)
        {
            uint64_t hash = PsoHash(key);
            auto cached = m_drawPipelines.find(hash);
            if (cached != m_drawPipelines.end())
                return cached->second->state ? cached->second.get() : nullptr;

            auto pipeline = std::make_unique<D3D12Pipeline>();
            std::string untranslated;  // the first reason, if any
            if (key.vs)
            {
                for (uint32_t i = 0; i < std::min(key.colorCount, 4u) && untranslated.empty(); i++)
                {
                    uint32_t blend = key.blend[i];
                    if (!BlendFactorKnown(blend & 0x1F) || !BlendFactorKnown((blend >> 8) & 0x1F) || !BlendFactorKnown((blend >> 16) & 0x1F) ||
                        !BlendFactorKnown((blend >> 24) & 0x1F))
                        untranslated = "a blend factor (RB_BLENDCONTROL " + std::to_string(blend) + ")";
                    else if (((blend >> 5) & 7) > 4 || ((blend >> 21) & 7) > 4)
                        untranslated = "a blend operation (RB_BLENDCONTROL " + std::to_string(blend) + ")";
                    else if (!FormatSupported(DXGI_FORMAT(key.colorFormats[i]), USE_RENDER_TARGET))
                        untranslated = "render target format DXGI " + std::to_string(key.colorFormats[i]);
                }
                if (untranslated.empty() && key.depthFormat && !FormatSupported(DXGI_FORMAT(key.depthFormat), USE_DEPTH_STENCIL))
                    untranslated = "depth format DXGI " + std::to_string(key.depthFormat);
            }
            if (!untranslated.empty())
            {
                if (WarnOnce("rf6 " + untranslated))
                    Log("[gpu] Direct3D 12: draws that need %s are not drawn (not translated)", untranslated.c_str());
            }
            else if (key.vs)
            {
                D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
                desc.pRootSignature = m_drawRootSignature.Get();
                auto bytecode = [](Shader* shader) {
                    const auto& code = static_cast<D3D12Shader*>(shader)->code;
                    return D3D12_SHADER_BYTECODE{ code.data(), code.size() };
                };
                desc.VS = bytecode(key.vs);
                if (key.ps)
                    desc.PS = bytecode(key.ps);
                if (key.gs)
                    desc.GS = bytecode(key.gs);
                desc.BlendState.IndependentBlendEnable = TRUE;
                for (uint32_t i = 0; i < 4; i++)
                {
                    // RB_BLENDCONTROL: color src/op/dst in the low half, alpha in the high half; ONE, ADD, ZERO is no blending.
                    auto& rt = desc.BlendState.RenderTarget[i];
                    uint32_t blend = key.blend[i];
                    uint32_t colorSrc = blend & 0x1F, colorOp = (blend >> 5) & 7, colorDst = (blend >> 8) & 0x1F;
                    uint32_t alphaSrc = (blend >> 16) & 0x1F, alphaOp = (blend >> 21) & 7, alphaDst = (blend >> 24) & 0x1F;
                    rt.BlendEnable = !(colorSrc == 1 && colorDst == 0 && colorOp == 0 && alphaSrc == 1 && alphaDst == 0 && alphaOp == 0);
                    rt.SrcBlend = BlendFactor(colorSrc, false);
                    rt.DestBlend = BlendFactor(colorDst, false);
                    rt.BlendOp = BlendOp(colorOp);
                    rt.SrcBlendAlpha = BlendFactor(alphaSrc, true);
                    rt.DestBlendAlpha = BlendFactor(alphaDst, true);
                    rt.BlendOpAlpha = BlendOp(alphaOp);
                    rt.RenderTargetWriteMask = UINT8((key.colorMask >> (4 * i)) & 0xF);
                }
                desc.SampleMask = UINT_MAX;
                desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
                desc.RasterizerState.CullMode = key.cull == Cull::Front ? D3D12_CULL_MODE_FRONT : key.cull == Cull::Back ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
                desc.RasterizerState.DepthBias = key.depthBias;
                desc.RasterizerState.SlopeScaledDepthBias = key.slopeBias;
                desc.RasterizerState.DepthClipEnable = key.depthClip;
                uint32_t depthControl = key.depthControl;
                if (key.depthFormat)
                {
                    desc.DepthStencilState.DepthEnable = (depthControl >> 1) & 1;
                    desc.DepthStencilState.DepthWriteMask = ((depthControl >> 2) & 1) ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
                    desc.DepthStencilState.DepthFunc = CompareFunc((depthControl >> 4) & 7);
                    desc.DepthStencilState.StencilEnable = depthControl & 1;
                    desc.DepthStencilState.StencilReadMask = UINT8((key.stencilMasks >> 8) & 0xFF);
                    desc.DepthStencilState.StencilWriteMask = UINT8((key.stencilMasks >> 16) & 0xFF);
                    desc.DepthStencilState.FrontFace.StencilFunc = CompareFunc((depthControl >> 8) & 7);
                    desc.DepthStencilState.FrontFace.StencilFailOp = StencilOp(depthControl >> 11);
                    desc.DepthStencilState.FrontFace.StencilPassOp = StencilOp(depthControl >> 14);
                    desc.DepthStencilState.FrontFace.StencilDepthFailOp = StencilOp(depthControl >> 17);
                    if ((depthControl >> 7) & 1)  // separate back-face stencil
                    {
                        desc.DepthStencilState.BackFace.StencilFunc = CompareFunc((depthControl >> 20) & 7);
                        desc.DepthStencilState.BackFace.StencilFailOp = StencilOp(depthControl >> 23);
                        desc.DepthStencilState.BackFace.StencilPassOp = StencilOp(depthControl >> 26);
                        desc.DepthStencilState.BackFace.StencilDepthFailOp = StencilOp(depthControl >> 29);
                    }
                    else
                    {
                        desc.DepthStencilState.BackFace = desc.DepthStencilState.FrontFace;
                    }
                    desc.DSVFormat = DXGI_FORMAT(key.depthFormat);
                }
                desc.IBStripCutValue = key.stripCut ? D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF : D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
                desc.PrimitiveTopologyType = TopologyType(key.topology);
                desc.NumRenderTargets = key.colorCount;
                for (uint32_t i = 0; i < key.colorCount; i++)
                    desc.RTVFormats[i] = DXGI_FORMAT(key.colorFormats[i]);
                desc.SampleDesc.Count = 1;
                LARGE_INTEGER start, end;
                QueryPerformanceCounter(&start);
                HRESULT result = m_device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline->state));
                QueryPerformanceCounter(&end);
                m_timing.pipelines++;  // RF-8: the recording thread compiles it
                m_timing.pipelineTicks += uint64_t(end.QuadPart - start.QuadPart);
                if (FAILED(result))
                {
                    // RF-7: once per shader, however many pipelines it is in.
                    bool newVs = WarnOnce("rf7 vs " + std::to_string(key.vsHash));
                    bool newPs = key.ps && WarnOnce("rf7 ps " + std::to_string(key.psHash));
                    if (newVs || newPs)
                        Log("[gpu] Direct3D 12: no pipeline for vs %016llx ps %016llx (0x%08X): what uses them is not drawn",
                            (unsigned long long)key.vsHash, (unsigned long long)key.psHash, unsigned(result));
                }
                else
                    m_stats.drawPipelines++;
            }
            Pipeline* out = pipeline->state ? pipeline.get() : nullptr;
            m_drawPipelines[hash] = std::move(pipeline);
            return out;
        }

        // shortcut: every piece of state is recorded on every draw (no change tracking), which is slower in the 3D scenes (~1,600
        // draws per frame); a later spec can add it back.
        void D3D12Device::Draw(const DrawCall& draw)
        {
            // Samplers first: a full sampler heap waits for the GPU (WaitIdle submits what is recorded), which must not cut a draw.
            // A block of 32 per distinct set of samplers; the slots past textureCount get the default description.
            constexpr uint32_t samplerCount = 32;
            uint64_t samplerKey = HashBytes(draw.samplers, sizeof(SamplerDesc) * std::min(draw.textureCount, 32u), samplerCount);
            uint32_t samplerBase;
            auto samplerIt = m_samplerBlocks.find(samplerKey);
            if (samplerIt != m_samplerBlocks.end())
            {
                samplerBase = samplerIt->second;
            }
            else
            {
                if (m_samplerCount + samplerCount > SAMPLER_DESCRIPTORS)
                {
                    WaitIdle();
                    m_samplerBlocks.clear();
                    m_samplerCount = 0;
                }
                samplerBase = m_samplerCount;
                m_samplerCount += samplerCount;
                for (uint32_t i = 0; i < samplerCount; i++)
                {
                    D3D12_CPU_DESCRIPTOR_HANDLE handle = m_samplerHeap->GetCPUDescriptorHandleForHeapStart();
                    handle.ptr += size_t(samplerBase + i) * m_samplerSize;
                    D3D12_SAMPLER_DESC desc = SamplerOf(i < draw.textureCount ? draw.samplers[i] : SamplerDesc{});
                    m_device->CreateSampler(&desc, handle);
                }
                m_samplerBlocks[samplerKey] = samplerBase;
            }

            // Textures: their state first (a barrier needs no descriptor), then one table of 32 with only the used slots written.
            uint32_t srvs[32];
            uint32_t textureCount = std::min(draw.textureCount, 32u);
            for (uint32_t i = 0; i < textureCount; i++)
            {
                auto* texture = static_cast<D3D12Texture*>(draw.textures[i].texture);
                if (texture)
                    Transition(texture, STATE_SHADER_RESOURCE);
                srvs[i] = ViewSrv(texture, draw.textures[i].view);
            }
            uint32_t descriptors = AllocateDescriptors(32);
            for (uint32_t i = 0; i < textureCount; i++)
                m_device->CopyDescriptorsSimple(1, GpuHeapCpu(descriptors + i), m_cpuSrvs.Cpu(srvs[i]), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            // Resource Binding Tier 1: every descriptor of the table must be valid, so the unused slots get the null SRV.
            for (uint32_t i = textureCount; i < 32; i++)
                m_device->CopyDescriptorsSimple(1, GpuHeapCpu(descriptors + i), m_cpuSrvs.Cpu(m_nullTextureSrv), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[4];
            for (uint32_t i = 0; i < draw.colorCount; i++)
            {
                if (auto* target = static_cast<D3D12Texture*>(draw.colorTargets[i]))
                {
                    Transition(target, D3D12_RESOURCE_STATE_RENDER_TARGET);
                    rtvs[i] = Rtv(target);
                }
                else
                {
                    rtvs[i] = m_nullRtv;
                }
            }

            auto* depthTarget = static_cast<D3D12Texture*>(draw.depthTarget);
            D3D12_CPU_DESCRIPTOR_HANDLE dsv{};
            if (depthTarget)
            {
                Transition(depthTarget, D3D12_RESOURCE_STATE_DEPTH_WRITE);
                dsv = Dsv(depthTarget);
            }

            BindDescriptorHeaps();
            m_list->SetGraphicsRootSignature(m_drawRootSignature.Get());
            D3D12_INDEX_BUFFER_VIEW indices{ m_uploadGpu, uint32_t(UPLOAD_RING_SIZE), DXGI_FORMAT_R32_UINT };
            m_list->IASetIndexBuffer(&indices);  // the whole upload ring; a draw starts at its own indices
            m_list->SetGraphicsRootConstantBufferView(ROOT_SYSTEM_CONSTANTS, m_uploadGpu + draw.systemConstants);
            m_list->SetGraphicsRootConstantBufferView(ROOT_VS_CONSTANTS, m_uploadGpu + draw.vertexConstants);
            m_list->SetGraphicsRootConstantBufferView(ROOT_PS_CONSTANTS, m_uploadGpu + draw.pixelConstants);
            m_list->SetGraphicsRootDescriptorTable(ROOT_TEXTURES, GpuHeapGpu(descriptors));
            m_list->SetGraphicsRootDescriptorTable(ROOT_VERTEX_DATA, GpuHeapGpu(VERTEX_DATA_DESCRIPTOR));
            D3D12_GPU_DESCRIPTOR_HANDLE samplerHandle = m_samplerHeap->GetGPUDescriptorHandleForHeapStart();
            samplerHandle.ptr += uint64_t(samplerBase) * m_samplerSize;
            m_list->SetGraphicsRootDescriptorTable(ROOT_SAMPLERS, samplerHandle);
            m_list->SetGraphicsRoot32BitConstants(ROOT_EFFECTS, graphics::EFFECT_CONSTANT_DWORDS, &draw.effects, 0);
            m_list->SetPipelineState(static_cast<D3D12Pipeline*>(draw.pipeline)->state.Get());
            m_list->OMSetRenderTargets(draw.colorCount, draw.colorCount ? rtvs : nullptr, FALSE, depthTarget ? &dsv : nullptr);
            m_list->OMSetStencilRef(draw.stencilRef);
            m_list->OMSetBlendFactor(draw.blendFactor);
            D3D12_VIEWPORT viewport{ draw.viewport.x, draw.viewport.y, draw.viewport.width, draw.viewport.height, draw.viewport.minDepth, draw.viewport.maxDepth };
            D3D12_RECT scissor{ draw.scissor.left, draw.scissor.top, draw.scissor.right, draw.scissor.bottom };
            m_list->RSSetViewports(1, &viewport);
            m_list->RSSetScissorRects(1, &scissor);
            m_list->IASetPrimitiveTopology(PrimitiveTopology(draw.topology));
            m_list->DrawIndexedInstanced(draw.indexCount, 1, draw.firstIndex, 0, 0);
        }

        Pipeline* D3D12Device::CreatePassPipeline(const PassPipelineDesc& desc)
        {
            if (m_failShaders)
            {
                static bool logged = false;
                if (!logged)
                {
                    logged = true;
                    Log("[gpu] Direct3D 12: %s: shader compilation failed on purpose", desc.name.c_str());
                }
                return nullptr;
            }
            auto pipeline = std::make_unique<D3D12Pipeline>();
            pipeline->textureCount = desc.textureCount;
            pipeline->bufferInput = desc.bufferInput;

            // Root signature: [0] CBV b0, [1] SRV table t0.. (or a root SRV for the buffer), static samplers.
            D3D12_DESCRIPTOR_RANGE range{ D3D12_DESCRIPTOR_RANGE_TYPE_SRV, std::max(desc.textureCount, 1u), 0, 0, 0 };
            D3D12_ROOT_PARAMETER parameters[2]{};
            parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
            parameters[0].Descriptor = { 0, 0 };
            parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            if (desc.bufferInput)
            {
                parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
                parameters[1].Descriptor = { 0, 0 };
            }
            else
            {
                parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
                parameters[1].DescriptorTable = { 1, &range };
            }
            parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            D3D12_STATIC_SAMPLER_DESC samplers[2]{};
            for (uint32_t i = 0; i < desc.samplerCount; i++)
            {
                D3D12_SAMPLER_DESC sampler = SamplerOf(desc.samplers[i]);
                samplers[i].Filter = sampler.Filter;
                samplers[i].AddressU = sampler.AddressU;
                samplers[i].AddressV = sampler.AddressV;
                samplers[i].AddressW = sampler.AddressW;
                samplers[i].MaxAnisotropy = sampler.MaxAnisotropy;
                samplers[i].MinLOD = sampler.MinLOD;
                samplers[i].MaxLOD = sampler.MaxLOD;
                samplers[i].ShaderRegister = i;
                samplers[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            }
            D3D12_ROOT_SIGNATURE_DESC rootDesc{ 2, parameters, desc.samplerCount, samplers, D3D12_ROOT_SIGNATURE_FLAG_NONE };
            ComPtr<ID3DBlob> blob, errors;
            if (FAILED(D3D12SerializeRootSignature(&rootDesc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors)) ||
                FAILED(m_device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&pipeline->rootSignature))))
            {
                Log("[gpu] %s: could not create the root signature", desc.name.c_str());
                return nullptr;
            }

            std::vector<uint8_t> vsCode, psCode;
            LARGE_INTEGER start, end;
            QueryPerformanceCounter(&start);
            uint64_t dxilBefore = m_stats.dxilCompiles;
            if (desc.vs.empty())
                vsCode = m_fullscreenVs;
            else if (!CompileDxil(Stage::Vertex, desc.vs, (desc.name + "_vs").c_str(), m_shaderCache, m_dxilCache, vsCode, &m_stats.dxilCompiles,
                                &m_stats.dxilFailures, nullptr))
                return nullptr;
            if (!CompileDxil(Stage::Pixel, desc.ps, (desc.name + "_ps").c_str(), m_shaderCache, m_dxilCache, psCode, &m_stats.dxilCompiles,
                           &m_stats.dxilFailures, nullptr))
                return nullptr;
            D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
            pso.pRootSignature = pipeline->rootSignature.Get();
            pso.VS = { vsCode.data(), vsCode.size() };
            pso.PS = { psCode.data(), psCode.size() };
            for (uint32_t i = 0; i < desc.targetCount; i++)
            {
                D3D12_RENDER_TARGET_BLEND_DESC& blend = pso.BlendState.RenderTarget[i];
                blend.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
                if (desc.blend == Blend::Alpha)
                {
                    blend.BlendEnable = TRUE;
                    blend.SrcBlend = D3D12_BLEND_SRC_ALPHA;
                    blend.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
                    blend.BlendOp = D3D12_BLEND_OP_ADD;
                    blend.SrcBlendAlpha = D3D12_BLEND_ZERO;
                    blend.DestBlendAlpha = D3D12_BLEND_ONE;
                    blend.BlendOpAlpha = D3D12_BLEND_OP_ADD;
                }
                pso.RTVFormats[i] = desc.targets[i];
            }
            pso.SampleMask = UINT_MAX;
            pso.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
            pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
            pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            pso.NumRenderTargets = desc.targetCount;
            pso.SampleDesc.Count = 1;
            HRESULT created = m_device->CreateGraphicsPipelineState(&pso, IID_PPV_ARGS(&pipeline->state));
            if (FAILED(created))
            {
                Log("[gpu] %s: could not create the pipeline (0x%08X)", desc.name.c_str(), unsigned(created));
                return nullptr;
            }
            QueryPerformanceCounter(&end);
            m_timing.pipelines++;  // RF-8 (the DXIL it compiled counts as shaders)
            m_timing.pipelineTicks += uint64_t(end.QuadPart - start.QuadPart);
            m_timing.shaders += uint32_t(m_stats.dxilCompiles - dxilBefore);
            m_stats.passPipelines++;
            m_passPipelines.push_back(std::move(pipeline));
            return m_passPipelines.back().get();
        }

        void D3D12Device::DrawPass(const PassDraw& pass)
        {
            auto* pipeline = static_cast<D3D12Pipeline*>(pass.pipeline);
            if (!pipeline)
                return;
            // The textures it samples (a slot without one samples the null texture, transparent black).
            uint32_t descriptors = 0;
            if (!pipeline->bufferInput)
            {
                uint32_t count = std::max(pipeline->textureCount, 1u);
                uint32_t srvs[8];
                for (uint32_t i = 0; i < count && i < 8; i++)
                {
                    auto* texture = i < pass.textureCount ? static_cast<D3D12Texture*>(pass.textures[i].texture) : nullptr;
                    if (texture)
                        Transition(texture, STATE_SHADER_RESOURCE);
                    srvs[i] = ViewSrv(texture, i < pass.textureCount ? pass.textures[i].view : View::Default);
                }
                descriptors = AllocateDescriptors(count);
                for (uint32_t i = 0; i < count; i++)
                    m_device->CopyDescriptorsSimple(1, GpuHeapCpu(descriptors + i), m_cpuSrvs.Cpu(srvs[i]), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            }
            D3D12_GPU_VIRTUAL_ADDRESS constants = m_uploadGpu;
            if (pass.constantBytes)
            {
                Upload upload = AllocateUpload(AlignUp64(pass.constantBytes, 256), 256);
                if (!upload.cpu)
                    return;
                memcpy(upload.cpu, pass.constants, pass.constantBytes);
                constants = m_uploadGpu + upload.offset;
            }
            D3D12_CPU_DESCRIPTOR_HANDLE rtvs[4];
            for (uint32_t i = 0; i < pass.targetCount; i++)
            {
                auto* target = static_cast<D3D12Texture*>(pass.targets[i]);
                Transition(target, D3D12_RESOURCE_STATE_RENDER_TARGET);
                rtvs[i] = Rtv(target);
            }
            BindDescriptorHeaps();
            m_list->SetGraphicsRootSignature(pipeline->rootSignature.Get());
            m_list->SetGraphicsRootConstantBufferView(0, constants);
            if (pipeline->bufferInput)
                m_list->SetGraphicsRootShaderResourceView(1, static_cast<D3D12Buffer*>(pass.buffer)->resource->GetGPUVirtualAddress() + pass.bufferOffset);
            else
                m_list->SetGraphicsRootDescriptorTable(1, GpuHeapGpu(descriptors));
            m_list->SetPipelineState(pipeline->state.Get());
            m_list->OMSetRenderTargets(pass.targetCount, rtvs, FALSE, nullptr);
            D3D12_VIEWPORT viewport{ pass.viewport.x, pass.viewport.y, pass.viewport.width, pass.viewport.height, 0.0f, 1.0f };
            D3D12_RECT scissor{ pass.scissor.left, pass.scissor.top, pass.scissor.right, pass.scissor.bottom };
            m_list->RSSetViewports(1, &viewport);
            m_list->RSSetScissorRects(1, &scissor);
            m_list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            m_list->DrawInstanced(pass.vertexCount, 1, 0, 0);
        }

        // Presentation ----------------------------------------------------------------------------

        void D3D12Device::LogSwapChain()
        {
            Log("[gpu] Direct3D 12: swap chain %ux%u, %u buffers, flip %s, %s", m_backBufferWidth, m_backBufferHeight, BACK_BUFFER_COUNT,
                (m_safe & SAFE_PRESENT) ? "sequential" : "discard", m_swapFlags ? "tearing allowed" : "no tearing");
        }

        void D3D12Device::CreateBackBuffers()
        {
            for (uint32_t i = 0; i < BACK_BUFFER_COUNT; i++)
            {
                D3D12Texture& buffer = m_backBuffers[i];
                m_swapChain->GetBuffer(i, IID_PPV_ARGS(&buffer.resource));
                buffer.desc.width = m_backBufferWidth;
                buffer.desc.height = m_backBufferHeight;
                buffer.desc.format = DXGI_FORMAT_R8G8B8A8_UNORM;
                buffer.desc.usage = USAGE_RENDER_TARGET;
                buffer.desc.name = "back buffer";
                buffer.state = D3D12_RESOURCE_STATE_PRESENT;
                if (buffer.rtv == NO_DESCRIPTOR)
                    buffer.rtv = m_rtvs.Allocate();
                m_device->CreateRenderTargetView(buffer.resource.Get(), nullptr, m_rtvs.Cpu(buffer.rtv));
            }
        }

        bool D3D12Device::Resize(uint32_t width, uint32_t height)
        {
            if (!width || !height || (width == m_backBufferWidth && height == m_backBufferHeight) ||
                (width == m_failedWidth && height == m_failedHeight))
                return false;
            WaitIdle();
            // The swap chain can only be resized once every reference to its buffers is gone.
            for (auto& buffer : m_backBuffers)
                buffer.resource.Reset();
            HRESULT result = m_swapChain->ResizeBuffers(BACK_BUFFER_COUNT, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, m_swapFlags);
            if (SUCCEEDED(result))
            {
                m_backBufferWidth = width;
                m_backBufferHeight = height;
                m_failedWidth = m_failedHeight = 0;
                LogSwapChain();
            }
            else
            {
                // Keep the old size (the buffers are taken again below) and do not retry this size every frame.
                CheckDeviceLost("resizing the swap chain");
                m_failedWidth = width;
                m_failedHeight = height;
                Log("[gpu] Direct3D 12: could not resize the swap chain to %ux%u (0x%08X): keeping %ux%u", width, height, unsigned(result),
                    m_backBufferWidth, m_backBufferHeight);
            }
            CreateBackBuffers();
            return SUCCEEDED(result);
        }

        void D3D12Device::Present(bool vsync)
        {
            if (m_removeOnPurpose && ++m_presentCount == REMOVE_AT_PRESENT)
            {
                ComPtr<ID3D12Device5> device5;
                if (SUCCEEDED(m_device.As(&device5)))
                {
                    Log("[gpu] CRASH_D3D12_FAIL=removed: Direct3D 12 removes the device on purpose (Present %u)", REMOVE_AT_PRESENT);
                    device5->RemoveDevice();
                }
                else
                    Log("[gpu] CRASH_D3D12_FAIL=removed: this Windows has no ID3D12Device5, the device stays");
            }
            D3D12Texture* backBuffer = &m_backBuffers[m_swapChain->GetCurrentBackBufferIndex()];
            Transition(backBuffer, D3D12_RESOURCE_STATE_PRESENT);
            LARGE_INTEGER submitStart, submitEnd, presentEnd;
            QueryPerformanceCounter(&submitStart);
            Submit();
            QueryPerformanceCounter(&submitEnd);
            // V-Sync: wait for the refresh. Without it Present(0), with tearing when the swap chain was made for it (RF-4); the
            // choice is per call, so a V-Sync change in SETTINGS counts on the next frame.
            bool tearing = !vsync && (m_swapFlags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING);
            int mode = vsync ? 0 : tearing ? 1 : 2;
            if (mode != m_presentMode)
            {
                m_presentMode = mode;
                Log("[gpu] Direct3D 12: presenting %s", mode == 0 ? "with V-Sync" : mode == 1 ? "without V-Sync, with tearing" : "without V-Sync, no tearing");
            }
            HRESULT presented = m_swapChain->Present(vsync ? 1 : 0, tearing ? DXGI_PRESENT_ALLOW_TEARING : 0);
            if (FAILED(presented) && !m_presentFailed)
            {
                CheckDeviceLost("presenting");
                m_presentFailed = true;
                Log("[gpu] Direct3D 12: Present failed (0x%08X)", unsigned(presented));
            }
            QueryPerformanceCounter(&presentEnd);
            m_timing.submitTicks += uint64_t(submitEnd.QuadPart - submitStart.QuadPart);
            m_timing.presentTicks = std::max(m_timing.presentTicks, uint64_t(presentEnd.QuadPart - submitEnd.QuadPart));

            // At most two frames in flight (RF-5; by code review: the loop below waits for the oldest fence as soon as a third is
            // queued). The safe mode's "serial" waits for this one too.
            m_frameFences.push_back(m_fenceValue);
            LARGE_INTEGER waitStart, waitEnd;
            QueryPerformanceCounter(&waitStart);
            while (m_frameFences.size() > ((m_safe & SAFE_SERIAL) ? 0u : 2u))
            {
                WaitForFence(m_frameFences.front());
                m_frameFences.pop_front();
            }
            QueryPerformanceCounter(&waitEnd);
            m_timing.frameWaitTicks += uint64_t(waitEnd.QuadPart - waitStart.QuadPart);
            RetireCompleted();
        }

        // Timestamps ------------------------------------------------------------------------------

        void D3D12Device::WriteTimestamp(uint32_t index)
        {
            if (m_queryHeap && index < TIMESTAMPS)
                m_list->EndQuery(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, index);
        }

        void D3D12Device::ResolveTimestamps(uint32_t first, uint32_t count)
        {
            if (m_queryHeap && first + count <= TIMESTAMPS)
                m_list->ResolveQueryData(m_queryHeap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, first, count, m_queryReadback.Get(), uint64_t(first) * 8);
        }

        bool D3D12Device::ReadTimestamps(uint32_t first, uint32_t count, uint64_t* ticks)
        {
            if (!m_queryHeap || first + count > TIMESTAMPS)
                return false;
            uint64_t* values = nullptr;
            D3D12_RANGE range{ size_t(first) * 8, size_t(first + count) * 8 };
            if (FAILED(m_queryReadback->Map(0, &range, reinterpret_cast<void**>(&values))))
                return false;
            memcpy(ticks, values + first, size_t(count) * 8);
            D3D12_RANGE none{ 0, 0 };
            m_queryReadback->Unmap(0, &none);
            return true;
        }

        bool D3D12Device::ReadBack(Texture* texture, std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height)
        {
            auto* source = static_cast<D3D12Texture*>(texture);
            D3D12_RESOURCE_DESC desc = source->resource->GetDesc();
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint;
            uint64_t totalBytes;
            m_device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &totalBytes);
            D3D12_RESOURCE_DESC bufferDesc{};
            bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bufferDesc.Width = totalBytes;
            bufferDesc.Height = 1;
            bufferDesc.DepthOrArraySize = 1;
            bufferDesc.MipLevels = 1;
            bufferDesc.SampleDesc.Count = 1;
            bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            D3D12_HEAP_PROPERTIES heap{ D3D12_HEAP_TYPE_READBACK };
            ComPtr<ID3D12Resource> readback;
            if (FAILED(m_device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback))))
            {
                CheckDeviceLost("reading a capture");  // a lost device fails here too: do not lose the capture in silence
                return false;
            }
            D3D12_RESOURCE_STATES before = source->state;
            Transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE);
            D3D12_TEXTURE_COPY_LOCATION destination{};
            destination.pResource = readback.Get();
            destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            destination.PlacedFootprint = footprint;
            D3D12_TEXTURE_COPY_LOCATION sourceLocation{};
            sourceLocation.pResource = source->resource.Get();
            sourceLocation.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            m_list->CopyTextureRegion(&destination, 0, 0, 0, &sourceLocation, nullptr);
            Transition(source, before);
            WaitIdle();
            CheckDeviceLost("reading a capture");  // before copying to the image: no half a PNG
            void* data;
            D3D12_RANGE range{ 0, SIZE_T(totalBytes) };
            if (FAILED(readback->Map(0, &range, &data)))
                return false;
            width = footprint.Footprint.Width;
            height = footprint.Footprint.Height;
            rgba.resize(size_t(width) * height * 4);
            for (uint32_t y = 0; y < height; y++)
                memcpy(rgba.data() + size_t(y) * width * 4, static_cast<uint8_t*>(data) + footprint.Offset + size_t(y) * footprint.Footprint.RowPitch,
                       size_t(width) * 4);
            readback->Unmap(0, nullptr);
            if (desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM)
            {
                for (size_t i = 0; i < rgba.size(); i += 4)
                    std::swap(rgba[i], rgba[i + 2]);
            }
            return true;
        }

        void D3D12Device::LogStats()
        {
            Log("[gpu] d3d12: %llu submits, %llu pass pipelines, %llu draw pipelines, %llu DXIL compiles (%llu failed)", (unsigned long long)m_stats.submits,
                (unsigned long long)m_stats.passPipelines, (unsigned long long)m_stats.drawPipelines, (unsigned long long)m_stats.dxilCompiles,
                (unsigned long long)m_stats.dxilFailures);
            Log("[gpu] Direct3D 12: game shaders: %llu compiled, %llu from cache, %llu failed", (unsigned long long)m_stats.shaderCompiles,
                (unsigned long long)m_stats.shaderCached, (unsigned long long)m_stats.shaderFailures);
        }

        // RF-12: Direct3D 12 compiles shaders on the spot whatever graphics.shader_compilation says (it never changes the setting).
        void D3D12Device::NoteShaderCompilation(ShaderCompilation mode)
        {
            if (mode == ShaderCompilation::Sync || m_compilationNoted)
                return;
            m_compilationNoted = true;
            Log("[gpu] Direct3D 12: graphics.shader_compilation=%s asks for asynchronous compilation; Direct3D 12 compiles synchronously",
                mode == ShaderCompilation::Aggressive ? "aggressive" : "async");
        }

        // Startup ---------------------------------------------------------------------------------

        // Spec 006 RF-2: Vulkan's rule (ChoosePhysicalDevice). No software adapters; shared memory (UMA) counts as integrated (2), else
        // discrete (3); CRASH_D3D12_DEVICE=<part of the name> adds 10 (case-sensitive, like CRASH_VULKAN_DEVICE); the highest wins, the first on a tie.
        bool D3D12Device::ChooseAdapter()
        {
            const char* wanted = getenv("CRASH_D3D12_DEVICE");
            if (wanted && !*wanted)
                wanted = nullptr;
            std::vector<ComPtr<IDXGIAdapter1>> adapters;
            for (UINT i = 0; ; i++)
            {
                ComPtr<IDXGIAdapter1> adapter;
                if (m_factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND)
                    break;
                if (adapter)
                    adapters.push_back(adapter);
            }
            int bestScore = -1;
            bool matched = false;
            size_t candidates = 0;
            std::vector<std::string> lines;
            for (const ComPtr<IDXGIAdapter1>& adapter : adapters)
            {
                DXGI_ADAPTER_DESC1 desc{};
                adapter->GetDesc1(&desc);
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
                    continue;
                std::string name = plat::NarrowUtf8(desc.Description);
                ComPtr<ID3D12Device> device;
                if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))))
                {
                    Log("[gpu] Direct3D 12: %s: no feature level 11_0", name.c_str());
                    continue;
                }
                D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
                device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture));
                int score = architecture.UMA ? 2 : 3;
                candidates++;
                lines.push_back(name + (score == 3 ? " (discrete)" : " (integrated)"));
                if (wanted && name.find(wanted) != std::string::npos)
                {
                    score += 10;
                    matched = true;
                }
                if (score > bestScore)
                {
                    bestScore = score;
                    m_adapter = adapter;
                    m_adapterDesc = desc;
                    m_device = device;
                }
            }
            if (lines.size() > 1)  // like Vulkan: candidates are listed only when there is a choice
                for (const std::string& line : lines)
                    Log("[gpu] Direct3D 12: GPU candidate: %s", line.c_str());
            if (wanted && !matched && candidates)
                Log("[gpu] Direct3D 12: CRASH_D3D12_DEVICE=%s matches no adapter; the default rule applies", wanted);
            return m_device != nullptr;
        }

        // Every failure says why and returns false: the renderer then starts Vulkan in the same window.
        bool D3D12Device::Init(const DeviceInfo& info)
        {
            m_hwnd = HWND(info.window);
            m_safe = info.safeMode;
            // RF-6: what D3D12 has no equivalent for is logged, never an error.
            if (m_safe)
            {
                const uint32_t applies = SAFE_BARRIERS | SAFE_SERIAL | SAFE_PRESENT;
                Log("[gpu] Direct3D 12: GPU safe mode: applied %s; no effect here (no equivalent) %s", SafeModeNames(m_safe & applies).c_str(),
                    SafeModeNames(m_safe & ~applies).c_str());
            }
            m_shaderCache = info.shaderCache;
            // CRASH_D3D12_FAIL=shaders falls back to Vulkan: no DXIL cache is created or touched.
            m_dxilCache = m_failShaders ? std::filesystem::path() : DxilCacheFolder(info.shaderCache);
            NoteShaderCompilation(info.shaderCompilation);
            // No debug layer (spec 004): info.debug is a Vulkan switch for now.
            if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory))) || !ChooseAdapter())
            {
                Log("[gpu] Direct3D 12: no hardware adapter with feature level 11_0");
                return false;
            }
            // The shaders are DXIL (Shader Model 6).
            D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{ D3D_SHADER_MODEL_6_0 };
            if (FAILED(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel))) ||
                shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_0)
            {
                Log("[gpu] Direct3D 12: the GPU has no Shader Model 6.0");
                return false;
            }
            // Only now: a fallback to Vulkan must not leave an empty shadercache_dxil behind.
            if (!m_dxilCache.empty())
            {
                std::error_code error;
                std::filesystem::create_directories(m_dxilCache, error);  // without it, shaders compile every start
            }
            m_adapterName = plat::NarrowUtf8(m_adapterDesc.Description);  // whole, in UTF-8 (a fixed buffer cut non-ASCII names)

            D3D12_COMMAND_QUEUE_DESC queueDesc{ D3D12_COMMAND_LIST_TYPE_DIRECT };
            if (FAILED(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_queue))))
            {
                Log("[gpu] Direct3D 12: could not create the command queue");
                return false;
            }
            m_fenceEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!m_fenceEvent || FAILED(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence))))
            {
                Log("[gpu] Direct3D 12: could not create the fence");
                return false;
            }
            if (FAILED(m_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_allocator))) ||
                FAILED(m_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, m_allocator.Get(), nullptr, IID_PPV_ARGS(&m_list))))
            {
                Log("[gpu] Direct3D 12: could not create the command list");
                return false;
            }

            // Descriptor heaps: render target views (the back buffers'), depth-stencil views, CPU-only SRVs, the shader-visible SRV heap and
            // the shader-visible sampler heap.
            auto makeHeap = [this](D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t count, bool shaderVisible, ComPtr<ID3D12DescriptorHeap>& heap) {
                D3D12_DESCRIPTOR_HEAP_DESC desc{ type, count, shaderVisible ? D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE : D3D12_DESCRIPTOR_HEAP_FLAG_NONE };
                return SUCCEEDED(m_device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));
            };
            if (!makeHeap(D3D12_DESCRIPTOR_HEAP_TYPE_RTV, RTV_DESCRIPTORS, false, m_rtvs.heap) ||
                !makeHeap(D3D12_DESCRIPTOR_HEAP_TYPE_DSV, DSV_DESCRIPTORS, false, m_dsvs.heap) ||
                !makeHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, CPU_SRV_DESCRIPTORS, false, m_cpuSrvs.heap) ||
                !makeHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, SHADER_VISIBLE_DESCRIPTORS, true, m_gpuSrvHeap) ||
                !makeHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, SAMPLER_DESCRIPTORS, true, m_samplerHeap))
            {
                Log("[gpu] Direct3D 12: could not create the descriptor heap");
                return false;
            }
            m_srvSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            m_samplerSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
            m_rtvs.size = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            m_rtvs.capacity = RTV_DESCRIPTORS;
            m_rtvs.name = "RTV";
            m_dsvs.size = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
            m_dsvs.capacity = DSV_DESCRIPTORS;
            m_dsvs.name = "DSV";
            m_cpuSrvs.size = m_srvSize;
            m_cpuSrvs.capacity = CPU_SRV_DESCRIPTORS;
            m_cpuSrvs.name = "CPU SRV";
            m_descriptorRing.Init(VERTEX_DATA_DESCRIPTOR);

            // The null texture (a pass without an image samples transparent black).
            m_nullTextureSrv = m_cpuSrvs.Allocate();
            D3D12_SHADER_RESOURCE_VIEW_DESC nullTexture{};
            nullTexture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            nullTexture.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            nullTexture.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            nullTexture.Texture2D.MipLevels = 1;
            m_device->CreateShaderResourceView(nullptr, &nullTexture, m_cpuSrvs.Cpu(m_nullTextureSrv));
            m_nullRtv = m_rtvs.Cpu(m_rtvs.Allocate());
            D3D12_RENDER_TARGET_VIEW_DESC nullRtv{};
            nullRtv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            nullRtv.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            m_device->CreateRenderTargetView(nullptr, &nullRtv, m_nullRtv);

            // Upload ring.
            D3D12_RESOURCE_DESC bufferDesc{};
            bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            bufferDesc.Width = UPLOAD_RING_SIZE;
            bufferDesc.Height = 1;
            bufferDesc.DepthOrArraySize = 1;
            bufferDesc.MipLevels = 1;
            bufferDesc.SampleDesc.Count = 1;
            bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            D3D12_HEAP_PROPERTIES uploadHeap{ D3D12_HEAP_TYPE_UPLOAD };
            D3D12_RANGE noRead{ 0, 0 };
            if (FAILED(m_device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&m_uploadBuffer))) ||
                FAILED(m_uploadBuffer->Map(0, &noRead, reinterpret_cast<void**>(&m_uploadCpu))))
            {
                Log("[gpu] Direct3D 12: could not create the upload buffer");
                return false;
            }
            m_uploadGpu = m_uploadBuffer->GetGPUVirtualAddress();
            m_uploadRing.Init(UPLOAD_RING_SIZE);
            // One raw SRV over the whole upload ring in the last shader-visible slot (the descriptor ring stops before it): a
            // draw finds its vertex data by the offsets in the system constants, with no view made per draw.
            D3D12_SHADER_RESOURCE_VIEW_DESC vertexData{};
            vertexData.Format = DXGI_FORMAT_R32_TYPELESS;
            vertexData.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            vertexData.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            vertexData.Buffer.NumElements = uint32_t(UPLOAD_RING_SIZE / 4);
            vertexData.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            m_device->CreateShaderResourceView(m_uploadBuffer.Get(), &vertexData, GpuHeapCpu(VERTEX_DATA_DESCRIPTOR));
            if (!CreateDrawRootSignature())
            {
                Log("[gpu] Direct3D 12: could not create the draw root signature");
                return false;
            }

            // The full-screen vertex shader: if even that does not compile, nothing can be shown (the start fails).
            if (!CompileDxil(Stage::Vertex, FULLSCREEN_VS, "blit_vs", m_shaderCache, m_dxilCache, m_fullscreenVs, &m_stats.dxilCompiles,
                            &m_stats.dxilFailures, nullptr))
                return false;
            // Direct3D 12 refuses DXIL that dxil.dll (next to the exe) did not sign: say so up front. The hash alone
            // does not tell: without dxil.dll, DXC's built-in validator still fills in a digest, so the DLL is checked too.
            if (DxilSigned(m_fullscreenVs) && DxilDllFound())
                Log("[gpu] Direct3D 12: DXIL signed by dxil.dll");
            else
                Log("[gpu] Direct3D 12: DXIL NOT signed (no dxil.dll next to the exe): some Windows versions may refuse the shaders");

            // Swap chain: flip model, borderless window as on Vulkan, no tearing (spec 004).
            DXGI_SWAP_CHAIN_DESC1 swapDesc{};
            swapDesc.Width = info.width;
            swapDesc.Height = info.height;
            swapDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            swapDesc.SampleDesc.Count = 1;
            swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swapDesc.BufferCount = BACK_BUFFER_COUNT;
            swapDesc.SwapEffect = (m_safe & SAFE_PRESENT) ? DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL : DXGI_SWAP_EFFECT_FLIP_DISCARD;
            ComPtr<IDXGIFactory5> factory5;
            BOOL tearing = FALSE;
            if (SUCCEEDED(m_factory.As(&factory5)) &&
                FAILED(factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))))
                tearing = FALSE;
            if (m_safe & SAFE_PRESENT)
                tearing = FALSE;  // safe mode "present": no tearing, sequential flip
            m_swapFlags = tearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;
            swapDesc.Flags = m_swapFlags;
            ComPtr<IDXGISwapChain1> swapChain;
            if (FAILED(m_factory->CreateSwapChainForHwnd(m_queue.Get(), m_hwnd, &swapDesc, nullptr, nullptr, &swapChain)) ||
                FAILED(swapChain.As(&m_swapChain)))
            {
                Log("[gpu] Direct3D 12: could not create the swap chain");
                return false;
            }
            m_factory->MakeWindowAssociation(m_hwnd, DXGI_MWA_NO_ALT_ENTER);  // Alt+Enter is the window's, not DXGI's
            m_backBufferWidth = info.width;
            m_backBufferHeight = info.height;
            CreateBackBuffers();
            LogSwapChain();

            // Timestamps: without them TimestampFrequency() stays 0 and the callers skip their timing.
            D3D12_QUERY_HEAP_DESC queryDesc{ D3D12_QUERY_HEAP_TYPE_TIMESTAMP, TIMESTAMPS, 0 };
            bufferDesc.Width = TIMESTAMPS * sizeof(uint64_t);
            D3D12_HEAP_PROPERTIES readbackHeap{ D3D12_HEAP_TYPE_READBACK };
            if (SUCCEEDED(m_queue->GetTimestampFrequency(&m_timestampFrequency)) && SUCCEEDED(m_device->CreateQueryHeap(&queryDesc, IID_PPV_ARGS(&m_queryHeap))))
            {
                if (FAILED(m_device->CreateCommittedResource(&readbackHeap, D3D12_HEAP_FLAG_NONE, &bufferDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&m_queryReadback))))
                {
                    m_queryHeap.Reset();
                    m_timestampFrequency = 0;
                }
            }
            return true;
        }
    }

    bool TestCompileDxil(Stage stage, const std::string& hlsl, const char* name, const std::filesystem::path& shaderCache)
    {
        std::filesystem::path cache = DxilCacheFolder(shaderCache);
        std::error_code error;
        if (!cache.empty())
            std::filesystem::create_directories(cache, error);
        std::vector<uint8_t> code;
        return CompileDxil(stage, hlsl, name, shaderCache, cache, code, nullptr, nullptr, nullptr);
    }

    Device* CreateD3D12(const DeviceInfo& info)
    {
        // Test hook (spec 001): make the start fail on purpose, to see the fallback to Vulkan.
        bool failShaders = false, removeDevice = false;
        if (const char* fail = getenv("CRASH_D3D12_FAIL"); fail && *fail)
        {
            if (plat::CompareNoCase(fail, "device") == 0)
            {
                Log("[gpu] CRASH_D3D12_FAIL=device: Direct3D 12 fails on purpose");
                return nullptr;
            }
            if (plat::CompareNoCase(fail, "shaders") == 0)
            {
                Log("[gpu] CRASH_D3D12_FAIL=shaders: the Direct3D 12 shaders fail on purpose");
                failShaders = true;
            }
            else if (plat::CompareNoCase(fail, "removed") == 0)
                removeDevice = true;
            else
                Log("[gpu] CRASH_D3D12_FAIL=%s ignored (device | shaders | removed)", fail);
        }
        auto device = std::make_unique<D3D12Device>();
        if (failShaders)
            device->FailShaders();
        if (removeDevice)
            device->RemoveOnPurpose();
        if (!device->Init(info))
            return nullptr;
        return device.release();
    }
}

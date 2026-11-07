// Direct3D 12 backend of the rendering hardware interface (rhi.h), Windows only.
//
// Spec 001 skeleton: the device, one direct queue and command list, a flip-model swap chain, the
// upload ring, clears and the read-back of the back buffer (screenshots). Pass pipelines come with
// the DXIL shaders (CreatePassPipeline is a stub for now, so the port's layer is not drawn yet).
// What the game's own draws need (textures, depth, shaders, draw pipelines: specs 002-003) is a stub
// that does nothing: the renderer (xenos_renderer.cpp, "presenter only") never sends it a draw.
//
// Adapted from the backend this project had until 2026-10-23 (4cac6ecb22f7:runtime/gpu/rhi_d3d12.cpp).

#include "rhi.h"
#include "rhi_ring.h"
#include "../diagnostics.h"
#include "../platform/platform.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace gpu::rhi
{
    namespace
    {
        constexpr uint32_t BACK_BUFFER_COUNT = 2;
        constexpr uint64_t UPLOAD_RING_SIZE = 256ull << 20;
        constexpr uint32_t RTV_DESCRIPTORS = 4096;
        constexpr uint32_t NO_DESCRIPTOR = UINT32_MAX;

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
            D3D12_CPU_DESCRIPTOR_HANDLE Cpu(uint32_t index) const
            {
                D3D12_CPU_DESCRIPTOR_HANDLE handle = heap->GetCPUDescriptorHandleForHeapStart();
                handle.ptr += size_t(index) * size;
                return handle;
            }
        };

        // Only the swap chain's back buffers exist as textures for now.
        class D3D12Texture : public Texture
        {
        public:
            ComPtr<ID3D12Resource> resource;
            D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
            uint32_t rtv = NO_DESCRIPTOR;
        };

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

            const std::string& AdapterName() const override { return m_adapterName; }

            void Submit() override;
            void WaitIdle() override;
            Upload AllocateUpload(uint64_t size, uint64_t alignment) override;
            void HoldUploads(bool hold) override { hold ? m_uploadRing.Hold() : m_uploadRing.Release(); }

            // Textures and buffers of the game's draws and of the overlay: none yet (specs 002-003, DXIL task).
            Texture* CreateTexture(const TextureDesc&) override { return nullptr; }
            void DestroyTexture(Texture*) override {}
            Texture* NullTexture() override { return nullptr; }
            Buffer* CreateBuffer(uint64_t) override { return nullptr; }
            void DestroyBuffer(Buffer*) override {}

            void CopyUploadToTexture(Texture*, uint32_t, uint64_t, uint32_t) override {}
            void CopyTexture(Texture*, Texture*, const Rect*) override {}
            void ClearColor(Texture* texture, const float color[4], const Rect* rect) override;
            void ClearDepthStencil(Texture*, float, uint8_t, const Rect*) override {}

            Shader* CreateShader(Stage, const std::string&, const char*, Layout) override { return nullptr; }
            Pipeline* GetDrawPipeline(const DrawPipelineKey&, bool, bool*) override { return nullptr; }
            void PrecompileProgress(uint32_t& done, uint32_t& total) override { done = total = 0; }
            void SetShaderCompilation(ShaderCompilation) override {}
            void PrewarmShader(Stage, const std::string&) override {}
            // Pass pipelines (the present, the overlay) need the DXIL shaders: next task.
            Pipeline* CreatePassPipeline(const PassPipelineDesc&) override { return nullptr; }

            void Draw(const DrawCall&) override {}
            void DrawPass(const PassDraw&) override {}  // no pass pipeline exists yet

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

            // Timestamps and native access are for the lighting remaster and the upscalers (specs 004, issues #6/#7).
            void WriteTimestamp(uint32_t) override {}
            void ResolveTimestamps(uint32_t, uint32_t) override {}
            bool ReadTimestamps(uint32_t, uint32_t, uint64_t*) override { return false; }
            double TimestampFrequency() const override { return 0.0; }

            NativeHandles Native() override { return {}; }
            bool DeviceExtensionEnabled(const char*) const override { return false; }
            void* NativeTexture(Texture*) override { return nullptr; }
            void* NativeView(Texture*) override { return nullptr; }
            void PrepareNative(Texture*, NativeState) override {}
            void InvalidateState() override {}

            Timing TakeTiming() override
            {
                Timing timing = m_timing;
                m_timing = {};
                return timing;
            }
            void LogStats() override;

        private:
            void WaitForFence(uint64_t value);
            void RetireCompleted();
            void Transition(D3D12Texture* texture, D3D12_RESOURCE_STATES state);
            void CreateBackBuffers();

            std::string m_adapterName;
            HWND m_hwnd = nullptr;
            ComPtr<IDXGIFactory4> m_factory;
            ComPtr<ID3D12Device> m_device;
            ComPtr<ID3D12CommandQueue> m_queue;
            ComPtr<IDXGISwapChain3> m_swapChain;
            D3D12Texture m_backBuffers[BACK_BUFFER_COUNT];
            uint32_t m_backBufferWidth = 1280, m_backBufferHeight = 720;
            ComPtr<ID3D12Fence> m_fence;
            uint64_t m_fenceValue = 0;
            HANDLE m_fenceEvent = nullptr;
            ComPtr<ID3D12GraphicsCommandList> m_list;
            ComPtr<ID3D12CommandAllocator> m_allocator;
            std::deque<std::pair<ComPtr<ID3D12CommandAllocator>, uint64_t>> m_allocatorPool;
            std::deque<uint64_t> m_frameFences;

            DescriptorPool m_rtvs;

            // Upload ring (one upload heap buffer, mapped for its whole life).
            ComPtr<ID3D12Resource> m_uploadBuffer;
            uint8_t* m_uploadCpu = nullptr;
            Ring m_uploadRing;

            Timing m_timing;
            struct
            {
                uint64_t submits = 0, passPipelines = 0, dxilCompiles = 0, dxilFailures = 0;
            } m_stats;
        };

        // Submission ------------------------------------------------------------------------------

        void D3D12Device::WaitForFence(uint64_t value)
        {
            if (m_fence->GetCompletedValue() >= value)
                return;
            m_fence->SetEventOnCompletion(value, m_fenceEvent);
            WaitForSingleObject(m_fenceEvent, INFINITE);
        }

        void D3D12Device::RetireCompleted()
        {
            m_uploadRing.Retire(m_fence->GetCompletedValue());
        }

        void D3D12Device::Submit()
        {
            m_list->Close();
            ID3D12CommandList* lists[] = { m_list.Get() };
            m_queue->ExecuteCommandLists(1, lists);
            m_fenceValue++;
            m_queue->Signal(m_fence.Get(), m_fenceValue);
            m_uploadRing.OnSubmit(m_fenceValue);
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
        }

        void D3D12Device::WaitIdle()
        {
            Submit();
            WaitForFence(m_fenceValue);
            RetireCompleted();
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
            m_list->ResourceBarrier(1, &barrier);
            texture->state = state;
        }

        void D3D12Device::ClearColor(Texture* texture, const float color[4], const Rect* rect)
        {
            auto* d3dTexture = static_cast<D3D12Texture*>(texture);
            Transition(d3dTexture, D3D12_RESOURCE_STATE_RENDER_TARGET);
            D3D12_RECT d3dRect = rect ? D3D12_RECT{ rect->left, rect->top, rect->right, rect->bottom } : D3D12_RECT{};
            m_list->ClearRenderTargetView(m_rtvs.Cpu(d3dTexture->rtv), color, rect ? 1 : 0, rect ? &d3dRect : nullptr);
        }

        // Presentation ----------------------------------------------------------------------------

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
            if (!width || !height || (width == m_backBufferWidth && height == m_backBufferHeight))
                return false;
            WaitIdle();
            // The swap chain can only be resized once every reference to its buffers is gone.
            for (auto& buffer : m_backBuffers)
                buffer.resource.Reset();
            if (SUCCEEDED(m_swapChain->ResizeBuffers(BACK_BUFFER_COUNT, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, 0)))
            {
                m_backBufferWidth = width;
                m_backBufferHeight = height;
            }
            CreateBackBuffers();
            return true;
        }

        void D3D12Device::Present(bool vsync)
        {
            D3D12Texture* backBuffer = &m_backBuffers[m_swapChain->GetCurrentBackBufferIndex()];
            Transition(backBuffer, D3D12_RESOURCE_STATE_PRESENT);
            LARGE_INTEGER submitStart, submitEnd, presentEnd;
            QueryPerformanceCounter(&submitStart);
            Submit();
            QueryPerformanceCounter(&submitEnd);
            // No tearing and no exclusive fullscreen (spec 004): V-Sync off just means Present(0).
            m_swapChain->Present(vsync ? 1 : 0, 0);
            QueryPerformanceCounter(&presentEnd);
            m_timing.submitTicks += uint64_t(submitEnd.QuadPart - submitStart.QuadPart);
            m_timing.presentTicks = std::max(m_timing.presentTicks, uint64_t(presentEnd.QuadPart - submitEnd.QuadPart));

            // At most two frames in flight.
            m_frameFences.push_back(m_fenceValue);
            LARGE_INTEGER waitStart, waitEnd;
            QueryPerformanceCounter(&waitStart);
            while (m_frameFences.size() > 2)
            {
                WaitForFence(m_frameFences.front());
                m_frameFences.pop_front();
            }
            QueryPerformanceCounter(&waitEnd);
            m_timing.frameWaitTicks += uint64_t(waitEnd.QuadPart - waitStart.QuadPart);
            RetireCompleted();
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
                return false;
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
            Log("[gpu] d3d12: %llu submits, %llu pass pipelines, %llu DXIL compiles (%llu failed)", (unsigned long long)m_stats.submits,
                (unsigned long long)m_stats.passPipelines, (unsigned long long)m_stats.dxilCompiles, (unsigned long long)m_stats.dxilFailures);
        }

        // Startup ---------------------------------------------------------------------------------

        // Every failure says why and returns false: the renderer then starts Vulkan in the same window.
        bool D3D12Device::Init(const DeviceInfo& info)
        {
            m_hwnd = HWND(info.window);
            // No debug layer (spec 004): info.debug is a Vulkan switch for now.
            if (FAILED(CreateDXGIFactory2(0, IID_PPV_ARGS(&m_factory))) ||
                FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&m_device))))
            {
                Log("[gpu] Direct3D 12: no device with feature level 11_0");
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
            DXGI_ADAPTER_DESC1 adapterDesc{};
            ComPtr<IDXGIAdapter1> adapter;
            if (SUCCEEDED(m_factory->EnumAdapterByLuid(m_device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
                adapter->GetDesc1(&adapterDesc);
            char adapterName[128];
            WideCharToMultiByte(CP_UTF8, 0, adapterDesc.Description, -1, adapterName, sizeof(adapterName), nullptr, nullptr);
            m_adapterName = adapterName;

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

            // Descriptor heap: render target views (the back buffers'). The shader-visible ones come with the shaders.
            D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{ D3D12_DESCRIPTOR_HEAP_TYPE_RTV, RTV_DESCRIPTORS, D3D12_DESCRIPTOR_HEAP_FLAG_NONE };
            if (FAILED(m_device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&m_rtvs.heap))))
            {
                Log("[gpu] Direct3D 12: could not create the descriptor heap");
                return false;
            }
            m_rtvs.size = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            m_rtvs.capacity = RTV_DESCRIPTORS;
            m_rtvs.name = "RTV";

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
            m_uploadRing.Init(UPLOAD_RING_SIZE);

            // Swap chain: flip model, borderless window as on Vulkan, no tearing (spec 004).
            DXGI_SWAP_CHAIN_DESC1 swapDesc{};
            swapDesc.Width = info.width;
            swapDesc.Height = info.height;
            swapDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            swapDesc.SampleDesc.Count = 1;
            swapDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
            swapDesc.BufferCount = BACK_BUFFER_COUNT;
            swapDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
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
            return true;
        }
    }

    Device* CreateD3D12(const DeviceInfo& info)
    {
        // Test hook (spec 001): make the start fail on purpose, to see the fallback to Vulkan.
        // "shaders" is for the DXIL task and does nothing yet.
        if (const char* fail = getenv("CRASH_D3D12_FAIL"); fail && *fail)
        {
            if (plat::CompareNoCase(fail, "device") == 0)
            {
                Log("[gpu] CRASH_D3D12_FAIL=device: Direct3D 12 fails on purpose");
                return nullptr;
            }
            Log("[gpu] CRASH_D3D12_FAIL=%s ignored (device | shaders)", fail);
        }
        auto device = std::make_unique<D3D12Device>();
        if (!device->Init(info))
            return nullptr;
        return device.release();
    }
}

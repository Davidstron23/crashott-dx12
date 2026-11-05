# 0001. Direct3D 12 as a second backend of the existing renderer

- Status: accepted
- Date: 2026-11-05

## Context

CrashRecomp v1.07 renders only through Vulkan 1.3. Its Xenos renderer (`runtime/gpu/xenos_renderer.cpp`)
and its passes (lighting, SMAA, overlay, upscalers) talk to the GPU through a small rendering hardware
interface, `runtime/gpu/rhi.h`, implemented by `rhi_vulkan.cpp`. Upstream had a Direct3D 12 backend on
that same interface until 2026-10-23 (`rhi_d3d12.cpp`, removed in commit `d98d8f6`), which compiled its
shaders with FXC (Shader Model 5.1). Since then the interface gained asynchronous pipeline compiling, a
GPU safe mode, upload holds and debug hooks.

We want a D3D12 build for PCs without a working Vulkan driver and for PC tools and mods that only hook
Direct3D.

## Decision

- **Base:** upstream tag `v1.07`, the version the recompiled code (`generated/`) on the development
  machine comes from. `main` starts there; upstream changes are not merged until the D3D12 renderer is done.
- **Backend:** bring back `rhi_d3d12.cpp` from upstream's history and adapt it to the v1.07 interface,
  instead of writing a new one or translating Vulkan calls at run time. The renderer above the interface
  does not change.
- **Shaders:** the same HLSL (Shader Model 6.0, D3D register model) compiled to DXIL with the same
  `dxcompiler.dll` the Vulkan backend uses for SPIR-V. FXC is not brought back.
- **Choice at run time:** a `renderer` setting (`vulkan` by default, `d3d12`), both backends in the same
  executable. If D3D12 cannot start, the game falls back to Vulkan and logs why.
- **Toolchain:** unchanged (llvm-mingw clang, CMake, Ninja), D3D12 and DXGI from the Windows SDK headers
  that ship with it.

## Alternatives considered

- **A translation layer (Vulkan on D3D12, or the reverse, at run time):** no code to write, but adds a
  dependency and a layer between the game and the mods that want to hook D3D12.
- **Rewriting the backend from scratch:** the removed one already solved resource states, the upload ring
  and the Xenos binding model on D3D12; adapting it is less work and less risk.
- **FXC (Shader Model 5.1) as before:** the v1.07 shaders were written and tested for DXC; keeping one
  compiler for both backends avoids two sets of shader bugs.
- **Basing on upstream's latest release:** newer runtime, but the recompiled code would have to be
  generated again first, and the game re-checked before any renderer work.

## Consequences

- Upstream fixes after v1.07 have to be brought over by hand while the fork stays on v1.07.
- DLSS and FSR need their D3D12 paths (NVIDIA's NGX D3D12 API, AMD's FidelityFX DX12 library), which
  upstream never had in its current form.
- The CI cannot build the game itself (it needs the recompiled code, made from each user's own copy of
  the game); it builds what it can without it.

# AMD FSR 3.1 upscaler (FidelityFX API, Vulkan and Direct3D 12)

Subset of AMD's FidelityFX SDK, https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK, tag
`v1.1.4`, downloaded 2026-10-23 (the Direct3D 12 pieces 2026-11-16; the newer SDK 2.x only ships DirectX 12 DLLs):

- `include/ffx_api/`: `ffx_api.h`, `ffx_api_types.h`, `ffx_upscale.h` and `vk/ffx_api_vk.h` and `dx12/ffx_api_dx12.h`, from
  `ffx-api/include/ffx_api/` (unchanged; they include each other relatively);
- `bin/amd_fidelityfx_vk.dll`: `PrebuiltSignedDLL/amd_fidelityfx_vk.dll`, signed by AMD, with the
  FSR 3.1.4 upscaler (and FSR 2.3.3);
- `bin/amd_fidelityfx_dx12.dll`: `PrebuiltSignedDLL/amd_fidelityfx_dx12.dll` of the same tag, with a
  valid Authenticode signature from Advanced Micro Devices (checked when first downloaded);
- `license.md`: the SDK's `LICENSE.txt` (MIT).

`runtime/gpu/upscaler.cpp` loads the DLL at run time (nothing is linked) and asks for the FSR 3.1.x
upscaler explicitly, on the renderer's own Vulkan or Direct3D 12 device (only the DLL of the API in use is loaded). Frame generation is not used. The CMake
build copies the DLLs next to `CrashRecomp.exe`; `tools/setup.py` downloads these files (pinned
SHA-256). Folders left by older checkouts (`Kits/`, the `*_dx12.dll`) are not used any more.

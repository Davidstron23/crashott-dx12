# 0002. Sign the DXIL of the Direct3D 12 renderer with dxil.dll

- Status: accepted
- Date: 2026-11-07

## Context

The Direct3D 12 renderer (ADR 0001) compiles its HLSL to DXIL with `dxcompiler.dll`, the same compiler the
Vulkan backend uses for SPIR-V. Direct3D 12 only accepts DXIL that carries a validator signature in its
container. `dxcompiler.dll` signs the shader with `dxil.dll` (Microsoft's DXIL validator) when it finds it
next to it. The Windows SDK's copy of DXC is not an option (no SPIR-V backend, see
`thirdparty/dxc/README.md`).

Measured on the development machine (Windows 11 build 26300, DXC 1.9.2609): without `dxil.dll` DXC still
fills in the same digest, without a warning, and Direct3D 12 accepted the pipelines. Whether Windows 10 or a
Windows 11 without developer mode does is not known, so the DLL is shipped anyway and the log checks for it.

## Decision

- Ship `dxil.dll` next to `CrashRecomp.exe`. It comes from the same official DXC release as
  `dxcompiler.dll` (`dxc_2026_09_29.zip`, `bin/x64/dxil.dll`), whose SHA-256 is pinned in `tools/setup.py`
  (`DXIL_SHA256`), under the same Microsoft license that already ships with DXC (`LICENSE-MS.txt`).
- `tools/setup.py` and `tools/dev_setup.py` fetch it, CMake copies it, `make_dist.py` and
  `make_installer.py` already carry every DLL of `build\` and `thirdparty/dxc/bin`.
- The backend logs whether the DXIL came out signed: `DxilSigned` (a "DXBC" container whose digest is
  neither all zeros nor the "bypass" pattern of unvalidated shaders) and `dxil.dll` loadable from the exe's folder.

## Alternatives considered

- **The "bypass" hash (an unvalidated container with a fixed digest):** needs a recent Agility SDK and
  Windows 11 24H2 or later, while the game supports Windows 10 and 11.
- **Our own validator or no validation (developer mode):** unreliable on players' PCs, and unsafe.
- **FXC (DXBC, Shader Model 5.1):** needs no DLL (it is part of Windows), but then two compilers and two
  sets of shader bugs, as ADR 0001 already decided against.

## Consequences

- One more redistributable DLL (about 2 MB) in the installer and in `dist\`; the notices list it.
- Without `dxil.dll` the log says `DXIL NOT signed`; if Direct3D 12 then refuses the shaders, the game falls
  back to Vulkan (spec 001, task 3).
- A new DXC release means updating `DXC_ZIP`, its SHA-256 and `DXIL_SHA256` together.

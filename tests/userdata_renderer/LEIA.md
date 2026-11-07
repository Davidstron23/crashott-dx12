Fixtures of spec 001 (task T4): a settings.toml each, copied by run_game with `userdata_from`. They only differ from
`tests/userdata_base/settings.toml` in the `renderer` line:

- `dx12/`: `renderer = "DX12"` (an alias, in capitals)
- `foo/`: `renderer = "foo"` (not a renderer)
- `vulkan/`: `renderer = "vulkan"`

No saves and nothing of the game.

Spec 001, task T5 (the RENDERER row, SETTINGS drawn by Direct3D 12): same file, other lines changed, and `fps_limit = 30`
so the menu presses of the profiles are not lost at ~900 fps:

- `d3d12_full/`: `renderer = "d3d12"`, `upscaler = "fsr"`, `antialiasing = "smaa"`, `ssao = "high"`, `gi = true`
- `d3d12_dlss/`: `renderer = "d3d12"`, `upscaler = "dlss"`

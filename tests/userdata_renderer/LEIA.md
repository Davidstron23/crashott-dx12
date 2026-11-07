Fixtures of spec 001 (task T4): a settings.toml each, copied by run_game with `userdata_from`. They only differ from
`tests/userdata_base/settings.toml` in the `renderer` line:

- `dx12/`: `renderer = "DX12"` (an alias, in capitals)
- `foo/`: `renderer = "foo"` (not a renderer)
- `vulkan/`: `renderer = "vulkan"`

No saves and nothing of the game.

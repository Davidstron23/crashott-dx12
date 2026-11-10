#pragma once

// Parity tool (spec 003, tools/parity.py): compares the image of one game frame between the two
// graphics APIs, bit by bit where it can.
//
// Two games never draw the same frame (the recompiled mftb reads the CPU's timestamp counter), so the
// frame is captured once and replayed:
//   CRASH_PARITY_CAPTURE=<times> (the syntax of CRASH_SCREENSHOT_TIMES): at each time the game is held
//     in the first kick (CP_RB_WPTR write) that publishes its next VdSwap's swap token, the command
//     processor runs up to that frame's swap, and the frame is written to <CRASH_SCREENSHOT_DIR>\parity_<n>\:
//     the registers at the start of the frame, the packets it executed and the physical memory at its end,
//     the image and the draws as they were in the live run. Then the game goes on.
//   --parity-replay <parity_n folder> (with --renderer): runs the packets on their own five times with
//     the renderer asked for (pass 4 writes the image, pass 5 one digest per draw) and writes
//     replay_<api>.png and replay_<api>_draws.csv there. Exit 0, or 2 if the replay's packets are
//     not the live ones.
// Neither is set: nothing here runs.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu::parity
{
    // What the command processor keeps besides the registers: the tiling bins.
    struct CommandState
    {
        uint64_t binMask, binSelect;
    };

    // One draw or resolve packet (xenos_renderer.cpp, before any discard).
    struct Row
    {
        bool resolve;
        uint64_t vs, ps;
        uint32_t base, format, width, height, msaa, depthBase;  // of colour target 0
        const char* reason;  // why nothing was drawn, or ""
        bool hasDigest;
        uint64_t digest;
    };

    extern bool g_active;  // set once, before the threads start
    inline bool Active() { return g_active; }

    void Init();  // reads CRASH_PARITY_CAPTURE (gpu::Init)
    // Game thread: VdSwap number `swap`; a kick (the CP_RB_WPTR publication), where the game is held.
    void OnVdSwap(uint64_t swap, uint32_t token);
    void OnKick();
    // Command processor thread, around the renderer's Swap for an XE_SWAP packet.
    void BeginSwap(const CommandState& bins);
    void EndSwap();

    // Renderer: the draws of the frame, and the image to save at this swap ("" = none).
    void NoteRow(const Row& row);
    bool WantDigests();
    uint64_t Digest(const uint8_t* bytes, size_t size);
    const char* ImagePath();

    // main.cpp: --parity-replay forces graphics.shader_compilation = sync (before the settings load).
    void PrepareReplay(int argc, char** argv);
    int Replay(const char* folder);
    // command_processor.cpp: the packets executed since the last swap (indirect buffers inlined, host order); runs
    // `dwords` of them from guest memory at `stream` here, without waiting for the CPU or the interrupts; and
    // whether the ring written between write pointers `from` and `to` calls an indirect buffer that holds
    // the physical address `address`.
    const std::vector<uint32_t>& ExecutedStream();
    void ClearExecutedStream();
    void ReplayCommands(const CommandState& start, uint32_t stream, uint32_t dwords);
    bool PublishedContains(uint32_t address, uint32_t from, uint32_t to);
}

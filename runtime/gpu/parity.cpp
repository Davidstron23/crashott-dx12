// Parity tool: see parity.h.

#include "parity.h"
#include "gpu.h"
#include "renderer.h"
#include "../diagnostics.h"
#include "../kernel/input.h"
#include "../kernel/memory.h"
#include "../platform/crt.h"
#include "../platform/platform.h"
#include "../settings/direct_boot.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace gpu::parity
{
    bool g_active;

    namespace
    {
        namespace fs = std::filesystem;
        enum class Mode { Off, Capture, Replay };
        Mode g_mode = Mode::Off;
        constexpr uint32_t MAGIC = 0x59545250;  // 'PRTY'
        constexpr uint32_t BLOCK = 0x10000;     // memory is written in 64 KB blocks, the empty ones skipped
        constexpr int PASSES = 5;
        constexpr uint32_t STREAM_BASE = 0x30000000;  // where the replay puts the packets (nothing else uses it there)

        // What the command processor and the registers were at the end of an XE_SWAP: where the next frame starts.
        struct Snapshot
        {
            CommandState command{};
            std::vector<uint32_t> registers;
        };
        void Take(Snapshot& snapshot, const CommandState& command)
        {
            snapshot.command = command;
            snapshot.registers.resize(REGISTER_COUNT);
            for (uint32_t i = 0; i < REGISTER_COUNT; i++)
                snapshot.registers[i] = ReadRegister(i);
        }

        // The draws of the frame: "index,type,vs,ps,rt" (what must match the live run) and "reason,digest".
        struct Line
        {
            std::string key, tail;
        };
        std::vector<Line> g_rows;
        void WriteRows(const fs::path& path)
        {
            std::ofstream out(path);
            out << "index,type,vs,ps,rt,reason,digest\n";
            for (size_t i = 0; i < g_rows.size(); i++)
                out << i << ',' << g_rows[i].key << ',' << g_rows[i].tail << '\n';
        }

        // Capture ---------------------------------------------------------------------------------
        std::vector<boot::Anchored> g_anchors;
        std::vector<int64_t> g_times;
        fs::path g_directory;
        int g_captured;
        std::atomic<bool> g_holdRequested;
        std::atomic<uint64_t> g_target;  // number of the swap to capture, 0 = none
        uint32_t g_token;                // physical address of its swap token (VdSwap's buffer)
        uint32_t g_lastWritePointer;     // CP_RB_WPTR at the previous kick
        uint64_t g_swaps;                // XE_SWAPs executed (the same count as the game's VdSwaps)
        bool g_wanted;                   // the swap being executed is the target
        Snapshot g_start;                // the end of the previous swap
        CommandState g_bins;             // the tiling bins at the swap being executed
        std::mutex g_mutex;
        std::condition_variable g_condition;
        bool g_held, g_release;
        std::chrono::steady_clock::time_point g_heldSince;
        fs::path g_frameDirectory;
        std::string g_imagePath;

        // The committed physical memory, the empty 64 KB blocks skipped: {u32 guest, u32 bytes, data}...
        uint64_t WriteMemory(const fs::path& path)
        {
            static const std::vector<uint8_t> zero(BLOCK);
            FILE* out = plat::OpenCFile(path, "wb");
            if (!out)
                return 0;
            uint64_t total = 0;
            uint32_t runStart = 0, runBytes = 0;
            auto flush = [&] {
                if (!runBytes)
                    return;
                fwrite(&runStart, 4, 1, out);
                fwrite(&runBytes, 4, 1, out);
                fwrite(g_memory.Translate(runStart), 1, runBytes, out);
                total += runBytes;
                runBytes = 0;
            };
            for (uint64_t offset = 0; offset < PHYSICAL_SIZE; offset += BLOCK)
            {
                uint32_t guest = uint32_t(PHYSICAL_VIEW_A + offset);
                if (memcmp(g_memory.Translate(guest), zero.data(), BLOCK) == 0)
                {
                    flush();
                    continue;
                }
                if (!runBytes)
                    runStart = guest;
                runBytes += BLOCK;
                if (runBytes == 0x40000000)  // keep the record's size in 32 bits
                    flush();
            }
            flush();
            fclose(out);
            return total;
        }

        void WriteCapture()
        {
            if (g_start.registers.empty())
            {
                Log("[parity] swap #%llu: no earlier swap to start from", (unsigned long long)g_swaps);
                return;
            }
            FILE* state = plat::OpenCFile(g_frameDirectory / "state.bin", "wb");
            if (!state)
            {
                Log("[parity] could not write %s", (g_frameDirectory / "state.bin").string().c_str());
                return;
            }
            fwrite(&MAGIC, 4, 1, state);
            fwrite(&g_start.command, sizeof(CommandState), 1, state);
            fwrite(g_start.registers.data(), 4, g_start.registers.size(), state);
            fclose(state);
            const std::vector<uint32_t>& stream = ExecutedStream();
            FILE* packets = plat::OpenCFile(g_frameDirectory / "stream.bin", "wb");
            if (packets)
            {
                fwrite(stream.data(), 4, stream.size(), packets);
                fclose(packets);
            }
            uint64_t bytes = WriteMemory(g_frameDirectory / "memory.bin");
            WriteRows(g_frameDirectory / "live_draws.csv");
            const char* settings = getenv("CRASH_SETTINGS");
            std::ofstream(g_frameDirectory / "capture.txt") << "version=1\napi=" << renderer::ApiName() << "\nswap=" << g_swaps
                                                              << "\nsettings=" << (settings ? settings : "") << '\n';
            Log("[parity] frame %d captured (swap #%llu, %zu packets, %.0f MB of memory) in %s", g_captured, (unsigned long long)g_swaps,
                g_rows.size(), double(bytes) / 1048576.0, g_frameDirectory.string().c_str());
        }

        // Replay ----------------------------------------------------------------------------------
        int g_pass;
        fs::path g_replayDirectory;
        std::string g_replayImage;

        bool ReadState(const fs::path& path, Snapshot& snapshot)
        {
            FILE* in = plat::OpenCFile(path, "rb");
            if (!in)
                return false;
            uint32_t magic = 0;
            snapshot.registers.resize(REGISTER_COUNT);
            bool ok = fread(&magic, 4, 1, in) == 1 && magic == MAGIC && fread(&snapshot.command, sizeof(CommandState), 1, in) == 1 &&
                      fread(snapshot.registers.data(), 4, REGISTER_COUNT, in) == REGISTER_COUNT;
            fclose(in);
            return ok;
        }

        // The packets of the frame, in guest memory (big-endian, as the ring has them): returns their dwords.
        uint32_t ReadStream(const fs::path& path)
        {
            FILE* in = plat::OpenCFile(path, "rb");
            if (!in)
                return 0;
            std::vector<uint32_t> packets;
            uint32_t chunk[4096];
            while (size_t got = fread(chunk, 4, 4096, in))
                packets.insert(packets.end(), chunk, chunk + got);
            fclose(in);
            if (packets.empty() || !g_memory.Commit(STREAM_BASE, packets.size() * 4))
                return 0;
            for (size_t i = 0; i < packets.size(); i++)
                StoreBE32(STREAM_BASE + uint32_t(i) * 4, packets[i]);
            return uint32_t(packets.size());
        }

        bool ReadMemory(const fs::path& path)
        {
            FILE* in = plat::OpenCFile(path, "rb");
            if (!in)
                return false;
            uint32_t record[2];
            while (fread(record, 4, 2, in) == 2)
            {
                if (record[0] < PHYSICAL_VIEW_A || uint64_t(record[0]) + record[1] > uint64_t(PHYSICAL_VIEW_A) + PHYSICAL_SIZE ||
                    fread(g_memory.Translate(record[0]), 1, record[1], in) != record[1])
                {
                    fclose(in);
                    return false;
                }
            }
            fclose(in);
            return true;
        }

        // The packets (index, type, shaders, target) of the live run, to check the replay ran the same ones.
        bool SameAsLive(const fs::path& path)
        {
            std::ifstream in(path);
            std::string line;
            std::getline(in, line);  // header
            for (size_t i = 0; i < g_rows.size(); i++)
            {
                std::string expected = std::to_string(i) + ',' + g_rows[i].key + ',';
                if (!std::getline(in, line) || line.compare(0, expected.size(), expected) != 0)
                {
                    Log("[parity] packet %zu differs from the live run: replay '%s'; live '%s'", i, expected.c_str(), in ? line.c_str() : "(none)");
                    return false;
                }
            }
            if (std::getline(in, line))
            {
                Log("[parity] the live run has more packets than the replay (%zu)", g_rows.size());
                return false;
            }
            return true;
        }
    }

    void Init()
    {
        const char* times = getenv("CRASH_PARITY_CAPTURE");
        if (!times || !*times)
            return;
        // The times of CRASH_SCREENSHOT_TIMES: input-clock seconds, or seconds after a test event.
        for (const char* p = times; *p;)
        {
            boot::Anchored time;
            if (!boot::ParseAnchored(p, 1000.0, time))
                break;
            if (time.event.empty())
                g_times.push_back(time.offsetMs);
            else
                g_anchors.push_back(time);
            p = *p ? p + 1 : p;
        }
        std::sort(g_times.begin(), g_times.end());
        const char* directory = getenv("CRASH_SCREENSHOT_DIR");
        g_directory = directory ? fs::path(directory) : fs::current_path();
        g_mode = Mode::Capture;
        g_active = true;
        Log("[parity] capture on: %zu frame(s) to %s", g_times.size() + g_anchors.size(), g_directory.string().c_str());
    }

    void OnVdSwap(uint64_t swap, uint32_t token)
    {
        if (g_mode != Mode::Capture || g_target)
            return;
        for (auto it = g_anchors.begin(); it != g_anchors.end();)
        {
            int64_t at = boot::Resolve(*it);
            if (at < 0)
            {
                ++it;
                continue;
            }
            g_times.insert(std::upper_bound(g_times.begin(), g_times.end(), at), at);
            it = g_anchors.erase(it);
        }
        if (g_times.empty() || input::ClockMs() < g_times.front())
            return;
        g_times.erase(g_times.begin());
        g_token = token;
        g_target = swap;
        g_holdRequested = true;
    }

    // The game thread, in the first kick (the CP_RB_WPTR write) that publishes the target's swap token (the
    // game builds its frame in segments and kicks several per frame, the token's one after its VdSwap): the
    // command processor reaches the swap, and the game waits here until EndSwap has the frame's memory. The
    // clock of the input script then skips the wait.
    void OnKick()
    {
        if (g_mode != Mode::Capture)
            return;
        uint32_t writePointer = ReadRegister(REG_CP_RB_WPTR), previous = g_lastWritePointer;
        g_lastWritePointer = writePointer;
        if (!g_holdRequested || !PublishedContains(g_token, previous, writePointer))
            return;
        g_holdRequested = false;
        std::unique_lock lock(g_mutex);
        g_held = true;
        g_heldSince = std::chrono::steady_clock::now();
        g_condition.notify_all();
        Log("[parity] game held at the kick after VdSwap #%llu", (unsigned long long)g_target.load());
        bool released = g_condition.wait_for(lock, std::chrono::seconds(30), [] { return g_release; });
        g_held = g_release = false;
        if (!released)
        {
            Log("[parity] the command processor did not reach the swap in 30 s: frame dropped");
            g_target = 0;
        }
        input::ShiftClock(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - g_heldSince).count());
    }

    void BeginSwap(const CommandState& bins)
    {
        if (g_mode != Mode::Capture)
            return;
        g_bins = bins;
        g_wanted = ++g_swaps == g_target;
        if (!g_wanted)
            return;
        g_frameDirectory = g_directory / ("parity_" + std::to_string(++g_captured));
        std::error_code error;
        fs::create_directories(g_frameDirectory, error);
        g_imagePath = (g_frameDirectory / "live.png").string();
    }

    void EndSwap()
    {
        if (g_mode != Mode::Capture)
            return;
        if (g_wanted)
        {
            std::unique_lock lock(g_mutex);
            // The game must be in the trap by now; if the write slipped past it, it kept running.
            if (g_condition.wait_for(lock, std::chrono::seconds(2), [] { return g_held; }))
            {
                WriteCapture();
                g_release = true;
                g_condition.notify_all();
            }
            else
            {
                g_holdRequested = false;
                Log("[parity] swap #%llu: the game was not held (the kick slipped past the trap): frame dropped", (unsigned long long)g_swaps);
            }
            g_target = 0;
            g_wanted = false;
            g_imagePath.clear();
        }
        g_rows.clear();
        ClearExecutedStream();
        Take(g_start, g_bins);
    }

    void NoteRow(const Row& row)
    {
        char key[128], tail[48];
        snprintf(key, sizeof(key), "%s,%016llx,%016llx,c%u:f%u:%ux%u:m%u:z%u", row.resolve ? "resolve" : "draw", (unsigned long long)row.vs,
                 (unsigned long long)row.ps, row.base, row.format, row.width, row.height, row.msaa, row.depthBase);
        if (row.hasDigest)
            snprintf(tail, sizeof(tail), "%s,%016llx", row.reason, (unsigned long long)row.digest);
        else
            snprintf(tail, sizeof(tail), "%s,n/c", row.reason);
        g_rows.push_back({ key, tail });
    }

    bool WantDigests()
    {
        return g_mode == Mode::Replay && g_pass == PASSES;
    }

    // FNV-1a over 64-bit words: a different word always gives a different digest.
    uint64_t Digest(const uint8_t* bytes, size_t size)
    {
        uint64_t hash = 0xCBF29CE484222325ull;
        for (size_t i = 0; i + 8 <= size; i += 8)
        {
            uint64_t word;
            memcpy(&word, bytes + i, 8);
            hash = (hash ^ word) * 0x100000001B3ull;
        }
        return hash;
    }

    const char* ImagePath()
    {
        if (g_mode == Mode::Capture)
            return g_wanted ? g_imagePath.c_str() : "";
        return g_mode == Mode::Replay && g_pass == PASSES - 1 ? g_replayImage.c_str() : "";
    }

    void PrepareReplay(int argc, char** argv)
    {
        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i], "--parity-replay") != 0)
                continue;
            // The replay compiles where it first needs a shader: nothing may be half compiled in the image.
            const char* current = getenv("CRASH_SETTINGS");
            std::string value = std::string(current ? current : "") + ";graphics.shader_compilation=sync";
#ifdef _WIN32
            _putenv_s("CRASH_SETTINGS", value.c_str());  // the CRT's copy, which getenv reads (SetEnvironmentVariable does not touch it)
#else
            setenv("CRASH_SETTINGS", value.c_str(), 1);
#endif
        }
    }

    int Replay(const char* folder)
    {
        g_mode = Mode::Replay;
        g_active = true;
        g_replayDirectory = folder;
        Snapshot start;
        if (!ReadState(g_replayDirectory / "state.bin", start))
        {
            Log("[parity] cannot read %s", (g_replayDirectory / "state.bin").string().c_str());
            return 1;
        }
        InitReplay();
        std::string api = renderer::ApiName();
        Log("[parity] renderer in use: %s", api.c_str());
        g_replayImage = (g_replayDirectory / ("replay_" + api + ".png")).string();
        uint32_t dwords = ReadStream(g_replayDirectory / "stream.bin");
        if (!dwords || !ReadMemory(g_replayDirectory / "memory.bin"))
        {
            Log("[parity] cannot read the packets or the memory in %s", g_replayDirectory.string().c_str());
            return 1;
        }
        // Passes 1-3 warm up (shaders, pipelines, textures); 4 saves the image; 5 takes a digest per draw. The
        // memory is not restored: what the GPU writes to it (fences) is not read back by the draws.
        for (g_pass = 1; g_pass <= PASSES; g_pass++)
        {
            for (uint32_t i = 0; i < REGISTER_COUNT; i++)
                WriteRegister(i, start.registers[i]);
            g_rows.clear();
            ReplayCommands(start.command, STREAM_BASE, dwords);
        }
        WriteRows(g_replayDirectory / ("replay_" + api + "_draws.csv"));
        bool same = SameAsLive(g_replayDirectory / "live_draws.csv");
        Log("[parity] replay done: %zu packets, %s the live run", g_rows.size(), same ? "the same as" : "NOT the same as");
        return same ? 0 : 2;
    }
}

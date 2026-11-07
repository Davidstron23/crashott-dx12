// The GPU and what it can run, for the launcher (which upscalers its options offer) and the
// installer (whether the DLSS files are installed at all). Included inside launcher.cpp's anonymous
// namespace.
//
//   - DetectGpu / GpuInfo: the GPU the game will pick, its vendor and whether its driver has the
//     extensions FSR 3.1 calls (runtime/gpu/gpu_detect.h, shared with the runtime's Direct3D 12 path).
//   - StartDlssProbe / DlssProbeResult: NGX's own check that DLSS runs here (an NVIDIA RTX GPU and a
//     driver with DLSS), done by CrashDlss.dll's CrashDlss_Probe in a child process (this exe with
//     --probe-dlss), so a driver that crashes in NGX can't take the window with it.

using gpu::GpuInfo;
using gpu::DetectGpu;

// The DLSS module's file name: CrashDlss.dll, or libCrashDlss.so next to NVIDIA's libnvidia-ngx-dlss.so.*.
#ifdef _WIN32
const char* DLSS_MODULE = "CrashDlss.dll";
#else
const char* DLSS_MODULE = "libCrashDlss.so";
#endif

// --probe-dlss <folder> <result file>: the child process. Writes "1|0\n<gpu>\n<reason>\n" and exits
// with 0 when DLSS runs here.
int RunDlssProbe(const fs::path& folder, const fs::path& result)
{
#ifndef _WIN32
    int available = 0;
    char gpu[256] = "", reason[512] = "libCrashDlss.so could not be loaded";
    if (void* module = plat::OpenLibrary(folder / DLSS_MODULE))
    {
        auto probe = reinterpret_cast<CrashDlss_ProbeFn>(plat::FindSymbol(module, "CrashDlss_Probe"));
        if (probe)
        {
            fs::path data = fs::temp_directory_path() / "CrashRecomp_ngx";
            available = probe(folder.wstring().c_str(), data.wstring().c_str(), gpu, sizeof(gpu), reason, sizeof(reason));
            std::error_code error;
            fs::remove_all(data, error);  // NGX's logs of this check
        }
        else
            snprintf(reason, sizeof(reason), "libCrashDlss.so has no CrashDlss_Probe");
    }
    else
        snprintf(reason, sizeof(reason), "%s", plat::LastErrorText().c_str());
    if (FILE* f = fopen(result.c_str(), "wb"))
    {
        fprintf(f, "%d\n%s\n%s\n", available, gpu, reason);
        fclose(f);
    }
    return available ? 0 : 1;
#else
    int available = 0;
    char gpu[256] = "", reason[512] = "CrashDlss.dll could not be loaded";
    SetDllDirectoryW(folder.c_str());  // nvngx_dlss.dll and what CrashDlss.dll loads, from there
    HMODULE module = LoadLibraryExW((folder / L"CrashDlss.dll").c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (module)
    {
        auto probe = reinterpret_cast<CrashDlss_ProbeFn>(GetProcAddress(module, "CrashDlss_Probe"));
        if (probe)
        {
            wchar_t temp[MAX_PATH];
            GetTempPathW(MAX_PATH, temp);
            fs::path data = fs::path(temp) / L"CrashRecomp_ngx";
            available = probe(folder.c_str(), data.c_str(), gpu, sizeof(gpu), reason, sizeof(reason));
            std::error_code error;
            fs::remove_all(data, error);  // NGX's logs of this check
        }
        else
            snprintf(reason, sizeof(reason), "CrashDlss.dll has no CrashDlss_Probe");
    }
    if (FILE* f = _wfopen(result.c_str(), L"wb"))
    {
        fprintf(f, "%d\n%s\n%s\n", available, gpu, reason);
        fclose(f);
    }
    return available ? 0 : 1;
#endif
}

struct DlssProbe
{
    std::atomic<int> state{ 0 };  // 0 not started, 1 running, 2 done
    bool available = false;
    std::string gpu, reason;
    std::thread thread;
};
DlssProbe g_dlssProbe;

// Runs the probe for the CrashDlss.dll + nvngx_dlss.dll in `folder` on a thread (state 2 when done).
void StartDlssProbe(const fs::path& folder)
{
    if (g_dlssProbe.state != 0)
        return;
    g_dlssProbe.state = 1;
#ifndef _WIN32
    g_dlssProbe.thread = std::thread([folder] {
        fs::path result = fs::temp_directory_path() / ("CrashRecomp_dlss_probe_" + std::to_string(getpid()) + ".txt");
        bool available = false;
        std::string gpu, reason = "the check did not run";
        if (pid_t pid = Spawn(plat::ExecutablePath(), { "--probe-dlss", folder.string(), result.string() }, folder))
        {
            int status = 0;
            bool finished = false;
            for (int waited = 0; waited < 2000 && !finished; waited++)  // 20 s
            {
                finished = waitpid(pid, &status, WNOHANG) == pid;
                if (!finished)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (!finished)
            {
                kill(pid, SIGKILL);
                waitpid(pid, &status, 0);
                reason = "the check did not finish (20 s)";
            }
            else
            {
                int code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
                reason = "the check crashed (code " + std::to_string(code) + ")";
                if (FILE* f = fopen(result.c_str(), "rb"))
                {
                    char line[3][512] = {};
                    for (auto& l : line)
                        if (fgets(l, sizeof(l), f))
                            l[strcspn(l, "\r\n")] = 0;
                    fclose(f);
                    available = code == 0 && line[0][0] == '1';
                    gpu = line[1], reason = line[2];
                }
            }
        }
        std::error_code error;
        fs::remove(result, error);
        g_dlssProbe.available = available;
        g_dlssProbe.gpu = gpu;
        g_dlssProbe.reason = reason;
        Log("[gpu] DLSS check (%s): %s - %s", folder.string().c_str(), available ? "available" : "not available", reason.c_str());
        g_dlssProbe.state = 2;
    });
#else
    g_dlssProbe.thread = std::thread([folder] {
        wchar_t temp[MAX_PATH], self[MAX_PATH];
        GetTempPathW(MAX_PATH, temp);
        GetModuleFileNameW(nullptr, self, MAX_PATH);
        fs::path result = fs::path(temp) / (L"CrashRecomp_dlss_probe_" + std::to_wstring(GetCurrentProcessId()) + L".txt");
        std::wstring command = L"\"" + std::wstring(self) + L"\" --probe-dlss \"" + folder.wstring() + L"\" \"" + result.wstring() + L"\"";
        STARTUPINFOW si{ sizeof(si) };
        PROCESS_INFORMATION pi{};
        bool available = false;
        std::string gpu, reason = "the check did not run";
        if (CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, folder.c_str(), &si, &pi))
        {
            if (WaitForSingleObject(pi.hProcess, 20000) != WAIT_OBJECT_0)
            {
                TerminateProcess(pi.hProcess, 1);
                reason = "the check did not finish (20 s)";
            }
            else
            {
                DWORD code = 1;
                GetExitCodeProcess(pi.hProcess, &code);
                reason = "the check crashed (code " + std::to_string(code) + ")";
                if (FILE* f = _wfopen(result.c_str(), L"rb"))
                {
                    char line[3][512] = {};
                    for (auto& l : line)
                        if (fgets(l, sizeof(l), f))
                            l[strcspn(l, "\r\n")] = 0;
                    fclose(f);
                    available = code == 0 && line[0][0] == '1';
                    gpu = line[1], reason = line[2];
                }
            }
            CloseHandle(pi.hThread);
            CloseHandle(pi.hProcess);
        }
        std::error_code error;
        fs::remove(result, error);
        g_dlssProbe.available = available;
        g_dlssProbe.gpu = gpu;
        g_dlssProbe.reason = reason;
        Log("[gpu] DLSS check (%s): %s - %s", Narrow(folder.wstring()).c_str(), available ? "available" : "not available", reason.c_str());
        g_dlssProbe.state = 2;
    });
#endif
}

void JoinDlssProbe()
{
    if (g_dlssProbe.thread.joinable())
        g_dlssProbe.thread.join();
}

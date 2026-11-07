#pragma once

// Which graphics API the renderer starts with (Vulkan, or Direct3D 12 on Windows), as a pure
// function of where the player asked for it: no file, no environment and no log in here, so it can
// be tested alone. xenos_renderer.cpp fills the Request, calls Resolve and writes the result to the log.
//
// Priority, highest first: --renderer, CRASH_RENDERER, CRASH_SETTINGS, settings.toml, Vulkan.
// A value that isn't a renderer is skipped (Choice::ignored says so) and the next source is tried.

#include "../settings/settings.h"

#include <cstdint>
#include <string>
#include <vector>

namespace gpu::choice
{
    enum class Source : uint8_t { Argument, Environment, SettingsOverride, SettingsFile, Default };
    // "--renderer", "CRASH_RENDERER", "CRASH_SETTINGS", "settings.toml", "default".
    const char* SourceLabel(Source source);

    struct Request
    {
        bool argumentGiven = false;          // --renderer on the command line
        const char* argument = nullptr;      // its value; null: given without one
        const char* environment = nullptr;   // CRASH_RENDERER; null or "": not set
        settings::RendererOrigin settings = settings::RendererOrigin::None;
        settings::Renderer settingsRenderer = settings::Renderer::Vulkan;
    };

    struct Choice
    {
        settings::Renderer renderer = settings::Renderer::Vulkan;
        Source source = Source::Default;
        std::vector<std::string> ignored;    // full log lines, in priority order
    };

    Choice Resolve(const Request& request);
}

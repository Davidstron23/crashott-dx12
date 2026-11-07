#include "renderer_choice.h"

namespace gpu::choice
{
    const char* SourceLabel(Source source)
    {
        switch (source)
        {
        case Source::Argument: return "--renderer";
        case Source::Environment: return "CRASH_RENDERER";
        case Source::SettingsOverride: return "CRASH_SETTINGS";
        case Source::SettingsFile: return "settings.toml";
        default: return "default";
        }
    }

    Choice Resolve(const Request& request)
    {
        Choice choice;
        settings::Renderer renderer;
        // Only the environment and the default so far: the argument and the settings come with the
        // other sources (spec 001, task 4), in the order of the header.
        if (request.environment && request.environment[0])
        {
            if (settings::ParseRenderer(request.environment, renderer))
            {
                choice.renderer = renderer;
                choice.source = Source::Environment;
                return choice;
            }
            choice.ignored.push_back(std::string("[gpu] CRASH_RENDERER \"") + request.environment +
                                     "\" is not a renderer (vulkan | d3d12): ignored");
        }
        return choice;
    }
}

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
        // The sources in the order of the header; a value that isn't a renderer is noted and skipped.
        if (request.argumentGiven)
        {
            if (!request.argument)
                choice.ignored.push_back("[gpu] --renderer has no value: ignored");
            else if (settings::ParseRenderer(request.argument, renderer))
            {
                choice.renderer = renderer;
                choice.source = Source::Argument;
                return choice;
            }
            else
                choice.ignored.push_back(std::string("[gpu] --renderer \"") + request.argument +
                                         "\" is not a renderer (vulkan | d3d12): ignored");
        }
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
        if (request.settings != settings::RendererOrigin::None)
        {
            choice.renderer = request.settingsRenderer;
            choice.source = request.settings == settings::RendererOrigin::Override ? Source::SettingsOverride : Source::SettingsFile;
        }
        return choice;
    }
}

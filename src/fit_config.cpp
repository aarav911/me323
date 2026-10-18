#include "fit_config.h"

#include <cmath>
#include <fstream>

bool save_configuration(
    const FitConfiguration& config, const std::string& path, std::string& error)
{
    if (config.mode == FitMode::Stretch && !config.has_stretch())
    {
        error = "Stretch mode needs three positive stretch factors.";
        return false;
    }
    std::ofstream file(path);
    if (!file)
    {
        error = "Could not open " + path + " for writing.";
        return false;
    }

    file.precision(17);
    if (config.mode == FitMode::Zoom)
    {
        file << "ME323_FIT_CONFIG 1\n"
             << "head_scale " << config.head_scale << '\n'
             << "helmet_scale " << config.helmet_scale << '\n'
             << "gap " << config.gap << '\n'
             << "head_translation " << config.head_translation.x() << ' '
             << config.head_translation.y() << ' ' << config.head_translation.z() << '\n'
             << "helmet_translation " << config.helmet_translation.x() << ' '
             << config.helmet_translation.y() << ' ' << config.helmet_translation.z() << '\n';
    }
    else
    {
        file << "ME323_FIT_CONFIG 2\n"
             << "fit_mode stretch\n"
             << "head_scale " << config.head_scale << '\n'
             << "helmet_scale " << config.helmet_scale << '\n'
             << "gap " << config.gap << '\n'
             << "head_translation " << config.head_translation.x() << ' '
             << config.head_translation.y() << ' ' << config.head_translation.z() << '\n'
             << "helmet_translation " << config.helmet_translation.x() << ' '
             << config.helmet_translation.y() << ' ' << config.helmet_translation.z() << '\n'
             << "helmet_stretch " << config.helmet_stretch.x() << ' '
             << config.helmet_stretch.y() << ' ' << config.helmet_stretch.z() << '\n'
             << "helmet_pitch " << config.helmet_pitch_degrees << '\n'
             << "helmet_offset " << config.helmet_offset.x() << ' '
             << config.helmet_offset.y() << ' ' << config.helmet_offset.z() << '\n';
    }
    if (!file)
    {
        error = "Could not write " + path + ".";
        return false;
    }
    return true;
}

bool load_configuration(
    FitConfiguration& config, const std::string& path, std::string& error)
{
    std::ifstream file(path);
    if (!file)
    {
        error = "Could not open " + path + ".";
        return false;
    }

    std::string key;
    std::string header;
    int version = 0;
    FitConfiguration loaded;
    if (!(file >> header >> version) || header != "ME323_FIT_CONFIG" ||
        (version != 1 && version != 2))
    {
        error = path + " is not an ME323 fit configuration (version 1 or 2).";
        return false;
    }
    if (version == 2)
    {
        std::string mode;
        if (!(file >> key >> mode) || key != "fit_mode" ||
            (mode != "stretch" && mode != "zoom"))
        {
            error = path + " is invalid or incomplete (fit_mode).";
            return false;
        }
        loaded.mode = mode == "stretch" ? FitMode::Stretch : FitMode::Zoom;
    }
    if (!(file >> key >> loaded.head_scale) || key != "head_scale" ||
        !(file >> key >> loaded.helmet_scale) || key != "helmet_scale" ||
        !(file >> key >> loaded.gap) || key != "gap" ||
        !(file >> key >> loaded.head_translation.x()
               >> loaded.head_translation.y() >> loaded.head_translation.z()) ||
        key != "head_translation" ||
        !(file >> key >> loaded.helmet_translation.x()
               >> loaded.helmet_translation.y() >> loaded.helmet_translation.z()) ||
        key != "helmet_translation")
    {
        error = path + " is invalid or incomplete.";
        return false;
    }
    if (version == 2 &&
        (!(file >> key >> loaded.helmet_stretch.x()
                >> loaded.helmet_stretch.y() >> loaded.helmet_stretch.z()) ||
         key != "helmet_stretch" ||
         !(file >> key >> loaded.helmet_pitch_degrees) || key != "helmet_pitch" ||
         !(file >> key >> loaded.helmet_offset.x()
                >> loaded.helmet_offset.y() >> loaded.helmet_offset.z()) ||
         key != "helmet_offset"))
    {
        error = path + " is invalid or incomplete (stretch fields).";
        return false;
    }

    std::string extra;
    if (file >> extra)
    {
        error = path + " contains unexpected extra data.";
        return false;
    }
    if (!std::isfinite(loaded.head_scale) || loaded.head_scale < 0.1f ||
        loaded.head_scale > 3.0f || !std::isfinite(loaded.helmet_scale) ||
        loaded.helmet_scale < 0.1f || loaded.helmet_scale > 3.0f ||
        !std::isfinite(loaded.gap) || loaded.gap < 0.0 ||
        !loaded.head_translation.allFinite() ||
        !loaded.helmet_translation.allFinite())
    {
        error = path + " contains values outside the supported range.";
        return false;
    }
    if (version == 2 &&
        (!loaded.helmet_stretch.allFinite() || !loaded.helmet_offset.allFinite() ||
         !std::isfinite(loaded.helmet_pitch_degrees) ||
         std::abs(loaded.helmet_pitch_degrees) > 90.0 ||
         (loaded.mode == FitMode::Stretch && !loaded.has_stretch())))
    {
        error = path + " contains stretch values outside the supported range.";
        return false;
    }

    config = loaded;
    return true;
}

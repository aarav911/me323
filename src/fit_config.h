#pragma once

#include <Eigen/Core>

#include <string>

// How the stock helmet is fitted around the head.
//   Zoom    (the original): one scale for all three axes, chosen so the helmet
//           is 2 x gap taller than the head, its top `gap` above the head's
//           top, then moved by helmet_translation.
//   Stretch (step 3, option 2): the helmet, centred on its own bounding box,
//           is stretched along its own axes by helmet_stretch (absolute factors
//           from the helmet file's units to the head's), tilted about the
//           side-to-side axis by helmet_pitch_degrees (+ = front up), and its
//           box centre put at the head's box centre + helmet_offset. The values
//           come from analysis_tools/stretch_fit_study.py --write-config.
enum class FitMode
{
    Zoom,
    Stretch
};

struct FitConfiguration
{
    float head_scale = 1.0f;
    float helmet_scale = 1.0f;
    double gap = 0.005;
    Eigen::Vector3d head_translation = Eigen::Vector3d::Zero();
    Eigen::Vector3d helmet_translation = Eigen::Vector3d::Zero();

    FitMode mode = FitMode::Zoom;
    Eigen::Vector3d helmet_stretch = Eigen::Vector3d::Zero();   // zero = not set
    double helmet_pitch_degrees = 0.0;
    Eigen::Vector3d helmet_offset = Eigen::Vector3d::Zero();

    bool has_stretch() const
    {
        return (helmet_stretch.array() > 0.0).all() && helmet_stretch.allFinite();
    }
};

// File format. Zoom mode is written as version 1, byte for byte what earlier
// builds write and read; stretch mode needs version 2, which older builds
// reject. load_configuration() reads both.
//
//   ME323_FIT_CONFIG 2
//   fit_mode stretch|zoom
//   head_scale <s>
//   helmet_scale <s>
//   gap <g>
//   head_translation <x> <y> <z>
//   helmet_translation <x> <y> <z>
//   helmet_stretch <sx> <sy> <sz>
//   helmet_pitch <degrees>
//   helmet_offset <x> <y> <z>
bool save_configuration(
    const FitConfiguration& config, const std::string& path, std::string& error);
bool load_configuration(
    FitConfiguration& config, const std::string& path, std::string& error);
